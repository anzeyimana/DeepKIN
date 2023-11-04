import gc
import math
from itertools import accumulate

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import custom_fwd
from torch.nn.utils.rnn import pad_sequence

from deepkin.clib.libkinlp.kinlpy import parse_text_to_morpho_sentence, ParsedMorphoSentence, BOS_ID, EOS_ID
from deepkin.models.kb_transformers import TransformerDecoderLayer, TransformerDecoder, init_bert_params
from deepkin.models.modules import PositionEncoding, MorphoEncoder, MorphoGPTPredictor, BaseConfig, \
    GPTClassificationHead


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class KinyaGPTEncoder(nn.Module):
    def __init__(self, args, cfg:BaseConfig):
        super(KinyaGPTEncoder, self).__init__()
        self.morpho_encoder = MorphoEncoder(args,cfg)
        self.stem_embedding = nn.Embedding(cfg.tot_num_stems, args.stem_dim_hidden, padding_idx=0)
        self.hidden_dim = (args.morpho_dim_hidden * 4) + args.stem_dim_hidden # 128 x 4 + 256 = 768
        self.num_heads = args.main_sequence_encoder_num_heads

        self.pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            False)
        decoder_layer = TransformerDecoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                          dim_feedforward=args.main_sequence_encoder_dim_ffn,
                                                          dropout=args.main_sequence_encoder_dropout, activation="gelu")
        self.main_sequence_encoder = TransformerDecoder(decoder_layer, args.main_sequence_encoder_num_layers)

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, lm_morphs, pos_tags, stems, input_sequence_lengths,
                afx_padded, m_masks_padded, input_masks_padded, decoder_mask,
                input_with_eos=True):
        device = stems.device
        morpho_input = self.morpho_encoder(stems, lm_morphs, pos_tags, afx_padded, m_masks_padded) # [4, L, E1]
        stem_input = self.stem_embedding(stems) # [L, E2]

        morpho_input = morpho_input.permute(1, 0, 2) # ==> [L, 4, E1]
        L = morpho_input.size(0)
        morpho_input = morpho_input.contiguous().view(L, -1)  # (L, 4E1)

        input_sequences = torch.cat((morpho_input, stem_input), 1) # [L, E'=4E1+E2]

        lists = input_sequences.split(input_sequence_lengths, 0) # len(input_sequence_lengths)
        dec_input = pad_sequence(lists, batch_first=False)

        abs_pos_bias = self.pos_encoder(dec_input)

        # in, out: ->shape: L x N x E, with L = max sequence length
        transformer_output = self.main_sequence_encoder(dec_input, tgt_mask=decoder_mask, tgt_attn_bias=abs_pos_bias, tgt_key_padding_mask=input_masks_padded) # Shape: L x N x E, with L = max sequence length

        return transformer_output # (L,N,E)


class KinyaGPT(nn.Module):
    def __init__(self, args, cfg:BaseConfig):
        super(KinyaGPT, self).__init__()
        self.encoder = KinyaGPTEncoder(args, cfg)
        self.predictor = MorphoGPTPredictor(self.encoder.stem_embedding.weight,
                                            self.encoder.morpho_encoder.pos_tag_embedding.weight,
                                            self.encoder.morpho_encoder.lm_morph_one_embedding.weight,
                                            self.encoder.morpho_encoder.affixes_embedding.weight,
                                            self.encoder.hidden_dim,
                                            args.layernorm_epsilon)
    @custom_fwd
    def forward(self, lm_morphs, pos_tags, tokens_lengths, stems, input_sequence_lengths, affixes_prob,
                afx_padded, m_masks_padded, input_masks_padded, decoder_mask):
        tr_hidden_state = self.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                       afx_padded, m_masks_padded, input_masks_padded, decoder_mask,
                                       input_with_eos=True)
        # Returns hidden_state (L,N,E) and (stem_loss,afset_loss) or (next_stem,next_afset)
        return self.predictor(tr_hidden_state, input_sequence_lengths, stems, pos_tags, lm_morphs, affixes_prob, tokens_lengths)

    def predict(self, lm_morphs, pos_tags, stems, input_sequence_lengths,
                afx_padded, m_masks_padded, input_masks_padded, decoder_mask):
        tr_hidden_state = self.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                       afx_padded, m_masks_padded, input_masks_padded, decoder_mask,
                                       input_with_eos=False)
        (next_stems, next_pos_tags, next_lm_morphs, next_affixes) = self.predictor.predict(tr_hidden_state, input_sequence_lengths)
        return tr_hidden_state, (next_stems, next_pos_tags, next_lm_morphs, next_affixes)

    def score_last_token_log10(self, ffi, lib, txt, cfg, device) -> float:
        sentence = parse_text_to_morpho_sentence(ffi, lib, txt)

        (in_afsets,
         in_pos_tags,
         in_affixes,
         in_tokens_lengths,
         in_stems) = prepare_score_input(sentence, omit_last=True)

        (in_afsets,
         in_pos_tags,
         in_affixes,
         in_tokens_lengths,
         in_stems) = ([BOS_ID] + in_afsets,
                      [BOS_ID] + in_pos_tags,
                      in_affixes,
                      [0] + in_tokens_lengths,
                      [BOS_ID] + in_stems)

        (lm_morphs, pos_tags, tokens_lengths, stems,
         input_sequence_lengths, affixes_prob,
         afx_padded, m_masks_padded, input_masks_padded,
         decoder_mask) = create_score_batch(in_afsets, in_pos_tags, in_affixes, in_tokens_lengths, in_stems, cfg, device)

        (tr_hidden_state,
         (next_stems,
          next_pos_tags,
          next_lm_morphs,
          next_affixes)) = self.predict(lm_morphs, pos_tags, stems,
                                        input_sequence_lengths, afx_padded, m_masks_padded,
                                        input_masks_padded, decoder_mask)

        (out_afsets,
         out_pos_tags,
         out_affixes,
         out_tokens_lengths,
         out_stems) = prepare_score_output(sentence)

        log_score = compute_prediction_score(next_stems, next_pos_tags, next_lm_morphs, next_affixes,
                                             out_afsets, out_pos_tags, out_affixes, out_stems)
        return log_score/math.log(10)

    def score_all_tokens_log10(self, ffi, lib, txt, cfg, device) -> float:
        sentence = parse_text_to_morpho_sentence(ffi, lib, txt)

        (in_afsets,
         in_pos_tags,
         in_affixes,
         in_tokens_lengths,
         in_stems) = prepare_score_input(sentence, omit_last=False)

        (in_afsets,
         in_pos_tags,
         in_affixes,
         in_tokens_lengths,
         in_stems) = ([BOS_ID] + in_afsets + [EOS_ID],
                      [BOS_ID] + in_pos_tags + [EOS_ID],
                      in_affixes,
                      [0] + in_tokens_lengths + [0],
                      [BOS_ID] + in_stems + [EOS_ID])

        (lm_morphs, pos_tags, tokens_lengths, stems,
         input_sequence_lengths, affixes_prob,
         afx_padded, m_masks_padded, input_masks_padded,
         decoder_mask) = create_score_batch(in_afsets, in_pos_tags, in_affixes, in_tokens_lengths, in_stems, cfg, device)

        losses, nll_losses = self(lm_morphs, pos_tags, tokens_lengths, stems, input_sequence_lengths, affixes_prob,
                                  afx_padded, m_masks_padded, input_masks_padded, decoder_mask)

        log_score = -(sum([l.cpu().item() for l in nll_losses]) / len(nll_losses))
        return log_score/math.log(10)

class KinyaGPT_SequenceClassifier(nn.Module):

    def __init__(self, args, cfg:BaseConfig, num_classes: int, encoder:KinyaGPTEncoder= None):
        super(KinyaGPT_SequenceClassifier, self).__init__()
        self.encoder_fine_tune = args.encoder_fine_tune
        self.hidden_dim = (args.morpho_dim_hidden * 4) + args.stem_dim_hidden  # 128 x 4 + 256 = 768
        if (encoder is not None):
           self.encoder = encoder
        else:
            if self.encoder_fine_tune:
                self.encoder = KinyaGPTEncoder(args, cfg)
        self.cls_head = GPTClassificationHead(self.hidden_dim, num_classes * 8, num_classes,
                                              pooler_dropout=args.pooler_dropout, head_trunk=args.head_trunk)

    @custom_fwd
    def forward(self, lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths, shared_encoder=None):
        device = stems.device
        afx = affixes.split(tokens_lengths)
        # [[2,4,5], [6,7]]
        afx_padded = pad_sequence(afx, batch_first=False)
        afx_padded = afx_padded.to(device, dtype=torch.long)
        # afx_padded: (M,L), M: max morphological length
        # m_masks_padded = None
        # if afx_padded.nelement() > 0:
        m_masks = [torch.zeros((x + 4), dtype=torch.bool, device=stems.device) for x in tokens_lengths]
        m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M)

        input_with_eos = True
        seq_len = max(input_sequence_lengths)
        # Using length: length-1 sot that last <EOS> token doesn't get processed
        input_masks = [torch.zeros(length, dtype=torch.bool, device=stems.device) for length in input_sequence_lengths]
        if input_with_eos:
            for i in range(len(input_masks)):
                input_masks[i][-1] = True
        input_masks_padded = pad_sequence(input_masks, batch_first=True, padding_value=1)  # Shape: N x S

        decoder_mask = generate_square_subsequent_mask(seq_len, stems.device)
        tr_hidden_state = self.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                       afx_padded, m_masks_padded, input_masks_padded, decoder_mask,
                                       input_with_eos=True) #(L,N,E)
        cls_scores = self.cls_head(tr_hidden_state, input_sequence_lengths)
        return cls_scores #(N,|Classes|)

def KinyaGPT_from_pretrained(args, cfg:BaseConfig, pretrained_model_file):
    gpt_model = KinyaGPT(args,cfg)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    kb_state_dict = torch.load(pretrained_model_file, map_location=map_location)
    gpt_model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()
    return gpt_model

def KinyaGPT_SequenceClassifier_from_pretrained(num_classes, device, args, cfg: BaseConfig, pretrained_model_file) -> KinyaGPT_SequenceClassifier:
    classifier_model = KinyaGPT_SequenceClassifier(args, cfg, num_classes).to(device)
    if args.encoder_fine_tune:
        pretrained_model = KinyaGPT_from_pretrained(args, cfg, pretrained_model_file)
        classifier_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        del pretrained_model
        gc.collect()
    return classifier_model


def compute_prediction_score(next_stems, next_pos_tags, next_lm_morphs, next_affixes,
                             out_afsets, out_pos_tags, out_affixes, out_stems) -> float:
    stem_score = math.log(next_stems[0,out_stems[0]].cpu().item())
    pos_score = math.log(next_pos_tags[0,out_pos_tags[0]].cpu().item())
    afset_score = math.log(next_lm_morphs[0,out_afsets[0]].cpu().item())
    affix_score = sum([stem_score, pos_score, afset_score]) / 3.0
    affix_score_list = next_affixes[0,out_affixes].cpu().tolist()
    if (len(affix_score_list) > 0):
        affix_score = sum(affix_score_list) / len(affix_score_list)
    return sum([stem_score, pos_score, afset_score, affix_score]) / 4.0

def prepare_score_input(sentence: ParsedMorphoSentence, omit_last=True):
    lm_morphs = []
    pos_tags = []
    stems = []
    affixes = []
    tokens_lengths = []

    tokens = sentence.tokens
    if omit_last:
        tokens = sentence.tokens[:-1]
    for token in tokens:
        lm_morphs.append(token.lm_morph_id)
        pos_tags.append(token.pos_tag_id)
        stems.append(token.stem_id)
        affixes.extend(token.affixes)
        tokens_lengths.append(len(token.affixes))
        for tid in token.extra_tokens_ids:
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append(tid)
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))

    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems)

def prepare_score_output(sentence: ParsedMorphoSentence):
    lm_morphs = []
    pos_tags = []
    stems = []
    affixes = []
    tokens_lengths = []

    token = sentence.tokens[-1]
    lm_morphs.append(token.lm_morph_id)
    pos_tags.append(token.pos_tag_id)
    stems.append(token.stem_id)
    affixes.extend(token.affixes)
    tokens_lengths.append(len(token.affixes))
    for tid in token.extra_tokens_ids:
        lm_morphs.append(token.lm_morph_id)
        pos_tags.append(token.pos_tag_id)
        stems.append(tid)
        affixes.extend(token.affixes)
        tokens_lengths.append(len(token.affixes))

    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems)

def create_score_batch(afsets, pos_tags, affixes, tokens_lengths, stems, cfg, device):
    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []

    batch_input_sequence_lengths = []

    batch_lm_morphs.extend(afsets)
    batch_pos_tags.extend(pos_tags)
    batch_affixes.extend(affixes)
    batch_tokens_lengths.extend(tokens_lengths)
    batch_stems.extend(stems)

    batch_input_sequence_lengths.append(len(tokens_lengths))

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes)#.to(device)
    stems = torch.tensor(batch_stems).to(device)

    pred_affixes_list = [batch_affixes[x - y: x] for x, y in zip(accumulate(batch_tokens_lengths), batch_tokens_lengths)]
    afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
    for i,lst in enumerate(pred_affixes_list):
        if (len(lst) > 0):
            afx_prob[i,lst] = 1.0
    affixes_prob = afx_prob.to(device, dtype=torch.float)

    if sum(tokens_lengths) > 0:
        afx = affixes.split(tokens_lengths)
        # [[2,4,5], [6,7]]
        afx_padded = pad_sequence(afx, batch_first=False)
        afx_padded = afx_padded.to(device)
    else:
        afx_padded = torch.tensor([]).to(device)
    # afx_padded: (M,L), M: max morphological length
    m_masks_padded = None
    if afx_padded.nelement() > 0:
        m_masks = [torch.zeros((x + 4), dtype=torch.bool, device=stems.device) for x in tokens_lengths]
        m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M)

    input_with_eos = True
    seq_len = max(input_sequence_lengths)
    # Using length: length-1 sot that last <EOS> token doesn't get processed
    input_masks = [torch.zeros(length, dtype=torch.bool, device=stems.device) for length in input_sequence_lengths]
    if input_with_eos:
        for i in range(len(input_masks)):
            input_masks[i][-1] = True
    input_masks_padded = pad_sequence(input_masks, batch_first=True, padding_value=1)  # Shape: N x S

    decoder_mask = generate_square_subsequent_mask(seq_len, stems.device)

    data_item = (lm_morphs, pos_tags, tokens_lengths, stems, input_sequence_lengths, affixes_prob,
                 afx_padded, m_masks_padded, input_masks_padded, decoder_mask)
    return data_item