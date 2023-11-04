import gc

import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
from torch.nn.utils.rnn import pad_sequence

from deepkin.models.modules import PositionEncoding, ClassificationHead, TokenClassificationHead, MorphoEncoder, MorphoMLMPredictor, BaseConfig
from deepkin.models.kb_transformers import TransformerEncoderLayer, TransformerEncoder, init_bert_params

class KinyaBERTEncoder(nn.Module):
    def __init__(self, args, cfg:BaseConfig):
        super(KinyaBERTEncoder, self).__init__()
        self.morpho_encoder = MorphoEncoder(args,cfg)
        self.stem_embedding = nn.Embedding(cfg.tot_num_stems, args.stem_dim_hidden, padding_idx=0)
        self.hidden_dim = (args.morpho_dim_hidden * 4) + args.stem_dim_hidden # 128 x 4 + 256 = 768
        self.num_heads = args.main_sequence_encoder_num_heads
        self.pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            args.main_sequence_encoder_max_seq_len,
                                            args.main_sequence_encoder_rel_pos_bins,
                                            args.main_sequence_encoder_max_rel_pos,
                                            True)
        encoder_layers = TransformerEncoderLayer(self.hidden_dim, args.main_sequence_encoder_num_heads,
                                                 dim_feedforward=args.main_sequence_encoder_dim_ffn, dropout=args.main_sequence_encoder_dropout,
                                                 activation="gelu")
        self.main_sequence_encoder = TransformerEncoder(encoder_layers, args.main_sequence_encoder_num_layers)
        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, lm_morphs, pos_tags, stems, input_sequence_lengths,
                afx_padded, m_masks_padded, masks_padded):
        morpho_input = self.morpho_encoder(stems, lm_morphs, pos_tags, afx_padded, m_masks_padded) # [4, L, E1]
        stem_input = self.stem_embedding(stems) # [L, E2]

        morpho_input = morpho_input.permute(1, 0, 2) # ==> [L, 4, E1]
        L = morpho_input.size(0)
        morpho_input = morpho_input.contiguous().view(L, -1)  # (L, 4E1)

        input_sequences = torch.cat((morpho_input, stem_input), 1) # [L, E'=4E1+E2]

        lists = input_sequences.split(input_sequence_lengths, 0) # len(input_sequence_lengths)
        tr_padded = pad_sequence(lists, batch_first=False) # [B, L, E], here new E=4E1+E2 and L=actual length of sequence

        abs_pos_bias = self.pos_encoder(tr_padded)

        output = self.main_sequence_encoder(tr_padded, attn_bias = abs_pos_bias, src_key_padding_mask = masks_padded) # Shape: L x N x E, with L = max sequence length
        return output

class KinyaBERT_PretrainModel(nn.Module):
    def __init__(self, args, cfg:BaseConfig):
        super(KinyaBERT_PretrainModel, self).__init__()
        self.encoder = KinyaBERTEncoder(args, cfg)
        self.predictor = MorphoMLMPredictor(self.encoder.stem_embedding.weight,
                                            self.encoder.morpho_encoder.pos_tag_embedding.weight,
                                            self.encoder.morpho_encoder.lm_morph_one_embedding.weight,
                                            self.encoder.morpho_encoder.affixes_embedding.weight,
                                            self.encoder.hidden_dim,
                                            args.layernorm_epsilon)
    @custom_fwd
    def forward(self, lm_morphs, pos_tags, stems, input_sequence_lengths,
                predicted_tokens_idx,
                predicted_tokens_affixes_idx,
                predicted_stems,
                predicted_pos_tags,
                predicted_lm_morphs,
                predicted_affixes_prob, afx_padded, m_masks_padded, masks_padded):
        tr_hidden_state = self.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                       afx_padded, m_masks_padded, masks_padded)
        return self.predictor(tr_hidden_state,
                              predicted_tokens_idx, predicted_tokens_affixes_idx, predicted_stems, predicted_pos_tags,
                              predicted_lm_morphs,
                              predicted_affixes_prob)

    def predict(self, lm_morphs, pos_tags, affixes, tokens_lengths, stems,
                input_sequence_lengths,
                seq_predicted_token_idx,
                max_predict_affixes,
                max_top_predictions=10):

        # Needed to fix decoding start bug
        # if len(affixes) == 0:
        #     affixes.append(0)
        #     tokens_lengths[-1] = 1
        afx = affixes.split(tokens_lengths)
        # [[2,4,5], [6,7]]
        afx_padded = pad_sequence(afx, batch_first=False)
        afx_padded = afx_padded.to(stems.device, dtype=torch.long)
        # afx_padded: (M,L), M: max morphological length
        # m_masks_padded = None
        # if afx_padded.nelement() > 0:
        m_masks = [torch.zeros((x + 4), dtype=torch.bool, device=stems.device) for x in tokens_lengths]
        m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M)

        masks = [torch.zeros(x, dtype=torch.bool, device=stems.device) for x in input_sequence_lengths]
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=1)  # Shape: N x S

        tr_hidden_state = self.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                afx_padded, m_masks_padded, masks_padded)
        ((stem_predictions,
          pos_tag_predictions,
          lm_morph_predictions,
          affixes_predictions),
         (stem_predictions_prob,
          pos_tag_predictions_prob,
          lm_morph_predictions_prob,
          affixes_predictions_prob)) =  self.predictor.predict(tr_hidden_state,
                                                                seq_predicted_token_idx,
                                                                max_predict_affixes,
                                                               max_top_predictions=max_top_predictions)
        return ((stem_predictions,
                 pos_tag_predictions,
                 lm_morph_predictions,
                 affixes_predictions),
               (stem_predictions_prob,
                pos_tag_predictions_prob,
                lm_morph_predictions_prob,
                affixes_predictions_prob))


class KinyaBERT_SequenceClassifier(nn.Module):

    def __init__(self, args, cfg:BaseConfig, num_classes: int, encoder:KinyaBERTEncoder= None):
        super(KinyaBERT_SequenceClassifier, self).__init__()
        self.encoder_fine_tune = args.encoder_fine_tune
        self.hidden_dim = (args.morpho_dim_hidden * 4) + args.stem_dim_hidden  # 128 x 4 + 256 = 768
        if (encoder is not None):
           self.encoder = encoder
        else:
            if self.encoder_fine_tune:
                self.encoder = KinyaBERTEncoder(args, cfg)
        self.cls_head = ClassificationHead(self.hidden_dim, num_classes * 8, num_classes,
                                           pooler_dropout=args.pooler_dropout, head_trunk=args.head_trunk)

    @custom_fwd
    def forward(self, lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths, shared_encoder=None):
        # Needed to fix decoding start bug
        # if len(affixes) == 0:
        #     affixes.append(0)
        #     tokens_lengths[-1] = 1
        afx = affixes.split(tokens_lengths)
        # [[2,4,5], [6,7]]
        afx_padded = pad_sequence(afx, batch_first=False)
        afx_padded = afx_padded.to(stems.device, dtype=torch.long)
        # afx_padded: (M,L), M: max morphological length
        # m_masks_padded = None
        # if afx_padded.nelement() > 0:
        m_masks = [torch.zeros((x + 4), dtype=torch.bool, device=stems.device) for x in tokens_lengths]
        m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M)

        masks = [torch.zeros(x, dtype=torch.bool, device=stems.device) for x in input_sequence_lengths]
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=1)  # Shape: N x S

        if shared_encoder is not None:
            if self.encoder_fine_tune:
                tr_hidden_state = shared_encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                 afx_padded, m_masks_padded, masks_padded)
            else:
                shared_encoder.eval()
                with torch.no_grad():
                    tr_hidden_state = shared_encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                     afx_padded, m_masks_padded, masks_padded)
        else:
            if not self.encoder_fine_tune:
                raise RuntimeError("No shared encoder provided when encoder_fine_tune=True")
            tr_hidden_state = self.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                           afx_padded, m_masks_padded, masks_padded)
        cls_scores = self.cls_head(tr_hidden_state)
        return cls_scores

class KinyaBERT_SequenceTagger(nn.Module):

    def __init__(self, args, cfg:BaseConfig, num_classes: int, encoder:KinyaBERTEncoder= None):
        super(KinyaBERT_SequenceTagger, self).__init__()
        self.encoder_fine_tune = args.encoder_fine_tune
        self.hidden_dim = (args.morpho_dim_hidden * 4) + args.stem_dim_hidden  # 128 x 4 + 256 = 768
        if (encoder is not None):
           self.encoder = encoder
        else:
            if self.encoder_fine_tune:
                self.encoder = KinyaBERTEncoder(args, cfg)
        self.tagger_head = TokenClassificationHead(self.hidden_dim, num_classes * 8, num_classes,
                                                   pooler_dropout=args.pooler_dropout, head_trunk=args.head_trunk)

    @custom_fwd
    def forward(self, lm_morphs, pos_tags, affixes, old_tokens_lengths, stems, input_sequence_lengths, shared_encoder=None):
        # Needed to fix decoding start bug
        assert affixes.nelement() == sum(old_tokens_lengths), "Mismatch token lengths affixes={} vs lengths={}".format(affixes.nelement(), sum(old_tokens_lengths))
        if (affixes.nelement() == 0) and (sum(old_tokens_lengths)==0) and (len(old_tokens_lengths) > 0):
            affixes = torch.tensor([0], dtype=torch.long, device=stems.device)
            tokens_lengths = [x for x in old_tokens_lengths]
            tokens_lengths[-1] = 1
        else:
            tokens_lengths = old_tokens_lengths
        afx = affixes.split(tokens_lengths)
        # [[2,4,5], [6,7]]
        afx_padded = pad_sequence(afx, batch_first=False)
        afx_padded = afx_padded.to(stems.device, dtype=torch.long)
        # afx_padded: (M,L), M: max morphological length
        # m_masks_padded = None
        # if afx_padded.nelement() > 0:
        m_masks = [torch.zeros((x + 4), dtype=torch.bool, device=stems.device) for x in tokens_lengths]
        m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M)

        masks = [torch.zeros(x, dtype=torch.bool, device=stems.device) for x in input_sequence_lengths]
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=1)  # Shape: N x S

        if shared_encoder is not None:
            if self.encoder_fine_tune:
                tr_hidden_state = shared_encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                 afx_padded, m_masks_padded, masks_padded)
            else:
                shared_encoder.eval()
                with torch.no_grad():
                    tr_hidden_state = shared_encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                     afx_padded, m_masks_padded, masks_padded)
        else:
            if not self.encoder_fine_tune:
                raise RuntimeError("No shared encoder provided when encoder_fine_tune=True")
            tr_hidden_state = self.encoder(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                           afx_padded, m_masks_padded, masks_padded)

        cls_scores = self.tagger_head(tr_hidden_state, input_sequence_lengths)
        return cls_scores

def KinyaBERT_from_pretrained(device, args, cfg: BaseConfig, pretrained_model_file) -> KinyaBERT_PretrainModel:
    pretrained_model = KinyaBERT_PretrainModel(args, cfg).to(device)
    kb_state_dict = torch.load(pretrained_model_file, map_location=device)
    pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()
    return pretrained_model

def KinyaBERT_large_from_pretrained(device, args, cfg: BaseConfig, pretrained_model_file) -> KinyaBERT_PretrainModel:
    # Store original config
    morpho_dim_hidden = args.morpho_dim_hidden
    stem_dim_hidden = args.stem_dim_hidden
    morpho_max_token_len = args.morpho_max_token_len
    morpho_rel_pos_bins = args.morpho_rel_pos_bins
    morpho_max_rel_pos = args.morpho_max_rel_pos
    main_sequence_encoder_max_seq_len = args.main_sequence_encoder_max_seq_len
    main_sequence_encoder_rel_pos_bins = args.main_sequence_encoder_rel_pos_bins
    main_sequence_encoder_max_rel_pos = args.main_sequence_encoder_max_rel_pos
    dataset_max_seq_len = args.dataset_max_seq_len
    morpho_dim_ffn = args.morpho_dim_ffn
    main_sequence_encoder_dim_ffn = args.main_sequence_encoder_dim_ffn
    morpho_num_heads = args.morpho_num_heads
    main_sequence_encoder_num_heads = args.main_sequence_encoder_num_heads
    morpho_num_layers = args.morpho_num_layers
    main_sequence_encoder_num_layers = args.main_sequence_encoder_num_layers

    # Use KinyaBERT large config
    args.morpho_dim_hidden = 160
    args.stem_dim_hidden = 320
    args.morpho_max_token_len = 24
    args.morpho_rel_pos_bins = 24
    args.morpho_max_rel_pos = 24
    args.main_sequence_encoder_max_seq_len = 512
    args.main_sequence_encoder_rel_pos_bins = 256
    args.main_sequence_encoder_max_rel_pos = 256
    args.dataset_max_seq_len = 512
    args.morpho_dim_ffn = 640
    args.main_sequence_encoder_dim_ffn = 3840
    args.morpho_num_heads = 4
    args.main_sequence_encoder_num_heads = 12
    args.morpho_num_layers = 6
    args.main_sequence_encoder_num_layers = 16

    pretrained_model = KinyaBERT_PretrainModel(args, cfg).to(device)
    kb_state_dict = torch.load(pretrained_model_file, map_location=device)
    pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()

    # Restore original config
    args.morpho_dim_hidden = morpho_dim_hidden
    args.stem_dim_hidden = stem_dim_hidden
    args.morpho_max_token_len = morpho_max_token_len
    args.morpho_rel_pos_bins = morpho_rel_pos_bins
    args.morpho_max_rel_pos = morpho_max_rel_pos
    args.main_sequence_encoder_max_seq_len = main_sequence_encoder_max_seq_len
    args.main_sequence_encoder_rel_pos_bins = main_sequence_encoder_rel_pos_bins
    args.main_sequence_encoder_max_rel_pos = main_sequence_encoder_max_rel_pos
    args.dataset_max_seq_len = dataset_max_seq_len
    args.morpho_dim_ffn = morpho_dim_ffn
    args.main_sequence_encoder_dim_ffn = main_sequence_encoder_dim_ffn
    args.morpho_num_heads = morpho_num_heads
    args.main_sequence_encoder_num_heads = main_sequence_encoder_num_heads
    args.morpho_num_layers = morpho_num_layers
    args.main_sequence_encoder_num_layers = main_sequence_encoder_num_layers

    return pretrained_model

def KinyaBERT_SequenceClassifier_from_pretrained(num_classes, device, args, cfg: BaseConfig, pretrained_model_file, pre_trained_model=None) -> KinyaBERT_SequenceClassifier:
    classifier_model = KinyaBERT_SequenceClassifier(args, cfg, num_classes).to(device)
    if pre_trained_model is not None:
        classifier_model.encoder.load_state_dict(pre_trained_model.encoder.state_dict())
        gc.collect()
        return classifier_model

    if args.encoder_fine_tune:
        kb_state_dict = torch.load(pretrained_model_file, map_location=device)
        pretrained_model = KinyaBERT_PretrainModel(args, cfg).to(device)
        pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])
        classifier_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        for ln in range(args.ft_reinit_layers):
            classifier_model.encoder.main_sequence_encoder.layers[-(ln+1)].apply(init_bert_params)
        del kb_state_dict
        del pretrained_model
        gc.collect()
    return classifier_model

def KinyaBERT_SequenceTagger_from_pretrained(num_classes, device, args, cfg: BaseConfig, pretrained_model_file) -> KinyaBERT_SequenceTagger:
    tagger_model = KinyaBERT_SequenceTagger(args, cfg, num_classes).to(device)
    if args.encoder_fine_tune:
        kb_state_dict = torch.load(pretrained_model_file, map_location=device)
        pretrained_model = KinyaBERT_PretrainModel(args, cfg).to(device)
        pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])
        tagger_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        for ln in range(args.ft_reinit_layers):
            tagger_model.encoder.main_sequence_encoder.layers[-(ln+1)].apply(init_bert_params)
        del kb_state_dict
        del pretrained_model
        gc.collect()
    return tagger_model
