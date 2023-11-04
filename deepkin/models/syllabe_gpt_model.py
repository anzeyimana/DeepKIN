import gc

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd
from torch.nn.utils.rnn import pad_sequence

from deepkin.models.kb_transformers import TransformerDecoderLayer, TransformerDecoder, init_bert_params
from deepkin.models.util import generate_input_key_padding_mask, generate_square_subsequent_mask
from deepkin.models.modules import PositionEncoding, BaseHeadTransform, label_smoothed_nll_loss
from deepkin.models.syllabe_vocab import syllbe_vocab_size


class SyllabeGPTPredictor(nn.Module):
    def __init__(self, syllabe_embedding_weights,
                 tr_d_model,
                 layernorm_epsilon):
        super(SyllabeGPTPredictor, self).__init__()

        self.syllabe_transform = BaseHeadTransform(tr_d_model, syllabe_embedding_weights.size(1), layernorm_epsilon)
        self.syllabe_decoder = nn.Linear(syllabe_embedding_weights.size(1), syllabe_embedding_weights.size(0), bias=False)
        self.syllabe_decoder.weight = syllabe_embedding_weights
        self.syllabe_decoder_bias = nn.Parameter(torch.zeros(syllabe_embedding_weights.size(0)))

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, tr_hidden_state, input_sequence_lengths,
                syllabes,
                epsilon_ls=0.1):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # Last <EOS> token not processed
        sub = 1
        hidden_states = [token_hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_syllabes = syllabes.split(input_sequence_lengths)
        target_syllabes = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_syllabes)]
        target_syllabes = torch.cat(target_syllabes, dim=0)

        syllabe_scores = self.syllabe_transform(batch_logits)
        syllabe_scores = self.syllabe_decoder(syllabe_scores) + self.syllabe_decoder_bias
        syllabe_scores = F.log_softmax(syllabe_scores, dim=1)
        loss_avg, nll_loss_avg = label_smoothed_nll_loss(syllabe_scores, target_syllabes, epsilon_ls)
        return loss_avg, nll_loss_avg

    def batched_nll_losses(self, tr_hidden_state, input_sequence_lengths, syllabes):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # Last <EOS> token not processed
        sub = 1
        hidden_states = [token_hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_syllabes = syllabes.split(input_sequence_lengths)
        target_syllabes = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_syllabes)]
        target_syllabes = torch.cat(target_syllabes, dim=0)

        syllabe_scores = self.syllabe_transform(batch_logits)
        syllabe_scores = self.syllabe_decoder(syllabe_scores) + self.syllabe_decoder_bias
        syllabe_scores = F.log_softmax(syllabe_scores, dim=1)

        losses = -syllabe_scores.gather(dim=-1, index=target_syllabes.unsqueeze(-1))# F.nll_loss(syllabe_scores, target_syllabes, reduction='none')
        start = 0
        nll_losses = []
        for i in range(N):
            end = start + (input_sequence_lengths[i] - sub)
            nll_losses.append(losses[start:end].mean().item())
            start = end
        return nll_losses

    def predict(self, tr_hidden_state, input_sequence_lengths):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # No <EOS> at end, so no sub
        sub = 0
        hidden_states = [token_hidden_state[i, (input_sequence_lengths[i] - sub - 1):(input_sequence_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)

        syllabe_scores = self.syllabe_transform(batch_logits)
        syllabe_scores = self.syllabe_decoder(syllabe_scores) + self.syllabe_decoder_bias
        next_syllabes = F.log_softmax(syllabe_scores, dim=1)

        return next_syllabes
    def predict_syllabe_lm(self, tr_hidden_state):
        syllabe_scores = self.syllabe_transform(tr_hidden_state)
        syllabe_scores = self.syllabe_decoder(syllabe_scores) + self.syllabe_decoder_bias
        syllabe_log_probs = F.log_softmax(syllabe_scores, dim=-1)
        return syllabe_log_probs


class SyllabeGPT(nn.Module):
    def __init__(self,
                 hidden_dim = 768,
                 num_heads = 8,
                 num_layers = 6,
                 dim_ffn = 3072,
                 dropout = 0.1,
                 max_seq_len = 1024,
                 rel_pos_bins = 512,
                 max_rel_pos = 512):
        super(SyllabeGPT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_syllabe_units = syllbe_vocab_size()
        self.syllabe_embedding = nn.Embedding(self.num_syllabe_units, self.hidden_dim, padding_idx=0)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_ffn = dim_ffn
        self.layernorm_epsilon = 1e-6

        self.pos_encoder = PositionEncoding(self.hidden_dim,
                                            self.num_heads,
                                            max_seq_len,
                                            rel_pos_bins,
                                            max_rel_pos,
                                            False)
        decoder_layer = TransformerDecoderLayer(self.hidden_dim, self.num_heads,
                                                          dim_feedforward=self.dim_ffn,
                                                          dropout=dropout, activation="gelu")
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers)

        self.predictor = SyllabeGPTPredictor(self.syllabe_embedding.weight, self.hidden_dim, self.layernorm_epsilon)

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, syllabe_ids, syllabe_id_lengths,
                input_masks_padded, decoder_mask,
                input_with_eos=True):
        x = self.syllabe_embedding(syllabe_ids) # [L, E2]
        lists = x.split(syllabe_id_lengths, 0) # len(input_sequence_lengths)
        dec_input = pad_sequence(lists, batch_first=False)
        abs_pos_bias = self.pos_encoder(dec_input)
        # in, out: ->shape: L x N x E, with L = max sequence length
        tr_hidden_state = self.decoder(dec_input,
                                       tgt_mask=decoder_mask.to(x.device),
                                       tgt_attn_bias=abs_pos_bias,
                                       tgt_key_padding_mask=input_masks_padded.to(x.device)) # Shape: L x N x E, with L = max sequence length
        loss, nll_loss = self.predictor(tr_hidden_state, syllabe_id_lengths, syllabe_ids)
        return loss, nll_loss

    def batched_nll_losses(self, syllabe_ids, syllabe_id_lengths,
                input_masks_padded, decoder_mask):
        x = self.syllabe_embedding(syllabe_ids) # [L, E2]
        lists = x.split(syllabe_id_lengths, 0) # len(input_sequence_lengths)
        dec_input = pad_sequence(lists, batch_first=False)
        abs_pos_bias = self.pos_encoder(dec_input)
        # in, out: ->shape: L x N x E, with L = max sequence length
        tr_hidden_state = self.decoder(dec_input,
                                       tgt_mask=decoder_mask.to(x.device),
                                       tgt_attn_bias=abs_pos_bias,
                                       tgt_key_padding_mask=input_masks_padded.to(x.device)) # Shape: L x N x E, with L = max sequence length
        return self.predictor.batched_nll_losses(tr_hidden_state, syllabe_id_lengths, syllabe_ids)

    def predict(self, syllabe_ids, syllabe_id_lengths, ignore_last=False):
        tgt_key_padding_mask = generate_input_key_padding_mask(syllabe_id_lengths, ignore_last=ignore_last)
        tgt_decoder_mask = generate_square_subsequent_mask(max(syllabe_id_lengths))

        x = self.syllabe_embedding(syllabe_ids)  # [L, E2]
        lists = x.split(syllabe_id_lengths, 0)  # len(input_sequence_lengths)
        dec_input = pad_sequence(lists, batch_first=False)
        abs_pos_bias = self.pos_encoder(dec_input)
        # in, out: ->shape: L x N x E, with L = max sequence length
        tr_hidden_state = self.decoder(dec_input,
                                       tgt_mask=tgt_decoder_mask.to(x.device),
                                       tgt_attn_bias=abs_pos_bias,
                                       tgt_key_padding_mask=tgt_key_padding_mask.to(x.device))  # Shape: L x N x E, with L = max sequence length
        next_syllabes = self.predictor.predict(tr_hidden_state, syllabe_id_lengths)
        return tr_hidden_state, next_syllabes
    def predict_syllabe_lm(self, syllabe_ids, syllabe_id_lengths, ignore_last=False):
        tgt_key_padding_mask = generate_input_key_padding_mask(syllabe_id_lengths, ignore_last=ignore_last)
        tgt_decoder_mask = generate_square_subsequent_mask(max(syllabe_id_lengths))

        x = self.syllabe_embedding(syllabe_ids)  # [L, E2]
        lists = x.split(syllabe_id_lengths, 0)  # len(input_sequence_lengths)
        dec_input = pad_sequence(lists, batch_first=False)
        abs_pos_bias = self.pos_encoder(dec_input)
        # in, out: ->shape: L x N x E, with L = max sequence length
        tr_hidden_state = self.decoder(dec_input,
                                       tgt_mask=tgt_decoder_mask.to(x.device),
                                       tgt_attn_bias=abs_pos_bias,
                                       tgt_key_padding_mask=tgt_key_padding_mask.to(x.device))  # Shape: L x N x E, with L = max sequence length

        syllabe_log_probs = self.predictor.predict_syllabe_lm(tr_hidden_state)
        return tr_hidden_state, syllabe_log_probs

def SyllabeGPT_from_pretrained(pretrained_model_file):
    gpt_model = SyllabeGPT()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    state_dict = torch.load(pretrained_model_file, map_location=map_location)
    gpt_model.load_state_dict(state_dict['model_state_dict'])
    del state_dict
    gc.collect()
    return gpt_model