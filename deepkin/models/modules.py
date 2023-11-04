import math

import torch
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm
from deepkin.models.kb_transformers import TransformerEncoderLayer, TransformerEncoder, init_bert_params, \
    tupe_relative_position_bucket
from torch import nn
from torch.cuda.amp import custom_fwd

EN_PAD_IDX = 1
KIN_PAD_IDX = 0

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@torch.jit.script
def f_gelu(x):
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)

@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return f_gelu(x)

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BaseConfig:
    def __init__(self):
        self.tot_num_lm_morphs = 24121
        self.tot_num_pos_tags = 156
        self.tot_num_stems = 32516
        self.tot_num_affixes = 403

class BaseHeadTransform(nn.Module):
    def __init__(self, tr_d_model, cls_ctxt_size, layernorm_epsilon):
        super(BaseHeadTransform, self).__init__()
        self.dense = nn.Linear(tr_d_model, cls_ctxt_size)
        self.layerNorm = FusedLayerNorm(cls_ctxt_size, eps=layernorm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.layerNorm(hidden_states)
        return hidden_states

class KINMT_HeadTransform(nn.Module):
    def __init__(self, tr_d_model, cls_ctxt_size, layernorm_epsilon, dropout=0.3):
        super(KINMT_HeadTransform, self).__init__()
        self.dense = nn.Linear(tr_d_model, cls_ctxt_size)
        self.layerNorm = FusedLayerNorm(cls_ctxt_size, eps=layernorm_epsilon)
        self.in_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.in_dropout(x)
        x = self.dense(x)
        x = self.layerNorm(x)
        x = torch.nn.functional.relu(x)
        return x

class TokenClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.3, head_trunk=False):
        super(TokenClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk

        if self.head_trunk:
            self.trunk_dense = nn.Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedLayerNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)

        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim if self.head_trunk else input_dim, num_classes)

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, features, input_sequence_lengths):
        # features.shape = S x N x E
        # Remove [CLS]
        # len already includes [CLS] in the sequence length count, so number of normal tokens here is (len-1)
        inputs = [features[1:len, i, :].contiguous().view(-1, self.input_dim) for i, len in
                  enumerate(input_sequence_lengths)]
        x = torch.cat(inputs, 0) #  B x E

        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)

        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.0, head_trunk=False):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk

        if self.head_trunk:
            self.trunk_dense = nn.Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedLayerNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)

        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim if self.head_trunk else input_dim, num_classes)

        self.apply(init_bert_params)

    def forward(self, features):
        # features.shape = S x N x E
        x = features[0, :, :]  # Take [CLS]

        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)

        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x
class GPTClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.0, head_trunk=False):
        super(GPTClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.head_trunk = head_trunk

        if self.head_trunk:
            self.trunk_dense = nn.Linear(input_dim, inner_dim)
            self.trunk_layerNorm = FusedLayerNorm(inner_dim)
            self.trunk_activation_fn = torch.tanh
            self.trunk_dropout = nn.Dropout(p=pooler_dropout)

        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim if self.head_trunk else input_dim, num_classes)

        self.apply(init_bert_params)

    def forward(self, features, input_sequence_lengths):
        # features.shape = L x N x E
        x = torch.cat([features[(ln-1):ln, n, :] for n,ln in enumerate(input_sequence_lengths)], 0)

        if self.head_trunk:
            x = self.trunk_dropout(x)
            x = self.trunk_dense(x)
            x = self.trunk_layerNorm(x)
            x = self.trunk_activation_fn(x)

        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x

# TUPE: https://arxiv.org/abs/2006.15595
# https://github.com/guolinke/TUPE/blob/master/fairseq/modules/transformer_sentence_encoder.py
class PositionEncoding(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_seq_len, rel_pos_bins, max_rel_pos, separate_cls_sos):
        super(PositionEncoding, self).__init__()
        self.separate_cls_sos = separate_cls_sos
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.attn_scale_factor = 2
        self.pos = nn.Embedding(self.max_seq_len + 1, self.hidden_dim)
        self.pos_q_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_k_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_scaling = float(self.hidden_dim / self.num_heads * self.attn_scale_factor) ** -0.5
        self.pos_ln = FusedLayerNorm(self.hidden_dim)
        self.tupe_rel_pos_bins = rel_pos_bins
        self.tupe_max_rel_pos = max_rel_pos
        self.relative_attention_bias = nn.Embedding(self.tupe_rel_pos_bins + 1, self.num_heads)
        seq_len = self.max_seq_len
        context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        self.rp_bucket = tupe_relative_position_bucket(
            relative_position,
            num_buckets=self.tupe_rel_pos_bins,
            max_distance=self.tupe_max_rel_pos
        )
        # others to [CLS]
        self.rp_bucket[:, 0] = self.tupe_rel_pos_bins
        # [CLS] to others, Note: self.tupe_rel_pos_bins // 2 is not used in relative_position_bucket
        self.rp_bucket[0, :] = self.tupe_rel_pos_bins // 2
        self.apply(init_bert_params)

    def get_tupe_rel_pos_bias(self, seq_len, device):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != device:
            self.rp_bucket = self.rp_bucket.to(device)
        # Adjusted because final x's shape is L x B X E
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_position_attn_bias(self, seq_len, batch_size, device):
        tupe_rel_pos_bias = self.get_tupe_rel_pos_bias(seq_len, device)
        if self.separate_cls_sos:
            # 0 is for other-to-cls 1 is for cls-to-other
            # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
            weight = self.pos_ln(self.pos.weight[:seq_len + 1, :])
            pos_q =  self.pos_q_linear(weight).view(seq_len + 1, self.num_heads, -1).transpose(0, 1) * self.pos_scaling
            pos_k =  self.pos_k_linear(weight).view(seq_len + 1, self.num_heads, -1).transpose(0, 1)
            abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
            # p_0 \dot p_0 is cls to others
            cls_2_other = abs_pos_bias[:, 0, 0]
            # p_1 \dot p_1 is others to cls
            other_2_cls = abs_pos_bias[:, 1, 1]
            # offset
            abs_pos_bias = abs_pos_bias[:, 1:, 1:]
            abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
            abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)
            tupe_rel_pos_bias += abs_pos_bias
        else:
            weight = self.pos_ln(self.pos.weight[:seq_len, :])
            pos_q = self.pos_q_linear(weight).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.pos_scaling
            pos_k = self.pos_k_linear(weight).view(seq_len, self.num_heads, -1).transpose(0, 1)
            abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
            tupe_rel_pos_bias += abs_pos_bias

        tupe_rel_pos_bias = tupe_rel_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, seq_len, seq_len)
        # Final shape: [batch_size x num_heads, from_seq_length, to_seq_length]
        return tupe_rel_pos_bias

    @custom_fwd
    def forward(self, hidden):
        # hidden shape for our Transformer models: (Len, Batch, Embed)
        device = hidden.device
        seq_len = hidden.size(0)
        batch_size = hidden.size(1)
        return self.get_position_attn_bias(seq_len, batch_size, device)


class MorphoEncoder(nn.Module):
    def __init__(self, args, cfg:BaseConfig):
        super(MorphoEncoder, self).__init__()
        self.pos_tag_embedding = nn.Embedding(cfg.tot_num_pos_tags, args.morpho_dim_hidden, padding_idx=KIN_PAD_IDX)
        self.lm_morph_one_embedding = nn.Embedding(cfg.tot_num_lm_morphs, args.morpho_dim_hidden, padding_idx=KIN_PAD_IDX)
        self.lm_morph_two_embedding = nn.Embedding(cfg.tot_num_lm_morphs, args.morpho_dim_hidden, padding_idx=KIN_PAD_IDX)
        self.stem_embedding = nn.Embedding(cfg.tot_num_stems, args.morpho_dim_hidden, padding_idx=KIN_PAD_IDX)
        self.affixes_embedding = nn.Embedding(cfg.tot_num_affixes, args.morpho_dim_hidden, padding_idx=KIN_PAD_IDX)
        # self.affixes_sum_bias = nn.Parameter(torch.zeros(args.morpho_dim_hidden))
        encoder_layers = TransformerEncoderLayer(args.morpho_dim_hidden, args.morpho_num_heads,
                                                dim_feedforward=args.morpho_dim_ffn, dropout=args.morpho_dropout,
                                                activation="gelu")
        self.encoder = TransformerEncoder(encoder_layers, args.morpho_num_layers)
        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, stems, lm_morphs, pos_tags, afx_padded, m_masks_padded):
        pos_tags = self.pos_tag_embedding(pos_tags)
        pos_tags = torch.unsqueeze(pos_tags, 0)

        lm_morphs1 = self.lm_morph_one_embedding(lm_morphs)
        lm_morphs1 = torch.unsqueeze(lm_morphs1, 0)

        lm_morphs2 = self.lm_morph_two_embedding(lm_morphs)
        lm_morphs2 = torch.unsqueeze(lm_morphs2, 0)

        stems = self.stem_embedding(stems)
        stems = torch.unsqueeze(stems, 0)

        # x_afx_bias = torch.unsqueeze(self.affixes_sum_bias, 0).expand(pos_tags.shape[1],-1).unsqueeze(0)
        # x_embed = torch.cat((stems, lm_morphs, pos_tags, x_afx_bias), 0)
        x_embed = torch.cat((pos_tags, lm_morphs1, lm_morphs2, stems), 0)

        if afx_padded.nelement() > 0:
            xm_affix = self.affixes_embedding(afx_padded)
            # xm_affix: (M,L,E)
            x_embed = torch.cat((x_embed, xm_affix), 0)

        output = self.encoder(x_embed, src_key_padding_mask=m_masks_padded)  # --> Shape: (4+Affixes_Len@Word, L, E)

        # xhead = output[:3, :, :]
        # xsum = torch.sum(output[3:, :, :], 0, keepdim=True) # Add bias term to the rest of affixes
        #
        # x_out = torch.cat((xhead, xsum), 0)

        x_out = output[:4, :, :]

        return x_out # shape: [4, L, E]


class MorphoMLMPredictor(nn.Module):
    def __init__(self, stem_embedding_weights,
                 pos_tag_embedding_weights,
                 lm_morph_embedding_weights,
                 affixes_embedding_weights,
                 tr_d_model,
                 layernorm_epsilon):
        super(MorphoMLMPredictor, self).__init__()

        self.stem_transform = BaseHeadTransform(tr_d_model, stem_embedding_weights.size(1), layernorm_epsilon)
        self.stem_decoder = nn.Linear(stem_embedding_weights.size(1), stem_embedding_weights.size(0), bias=False)
        self.stem_decoder.weight = stem_embedding_weights
        self.stem_decoder_bias = nn.Parameter(torch.zeros(stem_embedding_weights.size(0)))

        self.pos_tag_transform = BaseHeadTransform(tr_d_model, pos_tag_embedding_weights.size(1), layernorm_epsilon)
        self.pos_tag_decoder = nn.Linear(pos_tag_embedding_weights.size(1), pos_tag_embedding_weights.size(0), bias=False)
        self.pos_tag_decoder.weight = pos_tag_embedding_weights
        self.pos_tag_decoder_bias = nn.Parameter(torch.zeros(pos_tag_embedding_weights.size(0)))

        self.lm_morph_transform = BaseHeadTransform(tr_d_model, lm_morph_embedding_weights.size(1), layernorm_epsilon)
        self.lm_morph_decoder = nn.Linear(lm_morph_embedding_weights.size(1), lm_morph_embedding_weights.size(0), bias=False)
        self.lm_morph_decoder.weight = lm_morph_embedding_weights
        self.lm_morph_decoder_bias = nn.Parameter(torch.zeros(lm_morph_embedding_weights.size(0)))

        self.affixes_transform = BaseHeadTransform(tr_d_model, affixes_embedding_weights.size(1), layernorm_epsilon)
        self.affixes_decoder = nn.Linear(affixes_embedding_weights.size(1), affixes_embedding_weights.size(0), bias=False)
        self.affixes_decoder.weight = affixes_embedding_weights
        self.affixes_decoder_bias = nn.Parameter(torch.zeros(affixes_embedding_weights.size(0)))

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, tr_hidden_state,
                predicted_tokens_idx,
                predicted_tokens_affixes_idx,
                predicted_stems,
                predicted_pos_tags,
                predicted_lm_morphs,
                predicted_affixes_prob,
                return_logits = False):

        # print('tr_hidden_state.shape',tr_hidden_state.shape)
        # 1. Crop together predicted tokens
        # tr_hidden_state.shape = S x N x E
        token_hidden_state = tr_hidden_state.permute(1,0,2)
        # N x S x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        token_hidden_state = token_hidden_state.reshape(-1, token_hidden_state.shape[2])
        # predicted_state.shape: NS x E or B x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        # print('batch_predicted_token_idx',batch_predicted_token_idx)
        token_hidden_state = torch.index_select(token_hidden_state, 0, index=predicted_tokens_idx)
        # predicted_state.shape: B x E

        stem_predicted_state = self.stem_transform(token_hidden_state)
        stem_logits = self.stem_decoder(stem_predicted_state) + self.stem_decoder_bias
        stem_scores = F.log_softmax(stem_logits, dim=1)
        stem_loss_avg = F.nll_loss(stem_scores, predicted_stems)

        pos_tag_predicted_state = self.pos_tag_transform(token_hidden_state)
        pos_logits = self.pos_tag_decoder(pos_tag_predicted_state) + self.pos_tag_decoder_bias
        pos_tag_scores = F.log_softmax(pos_logits, dim=1)
        pos_tag_loss_avg = F.nll_loss(pos_tag_scores, predicted_pos_tags)

        lm_morph_predicted_state = self.lm_morph_transform(token_hidden_state)
        afset_logits = self.lm_morph_decoder(lm_morph_predicted_state) + self.lm_morph_decoder_bias
        lm_morph_scores = F.log_softmax(afset_logits, dim=1)
        lm_morph_loss_avg = F.nll_loss(lm_morph_scores, predicted_lm_morphs)

        affix_logits = None
        affixes_loss_avg = None
        if predicted_tokens_affixes_idx.nelement() > 0:
            affixes_hidden_state = torch.index_select(token_hidden_state, 0, index=predicted_tokens_affixes_idx)
            affixes_predicted_state = self.affixes_transform(affixes_hidden_state)
            affix_logits = self.affixes_decoder(affixes_predicted_state) + self.affixes_decoder_bias
            affixes_loss_avg = F.binary_cross_entropy_with_logits(affix_logits, predicted_affixes_prob)

        losses = [stem_loss_avg, pos_tag_loss_avg, lm_morph_loss_avg]
        if affixes_loss_avg is not None:
            losses.append(affixes_loss_avg)
        if return_logits:
            return losses, stem_logits, pos_logits, afset_logits, affix_logits
        else:
            return losses

    def predict(self, tr_hidden_state,
                seq_predicted_token_idx,
                max_predict_affixes,
                max_top_predictions = 10):
        # print('tr_hidden_state.shape',tr_hidden_state.shape)
        # 1. Crop together predicted tokens
        # tr_hidden_state.shape = S x N x E
        token_hidden_state = tr_hidden_state.permute(1,0,2)
        # N x S x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        token_hidden_state = token_hidden_state.reshape(-1, token_hidden_state.shape[2])
        # predicted_state.shape: NS x E or B x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        # print('batch_predicted_token_idx',batch_predicted_token_idx)
        token_hidden_state = torch.index_select(token_hidden_state, 0, index=seq_predicted_token_idx)
        # predicted_state.shape: B x E

        stem_predicted_state = self.stem_transform(token_hidden_state)
        stem_scores = self.stem_decoder(stem_predicted_state) + self.stem_decoder_bias
        stem_scores = F.softmax(stem_scores, dim=1)

        top_preds_prob, top_preds = torch.topk(stem_scores, max_top_predictions, dim=1)
        stem_predictions = []
        stem_predictions_prob = []
        for batch in range(top_preds.shape[0]):
            stem_predictions.append(top_preds[batch, :].cpu().tolist())
            stem_predictions_prob.append(top_preds_prob[batch, :].cpu().tolist())

        pos_tag_predicted_state = self.pos_tag_transform(token_hidden_state)
        pos_tag_scores = self.pos_tag_decoder(pos_tag_predicted_state) + self.pos_tag_decoder_bias
        pos_tag_scores = F.softmax(pos_tag_scores, dim=1)

        top_preds_prob, top_preds = torch.topk(pos_tag_scores, max_top_predictions, dim=1)
        pos_tag_predictions = []
        pos_tag_predictions_prob = []
        for batch in range(top_preds.shape[0]):
            pos_tag_predictions.append(top_preds[batch, :].cpu().tolist())
            pos_tag_predictions_prob.append(top_preds_prob[batch, :].cpu().tolist())

        lm_morph_predicted_state = self.lm_morph_transform(token_hidden_state)
        lm_morph_scores = self.lm_morph_decoder(lm_morph_predicted_state) + self.lm_morph_decoder_bias
        lm_morph_scores = F.softmax(lm_morph_scores, dim=1)

        top_preds_prob, top_preds = torch.topk(lm_morph_scores, max_top_predictions, dim=1)
        lm_morph_predictions = []
        lm_morph_predictions_prob = []
        for batch in range(top_preds.shape[0]):
            lm_morph_predictions.append(top_preds[batch, :].cpu().tolist())
            lm_morph_predictions_prob.append(top_preds_prob[batch, :].cpu().tolist())

        affixes_predicted_state = self.affixes_transform(token_hidden_state)
        affixes_scores = self.affixes_decoder(affixes_predicted_state) + self.affixes_decoder_bias
        affixes_scores = F.sigmoid(affixes_scores)
        top_affixes_prob, top_affixes = torch.topk(affixes_scores, max_predict_affixes, dim=1)
        affixes_predictions = []
        affixes_predictions_prob = []
        for batch in range(top_affixes.shape[0]):
            affixes_predictions.append(top_affixes[batch,:].cpu().tolist())
            affixes_predictions_prob.append(top_affixes_prob[batch,:].cpu().tolist())

        return ((stem_predictions,
                 pos_tag_predictions,
                 lm_morph_predictions,
                 affixes_predictions),
               (stem_predictions_prob,
                pos_tag_predictions_prob,
                lm_morph_predictions_prob,
                affixes_predictions_prob))

class MorphoGPTPredictor(nn.Module):
    def __init__(self, stem_embedding_weights,
                 pos_tag_embedding_weights,
                 lm_morph_embedding_weights,
                 affixes_embedding_weights,
                 tr_d_model,
                 layernorm_epsilon):
        super(MorphoGPTPredictor, self).__init__()

        self.stem_transform = BaseHeadTransform(tr_d_model, stem_embedding_weights.size(1), layernorm_epsilon)
        self.stem_decoder = nn.Linear(stem_embedding_weights.size(1), stem_embedding_weights.size(0), bias=False)
        self.stem_decoder.weight = stem_embedding_weights
        self.stem_decoder_bias = nn.Parameter(torch.zeros(stem_embedding_weights.size(0)))

        self.pos_tag_transform = BaseHeadTransform(tr_d_model, pos_tag_embedding_weights.size(1), layernorm_epsilon)
        self.pos_tag_decoder = nn.Linear(pos_tag_embedding_weights.size(1), pos_tag_embedding_weights.size(0), bias=False)
        self.pos_tag_decoder.weight = pos_tag_embedding_weights
        self.pos_tag_decoder_bias = nn.Parameter(torch.zeros(pos_tag_embedding_weights.size(0)))

        self.lm_morph_transform = BaseHeadTransform(tr_d_model, lm_morph_embedding_weights.size(1), layernorm_epsilon)
        self.lm_morph_decoder = nn.Linear(lm_morph_embedding_weights.size(1), lm_morph_embedding_weights.size(0), bias=False)
        self.lm_morph_decoder.weight = lm_morph_embedding_weights
        self.lm_morph_decoder_bias = nn.Parameter(torch.zeros(lm_morph_embedding_weights.size(0)))

        self.affixes_transform = BaseHeadTransform(tr_d_model, affixes_embedding_weights.size(1), layernorm_epsilon)
        self.affixes_decoder = nn.Linear(affixes_embedding_weights.size(1), affixes_embedding_weights.size(0), bias=False)
        self.affixes_decoder.weight = affixes_embedding_weights
        self.affixes_decoder_bias = nn.Parameter(torch.zeros(affixes_embedding_weights.size(0)))

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, tr_hidden_state, input_sequence_lengths,
                stems,
                pos_tags,
                lm_morphs,
                affixes_prob,
                tokens_lengths,
                epsilon_ls=0.1):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # Last <EOS> token not processed
        sub = 1
        hidden_states = [token_hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        affixes_lens = torch.tensor(tokens_lengths).split(input_sequence_lengths)
        affixes_lens = [tln[1:length] for length, tln in zip(input_sequence_lengths, affixes_lens)]
        affixes_lens = torch.cat(affixes_lens, dim=0)
        selected_affixes_idx = torch.tensor([i for i in range(affixes_lens.size(0)) if affixes_lens[i].item() > 0]).to(tr_hidden_state.device)

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_stems = stems.split(input_sequence_lengths)
        target_stems = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_stems)]
        target_stems = torch.cat(target_stems, dim=0)

        target_pos_tags = pos_tags.split(input_sequence_lengths)
        target_pos_tags = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_pos_tags)]
        target_pos_tags = torch.cat(target_pos_tags, dim=0)

        target_lm_morphs = lm_morphs.split(input_sequence_lengths)
        target_lm_morphs = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_lm_morphs)]
        target_lm_morphs = torch.cat(target_lm_morphs, dim=0)

        target_affixes_prob = affixes_prob.split(input_sequence_lengths)
        target_affixes_prob = [tns[1:length,:] for length, tns in zip(input_sequence_lengths, target_affixes_prob)]
        target_affixes_prob = torch.cat(target_affixes_prob, dim=0)

        stem_scores = self.stem_transform(batch_logits)
        stem_scores = self.stem_decoder(stem_scores) + self.stem_decoder_bias
        stem_scores = F.log_softmax(stem_scores, dim=1)
        # stem_loss_avg = F.nll_loss(stem_scores, target_stems)
        stem_loss_avg, stem_nll_loss_avg = label_smoothed_nll_loss(stem_scores, target_stems, epsilon_ls)

        pos_tag_scores = self.pos_tag_transform(batch_logits)
        pos_tag_scores = self.pos_tag_decoder(pos_tag_scores) + self.pos_tag_decoder_bias
        pos_tag_scores = F.log_softmax(pos_tag_scores, dim=1)
        # pos_tag_loss_avg = F.nll_loss(pos_tag_scores, target_pos_tags)
        pos_tag_loss_avg, pos_tag_nll_loss_avg = label_smoothed_nll_loss(pos_tag_scores, target_pos_tags, epsilon_ls)

        lm_morph_scores = self.lm_morph_transform(batch_logits)
        lm_morph_scores = self.lm_morph_decoder(lm_morph_scores) + self.lm_morph_decoder_bias
        lm_morph_scores = F.log_softmax(lm_morph_scores, dim=1)
        # lm_morph_loss_avg = F.nll_loss(lm_morph_scores, target_lm_morphs)
        lm_morph_loss_avg, lm_morph_nll_loss_avg = label_smoothed_nll_loss(lm_morph_scores, target_lm_morphs, epsilon_ls)

        # Find out how to predict affixes autoregressively
        affixes_loss_avg = None
        if selected_affixes_idx.size(0) > 0:
            affixes_batch_logits = batch_logits[selected_affixes_idx,:]
            target_affixes_prob = target_affixes_prob[selected_affixes_idx,:]
            affixes_predicted_state = self.affixes_transform(affixes_batch_logits)
            affixes_scores = self.affixes_decoder(affixes_predicted_state) + self.affixes_decoder_bias
            affixes_loss_avg = F.binary_cross_entropy_with_logits(affixes_scores, target_affixes_prob)

        losses = [stem_loss_avg, pos_tag_loss_avg, lm_morph_loss_avg]
        nll_losses = [stem_nll_loss_avg, pos_tag_nll_loss_avg, lm_morph_nll_loss_avg]
        if affixes_loss_avg is not None:
            losses.append(affixes_loss_avg)
        return losses, nll_losses

    def predict(self, tr_hidden_state, input_sequence_lengths):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # No <EOS> at end, so no sub
        sub = 0
        hidden_states = [token_hidden_state[i, (input_sequence_lengths[i] - sub - 1):(input_sequence_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)

        stem_scores = self.stem_transform(batch_logits)
        stem_scores = self.stem_decoder(stem_scores) + self.stem_decoder_bias
        next_stems = F.softmax(stem_scores, dim=1)

        pos_tag_scores = self.pos_tag_transform(batch_logits)
        pos_tag_scores = self.pos_tag_decoder(pos_tag_scores) + self.pos_tag_decoder_bias
        next_pos_tags = F.softmax(pos_tag_scores, dim=1)

        lm_morph_scores = self.lm_morph_transform(batch_logits)
        lm_morph_scores = self.lm_morph_decoder(lm_morph_scores) + self.lm_morph_decoder_bias
        next_lm_morphs = F.softmax(lm_morph_scores, dim=1)

        affixes_scores = self.affixes_transform(batch_logits)
        affixes_scores = self.affixes_decoder(affixes_scores) + self.affixes_decoder_bias
        next_affixes = F.sigmoid(affixes_scores)

        return (next_stems, next_pos_tags, next_lm_morphs, next_affixes)

class Kinya_TokenGenerator(nn.Module):
    def __init__(self, stem_embedding_weights,
                 pos_tag_embedding_weights,
                 lm_morph_embedding_weights,
                 affixes_embedding_weights,
                 tr_d_model,
                 layernorm_epsilon,
                 copy_tokens_embedding_weights = None):
        super(Kinya_TokenGenerator, self).__init__()

        self.stem_transform = BaseHeadTransform(tr_d_model, stem_embedding_weights.size(1), layernorm_epsilon)
        self.stem_decoder = nn.Linear(stem_embedding_weights.size(1), stem_embedding_weights.size(0), bias=False)
        self.stem_decoder.weight = stem_embedding_weights
        self.stem_decoder_bias = nn.Parameter(torch.zeros(stem_embedding_weights.size(0)))

        self.pos_tag_transform = BaseHeadTransform(tr_d_model, pos_tag_embedding_weights.size(1), layernorm_epsilon)
        self.pos_tag_decoder = nn.Linear(pos_tag_embedding_weights.size(1), pos_tag_embedding_weights.size(0), bias=False)
        self.pos_tag_decoder.weight = pos_tag_embedding_weights
        self.pos_tag_decoder_bias = nn.Parameter(torch.zeros(pos_tag_embedding_weights.size(0)))

        self.lm_morph_transform = BaseHeadTransform(tr_d_model, lm_morph_embedding_weights.size(1), layernorm_epsilon)
        self.lm_morph_decoder = nn.Linear(lm_morph_embedding_weights.size(1), lm_morph_embedding_weights.size(0), bias=False)
        self.lm_morph_decoder.weight = lm_morph_embedding_weights
        self.lm_morph_decoder_bias = nn.Parameter(torch.zeros(lm_morph_embedding_weights.size(0)))

        self.affixes_transform = BaseHeadTransform(tr_d_model, affixes_embedding_weights.size(1), layernorm_epsilon)
        self.affixes_decoder = nn.Linear(affixes_embedding_weights.size(1), affixes_embedding_weights.size(0), bias=False)
        self.affixes_decoder.weight = affixes_embedding_weights
        self.affixes_decoder_bias = nn.Parameter(torch.zeros(affixes_embedding_weights.size(0)))

        if copy_tokens_embedding_weights is None:
            self.copy_tokens = False
        else:
            self.copy_tokens = True
            self.copy_tokens_transform = BaseHeadTransform(tr_d_model, copy_tokens_embedding_weights.size(1), layernorm_epsilon)
            self.copy_tokens_decoder = nn.Linear(copy_tokens_embedding_weights.size(1), copy_tokens_embedding_weights.size(0), bias=False)
            self.copy_tokens_decoder.weight = copy_tokens_embedding_weights
            self.copy_tokens_decoder_bias = nn.Parameter(torch.zeros(copy_tokens_embedding_weights.size(0)))

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, tr_hidden_state, input_sequence_lengths,
                stems,
                pos_tags,
                lm_morphs,
                affixes_prob,
                copy_tokens_prob = None, # (LN,V)
                epsilon_ls=0.1):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # Last <EOS> token not processed
        sub = 1
        hidden_states = [token_hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_stems = stems.split(input_sequence_lengths)
        target_stems = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_stems)]
        target_stems = torch.cat(target_stems, dim=0)

        target_pos_tags = pos_tags.split(input_sequence_lengths)
        target_pos_tags = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_pos_tags)]
        target_pos_tags = torch.cat(target_pos_tags, dim=0)

        target_lm_morphs = lm_morphs.split(input_sequence_lengths)
        target_lm_morphs = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_lm_morphs)]
        target_lm_morphs = torch.cat(target_lm_morphs, dim=0)

        target_affixes_prob = affixes_prob.split(input_sequence_lengths)
        target_affixes_prob = [tns[1:length,:] for length, tns in zip(input_sequence_lengths, target_affixes_prob)]
        target_affixes_prob = torch.cat(target_affixes_prob, dim=0)

        stem_scores = self.stem_transform(batch_logits)
        stem_scores = self.stem_decoder(stem_scores) + self.stem_decoder_bias
        stem_scores = F.log_softmax(stem_scores, dim=1)
        # stem_loss_avg = F.nll_loss(stem_scores, target_stems)
        stem_loss_avg, stem_nll_loss_avg = label_smoothed_nll_loss(stem_scores, target_stems, epsilon_ls)

        pos_tag_scores = self.pos_tag_transform(batch_logits)
        pos_tag_scores = self.pos_tag_decoder(pos_tag_scores) + self.pos_tag_decoder_bias
        pos_tag_scores = F.log_softmax(pos_tag_scores, dim=1)
        # pos_tag_loss_avg = F.nll_loss(pos_tag_scores, target_pos_tags)
        pos_tag_loss_avg, pos_tag_nll_loss_avg = label_smoothed_nll_loss(pos_tag_scores, target_pos_tags, epsilon_ls)

        lm_morph_scores = self.lm_morph_transform(batch_logits)
        lm_morph_scores = self.lm_morph_decoder(lm_morph_scores) + self.lm_morph_decoder_bias
        lm_morph_scores = F.log_softmax(lm_morph_scores, dim=1)
        # lm_morph_loss_avg = F.nll_loss(lm_morph_scores, target_lm_morphs)
        lm_morph_loss_avg, lm_morph_nll_loss_avg = label_smoothed_nll_loss(lm_morph_scores, target_lm_morphs, epsilon_ls)

        affixes_predicted_state = self.affixes_transform(batch_logits)
        affixes_scores = self.affixes_decoder(affixes_predicted_state) + self.affixes_decoder_bias
        affixes_loss_avg = F.binary_cross_entropy_with_logits(affixes_scores, target_affixes_prob)

        if self.copy_tokens:
            target_copy_tokens_prob = copy_tokens_prob.split(input_sequence_lengths)
            target_copy_tokens_prob = [tns[1:length,:] for length, tns in zip(input_sequence_lengths, target_copy_tokens_prob)]
            target_copy_tokens_prob = torch.cat(target_copy_tokens_prob, dim=0)

            copy_tokens_predicted_state = self.copy_tokens_transform(batch_logits)
            copy_tokens_scores = self.copy_tokens_decoder(copy_tokens_predicted_state) + self.copy_tokens_decoder_bias
            copy_tokens_loss_avg = F.binary_cross_entropy_with_logits(copy_tokens_scores, target_copy_tokens_prob)

            losses = [stem_loss_avg, pos_tag_loss_avg, lm_morph_loss_avg, affixes_loss_avg, copy_tokens_loss_avg]
            nll_losses = [stem_nll_loss_avg, pos_tag_nll_loss_avg, lm_morph_nll_loss_avg, affixes_loss_avg, copy_tokens_loss_avg]
        else:
            losses = [stem_loss_avg, pos_tag_loss_avg, lm_morph_loss_avg, affixes_loss_avg]
            nll_losses = [stem_nll_loss_avg, pos_tag_nll_loss_avg, lm_morph_nll_loss_avg, affixes_loss_avg]

        return losses, nll_losses

    def predict(self, tr_hidden_state, input_sequence_lengths):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # No <EOS> at end, so no sub
        sub = 0
        hidden_states = [token_hidden_state[i, (input_sequence_lengths[i] - sub - 1):(input_sequence_lengths[i] - sub), :] for i in range(N)]
        batch_logits = torch.cat(hidden_states, dim=0)

        stem_scores = self.stem_transform(batch_logits)
        stem_scores = self.stem_decoder(stem_scores) + self.stem_decoder_bias
        next_stems = F.softmax(stem_scores, dim=1)

        pos_tag_scores = self.pos_tag_transform(batch_logits)
        pos_tag_scores = self.pos_tag_decoder(pos_tag_scores) + self.pos_tag_decoder_bias
        next_pos_tags = F.softmax(pos_tag_scores, dim=1)

        lm_morph_scores = self.lm_morph_transform(batch_logits)
        lm_morph_scores = self.lm_morph_decoder(lm_morph_scores) + self.lm_morph_decoder_bias
        next_lm_morphs = F.softmax(lm_morph_scores, dim=1)

        affixes_scores = self.affixes_transform(batch_logits)
        affixes_scores = self.affixes_decoder(affixes_scores) + self.affixes_decoder_bias
        next_affixes = F.sigmoid(affixes_scores)
        if self.copy_tokens:
            copy_tokens_scores = self.copy_tokens_transform(batch_logits)
            copy_tokens_scores = self.copy_tokens_decoder(copy_tokens_scores) + self.copy_tokens_decoder_bias
            next_copy_prob = F.sigmoid(copy_tokens_scores)

            return (next_stems, next_pos_tags, next_lm_morphs, next_affixes, next_copy_prob) #shape: (N,V)
        else:
            return (next_stems, next_pos_tags, next_lm_morphs, next_affixes, None) #shape: (N,V)

class Engl_TokenGenerator(nn.Module):
    def __init__(self, token_embedding_weights, tr_d_model, layernorm_epsilon,
                 copy_tokens_embedding_weights = None):
        super(Engl_TokenGenerator, self).__init__()

        self.token_transform = BaseHeadTransform(tr_d_model, token_embedding_weights.size(1), layernorm_epsilon)
        self.token_decoder = nn.Linear(token_embedding_weights.size(1), token_embedding_weights.size(0), bias=False)
        self.token_decoder.weight = token_embedding_weights
        self.token_decoder_bias = nn.Parameter(torch.zeros(token_embedding_weights.size(0)))

        if copy_tokens_embedding_weights is None:
            self.copy_tokens = False
        else:
            self.copy_tokens = True
            self.copy_tokens_transform = BaseHeadTransform(tr_d_model, copy_tokens_embedding_weights.size(1), layernorm_epsilon)
            self.copy_tokens_decoder = nn.Linear(copy_tokens_embedding_weights.size(1), copy_tokens_embedding_weights.size(0), bias=False)
            self.copy_tokens_decoder.weight = copy_tokens_embedding_weights
            self.copy_tokens_decoder_bias = nn.Parameter(torch.zeros(copy_tokens_embedding_weights.size(0)))

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, tr_hidden_state, tokens, input_sequence_lengths,
                copy_tokens_prob = None,
                epsilon_ls=0.1):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # Last <EOS> token not processed
        sub = 1
        hidden_states = [token_hidden_state[i, :(input_sequence_lengths[i] - sub), :] for i in range(N)]

        batch_logits = torch.cat(hidden_states, dim=0)
        # (B,E), B=Batch Size = sum([l-1 for l in input_sequence_lengths])

        target_tokens = tokens.split(input_sequence_lengths)
        target_tokens = [tns[1:length] for length, tns in zip(input_sequence_lengths, target_tokens)]
        target_tokens = torch.cat(target_tokens, dim=0)

        token_scores = self.token_transform(batch_logits)
        token_scores = self.token_decoder(token_scores) + self.token_decoder_bias
        token_scores = F.log_softmax(token_scores, dim=1)
        # token_loss_avg = F.nll_loss(token_scores, target_tokens)
        token_loss_avg, token_nll_loss_avg = label_smoothed_nll_loss(token_scores, target_tokens, epsilon_ls)

        if self.copy_tokens:
            target_copy_tokens_prob = copy_tokens_prob.split(input_sequence_lengths)
            target_copy_tokens_prob = [tns[1:length,:] for length, tns in zip(input_sequence_lengths, target_copy_tokens_prob)]
            target_copy_tokens_prob = torch.cat(target_copy_tokens_prob, dim=0)

            copy_tokens_predicted_state = self.copy_tokens_transform(batch_logits)
            copy_tokens_scores = self.copy_tokens_decoder(copy_tokens_predicted_state) + self.copy_tokens_decoder_bias
            copy_tokens_loss_avg = F.binary_cross_entropy_with_logits(copy_tokens_scores, target_copy_tokens_prob)

            losses = [token_loss_avg, copy_tokens_loss_avg]
            nll_losses = [token_nll_loss_avg, copy_tokens_loss_avg]
        else:
            losses = [token_loss_avg]
            nll_losses = [token_nll_loss_avg]

        return losses, nll_losses

    def predict(self, tr_hidden_state, input_sequence_lengths):
        token_hidden_state = tr_hidden_state.permute(1, 0, 2)  # L,N,E --> N,L,E
        N = token_hidden_state.size(0)
        # No <EOS> at end, so no sub
        sub = 0
        hidden_states = [token_hidden_state[i, (input_sequence_lengths[i] - sub - 1):(input_sequence_lengths[i] - sub), :] for i in range(N)]
        batch_logits = torch.cat(hidden_states, dim=0)

        token_scores = self.token_transform(batch_logits)
        token_scores = self.token_decoder(token_scores) + self.token_decoder_bias
        # next_tokens = F.softmax(token_scores, dim=1)

        if self.copy_tokens:
            copy_tokens_scores = self.copy_tokens_transform(batch_logits)
            copy_tokens_scores = self.copy_tokens_decoder(copy_tokens_scores) + self.copy_tokens_decoder_bias
            next_copy_prob = F.sigmoid(copy_tokens_scores)

            return token_scores, next_copy_prob
        else:
            return token_scores, None
