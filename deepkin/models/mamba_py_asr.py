# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from deepkin.models.kb_transformers import init_bert_params
from deepkin.models.mamba import Mamba, MambaConfig
from deepkin.utils.misc_functions import time_now


def compute_subsampling_output_sizes(input_sizes: List[int]) -> List[int]:
    kernel: int = 3
    stride: int = 2
    padding: int = 1
    output_sizes: List[int] = []
    for l in input_sizes:
        l1 = int(math.floor(((l + (2 * padding) - (kernel - 1) - 1) / stride) + 1))
        l2 = int(math.floor(((l1 + (2 * padding) - (kernel - 1) - 1) / stride) + 1))
        output_sizes.append(l2)
    return output_sizes

@torch.jit.script
def f_gelu(x):
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)

class BaseHeadTransform(nn.Module):
    def __init__(self, tr_d_model, cls_ctxt_size, layernorm_epsilon):
        super(BaseHeadTransform, self).__init__()
        self.dense = nn.Linear(tr_d_model, cls_ctxt_size)
        self.layerNorm = torch.nn.LayerNorm(cls_ctxt_size, eps=layernorm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.layerNorm(hidden_states)
        return hidden_states

class KinspeakSubSampler(nn.Module):
    def __init__(
            self,
            log_mel_spectrogram_dim: int=80,
            channel_multiple:int=32,
            embed_dim:int=768,
            pre_trained_kinspeak_model_path:str=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_sub_sampling_module = nn.Sequential(
            nn.Conv2d(1, channel_multiple, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(channel_multiple, channel_multiple, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.SiLU())
        self.features_dim = channel_multiple * compute_subsampling_output_sizes([log_mel_spectrogram_dim])[0]
        self.layer_norm = torch.nn.LayerNorm(self.features_dim)
        self.project_input = nn.Linear(self.features_dim, self.embed_dim)
        if pre_trained_kinspeak_model_path is not None:
            my_dict = self.state_dict()
            target_dict = torch.load(pre_trained_kinspeak_model_path, map_location='cpu')['model_state_dict']
            key_dict = dict()
            for key in my_dict.keys():
                key_dict[key] = 'ctxt_encoder.'+key
            for k in key_dict.keys():
                my_dict[k] = target_dict[key_dict[k]]
            self.load_state_dict(my_dict)
            print(time_now(), 'Loaded convolutional sub-sampling Layers from KinSPEAK!')

    def forward(self, log_mel_spectrograms: torch.Tensor) -> torch.Tensor: # N,F,L
        """
        Args:
            log_mel_spectrograms: torch.Tensor of shape (N,F,L)
        Returns:
            torch.Tensor of shape T,N,F
        """

        # Convolutional subsampling
        x = log_mel_spectrograms.unsqueeze(1)  # (N,1,F,L)
        x = self.conv_sub_sampling_module(x)  # (N,C',F',L')

        N, C, F, L = x.shape
        # print('N={}, C={}, F={}, L={}'.format(N, C, F, L), flush=True)
        x = x.reshape(N, -1, L)  # (N,CF,L)
        x = x.transpose(-2, -1)  # (N,L,CF) or (B,T,C)

        x = self.layer_norm(x)  # (B,T,C)
        x = self.project_input(x)  # (B,T,E)
        return x # (N,T,E)/(B,L,D)


class KinspeakAsrHead(nn.Module):
    def __init__(
            self,
            target_vocab_size: int,
            target_blank_id: int,
            embed_dim=768,
            pre_trained_kinspeak_model_path:str=None,
    ) -> None:
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.target_blank_id = target_blank_id
        self.embed_dim = embed_dim
        self.encoder_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.encoder_projection.apply(init_bert_params)
        self.acoustic_transform = BaseHeadTransform(self.embed_dim,
                                                    self.embed_dim,
                                                    1e-6)
        self.acoustic_decoder = nn.Linear(self.embed_dim, self.target_vocab_size)
        self.acoustic_transform.apply(init_bert_params)
        self.acoustic_decoder.apply(init_bert_params)
        if pre_trained_kinspeak_model_path is not None:
            my_dict = self.state_dict()
            target_dict = torch.load(pre_trained_kinspeak_model_path, map_location='cpu')['model_state_dict']
            key_dict = dict()
            for key in my_dict.keys():
                key_dict[key] = ''+key
            for k in key_dict.keys():
                my_dict[k] = target_dict[key_dict[k]]
            self.load_state_dict(my_dict)
            print(time_now(), 'Loaded acoustic predictor layers from KinSPEAK!')

    def forward(self, source_encoder_output: torch.Tensor, source_encoder_output_lengths: List[int],
                target_syllabe_ids:torch.Tensor, target_syllabe_id_lengths:List[int]) -> torch.Tensor:
        source_encoder_output = self.encoder_projection(source_encoder_output)  # (T,N,C) (N:batch_size,T:input_lengths,C:pred_classes)
        encoder_scores = self.acoustic_transform(source_encoder_output)
        encoder_scores = self.acoustic_decoder(encoder_scores)
        log_probs = F.log_softmax(encoder_scores, dim=-1)
        ctc_loss = F.ctc_loss(log_probs, target_syllabe_ids.to(dtype=torch.int32),
                              torch.tensor(source_encoder_output_lengths, dtype=torch.int32),
                              torch.tensor(target_syllabe_id_lengths, dtype=torch.int32),
                              blank=self.target_blank_id, reduction='mean', zero_infinity=True)
        return ctc_loss

    def infer_acoustic(self, source_encoder_output: torch.Tensor, source_encoder_output_lengths: List[int]) -> Tuple[torch.Tensor, List[int]]:
        source_encoder_output = self.encoder_projection(source_encoder_output) # (T,B,E)
        encoder_scores = self.acoustic_transform(source_encoder_output)
        encoder_scores = self.acoustic_decoder(encoder_scores)
        acoustic_log_probs = F.log_softmax(encoder_scores, dim=-1)
        return acoustic_log_probs, source_encoder_output_lengths


class MambaAsrModel(nn.Module):

    def __init__(
            self,
            mamba_config: MambaConfig,
            target_vocab_size: int,
            target_blank_id: int,
            log_mel_spectrogram_dim=80,
            channel_multiple=32,
            embed_dim=768,
            pre_trained_kinspeak_model_path="/home/nzeyi/KINLP/data/kinspeak_asr_syllabe_INTERSPEECH2023_MULTI_STAGE_JW_CV12_NST_ENCODER_ONLY_PRETRAINED_model_base_2024-01-13.pt_stage_10.pt",
    ) -> None:
        super().__init__()
        self.sub_sampler = KinspeakSubSampler(log_mel_spectrogram_dim=log_mel_spectrogram_dim,
                                             channel_multiple=channel_multiple, embed_dim=embed_dim,
                                             pre_trained_kinspeak_model_path=pre_trained_kinspeak_model_path)
        assert mamba_config.d_model == self.sub_sampler.embed_dim, "d_model ~ sub-sampler embedding dim mismatch"
        self.backbone = Mamba(mamba_config)
        self.asr_head = KinspeakAsrHead(target_vocab_size, target_blank_id,
                                        embed_dim=self.sub_sampler.embed_dim,
                                        pre_trained_kinspeak_model_path=pre_trained_kinspeak_model_path)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, log_mel_spectrograms: torch.Tensor,  # log_mel_spectrograms: (N,F,L)
                log_mel_spectrogram_lengths: List[int],
                target_syllabe_ids: torch.Tensor, target_syllabe_id_lengths: List[int]):
        source_encoder_output_lengths = compute_subsampling_output_sizes(log_mel_spectrogram_lengths)
        source_encoder_output = self.sub_sampler(log_mel_spectrograms) # (T,N,E)
        source_encoder_output = self.backbone(source_encoder_output)
        source_encoder_output = source_encoder_output.transpose(0, 1)  # (B,T,E) --> (T,B,E)

        ctc_loss = self.asr_head(source_encoder_output, source_encoder_output_lengths,
                                 target_syllabe_ids, target_syllabe_id_lengths)
        return ctc_loss

    def infer_acoustic(self, log_mel_spectrograms: torch.Tensor, #log_mel_spectrograms: (N,F,L)
                log_mel_spectrogram_lengths: List[int]):
        source_encoder_output_lengths = compute_subsampling_output_sizes(log_mel_spectrogram_lengths)
        source_encoder_output = self.sub_sampler(log_mel_spectrograms)  # (T,N,E)
        source_encoder_output = self.backbone(source_encoder_output)
        source_encoder_output = source_encoder_output.transpose(0, 1)  # (B,T,E) --> (T,B,E)

        (acoustic_log_probs,
         source_encoder_output_lengths) = self.asr_head.infer_acoustic(source_encoder_output,
                                                                       source_encoder_output_lengths)
        return (acoustic_log_probs, source_encoder_output_lengths)

class MambaMobileAsrModel(nn.Module):

    def __init__(
            self,
            mamba_config: MambaConfig,
            target_vocab_size: int,
            target_blank_id: int,
            log_mel_spectrogram_dim=80,
            channel_multiple=32,
            embed_dim=768,
            pre_trained_kinspeak_model_path="/home/nzeyi/KINLP/data/kinspeak_asr_syllabe_INTERSPEECH2023_MULTI_STAGE_JW_CV12_NST_ENCODER_ONLY_PRETRAINED_model_base_2024-01-13.pt_stage_10.pt",
            pre_trained_mamba_asr_model = None,
    ) -> None:
        super().__init__()
        self.sub_sampler = KinspeakSubSampler(log_mel_spectrogram_dim=log_mel_spectrogram_dim,
                                             channel_multiple=channel_multiple, embed_dim=embed_dim,
                                             pre_trained_kinspeak_model_path=pre_trained_kinspeak_model_path)
        assert mamba_config.d_model == self.sub_sampler.embed_dim, "d_model ~ sub-sampler embedding dim mismatch"
        self.backbone = Mamba(mamba_config)
        self.asr_head = KinspeakAsrHead(target_vocab_size, target_blank_id,
                                        embed_dim=self.sub_sampler.embed_dim,
                                        pre_trained_kinspeak_model_path=pre_trained_kinspeak_model_path)
        if pre_trained_mamba_asr_model is not None:
            asr_state_dict = torch.load(pre_trained_mamba_asr_model, map_location='cpu')
            self.load_state_dict(asr_state_dict['model_state_dict'])
        self.sample_rate = 16_000
        self.n_fft = 1024
        self.n_mels = 80
        self.win_length = self.sample_rate * 25 // 1000  # 25ms
        self.hop_length = self.sample_rate * 10 // 1000  # 10ms
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft,
                                           win_length=self.win_length,
                                           hop_length=self.hop_length, center=True, pad_mode="reflect", power=2.0,
                                           norm="slaney", onesided=True, n_mels=self.n_mels,
                                           mel_scale="htk", )
        self.batch_size = 1
        self.d_inner = mamba_config.d_inner
        self.d_state = mamba_config.d_state
        self.d_conv = mamba_config.d_conv
        self.n_layers = mamba_config.n_layers
        self.caches: List[Tuple[torch.Tensor, torch.Tensor]] = [(torch.zeros(self.batch_size, self.d_inner, self.d_state),
                                                                 torch.zeros(self.batch_size, self.d_inner, self.d_conv - 1)) for _ in range(self.n_layers)]
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Given an audi chunk, return log probabilities over the space of target vocab tokens
        log_eps = 1e-36
        log_mel_spectrograms = self.mel_spectrogram(x)  # (F,L)
        log_mel_spectrograms = torch.log(log_mel_spectrograms + log_eps)
        log_mel_spectrograms = log_mel_spectrograms.unsqueeze(0)
        L = log_mel_spectrograms.shape[-1]
        log_mel_spectrogram_lengths = [L]
        idx = 0 if (L < 8) else 1
        source_encoder_output_lengths = compute_subsampling_output_sizes(log_mel_spectrogram_lengths)

        inp = self.sub_sampler(log_mel_spectrograms)  # (N,F,L) -> (B,L,D)
        # print('N,F,L -> B,L,D:', log_mel_spectrograms.shape, '->', inp.shape)
        inp = inp[:, idx, :]
        if idx == 0:
            # i.e. Reset cache as we are starting a new audio chunk
            self.caches = [(torch.zeros(self.batch_size, self.d_inner, self.d_state),
                       torch.zeros(self.batch_size, self.d_inner, self.d_conv - 1)) for _ in range(self.n_layers)]

        inp, self.caches = self.backbone.step(inp, self.caches) # (B,D) -> (B,D)
        # This might not be necessary
        inp = inp.unsqueeze(0)  # (B,D) -> (T,B,D)

        (acoustic_log_probs,
         source_encoder_output_lengths) = self.asr_head.infer_acoustic(inp,
                                                                       source_encoder_output_lengths)
        return acoustic_log_probs.squeeze()

    @torch.jit.export
    def reset(self) -> None:
        self.caches = [(torch.zeros(self.batch_size, self.d_inner, self.d_state),
                        torch.zeros(self.batch_size, self.d_inner, self.d_conv - 1)) for _ in range(self.n_layers)]
