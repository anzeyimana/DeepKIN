import math
import os
from itertools import accumulate
import progressbar

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from deepkin.clib.libkinlp.kinlp_model import read_all_affixes, read_all_afsets, read_corr_table, id_to_affix, \
    decode_word, pos_tag_initials
from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib, parse_text_to_morpho_sentence
from deepkin.models.arguments import py_trainer_args
from deepkin.models.kinyagpt import KinyaGPT, generate_square_subsequent_mask
from deepkin.models.modules import BaseConfig
from deepkin.utils.misc_functions import normalize_kinya_text

english_BOS_idx = 0
english_EOS_idx = 2

PAD_ID = 0
UNK_ID = 1
MSK_ID = 2
BOS_ID = 3
EOS_ID = 4

def gpt_init_decode_model():
    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
    home_path = KINLP_HOME + "/"
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    build_kinlpy_lib()
    from kinlpy import ffi, lib

    lib.init_kinlp_socket()
    cfg = BaseConfig()

    args = py_trainer_args(list_args=[], silent=True)

    gpt_model = KinyaGPT(args, cfg).to(device)
    kb_state_dict = torch.load(home_path + "models/kinyagpt_final_2022-12-23_operated_full_base_2022-12-13.pt",
                               map_location=device)
    gpt_model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict

    f = open(home_path + "data/plain/stems_tokens_33K_vocab_2022-12-12.csv", 'r')
    text_lines = [line.rstrip('\n') for line in f.readlines()]
    f.close()
    stems_vocab = []
    for line in text_lines:
        double_comma_idx = line.find(",,")
        if double_comma_idx >= 0:
            stems_vocab.append(line[:(double_comma_idx + 1)])
        else:
            single_comma_idx = line.find(",")
            if single_comma_idx >= 0:
                stems_vocab.append(line[:single_comma_idx])
            else:
                raise ValueError('Invalid vocab file: ' + line)

    all_affixes = read_all_affixes(home_path + "data/plain/affixes_prob_file_2022-05-05.csv")
    all_afsets = read_all_afsets(home_path + "data/plain/afsets_prob_file_2022-05-05.csv")
    all_afsets_inverted_index = {a.key: a for a in all_affixes}

    afset_affix_corr = read_corr_table(home_path + "data/plain/afset_affix_corr_log_2022-12-08.txt")
    afset_stem_corr = read_corr_table(home_path + "data/plain/afset_stem_corr_log_2022-12-08.txt")
    pos_afset_corr = read_corr_table(home_path + "data/plain/pos_afset_corr_log_2022-12-08.txt")
    pos_stem_corr = read_corr_table(home_path + "data/plain/pos_stem_corr_log_2022-12-08.txt")

    afset_affix_slot_corr = set()
    for key in afset_affix_corr.keys():
        tk = key.split('-')
        af = id_to_affix(int(tk[1]), all_affixes)
        if af is not None:
            new_key = '{}-{}:{}'.format(tk[0], af.wt, af.slot)
            afset_affix_slot_corr.add(new_key)

    return (gpt_model, ffi, lib, device,
            stems_vocab, all_affixes, all_afsets, all_afsets_inverted_index,
            afset_affix_corr, afset_stem_corr, pos_afset_corr, pos_stem_corr,
            afset_affix_slot_corr)


def initial_outputs(sentence, model_setup):
    (gpt_model, ffi, lib, device,
     stems_vocab, all_affixes, all_afsets, all_afsets_inverted_index,
     afset_affix_corr, afset_stem_corr, pos_afset_corr, pos_stem_corr,
     afset_affix_slot_corr) = model_setup
    parsed_sentence = parse_text_to_morpho_sentence(ffi, lib, sentence)
    output_list = [(BOS_ID, BOS_ID, BOS_ID, [], '<s>')]
    for tok in parsed_sentence.tokens:
        output_list.append((tok.stem_id, tok.pos_tag_id, tok.lm_morph_id, tok.affixes, tok.raw_surface_form))
    outputs = [(output_list, 0.0, False)]
    init_length = len(output_list)
    return outputs, init_length


def batch_data(outputs, device):
    batch_stems = []
    batch_pos_tags = []
    batch_lm_morphs = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_input_sequence_lengths = []
    for rw, prob, eos in outputs:
        for stem, pos, afset, affixes, tkn in rw:
            batch_stems.append(stem)
            batch_pos_tags.append(pos)
            batch_lm_morphs.append(afset)
            batch_affixes.extend(affixes)
            batch_tokens_lengths.append(len(affixes))
        batch_input_sequence_lengths.append(len(rw))

    if len(batch_affixes) == 0:
        batch_affixes = [0]
        batch_tokens_lengths[-1] = 1

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes)#.to(device)
    stems = torch.tensor(batch_stems).to(device)

    cfg = BaseConfig()

    pred_affixes_list = [batch_affixes[x - y: x] for x, y in zip(accumulate(batch_tokens_lengths), batch_tokens_lengths)]
    afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
    for i,lst in enumerate(pred_affixes_list):
        if (len(lst) > 0):
            afx_prob[i,lst] = 1.0
    affixes_prob = afx_prob.to(device, dtype=torch.float)

    # Needed to fix decoding start bug
    # if len(affixes) == 0:
    #     affixes.append(0)
    #     tokens_lengths[-1] = 1
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

    data_item = (lm_morphs, pos_tags, tokens_lengths, stems, input_sequence_lengths, affixes_prob,
                 afx_padded, m_masks_padded, input_masks_padded, decoder_mask)
    return data_item


def join_token(txt, space):
    if (txt[:2] == '@@'):
        # txt = '<'+txt[2:]+'>'
        txt = txt[2:]
        space = False
    elif (txt[:1] == '▁'):
        # txt = '<'+txt[1:]+'>'
        txt = txt[1:]
    return (' ' + txt) if space else txt


def decode_kinya_sequence(seq, upper_first=True):
    ret = ''
    tag_dict = dict()
    for stem, pos, afset, affixes, txt in seq:
        tag_dict[txt] = pos_tag_initials(pos)
    for stem, pos, afset, affixes, txt in seq:
        if (txt != '<s>') and (txt != '</s>'):
            if len(ret) > 0:
                ret += join_token(txt, True)
            else:
                ret += join_token(txt, False)
    return normalize_kinya_text(ret, tag_dict=tag_dict, upper_first=upper_first)

def decode_batch_item(model_setup, logits, bidx, seq, log_prob, eos_flag, max_morpho_inference_table_length,
                      prob_cutoff=0.3, affix_prob_cutoff=0.3, affix_min_prob=0.3, lprob_score_delta=2.0):
    UNK_AFSET = 24118
    UNK_POS = 153

    (gpt_model, ffi, lib, device,
     stems_vocab, all_affixes, all_afsets, all_afsets_inverted_index,
     afset_affix_corr, afset_stem_corr, pos_afset_corr, pos_stem_corr,
     afset_affix_slot_corr) = model_setup

    (next_stems, next_pos_tags, next_lm_morphs, next_affixes) = logits

    new_outputs = []
    stem_predictions_prob, stem_predictions = torch.topk(next_stems[bidx, :],
                                                         min(next_stems.size(1), max_morpho_inference_table_length),
                                                         dim=-1)
    stem_predictions = stem_predictions.cpu().tolist()
    stem_predictions_prob = stem_predictions_prob.cpu().tolist()

    pos_tag_predictions_prob, pos_tag_predictions = torch.topk(next_pos_tags[bidx, :],
                                                               min(next_pos_tags.size(1),
                                                                   max_morpho_inference_table_length), dim=-1)
    pos_tag_predictions = pos_tag_predictions.cpu().tolist()
    pos_tag_predictions_prob = pos_tag_predictions_prob.cpu().tolist()

    # -----------------------------------------------------
    # Hack: Cap UNK POS tag prob to max non unk
    max_normal_pos_prob = max(
        [pp for xp, pp in zip(pos_tag_predictions, pos_tag_predictions_prob) if xp != UNK_POS])
    unk_pos_prob = max(
        [pp if (xp == UNK_POS) else 0.0 for xp, pp in zip(pos_tag_predictions, pos_tag_predictions_prob)])
    if unk_pos_prob > max_normal_pos_prob:
        unk_pos_prob = max_normal_pos_prob
    pos_tag_predictions_prob = [unk_pos_prob if (xp == UNK_POS) else pp for xp, pp in
                                zip(pos_tag_predictions, pos_tag_predictions_prob)]
    # -----------------------------------------------------

    lm_morph_predictions_prob, lm_morph_predictions = torch.topk(next_lm_morphs[bidx, :],
                                                                 min(next_lm_morphs.size(1),
                                                                     max_morpho_inference_table_length), dim=-1)
    lm_morph_predictions = lm_morph_predictions.cpu().tolist()
    lm_morph_predictions_prob = lm_morph_predictions_prob.cpu().tolist()

    # -----------------------------------------------------
    # Hack: Cap UNK AFSET prob to Max Non-Unk
    max_normal_afset_prob = max(
        [pp for xp, pp in zip(lm_morph_predictions, lm_morph_predictions_prob) if xp != UNK_AFSET])
    unk_afset_prob = max(
        [pp if (xp == UNK_AFSET) else 0.0 for xp, pp in zip(lm_morph_predictions, lm_morph_predictions_prob)])
    if unk_afset_prob > max_normal_afset_prob:
        unk_afset_prob = max_normal_afset_prob
    lm_morph_predictions_prob = [unk_afset_prob if (xp == UNK_AFSET) else pp for xp, pp in
                                 zip(lm_morph_predictions, lm_morph_predictions_prob)]
    # -----------------------------------------------------

    affixes_predictions_prob, affixes_predictions = torch.topk(next_affixes[bidx, :], 32, dim=-1)

    affixes_predictions = affixes_predictions.cpu().tolist()
    affixes_predictions_prob = affixes_predictions_prob.cpu().tolist()

    return_list = decode_word(stem_predictions, pos_tag_predictions,
                              lm_morph_predictions, affixes_predictions,
                              stem_predictions_prob, pos_tag_predictions_prob,
                              lm_morph_predictions_prob, affixes_predictions_prob,
                              stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                              pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                              ffi, lib,
                              prob_cutoff=prob_cutoff, affix_prob_cutoff=affix_prob_cutoff,
                              affix_min_prob=affix_min_prob, lprob_score_delta=lprob_score_delta)

    for ret_item in return_list:
        (gen_word,
         surface_forms,
         ret_stems,
         ret_pos_tags,
         ret_afsets,
         ret_affixes,
         ret_probs,
         ret_non_afset_affixes_stats_list) = ret_item
        for j in range(len(ret_stems)):
            if surface_forms[j][1]:
                new_word = surface_forms[j][0]
                new_seq = []
                new_seq.extend(seq)
                new_seq.append((ret_stems[j], ret_pos_tags[j], ret_afsets[j], ret_affixes[j], new_word))
                new_prob = ret_probs[j]

                if not np.isfinite(new_prob):
                    new_prob = 1e-20
                if new_prob <= 0.0:
                    new_prob = 1e-20
                new_prob = log_prob + math.log(new_prob)
                eos = (ret_stems[j] == EOS_ID) and (ret_pos_tags[j] == EOS_ID) and (ret_afsets[j] == EOS_ID)
                if eos:
                    alpha = 1.0
                    length_penalty = math.pow(((len(new_seq) + 5.0) / 6.0), alpha)
                    new_prob = (new_prob / length_penalty)
                new_outputs.append((new_seq, new_prob, eos))
    return new_outputs


def expand_kinya_outputs(model_setup, completed_outputs, outputs, logits,
                         _max_morpho_inference_table_length_,
                         max_batch_size, debug=False,
                         prob_cutoff=0.3, affix_prob_cutoff=0.3,
                         affix_min_prob=0.3, lprob_score_delta=2.0):
    # src_attn_weights: N,T,S
    N = len(outputs)
    new_outputs = []
    for bidx in range(N):
        seq, log_prob, eos_flag = outputs[bidx]
        batch_outs = decode_batch_item(model_setup, logits, bidx, seq, log_prob, eos_flag,
                                        _max_morpho_inference_table_length_,
                                        prob_cutoff=prob_cutoff, affix_prob_cutoff=affix_prob_cutoff,
                                        affix_min_prob=affix_min_prob, lprob_score_delta=lprob_score_delta)
        new_outputs.extend(batch_outs)
    new_completed_outputs = [x for x in new_outputs if x[2]]
    pending_outputs = [x for x in new_outputs if not x[2]]

    # Sort by output probability
    pending_outputs = sorted(pending_outputs, key=lambda x: x[1], reverse=True)
    pending_outputs = pending_outputs[:max_batch_size]

    completed_outputs = completed_outputs + new_completed_outputs
    completed_outputs = sorted(completed_outputs, key=lambda x: x[1], reverse=True)

    outputs = (pending_outputs)
    return outputs, completed_outputs


def gpt_predict(gpt_model, data_item):
    (lm_morphs, pos_tags, tokens_lengths, stems, input_sequence_lengths, affixes_prob,
     afx_padded, m_masks_padded, input_masks_padded, decoder_mask) = data_item
    (tr_hidden_state, logits) = gpt_model.predict(lm_morphs, pos_tags, stems, input_sequence_lengths,
                                                      afx_padded, m_masks_padded, input_masks_padded, decoder_mask)
    return logits
def gpt_auto_complete(model_setup,
                      sentence,
                      max_text_length,
                      max_completed,
                      max_morpho_inference_table_length=20,
                      max_batch_size=8,
                      debug=False,
                      show_progress=False,
                      upper_first = True,
                      prob_cutoff=0.3, affix_prob_cutoff=0.3,
                      affix_min_prob=0.3, lprob_score_delta=2.0):
    (gpt_model, ffi, lib, device,
     stems_vocab, all_affixes, all_afsets, all_afsets_inverted_index,
     afset_affix_corr, afset_stem_corr, pos_afset_corr, pos_stem_corr,
     afset_affix_slot_corr) = model_setup
    completed_outputs = []

    outputs, init_length = initial_outputs(sentence, model_setup)

    batch_data_item = batch_data(outputs, device)

    seq_count = 0
    if show_progress:
        with progressbar.ProgressBar(max_value=max_text_length+2, redirect_stdout=True) as bar:
            while True:
                bar.update(seq_count)
                with torch.no_grad():
                    logits = gpt_predict(gpt_model, batch_data_item)
                outputs, completed_outputs = expand_kinya_outputs(model_setup, completed_outputs, outputs, logits,
                                                                  max_morpho_inference_table_length, max_batch_size,
                                                                  debug=debug, prob_cutoff=prob_cutoff,
                                                                  affix_prob_cutoff=affix_prob_cutoff,
                                                                  affix_min_prob=affix_min_prob,
                                                                  lprob_score_delta=lprob_score_delta)
                seq_count += 1
                if (len(outputs) <= 0) or (seq_count >= max_text_length) or (len(completed_outputs) >= max_completed):
                    break
                batch_data_item = batch_data(outputs, device)
                if seq_count > 0:
                    debug = False
    else:
        while True:
            with torch.no_grad():
                logits = gpt_predict(gpt_model, batch_data_item)
            outputs, completed_outputs = expand_kinya_outputs(model_setup, completed_outputs, outputs, logits,
                                                              max_morpho_inference_table_length, max_batch_size,
                                                              debug=debug, prob_cutoff=prob_cutoff,
                                                              affix_prob_cutoff=affix_prob_cutoff,
                                                              affix_min_prob=affix_min_prob,
                                                              lprob_score_delta=lprob_score_delta)
            seq_count += 1
            if (len(outputs) <= 0) or (seq_count >= max_text_length) or (len(completed_outputs) >= max_completed):
                break
            batch_data_item = batch_data(outputs, device)
            if seq_count > 0:
                debug = False

    complete = sorted([(prob, decode_kinya_sequence(seq[init_length:], upper_first=upper_first)) for seq, prob, eos in completed_outputs], key=lambda x: x[0],
                      reverse=True)
    pending = sorted([(prob, decode_kinya_sequence(seq[init_length:], upper_first=upper_first)) for seq, prob, eos in outputs],
                     key=lambda x: x[0],
                     reverse=True)
    return complete, pending[:max_batch_size]
