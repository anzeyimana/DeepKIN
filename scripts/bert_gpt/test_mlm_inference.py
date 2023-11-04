from __future__ import print_function, division

import random
# Ignore warnings
import warnings
from typing import List

import numpy as np
import torch

from deepkin.clib.libkinlp.kinlp_model import read_all_affixes, read_all_afsets, read_corr_table, decode_word, \
    make_surface_form, id_to_affix
from deepkin.clib.libkinlp.kinlpy import ParsedMorphoSentence
from deepkin.models.arguments import py_trainer_args
from deepkin.models.data import prepare_mlm_data_from_sentence, mlm_model_predict
from deepkin.models.kinyabert import KinyaBERT_PretrainModel
from deepkin.models.modules import BaseConfig
from deepkin.utils.misc_functions import time_now

warnings.filterwarnings("ignore")
import os
# %%
from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib, parse_text_to_morpho_sentence

build_kinlpy_lib()
from kinlpy import ffi, lib

lib.init_kinlp_socket()
print('KINLPY Lib Ready!', flush=True)
cfg = BaseConfig()
print('BaseConfig: \n\ttot_num_stems: {}\n'.format(cfg.tot_num_stems),
      '\ttot_num_affixes: {}\n'.format(cfg.tot_num_affixes),
      '\ttot_num_lm_morphs: {}\n'.format(cfg.tot_num_lm_morphs),
      '\ttot_num_pos_tags: {}\n'.format(cfg.tot_num_pos_tags), flush=True)

KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
home_path = KINLP_HOME+"/"
USE_GPU = True

device = torch.device('cpu')
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using device: ', device, flush=True)

def get_mlm_data_item(parsed_sentences: List[ParsedMorphoSentence], max_seq_len):
    seq_lm_morphs = []
    seq_pos_tags = []
    seq_affixes = []
    seq_tokens_lengths = []
    seq_stems = []
    seq_predicted_tokens_idx = []
    seq_predicted_tokens_affixes_idx = []
    seq_predicted_stems = []
    seq_predicted_pos_tags = []
    seq_predicted_lm_morphs = []
    seq_predicted_affixes = []
    seq_predicted_affixes_lengths = []

    for sentence in parsed_sentences:
        (lm_morphs,
         pos_tags,
         affixes,
         tokens_lengths,
         stems,
         predicted_tokens_idx,
         predicted_tokens_affixes_idx,
         predicted_stems,
         predicted_pos_tags,
         predicted_lm_morphs,
         predicted_affixes,
         predicted_affixes_lengths) = prepare_mlm_data_from_sentence(sentence, (len(seq_stems) == 0), cfg)

        if (len(seq_tokens_lengths) + len(tokens_lengths)) <= max_seq_len:
            seq_predicted_tokens_affixes_idx.extend([len(seq_predicted_tokens_idx) + idx for idx in predicted_tokens_affixes_idx])
            seq_predicted_tokens_idx.extend([len(seq_tokens_lengths) + idx for idx in predicted_tokens_idx])
            seq_lm_morphs.extend(lm_morphs)
            seq_pos_tags.extend(pos_tags)
            seq_affixes.extend(affixes)
            seq_tokens_lengths.extend(tokens_lengths)
            seq_stems.extend(stems)
            seq_predicted_stems.extend(predicted_stems)
            seq_predicted_pos_tags.extend(predicted_pos_tags)
            seq_predicted_lm_morphs.extend(predicted_lm_morphs)
            seq_predicted_affixes.extend(predicted_affixes)
            seq_predicted_affixes_lengths.extend(predicted_affixes_lengths)
        else:
            break
    # Return
    data_item = (seq_lm_morphs,
                 seq_pos_tags,
                 seq_affixes,
                 seq_tokens_lengths,
                 seq_stems,
                 seq_predicted_tokens_idx,
                 seq_predicted_tokens_affixes_idx,
                 seq_predicted_stems,
                 seq_predicted_pos_tags,
                 seq_predicted_lm_morphs,
                 seq_predicted_affixes,
                 seq_predicted_affixes_lengths)
    return data_item

def better_mlm_inference_run(args, parsed_data, kinya_bert_model):
    import progressbar
    progressCount = 0

    stem_acc = 0.0
    stem_tot = 0.0

    pos_acc = 0.0
    pos_tot = 0.0

    afset_acc = 0.0
    afset_tot = 0.0

    affix_acc = 0.0
    affix_tot = 0.0

    token_acc = 0.0
    token_tot = 0.0

    f = open(home_path+"data/plain/stems_tokens_33K_vocab_2022-12-12.csv", 'r')
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

    all_affixes = read_all_affixes(home_path+"data/plain/affixes_prob_file_2022-05-05.csv")
    all_afsets = read_all_afsets(home_path+"data/plain/afsets_prob_file_2022-05-05.csv")
    all_afsets_inverted_index = {a.key: a for a in all_affixes}

    afset_affix_corr = read_corr_table(home_path+"data/plain/afset_affix_corr_log_2022-12-08.txt")
    afset_stem_corr = read_corr_table(home_path+"data/plain/afset_stem_corr_log_2022-12-08.txt")
    pos_afset_corr = read_corr_table(home_path+"data/plain/pos_afset_corr_log_2022-12-08.txt")
    pos_stem_corr = read_corr_table(home_path+"data/plain/pos_stem_corr_log_2022-12-08.txt")

    afset_affix_slot_corr = set()
    for key in afset_affix_corr.keys():
        tk = key.split('-')
        af = id_to_affix(int(tk[1]), all_affixes)
        if af is not None:
            new_key = '{}-{}:{}'.format(tk[0], af.wt, af.slot)
            afset_affix_slot_corr.add(new_key)

    print('Running inference ...', flush=True)

    with progressbar.ProgressBar(max_value=(args.num_inference_runs * len(parsed_data)), redirect_stdout=True) as bar:
        bar.update(progressCount)
        for _i_ in range(args.num_inference_runs):
            for di, it in enumerate(parsed_data):
                text_lines, parsed_sentences = it
                data_item = get_mlm_data_item(parsed_sentences, 510)

                (seq_lm_morphs,
                 seq_pos_tags,
                 seq_affixes,
                 seq_tokens_lengths,
                 seq_stems,
                 seq_predicted_tokens_idx,
                 seq_predicted_tokens_affixes_idx,
                 seq_predicted_stems,
                 seq_predicted_pos_tags,
                 seq_predicted_lm_morphs,
                 seq_predicted_affixes,
                 seq_predicted_affixes_lengths) = data_item

                if (len(seq_lm_morphs) > 10) and (len(seq_lm_morphs) < 512):
                    ((stem_predictions,
                      pos_tag_predictions,
                      lm_morph_predictions,
                      affixes_predictions),
                     (stem_predictions_prob,
                      pos_tag_predictions_prob,
                      lm_morph_predictions_prob,
                      affixes_predictions_prob)) = mlm_model_predict(data_item, kinya_bert_model, device, 24, max_top_predictions = 8)

                    print('Text # {} @ {}'.format(di, len(seq_lm_morphs)))
                    print(text_lines)

                    j = 0
                    k = 0
                    af_start = 0
                    paf_start = 0
                    for i in range(len(seq_stems)):
                        af_len = seq_tokens_lengths[i]
                        afx = seq_affixes[af_start:(af_start+af_len)]
                        af_start += af_len

                        # input_str = pos_tag_view(seq_pos_tags[i]) + '/' + afset_view(seq_lm_morphs[i], all_afsets) + '/' + \
                        #             stems_vocab[seq_stems[i]] + '/' + '-'.join([affix_view(a, all_affixes) for a in afx])

                        input_str, _ = make_surface_form(seq_stems[i], afx, stems_vocab, all_affixes, ffi, lib)

                        if i in seq_predicted_tokens_idx:
                            paf_len = 0
                            if j in seq_predicted_tokens_affixes_idx:
                                paf_len = seq_predicted_affixes_lengths[k]
                                k += 1
                            pafx = seq_predicted_affixes[paf_start:(paf_start + paf_len)]
                            paf_start += paf_len

                            # target_str = pos_tag_view(seq_predicted_pos_tags[j]) + '/' + \
                            #     afset_view(seq_predicted_lm_morphs[j], all_afsets) + '/' + \
                            #     stems_vocab[seq_predicted_stems[j]] + '/' +\
                            #     '-'.join([affix_view(a, all_affixes) for a in pafx])

                            target_str, _ = make_surface_form(seq_predicted_stems[j], pafx, stems_vocab, all_affixes, ffi, lib)

                            return_list = decode_word(stem_predictions[j], pos_tag_predictions[j],
                                                      lm_morph_predictions[j], affixes_predictions[j],
                                                      stem_predictions_prob[j], pos_tag_predictions_prob[j],
                                                      lm_morph_predictions_prob[j], affixes_predictions_prob[j],
                                                      stems_vocab, all_afsets, all_affixes, all_afsets_inverted_index,
                                                      pos_afset_corr, pos_stem_corr, afset_stem_corr, afset_affix_slot_corr,
                                                      ffi, lib,
                                                      prob_cutoff=0.8, affix_prob_cutoff=0.3,
                                                      affix_min_prob=0.1, lprob_score_delta=20.0)
                            if len(return_list) > 0:
                                (gen_word,
                                 surface_forms,
                                 ret_stems,
                                 ret_pos_tags,
                                 ret_afsets,
                                 ret_affixes,
                                 ret_probs,
                                 ret_non_afset_affixes_stats_list) = return_list[0]
                                if len(ret_stems) > 0:
                                    if seq_predicted_stems[j] == ret_stems[0]:
                                        stem_acc += 1.0
                                    if seq_predicted_pos_tags[j] == ret_pos_tags[0]:
                                        pos_acc += 1.0
                                    if seq_predicted_lm_morphs[j] == ret_afsets[0]:
                                        afset_acc += 1.0
                                    if target_str == gen_word:
                                        token_acc += 1.0
                                real_afx = set(pafx)
                                if len(ret_stems) > 0:
                                    pred_afx = set(ret_affixes[0])
                                    affix_acc += len(real_afx.intersection(pred_afx))

                                affix_tot += len(real_afx)
                                stem_tot += 1.0
                                pos_tot += 1.0
                                afset_tot += 1.0
                                token_tot += 1.0

                                print('{}\t{}\t->\t{}'.format(input_str, target_str, gen_word))

                                j += 1
                            else:
                                real_afx = set(pafx)
                                if len(ret_stems) > 0:
                                    pred_afx = set(ret_affixes[0])
                                    affix_acc += len(real_afx.intersection(pred_afx))

                                affix_tot += len(real_afx)
                                stem_tot += 1.0
                                pos_tot += 1.0
                                afset_tot += 1.0
                                token_tot += 1.0

                                print('{}\t{}\t->\t{}'.format(input_str, target_str, gen_word))

                                j += 1
                        else:
                            print(input_str)
                progressCount += 1
                bar.update(progressCount)
    return ((100.0*token_acc/token_tot), (100.0*stem_acc/stem_tot), ((100.0*pos_acc/pos_tot)), (100.0*afset_acc/afset_tot), (100.0*affix_acc/affix_tot))

def MLM_main_run(args, parsed_data, pretrained_model_file):
    pretrained_model = KinyaBERT_PretrainModel(args, cfg).to(device)
    kb_state_dict = torch.load(pretrained_model_file, map_location=device)
    pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])

    (token, stem, pos, afset, affix) = better_mlm_inference_run(args, parsed_data, pretrained_model)

    print('FINAL ACCURACIES ==> TOKEN: {:.2f}, STEM: {:.2f}, POS: {:.2f}, AFSET: {:.2f}, AFFIX: {:.2f}'.format(token, stem, pos, afset, affix), flush=True)


print(time_now(), 'Functions Ready!', flush=True)

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def test_all_MLM():
    import progressbar

    set_random_seeds(456)
    print(time_now(), 'Parsing data ...', flush=True)
    parsed_data = []
    progressCount = 0
    with progressbar.ProgressBar(max_value=1000, redirect_stdout=True) as bar:
        bar.update(progressCount)
        for docNum in range(1000):
            #docNum = random.randint(100, 500)
            input_file = home_path+"sample_docs/"+str(docNum)+".txt"
            f = open(input_file, 'r+')
            text_lines = [line.rstrip('\n') for line in f]
            f.close()
            parsed_tokens_lines = [parse_text_to_morpho_sentence(ffi, lib, line) for line in text_lines]
            parsed_data.append((text_lines,parsed_tokens_lines))
            progressCount += 1
            bar.update(progressCount)
            #break

    print(time_now(), 'Done parsing data!', flush=True)

    args = py_trainer_args()
    args.num_inference_runs = 1
    MLM_main_run(args, parsed_data, args.pretrained_model_file)

if __name__ == '__main__':
    test_all_MLM()

