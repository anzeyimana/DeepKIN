from __future__ import print_function, division

import random
import time
import warnings
from typing import List
import progressbar

import torch
from torch.utils.data import Dataset

from deepkin.clib.libkinlp.kinlpy import ParsedMorphoSentence, BOS_ID, EOS_ID, parse_text_to_morpho_sentence
from deepkin.utils.misc_functions import read_lines

# Ignore warnings
warnings.filterwarnings("ignore")

def prepare_cls_reg_input_segments(input_segments: List[ParsedMorphoSentence], max_len=512):
    lm_morphs = []
    pos_tags = []
    affixes = []
    tokens_lengths = []
    stems = []

    # Add <CLS> Token
    lm_morphs.append(BOS_ID)
    pos_tags.append(BOS_ID)
    stems.append(BOS_ID)
    tokens_lengths.append(0)

    started = False
    for segment in input_segments:
        if started:
            # Add <SEP> Token
            lm_morphs.append(EOS_ID)
            pos_tags.append(EOS_ID)
            stems.append(EOS_ID)
            tokens_lengths.append(0)
        for token in segment.tokens:
            if (len(tokens_lengths) >= max_len):
                break
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append(token.stem_id)
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))
            for tid in token.extra_tokens_ids:
                if (len(tokens_lengths) >= max_len):
                    break
                lm_morphs.append(token.lm_morph_id)
                pos_tags.append(token.pos_tag_id)
                stems.append(tid)
                affixes.extend(token.affixes)
                tokens_lengths.append(len(token.affixes))
        if (len(tokens_lengths) >= max_len):
            break
        started = True

    assert len(affixes) == sum(tokens_lengths), "@prepare_cls_reg_input_segments: Mismatch token lengths affixes={} vs lengths={}".format(
        len(affixes), sum(tokens_lengths))
    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems)

class ClsRegDataset(Dataset):

    def __init__(self, mydb, input_file_tsv,
                 requires_morpho_analysis=True,
                 label_dict=None,
                 regression_target=False,
                 multi_label_cls=False,
                 regression_scale_factor=1.0,
                 max_seq_len = 512,
                 task_id=None,
                 update_task_state_func = None):
        from kinlpy import ffi, lib
        lib.init_kinlp_socket()
        print('Morphokin library ready via Unix Socket!', flush=True)
        if (task_id is not None) and (update_task_state_func is not None):
            update_task_state_func(mydb, task_id, 'Reading input data', 0.0)
        lines = read_lines(input_file_tsv)
        lines_split = [line.split('\t') for line in lines]
        self.itemized_data = []
        start = time.time()
        with progressbar.ProgressBar(max_value=len(lines_split), redirect_stdout=True) as bar:
            for idx,cols in enumerate(lines_split):
                if ((idx % 100) == 0) and (task_id is not None) and (update_task_state_func is not None):
                    bar.update(idx)
                    now = time.time()
                    update_task_state_func(mydb, task_id, 'Parsing input data', 100.0 * (idx) / len(lines_split), eta = ((now-start) * (len(lines_split)-(idx)) / (idx+1)))
                if regression_target:
                    label = (float(cols[-1]) / regression_scale_factor)
                elif multi_label_cls:
                    label = [label_dict[l.strip()] for l in cols[-1].split(',')]
                else:
                    label = label_dict[cols[-1]]
                if requires_morpho_analysis:
                    inputs = prepare_cls_reg_input_segments(
                        [parse_text_to_morpho_sentence(ffi, lib, col) for col in cols[:-1]],
                        max_len=max_seq_len)
                else:
                    inputs = prepare_cls_reg_input_segments([ParsedMorphoSentence(txt) for txt in cols[:-1]],
                                                   max_len=max_seq_len)
                self.itemized_data.append((label,inputs))
        random.shuffle(self.itemized_data)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

def cls_reg_data_collate_wrapper(batch_items):
    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []

    batch_input_sequence_lengths = []
    batch_labels = []

    for bidx,data_item in enumerate(batch_items):
        (label,
         details) = data_item
        (seq_lm_morphs,
         seq_pos_tags,
         seq_affixes,
         seq_tokens_lengths,
         seq_stems) = details

        if label is not None:
            batch_labels.append(label)

        batch_lm_morphs.extend(seq_lm_morphs)
        batch_pos_tags.extend(seq_pos_tags)
        batch_affixes.extend(seq_affixes)
        batch_tokens_lengths.extend(seq_tokens_lengths)
        batch_stems.extend(seq_stems)

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))

    data_item = (batch_labels,
                 (batch_lm_morphs,
                 batch_pos_tags,
                 batch_affixes,
                 batch_tokens_lengths,
                 batch_stems,
                 batch_input_sequence_lengths))
    return data_item


def cls_reg_model_forward(batch_data_item, model, device):
    (batch_labels,
     details) = batch_data_item
    (batch_lm_morphs,
     batch_pos_tags,
     batch_affixes,
     batch_tokens_lengths,
     batch_stems,
     batch_input_sequence_lengths) = details

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes).to(device)
    stems = torch.tensor(batch_stems).to(device)

    scores = model(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths)
    return scores, batch_labels

def cls_reg_model_batch_predict(batch_data_item, cls_model, device):
    (batch_labels,
     details) = batch_data_item
    (batch_lm_morphs,
     batch_pos_tags,
     batch_affixes,
     batch_tokens_lengths,
     batch_stems,
     batch_input_sequence_lengths) = details

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes).to(device)
    stems = torch.tensor(batch_stems).to(device)

    scores = cls_model(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths)
    return scores

def cls_reg_model_predict(data_item, model, device):
    (true_label, details) = data_item
    (lm_morphs,
     pos_tags,
     affixes,
     tokens_lengths,
     stems) = details

    tokens_lengths = tokens_lengths
    input_sequence_lengths = [len(tokens_lengths)]

    lm_morphs = torch.tensor(lm_morphs).to(device)
    pos_tags = torch.tensor(pos_tags).to(device)
    affixes = torch.tensor(affixes).to(device)
    stems = torch.tensor(stems).to(device)

    scores = model(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths)
    predicted_label = torch.argmax(scores, dim=1)
    return scores, predicted_label.item(), true_label
