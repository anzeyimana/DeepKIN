from __future__ import print_function, division

import warnings
from typing import List

import torch
from torch.utils.data import Dataset

from deepkin.clib.libkinlp.kinlpy import ParsedMorphoSentence, BOS_ID, EOS_ID

# Ignore warnings
warnings.filterwarnings("ignore")

def prepare_cls_data(sentence0: ParsedMorphoSentence, sentence1: ParsedMorphoSentence=None, max_len=512, add_eos = False):
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

    for token in sentence0.tokens:
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

    if (sentence1 is not None) and (len(tokens_lengths) < max_len):
        # Add <SEP> Token
        lm_morphs.append(EOS_ID)
        pos_tags.append(EOS_ID)
        stems.append(EOS_ID)
        tokens_lengths.append(0)
        for token in sentence1.tokens:
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

    if add_eos:
        lm_morphs.append(EOS_ID)
        pos_tags.append(EOS_ID)
        stems.append(EOS_ID)
        tokens_lengths.append(0)
    assert len(affixes) == sum(tokens_lengths), "@prepare_cls_data: Mismatch token lengths affixes={} vs lengths={}".format(
        len(affixes), sum(tokens_lengths))
    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems)

class ClsDataset(Dataset):

    def __init__(self,
                 lines_input0: List[str], lines_input1: List[str] = None,
                 label_dict=None, label_lines=None,
                 regression_target=False,
                 regression_scale_factor=1.0,
                 max_seq_len = 512,
                 add_eos = False):
        cls_dict = {label_dict[k]:k for k in label_dict}
        if label_lines is None:
            label_lines = [cls_dict[0] for _ in range(len(lines_input0))]
        assert len(lines_input0) == len(label_lines), "Lines&labels not equal"
        if (lines_input1 is not None):
            assert len(lines_input0) == len(lines_input1), "Lines 0 and 1 not equal"
            self.itemized_data = [
                ((float(label) / regression_scale_factor) if regression_target else label_dict[label],
                 prepare_cls_data(ParsedMorphoSentence(line0), sentence1=ParsedMorphoSentence(line1),
                                  max_len=max_seq_len, add_eos=add_eos)) for
                label, line0, line1 in zip(label_lines, lines_input0, lines_input1)]
        else:
            self.itemized_data = [
                ((float(label) / regression_scale_factor) if regression_target else label_dict[label],
                 prepare_cls_data(ParsedMorphoSentence(line0), max_len=max_seq_len, add_eos=add_eos)) for
                label, line0 in zip(label_lines, lines_input0)]

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

def cls_data_collate_wrapper(batch_items):
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


def cls_model_forward(batch_data_item, model, shared_encoder, device):
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

    scores = model(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths,
                   shared_encoder=shared_encoder)
    return scores, batch_labels

def cls_model_predict(data_item, model, shared_encoder, device):
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

    scores = model(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths,
                   shared_encoder=shared_encoder)
    predicted_label = torch.argmax(scores, dim=1)
    return scores, predicted_label.item(), true_label
