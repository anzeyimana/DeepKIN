from __future__ import print_function, division

import random
from datetime import datetime
from typing import List

import torch
from deepkin.clib.libkinlp.kinlpy import ParsedMorphoSentence
from deepkin.models.cls_data_loaders import prepare_cls_data
from torch.utils.data import Dataset

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
import progressbar

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TagDataset(Dataset):

    def __init__(self,
                 input_lines: List[str],
                 tag_lines: List[str],
                 tags_dict):
        self.itemized_data = []
        num_lines = len(input_lines)
        assert len(input_lines) == len(tag_lines), "Mismatch between data and labels"
        with progressbar.ProgressBar(initial_value=0, max_value=(2*num_lines), redirect_stdout=True) as bar:
            bar.update(0)
            for idx in range(num_lines):
                if (len(input_lines[idx])>0):
                    sentence0 = ParsedMorphoSentence(input_lines[idx])
                    tags = [tg for tg in tag_lines[idx].split()]
                    assert (len(tags) == len(sentence0.tokens)), "Tag misalignment at example # {} '{}'".format(idx+1, input_lines[idx])
                    extended_tags_idx = []
                    for tag, token in zip(tags, sentence0.tokens):
                        if tag[0] == 'B':
                            extended_tags_idx.append(tags_dict[tag])
                            extended_tags_idx.extend([tags_dict[('I'+tag[1:])]] * (len(token.extra_tokens_ids)))
                        else:
                            extended_tags_idx.extend([tags_dict[tag]] * (len(token.extra_tokens_ids) + 1))
                    assert len(extended_tags_idx) == sum([(len(t.extra_tokens_ids)+1) for t in sentence0.tokens]), "Mismatch tags len vs tokens len"
                    self.itemized_data.append((extended_tags_idx, prepare_cls_data(sentence0)))
                    if (((idx+1) % 1000) == 0):
                        bar.update(idx+1)
        random.shuffle(self.itemized_data)
    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]


def tag_data_collate_wrapper(batch_items):
    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []

    batch_input_sequence_lengths = []
    batch_tags = []

    for bidx,data_item in enumerate(batch_items):
        (seq_tags,
         details) = data_item
        (seq_lm_morphs,
         seq_pos_tags,
         seq_affixes,
         seq_tokens_lengths,
         seq_stems) = details

        assert len(seq_affixes) == sum(
            seq_tokens_lengths), "@tag_data_collate_wrapper--item#{} stems: [{}]: Mismatch token lengths affixes={}({}) vs lengths={}({})".format(bidx,
                                                                                                                                          seq_stems,
            seq_affixes, len(seq_affixes), seq_tokens_lengths, sum(seq_tokens_lengths))

        assert len(seq_tags) == (len(seq_stems)-1)

        batch_tags.extend(seq_tags)


        batch_lm_morphs.extend(seq_lm_morphs)
        batch_pos_tags.extend(seq_pos_tags)
        batch_affixes.extend(seq_affixes)
        batch_tokens_lengths.extend(seq_tokens_lengths)
        batch_stems.extend(seq_stems)

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))

    assert len(batch_affixes) == sum(
        batch_tokens_lengths), "@tag_data_collate_wrapper: Mismatch token lengths affixes={} vs lengths={}".format(
        len(batch_affixes), sum(batch_tokens_lengths))
    data_item = (batch_tags,
                 (batch_lm_morphs,
                 batch_pos_tags,
                 batch_affixes,
                 batch_tokens_lengths,
                 batch_stems,
                 batch_input_sequence_lengths))
    return data_item


def tag_model_forward(batch_data_item, model, shared_encoder, device):
    (batch_tags,
     (batch_lm_morphs,
     batch_pos_tags,
     batch_affixes,
     batch_tokens_lengths,
     batch_stems,
     batch_input_sequence_lengths)) = batch_data_item

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes).to(device)
    stems = torch.tensor(batch_stems).to(device)

    assert affixes.nelement() == sum(tokens_lengths), "@tag_model_forward: Mismatch token lengths affixes={} vs lengths={}".format(
        affixes.nelement(), sum(tokens_lengths))

    scores = model(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths, shared_encoder=shared_encoder)
    return scores, batch_tags

def tag_model_predict(data_item, model, shared_encoder, device):
    (tags,
     (lm_morphs,
     pos_tags,
     affixes,
     tokens_lengths,
     stems)) = data_item

    tokens_lengths = tokens_lengths
    input_sequence_lengths = [len(tokens_lengths)]

    lm_morphs = torch.tensor(lm_morphs).to(device)
    pos_tags = torch.tensor(pos_tags).to(device)
    affixes = torch.tensor(affixes).to(device)
    stems = torch.tensor(stems).to(device)

    scores = model(lm_morphs, pos_tags, affixes, tokens_lengths, stems, input_sequence_lengths, shared_encoder=shared_encoder)
    predicted_labels = torch.argmax(scores, dim=1)
    assert len(tags) == predicted_labels.shape[-1]
    return scores, predicted_labels, tags

