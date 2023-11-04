import math
import random
import sys
from typing import List

import progressbar
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from deepkin.models.syllabe_vocab import BOS_ID, EOS_ID, text_to_id_sequence

def get_random_start_line(doc_ends, num_text_lines):
    # Go to the next starting line
    dcx = random.randint(0, len(doc_ends) - 1) % len(doc_ends)
    start_line = (doc_ends[dcx] + 1) % num_text_lines
    if random.random() < 0.8:  # 80% of the time, start from anywhere within the corpus.
        end_line = (doc_ends[(dcx + 1) % len(doc_ends)] - 1)
        if end_line <= start_line:
            end_line = num_text_lines - 1
        if end_line > start_line:
            if random.random() < 0.6:
                # Start from within 2/3 of the beginning of chosen the document
                max_line = start_line + int(2.0 * (end_line - start_line) / 3.0)
                start_line = random.randint(start_line, max_line)
            else:
                # Start from within 3/4 of the beginning of chosen the document
                max_line = start_line + int(3.0 * (end_line - start_line) / 4.0)
                start_line = random.randint(start_line, max_line) % num_text_lines

    return start_line

def gather_gpt_single_sequence(text_lines: List[str], doc_ends, max_seq_len) -> List[int]:
    num_text_lines = len(text_lines)
    seq_syllabe_ids = []

    start_line = get_random_start_line(doc_ends, len(text_lines))
    id_list = text_to_id_sequence(text_lines[start_line % num_text_lines])
    if len(id_list) <= 0:
        start_line = (start_line + 1) % num_text_lines
        id_list = text_to_id_sequence(text_lines[start_line % num_text_lines])

    while len(id_list) > 0:
        if (len(seq_syllabe_ids) + len(id_list) + 2) < max_seq_len:
            seq_syllabe_ids.extend(id_list)
        else:
            return [BOS_ID] + seq_syllabe_ids + [EOS_ID]
        start_line = (start_line + 1) % num_text_lines
        id_list = text_to_id_sequence(text_lines[start_line % num_text_lines])
    return [BOS_ID] + seq_syllabe_ids + [EOS_ID]

def gather_itemized_gpt_data(text_lines: List[str], doc_ends, max_seq_len, max_batch_items, bar=None, num_lines=sys.maxsize):
    itemized_data = []
    for i in range(max_batch_items):
        itemized_data.append((dist.get_rank(),gather_gpt_single_sequence(text_lines, doc_ends, max_seq_len)))
        if (((len(itemized_data) % (math.floor(0.1 * max_batch_items) + 1)) == 0) and (bar is not None)):
            bar.update(len(itemized_data))
            sys.stdout.flush()
    return itemized_data

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class SyllabeGPTDataset(Dataset):

    def __init__(self, text_lines: List[str], doc_ends,
                 max_batch_items,
                 max_seq_len = 512,
                 rank = 0):
        self.max_seq_len = max_seq_len
        self.max_batch_items = max_batch_items
        self.rank = dist.get_rank()
        if (dist.get_rank() == 0):
            with progressbar.ProgressBar(max_value=max_batch_items, redirect_stdout=True) as bar:
                bar.update(0)
                sys.stdout.flush()
                self.itemized_data = gather_itemized_gpt_data(text_lines, doc_ends, max_seq_len, max_batch_items,
                                                              bar=bar,
                                                              num_lines=sys.maxsize)
        else:
            self.itemized_data = gather_itemized_gpt_data(text_lines, doc_ends, max_seq_len, max_batch_items,
                                                          bar=None,
                                                          num_lines=sys.maxsize)
        random.shuffle(self.itemized_data)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

def syllabe_gpt_data_collate_wrapper(batch_items):
    device = batch_items[0][0]
    batch_syllabes_list = []
    batch_input_sequence_lengths = []

    for bidx,data_item in enumerate(batch_items):
        (rank,syllabe_ids_list) = data_item
        batch_syllabes_list.extend(syllabe_ids_list)
        batch_input_sequence_lengths.append(len(syllabe_ids_list))
    input_sequence_lengths = batch_input_sequence_lengths
    syllabes = torch.tensor(batch_syllabes_list).to(device)

    input_with_eos = True
    seq_len = max(input_sequence_lengths)
    # Using length: length-1 sot that last <EOS> token doesn't get processed
    input_masks = [torch.zeros(length, dtype=torch.bool, device=syllabes.device) for length in input_sequence_lengths]
    if input_with_eos:
        for i in range(len(input_masks)):
            input_masks[i][-1] = True
    input_masks_padded = pad_sequence(input_masks, batch_first=True, padding_value=1)  # Shape: N x S

    decoder_mask = generate_square_subsequent_mask(seq_len, syllabes.device)

    data_item = (syllabes, input_sequence_lengths, input_masks_padded, decoder_mask)
    return data_item
