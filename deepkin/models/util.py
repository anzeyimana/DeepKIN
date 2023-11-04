import torch
from torch.nn.utils.rnn import pad_sequence


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_input_key_padding_mask(input_lengths, ignore_last=False):
    input_masks = [torch.zeros(length, dtype=torch.bool) for length in input_lengths]
    if ignore_last:
        for i in range(len(input_masks)):
            if len(input_masks[i]) > 0:
                input_masks[i][-1] = True
    input_masks_padded = pad_sequence(input_masks, batch_first=True, padding_value=1)  # Shape: N x S
    return input_masks_padded
