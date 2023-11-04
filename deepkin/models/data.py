import math
import random
import sys
from itertools import accumulate
from typing import List

import progressbar
import torch
from torch.nn.utils.rnn import pad_sequence

from deepkin.clib.libkinlp.kinlpy import ParsedMorphoSentence, MSK_ID, BOS_ID, EOS_ID, NUM_SPECIAL_TOKENS
from deepkin.models.kinyabert import KinyaBERT_PretrainModel
from deepkin.models.kinyagpt import KinyaGPT
from deepkin.models.modules import BaseConfig
from torch.utils.data import Dataset
import torch.distributed as dist

def get_random_start_line(doc_ends, corpus_lines):
    # Go to the next starting line
    dcx = random.randint(0, len(doc_ends) - 1) % len(doc_ends)
    start_line = (doc_ends[dcx] + 1) % len(corpus_lines)
    if random.random() < 0.8:  # 80% of the time, start from anywhere within the corpus.
        end_line = (doc_ends[(dcx + 1) % len(doc_ends)] - 1)
        if end_line <= start_line:
            end_line = len(corpus_lines) - 1
        if end_line > start_line:
            if random.random() < 0.6:
                # Start from within 2/3 of the beginning of chosen the document
                max_line = start_line + int(2.0 * (end_line - start_line) / 3.0)
                start_line = random.randint(start_line, max_line)
            else:
                # Start from within 3/4 of the beginning of chosen the document
                max_line = start_line + int(3.0 * (end_line - start_line) / 4.0)
                start_line = random.randint(start_line, max_line) % len(corpus_lines)

    return start_line

def prepare_mlm_data_from_sentence(sentence: ParsedMorphoSentence, add_cls, cfg:BaseConfig):
    lm_morphs = []
    pos_tags = []
    affixes = []
    tokens_lengths = []
    stems = []

    predicted_tokens_idx = []
    predicted_tokens_affixes_idx = []
    predicted_stems = []
    predicted_pos_tags = []
    predicted_lm_morphs = []
    predicted_affixes = []
    predicted_affixes_lengths = []

    # Add <CLS> Token
    if add_cls:
        lm_morphs.append(BOS_ID)
        pos_tags.append(BOS_ID)
        stems.append(BOS_ID)
        tokens_lengths.append(0)

    if (len(sentence.tokens) == 0): # New document
        lm_morphs.append(EOS_ID)
        pos_tags.append(EOS_ID)
        stems.append(EOS_ID)
        tokens_lengths.append(0)
    else:
        for token in sentence.tokens:
            # Whole word masking, decide on masking per whole word
            unchanged = True
            predict = False
            rped = random.random()
            if (rped <= 0.15): # 15% of tokens are predicted
                predict = True
                rval = rped / 0.15
                if(rval < 0.8): # 80% of predicted tokens are masked
                    unchanged = False
                    lm_morphs.append(MSK_ID)
                    pos_tags.append(MSK_ID)
                    stems.append(MSK_ID)

                    v_afx = random.random()
                    if v_afx < 0.3: # Include Affixes for 30% of the time to enforce morphology learning
                        affixes.extend(token.affixes)
                        tokens_lengths.append(len(token.affixes))
                    else:
                        tokens_lengths.append(0)
                    # Whole word masking
                    for __ in token.extra_tokens_ids:
                        lm_morphs.append(MSK_ID)
                        pos_tags.append(MSK_ID)
                        stems.append(MSK_ID)
                        tokens_lengths.append(0)
                elif (rval < 0.9): # 10% are replaced by random tokens, 10% are left unchanged
                    unchanged = False
                    lm_morphs.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_lm_morphs - 1))
                    pos_tags.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_pos_tags - 1))
                    stems.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_stems - 1))

                    v_afx = random.random()
                    if v_afx < 0.3: # Include Affixes for 30% of the time to enforce morphology learning
                        affixes.extend(token.affixes)
                        tokens_lengths.append(len(token.affixes))
                    else:
                        num_afx = len((token.affixes))
                        tokens_lengths.append(num_afx)
                        for i in range(num_afx):
                            affixes.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_affixes - 1))
                    # Whole word shuffling
                    for __ in token.extra_tokens_ids:
                        lm_morphs.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_lm_morphs - 1))
                        pos_tags.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_pos_tags - 1))
                        stems.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_stems - 1))
                        v_afx = random.random()
                        if v_afx < 0.3:  # Include Affixes for 30% of the time to enforce morphology learning
                            affixes.extend(token.affixes)
                            tokens_lengths.append(len(token.affixes))
                        else:
                            num_afx = len((token.affixes))
                            tokens_lengths.append(num_afx)
                            for i in range(num_afx):
                                affixes.append(random.randint(NUM_SPECIAL_TOKENS, cfg.tot_num_affixes - 1))

            # When no re-sampling happens
            if(unchanged):
                lm_morphs.append(token.lm_morph_id)
                pos_tags.append(token.pos_tag_id)
                stems.append(token.stem_id)
                affixes.extend(token.affixes)
                tokens_lengths.append(len(token.affixes))
                for xtid in token.extra_tokens_ids:
                    lm_morphs.append(token.lm_morph_id)
                    pos_tags.append(token.pos_tag_id)
                    stems.append(xtid)
                    affixes.extend(token.affixes)
                    tokens_lengths.append(len(token.affixes))
            # For prediction tokens
            if(predict):
                predicted_stems.append(token.stem_id)
                predicted_pos_tags.append(token.pos_tag_id)
                predicted_lm_morphs.append(token.lm_morph_id)
                predicted_tokens_idx.append(len(tokens_lengths) - len(token.extra_tokens_ids) - 1)
                if (len(token.affixes) > 0):
                    predicted_affixes.extend(token.affixes)
                    predicted_tokens_affixes_idx.append(len(predicted_tokens_idx) - 1)
                    predicted_affixes_lengths.append(len(token.affixes))

                for cnt,xtid in enumerate(token.extra_tokens_ids):
                    predicted_stems.append(xtid)
                    predicted_pos_tags.append(token.pos_tag_id)
                    predicted_lm_morphs.append(token.lm_morph_id)
                    predicted_tokens_idx.append(len(tokens_lengths) - len(token.extra_tokens_ids) + cnt)
                    if (len(token.affixes) > 0):
                        predicted_affixes.extend(token.affixes)
                        predicted_tokens_affixes_idx.append(len(predicted_tokens_idx) - 1)
                        predicted_affixes_lengths.append(len(token.affixes))



    return (lm_morphs,
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
            predicted_affixes_lengths)

def gather_itemized_mlm_data(parsed_sentences: List[str], doc_ends, max_seq_len, max_batch_items, cfg:BaseConfig, bar=None, num_lines=sys.maxsize, stochastic=True):
    itemized_data = []

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

    if stochastic:
        dcx = random.randint(0, len(doc_ends) - 1) % len(doc_ends)
        sline = (doc_ends[dcx] + 1) % len(parsed_sentences)
        if random.random() < 0.8:  # 80% of the time, start from anywhere within the corpus.
            sline = random.randint(0, len(parsed_sentences) - 1) % len(parsed_sentences)
    else:
        sline = 0

    lcount = 0
    while (stochastic or (sline < len(parsed_sentences))):
        sentence = ParsedMorphoSentence(parsed_sentences[sline % len(parsed_sentences)])
        if stochastic:
            sline = (sline + 1) % len(parsed_sentences)
        else:
            sline += 1
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
        # End of document
        if ((len(seq_tokens_lengths) + len(tokens_lengths)) > max_seq_len) or ((len(sentence.tokens) == 0) and (not stochastic)):
            data_item = (dist.get_rank(),(seq_lm_morphs,
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
                         seq_predicted_affixes_lengths))
            if len(seq_tokens_lengths) > 0:
                itemized_data.append(data_item)
            else:
                continue

            if stochastic:
                dcx = random.randint(0, len(doc_ends) - 1) % len(doc_ends)
                sline = (doc_ends[dcx]+1) % len(parsed_sentences)
                if random.random() < 0.8: # 80% of the time, start from anywhere within the corpus.
                    sline = random.randint(0, len(parsed_sentences) - 1) % len(parsed_sentences)

            if (len(itemized_data) >= max_batch_items):
                if (bar is not None):
                    bar.update(len(itemized_data))
                    sys.stdout.flush()
                return itemized_data

            if(lcount >= num_lines):
                return itemized_data

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

            if (((len(itemized_data) % (math.floor(0.1 * max_batch_items) + 1)) == 0) and (bar is not None)):
                bar.update(len(itemized_data))
                sys.stdout.flush()
        else:
            if len(tokens_lengths) > 0:
                lcount += 1
                seq_predicted_tokens_affixes_idx.extend([len(seq_predicted_tokens_idx) + idx for idx in predicted_tokens_affixes_idx])
                seq_predicted_tokens_idx.extend([len(seq_tokens_lengths)+idx for idx in predicted_tokens_idx])
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

            if(lcount >= num_lines):
                data_item = (dist.get_rank(),(seq_lm_morphs,
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
                             seq_predicted_affixes_lengths))
                itemized_data.append(data_item)
                return itemized_data
    return itemized_data

global_cfg = BaseConfig()
def mlm_data_collate_wrapper(batch_items):
    device = batch_items[0][0]

    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []
    batch_predicted_tokens_idx = []
    batch_predicted_tokens_affixes_idx = []
    batch_predicted_stems = []
    batch_predicted_pos_tags = []
    batch_predicted_lm_morphs = []
    batch_predicted_affixes = []
    batch_predicted_affixes_lengths = []

    batch_input_sequence_lengths = []

    for bidx,data_item in enumerate(batch_items):
        (rank,(seq_lm_morphs,
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
         seq_predicted_affixes_lengths)) = data_item

        batch_predicted_tokens_affixes_idx.extend([(len(batch_predicted_tokens_idx) + t) for t in seq_predicted_tokens_affixes_idx])
        batch_predicted_tokens_idx.extend([(t, len(batch_input_sequence_lengths)) for t in seq_predicted_tokens_idx])

        batch_lm_morphs.extend(seq_lm_morphs)
        batch_pos_tags.extend(seq_pos_tags)
        batch_affixes.extend(seq_affixes)
        batch_tokens_lengths.extend(seq_tokens_lengths)
        batch_stems.extend(seq_stems)

        batch_predicted_stems.extend(seq_predicted_stems)
        batch_predicted_pos_tags.extend(seq_predicted_pos_tags)
        batch_predicted_lm_morphs.extend(seq_predicted_lm_morphs)
        batch_predicted_affixes.extend(seq_predicted_affixes)
        batch_predicted_affixes_lengths.extend(seq_predicted_affixes_lengths)

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes)# .to(device) not copied to GPU because not needed there
    stems = torch.tensor(batch_stems).to(device)
    predicted_stems = torch.tensor(batch_predicted_stems).to(device)
    predicted_lm_morphs = torch.tensor(batch_predicted_lm_morphs).to(device)
    predicted_pos_tags = torch.tensor(batch_predicted_pos_tags).to(device)

    predicted_tokens_affixes_idx = torch.tensor(batch_predicted_tokens_affixes_idx).to(device)

    predicted_tokens_idx = torch.tensor([s * max(batch_input_sequence_lengths) + t for t, s in batch_predicted_tokens_idx]).to(device)

    pred_affixes_list = [batch_predicted_affixes[x - y: x] for x, y in zip(accumulate(batch_predicted_affixes_lengths), batch_predicted_affixes_lengths)]
    afx_prob = torch.zeros(len(pred_affixes_list), global_cfg.tot_num_affixes)
    for i,lst in enumerate(pred_affixes_list):
        assert (len(lst) > 0)
        afx_prob[i,lst] = 1.0
    predicted_affixes_prob = afx_prob.to(device, dtype=torch.float)

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

    masks = [torch.zeros(x, dtype=torch.bool, device = stems.device) for x in input_sequence_lengths]
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=1) # Shape: N x S

    batch_data_item = (lm_morphs, pos_tags, stems, input_sequence_lengths,
                       predicted_tokens_idx, predicted_tokens_affixes_idx, predicted_stems,
                       predicted_pos_tags, predicted_lm_morphs,
                       predicted_affixes_prob, afx_padded, m_masks_padded, masks_padded)

    return batch_data_item

def mlm_data_collate_wrapper_dev(batch_items):

    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []
    batch_predicted_tokens_idx = []
    batch_predicted_tokens_affixes_idx = []
    batch_predicted_stems = []
    batch_predicted_pos_tags = []
    batch_predicted_lm_morphs = []
    batch_predicted_affixes = []
    batch_predicted_affixes_lengths = []

    batch_input_sequence_lengths = []

    for bidx,data_item in enumerate(batch_items):
        (rank,(seq_lm_morphs,
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
         seq_predicted_affixes_lengths)) = data_item

        batch_predicted_tokens_affixes_idx.extend([(len(batch_predicted_tokens_idx) + t) for t in seq_predicted_tokens_affixes_idx])
        batch_predicted_tokens_idx.extend([(t, len(batch_input_sequence_lengths)) for t in seq_predicted_tokens_idx])

        batch_lm_morphs.extend(seq_lm_morphs)
        batch_pos_tags.extend(seq_pos_tags)
        batch_affixes.extend(seq_affixes)
        batch_tokens_lengths.extend(seq_tokens_lengths)
        batch_stems.extend(seq_stems)

        batch_predicted_stems.extend(seq_predicted_stems)
        batch_predicted_pos_tags.extend(seq_predicted_pos_tags)
        batch_predicted_lm_morphs.extend(seq_predicted_lm_morphs)
        batch_predicted_affixes.extend(seq_predicted_affixes)
        batch_predicted_affixes_lengths.extend(seq_predicted_affixes_lengths)

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))

    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs)
    pos_tags = torch.tensor(batch_pos_tags)
    affixes = torch.tensor(batch_affixes)
    stems = torch.tensor(batch_stems)
    predicted_stems = torch.tensor(batch_predicted_stems)
    predicted_lm_morphs = torch.tensor(batch_predicted_lm_morphs)
    predicted_pos_tags = torch.tensor(batch_predicted_pos_tags)

    predicted_tokens_affixes_idx = torch.tensor(batch_predicted_tokens_affixes_idx)

    predicted_tokens_idx = torch.tensor([s * max(batch_input_sequence_lengths) + t for t, s in batch_predicted_tokens_idx])

    pred_affixes_list = [batch_predicted_affixes[x - y: x] for x, y in zip(accumulate(batch_predicted_affixes_lengths), batch_predicted_affixes_lengths)]
    afx_prob = torch.zeros(len(pred_affixes_list), global_cfg.tot_num_affixes)
    for i,lst in enumerate(pred_affixes_list):
        assert (len(lst) > 0)
        afx_prob[i,lst] = 1.0
    predicted_affixes_prob = afx_prob.to(dtype=torch.float)

    # Needed to fix decoding start bug
    # if len(affixes) == 0:
    #     affixes.append(0)
    #     tokens_lengths[-1] = 1
    afx = affixes.split(tokens_lengths)
    # [[2,4,5], [6,7]]
    afx_padded = pad_sequence(afx, batch_first=False)
    afx_padded = afx_padded.to(dtype=torch.long)
    # afx_padded: (M,L), M: max morphological length

    # m_masks_padded = None
    # if afx_padded.nelement() > 0:
    m_masks = [torch.zeros((x + 4), dtype=torch.bool) for x in tokens_lengths]
    m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)  # Shape: (L, 4+M)

    masks = [torch.zeros(x, dtype=torch.bool) for x in input_sequence_lengths]
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=1) # Shape: N x S

    batch_data_item = (lm_morphs, pos_tags, stems, input_sequence_lengths,
                       predicted_tokens_idx, predicted_tokens_affixes_idx, predicted_stems,
                       predicted_pos_tags, predicted_lm_morphs,
                       predicted_affixes_prob, afx_padded, m_masks_padded, masks_padded)

    return batch_data_item

def mlm_model_forward_dev(batch_data_item, model: KinyaBERT_PretrainModel, device, cfg: BaseConfig) -> List[torch.Tensor]:
    (lm_morphs, pos_tags, stems, input_sequence_lengths,
     predicted_tokens_idx, predicted_tokens_affixes_idx, predicted_stems,
     predicted_pos_tags, predicted_lm_morphs,
     predicted_affixes_prob, afx_padded, m_masks_padded, masks_padded) = batch_data_item

    if len([l for l in input_sequence_lengths if l == 0]) != 0:
        print('===>> ERROR ERROR ERROR ERROR ====================================> Invalid input_sequence_lengths:', input_sequence_lengths, flush=True)

    return model(lm_morphs.to(device), pos_tags.to(device), stems.to(device), input_sequence_lengths,
                 predicted_tokens_idx.to(device), predicted_tokens_affixes_idx.to(device), predicted_stems.to(device),
                 predicted_pos_tags.to(device), predicted_lm_morphs.to(device),
                 predicted_affixes_prob.to(device), afx_padded.to(device), m_masks_padded.to(device), masks_padded.to(device))

class MLMDataset(Dataset):

    def __init__(self, parsed_sentences: List[str], doc_ends,
                 cfg: BaseConfig,
                 max_batch_items,
                 max_seq_len = 512,
                 stochastic=True):
        global global_cfg
        global_cfg = cfg
        self.max_seq_len = max_seq_len
        self.max_batch_items = max_batch_items
        if (dist.get_rank() == 0) and stochastic:
            with progressbar.ProgressBar(max_value=max_batch_items, redirect_stdout=True) as bar:
                bar.update(0)
                sys.stdout.flush()
                self.itemized_data = gather_itemized_mlm_data(parsed_sentences, doc_ends, max_seq_len, max_batch_items,
                                                              cfg, bar=bar, num_lines=sys.maxsize,
                                                              stochastic=stochastic)
        else:
            self.itemized_data = gather_itemized_mlm_data(parsed_sentences, doc_ends, max_seq_len, max_batch_items,
                                                          cfg, bar=None, num_lines=sys.maxsize,
                                                          stochastic=stochastic)

        random.shuffle(self.itemized_data)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

def mlm_model_forward(batch_data_item, model: KinyaBERT_PretrainModel, device, cfg: BaseConfig) -> List[torch.Tensor]:
    (lm_morphs, pos_tags, stems, input_sequence_lengths,
     predicted_tokens_idx, predicted_tokens_affixes_idx, predicted_stems,
     predicted_pos_tags, predicted_lm_morphs,
     predicted_affixes_prob, afx_padded, m_masks_padded, masks_padded) = batch_data_item
    if len([l for l in input_sequence_lengths if l == 0]) != 0:
        print('===>> ERROR ERROR ERROR ERROR ====================================> Invalid input_sequence_lengths:', input_sequence_lengths, flush=True)
    # (batch_lm_morphs,
    #  batch_pos_tags,
    #  batch_affixes,
    #  batch_tokens_lengths,
    #  batch_stems,
    #  batch_predicted_tokens_idx,
    #  batch_predicted_tokens_affixes_idx,
    #  batch_predicted_stems,
    #  batch_predicted_pos_tags,
    #  batch_predicted_lm_morphs,
    #  batch_predicted_affixes,
    #  batch_predicted_affixes_lengths,
    #  batch_input_sequence_lengths) = batch_data_item
    #
    # tokens_lengths = batch_tokens_lengths
    # input_sequence_lengths = batch_input_sequence_lengths
    #
    # lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    # pos_tags = torch.tensor(batch_pos_tags).to(device)
    # affixes = torch.tensor(batch_affixes).to(device)
    # stems = torch.tensor(batch_stems).to(device)
    # predicted_stems = torch.tensor(batch_predicted_stems).to(device)
    # predicted_lm_morphs = torch.tensor(batch_predicted_lm_morphs).to(device)
    # predicted_pos_tags = torch.tensor(batch_predicted_pos_tags).to(device)
    #
    # predicted_tokens_affixes_idx = torch.tensor(batch_predicted_tokens_affixes_idx).to(device)
    #
    # predicted_tokens_idx = torch.tensor([s * max(batch_input_sequence_lengths) + t for t, s in batch_predicted_tokens_idx]).to(device)
    #
    # pred_affixes_list = [batch_predicted_affixes[x - y: x] for x, y in zip(accumulate(batch_predicted_affixes_lengths), batch_predicted_affixes_lengths)]
    # afx_prob = torch.zeros(len(pred_affixes_list), cfg.tot_num_affixes)
    # for i,lst in enumerate(pred_affixes_list):
    #     assert (len(lst) > 0)
    #     afx_prob[i,lst] = 1.0
    # predicted_affixes_prob = afx_prob.to(device, dtype=torch.float)

    return model(lm_morphs, pos_tags, stems, input_sequence_lengths,
                         predicted_tokens_idx, predicted_tokens_affixes_idx, predicted_stems,
                         predicted_pos_tags, predicted_lm_morphs,
                         predicted_affixes_prob, afx_padded, m_masks_padded, masks_padded)

def mlm_model_predict(seq_data_item, model: KinyaBERT_PretrainModel, device, max_predict_affixes, max_top_predictions=10):
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
     seq_predicted_affixes_lengths) = seq_data_item

    tokens_lengths = seq_tokens_lengths
    input_sequence_lengths = [len(seq_tokens_lengths)]

    lm_morphs = torch.tensor(seq_lm_morphs).to(device)
    pos_tags = torch.tensor(seq_pos_tags).to(device)
    affixes = torch.tensor(seq_affixes).to(device)
    stems = torch.tensor(seq_stems).to(device)

    predicted_tokens_idx = torch.tensor(seq_predicted_tokens_idx, dtype=torch.int32).to(device)

    with torch.no_grad():
        ((stem_predictions,
          pos_tag_predictions,
          lm_morph_predictions,
          affixes_predictions),
         (stem_predictions_prob,
          pos_tag_predictions_prob,
          lm_morph_predictions_prob,
          affixes_predictions_prob)) = model.predict(lm_morphs, pos_tags, affixes, tokens_lengths, stems,
                                                     input_sequence_lengths,
                                                     predicted_tokens_idx,
                                                     max_predict_affixes, max_top_predictions=max_top_predictions)
    return ((stem_predictions,
             pos_tag_predictions,
             lm_morph_predictions,
             affixes_predictions),
           (stem_predictions_prob,
            pos_tag_predictions_prob,
            lm_morph_predictions_prob,
            affixes_predictions_prob))

def prepare_gpt_data_from_sentence(sentence: ParsedMorphoSentence):
    lm_morphs = []
    pos_tags = []
    stems = []
    affixes = []
    tokens_lengths = []

    for token in sentence.tokens:
        lm_morphs.append(token.lm_morph_id)
        pos_tags.append(token.pos_tag_id)
        stems.append(token.stem_id)
        affixes.extend(token.affixes)
        tokens_lengths.append(len(token.affixes))
        for tid in token.extra_tokens_ids:
            lm_morphs.append(token.lm_morph_id)
            pos_tags.append(token.pos_tag_id)
            stems.append(tid)
            affixes.extend(token.affixes)
            tokens_lengths.append(len(token.affixes))

    return (lm_morphs,
            pos_tags,
            affixes,
            tokens_lengths,
            stems)

def gather_gpt_single_sequence(parsed_sentences: List[str], doc_ends, max_seq_len):
    seq_lm_morphs = []
    seq_pos_tags = []
    seq_affixes = []
    seq_tokens_lengths = []
    seq_stems = []

    start_line = get_random_start_line(doc_ends, parsed_sentences)
    sentence = ParsedMorphoSentence(parsed_sentences[start_line % len(parsed_sentences)])
    if len(sentence.tokens) <= 0:
        start_line = (start_line + 1) % len(parsed_sentences)
        sentence = ParsedMorphoSentence(parsed_sentences[start_line % len(parsed_sentences)])

    while len(sentence.tokens) > 0:
        (lm_morphs,
         pos_tags,
         affixes,
         tokens_lengths,
         stems) = prepare_gpt_data_from_sentence(sentence)
        if (len(seq_tokens_lengths) + len(tokens_lengths) + 2) < max_seq_len:
            seq_lm_morphs.extend(lm_morphs)
            seq_pos_tags.extend(pos_tags)
            seq_affixes.extend(affixes)
            seq_tokens_lengths.extend(tokens_lengths)
            seq_stems.extend(stems)
        else:
            if len(seq_tokens_lengths) > 0:
                data_item = ([BOS_ID] + seq_lm_morphs + [EOS_ID],
                             [BOS_ID] + seq_pos_tags + [EOS_ID],
                             seq_affixes,
                             [0] + seq_tokens_lengths + [0],
                             [BOS_ID] + seq_stems + [EOS_ID])
                return data_item
        start_line = (start_line + 1) % len(parsed_sentences)
        sentence = ParsedMorphoSentence(parsed_sentences[start_line % len(parsed_sentences)])
    data_item = ([BOS_ID] + seq_lm_morphs + [EOS_ID],
                 [BOS_ID] + seq_pos_tags + [EOS_ID],
                 seq_affixes,
                 [0] + seq_tokens_lengths + [0],
                 [BOS_ID] + seq_stems + [EOS_ID])
    return data_item

def gather_itemized_gpt_data(parsed_sentences: List[str], doc_ends, max_seq_len, max_batch_items, bar=None, num_lines=sys.maxsize):
    itemized_data = []
    for i in range(max_batch_items):
        itemized_data.append((dist.get_rank(),gather_gpt_single_sequence(parsed_sentences, doc_ends, max_seq_len)))
        if (((len(itemized_data) % (math.floor(0.1 * max_batch_items) + 1)) == 0) and (bar is not None)):
            bar.update(len(itemized_data))
            sys.stdout.flush()
    return itemized_data

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def gpt_data_collate_wrapper(batch_items):
    device = batch_items[0][0]
    batch_lm_morphs = []
    batch_pos_tags = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_stems = []

    batch_input_sequence_lengths = []

    for bidx,data_item in enumerate(batch_items):
        (rank,(seq_lm_morphs,
         seq_pos_tags,
         seq_affixes,
         seq_tokens_lengths,
         seq_stems)) = data_item

        batch_lm_morphs.extend(seq_lm_morphs)
        batch_pos_tags.extend(seq_pos_tags)
        batch_affixes.extend(seq_affixes)
        batch_tokens_lengths.extend(seq_tokens_lengths)
        batch_stems.extend(seq_stems)

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))

    # data_item = (batch_lm_morphs,
    #              batch_pos_tags,
    #              batch_affixes,
    #              batch_tokens_lengths,
    #              batch_stems,
    #              batch_input_sequence_lengths)
    tokens_lengths = batch_tokens_lengths
    input_sequence_lengths = batch_input_sequence_lengths

    lm_morphs = torch.tensor(batch_lm_morphs).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    affixes = torch.tensor(batch_affixes)#.to(device)
    stems = torch.tensor(batch_stems).to(device)

    pred_affixes_list = [batch_affixes[x - y: x] for x, y in zip(accumulate(batch_tokens_lengths), batch_tokens_lengths)]
    afx_prob = torch.zeros(len(pred_affixes_list), global_cfg.tot_num_affixes)
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

class GPTDataset(Dataset):

    def __init__(self, parsed_sentences: List[str], doc_ends,
                 cfg: BaseConfig,
                 max_batch_items,
                 max_seq_len = 512,
                 rank = 0):
        global global_cfg
        global_cfg = cfg
        self.max_seq_len = max_seq_len
        self.max_batch_items = max_batch_items
        self.rank = dist.get_rank()
        if (dist.get_rank() == 0):
            with progressbar.ProgressBar(max_value=max_batch_items, redirect_stdout=True) as bar:
                bar.update(0)
                sys.stdout.flush()
                self.itemized_data = gather_itemized_gpt_data(parsed_sentences, doc_ends, max_seq_len, max_batch_items,
                                                              bar=bar,
                                                              num_lines=sys.maxsize)
        else:
            self.itemized_data = gather_itemized_gpt_data(parsed_sentences, doc_ends, max_seq_len, max_batch_items,
                                                          bar=None,
                                                          num_lines=sys.maxsize)
        random.shuffle(self.itemized_data)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

def gpt_model_forward(batch_data_item, model: KinyaGPT, device, cfg: BaseConfig) -> List[torch.Tensor]:
    (lm_morphs, pos_tags, tokens_lengths, stems, input_sequence_lengths, affixes_prob,
     afx_padded, m_masks_padded, input_masks_padded, decoder_mask) = batch_data_item
    return model(lm_morphs, pos_tags, tokens_lengths, stems, input_sequence_lengths, affixes_prob,
                afx_padded, m_masks_padded, input_masks_padded, decoder_mask)
