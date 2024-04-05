from __future__ import print_function, division

import random
import time

import pandas as pd
from deepkin.utils.misc_functions import read_lines
from torch.utils.data import Dataset

from classification_regression_data_util import prepare_cls_reg_input_segments
from deepkin.clib.libkinlp.kinlpy import parse_text_to_morpho_sentence


class Semeval2024StrRegDataset(Dataset):

    def __init__(self, ffi, lib, mydb, input_file,
                 task_id=None,
                 update_task_state_func = None,
                 shuffle=False,
                 score_column='Score'):
        self.itemized_data = []
        if (task_id is not None) and (update_task_state_func is not None):
            update_task_state_func(mydb, task_id, 'Reading and parsing input data', 0.0)
        start = time.time()
        itr = 0
        if score_column is None:
            lines = read_lines(input_file)
            tot = len(lines)
            for line in lines:
                itr += 1
                inputs = [parse_text_to_morpho_sentence(ffi, lib, l) for l in line.split('\t')[:2]]
                self.itemized_data.append((float(line.split('\t')[-1]), prepare_cls_reg_input_segments(inputs)))
                if ((itr % 1000) == 0) and (task_id is not None) and (update_task_state_func is not None):
                    now = time.time()
                    update_task_state_func(mydb, task_id, 'Parsing input data', 100.0 * itr / tot, eta=((now - start) * (tot - itr) / itr))
        else:
            df = pd.read_csv(input_file)
            tot = df.shape[0]
            for text,score in zip(df['Text'].values,df[score_column].values):
                itr += 1
                inputs = []
                for line in text.split('\n'):
                    inputs.append(parse_text_to_morpho_sentence(ffi, lib, line))
                self.itemized_data.append((score, prepare_cls_reg_input_segments(inputs)))
                if ((itr % 1000) == 0) and (task_id is not None) and (update_task_state_func is not None):
                    now = time.time()
                    update_task_state_func(mydb, task_id, 'Parsing input data', 100.0 * itr / tot, eta=((now - start) * (tot - itr) / itr))
        if (task_id is not None) and (update_task_state_func is not None):
            update_task_state_func(mydb, task_id, 'Done Reading and parsing input data', 100.0, eta=0)
        if shuffle:
            random.shuffle(self.itemized_data)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

