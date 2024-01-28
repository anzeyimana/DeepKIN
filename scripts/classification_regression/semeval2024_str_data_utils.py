from __future__ import print_function, division

import random
import time

import pandas as pd
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
        df = pd.read_csv(input_file)
        start = time.time()
        itr = 0
        tot = df.shape[0]
        if score_column is None:
            for text in df['Text'].values:
                itr += 1
                inputs = []
                for line in text.split('\n'):
                    inputs.append(parse_text_to_morpho_sentence(ffi, lib, line))
                self.itemized_data.append((None, prepare_cls_reg_input_segments(inputs)))
        else:
            for text,score in zip(df['Text'].values,df[score_column].values):
                itr += 1
                inputs = []
                for line in text.split('\n'):
                    inputs.append(parse_text_to_morpho_sentence(ffi, lib, line))
                self.itemized_data.append((score, prepare_cls_reg_input_segments(inputs)))
                if ((itr % 1000) == 0) and (task_id is not None) and (update_task_state_func is not None):
                    now = time.time()
                    update_task_state_func(mydb, task_id, 'Parsing input data', 100.0 * itr / tot, eta=((now - start) * (tot - itr) / itr))
        if shuffle:
            random.shuffle(self.itemized_data)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

