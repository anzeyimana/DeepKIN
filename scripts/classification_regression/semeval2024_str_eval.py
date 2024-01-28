from argparse import ArgumentParser
from typing import List

import pandas as pd
import progressbar
import torch

from arguments_util import add_dropout_args, \
    add_finetune_and_inference_args, add_base_model_arch_args
from classification_regression_data_util import cls_reg_data_collate_wrapper, prepare_cls_reg_input_segments, \
    cls_reg_model_batch_predict
from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib, parse_text_to_morpho_sentence
from deepkin.models.kinyabert import KinyaBERT_SequenceClassifier
from deepkin.models.modules import BaseConfig
from deepkin.utils.misc_functions import time_now


def cls_reg_inference(ffi, lib, cls_reg_model, device, txtList: List[str]):
    max_seq_len = 512
    lines_split = [line.split('\n') for line in txtList]
    data_items = []
    for line_of_cols in lines_split:
        parsed_tokens = [parse_text_to_morpho_sentence(ffi, lib, col) for col in line_of_cols]
        data_items.append((0, prepare_cls_reg_input_segments(parsed_tokens,max_len=max_seq_len)))
    #Inference
    batch_data_item = cls_reg_data_collate_wrapper(data_items)
    cls_reg_model.eval()
    scores = cls_reg_model_batch_predict(batch_data_item, cls_reg_model, device)
    cls_labels = (args.regression_scale_factor * scores).cpu().tolist()
    return cls_labels

if __name__ == '__main__':
    build_kinlpy_lib()
    from kinlpy import ffi, lib
    lib.init_kinlp_socket()
    print('Morphokin library ready via Unix Socket!', flush=True)

    parser = ArgumentParser(description="PyTorch Classification/Regression Inference")
    parser = add_base_model_arch_args(parser)
    parser = add_dropout_args(parser)
    args = add_finetune_and_inference_args(parser)
    args.regression_scale_factor = 1.0
    args.cls_labels = '0'

    print(time_now(), f'Setting up regression model {args.pretrained_model_file} ...', flush=True)

    device = torch.device('cuda')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cls_reg_model = KinyaBERT_SequenceClassifier(args, BaseConfig(), 1).to(device)
    kb_state_dict = torch.load(args.pretrained_model_file, map_location=device)
    cls_reg_model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    cls_reg_model.eval()

    df = pd.read_csv(args.eval_dataset)
    f = open(f'pred_kin_a.csv','w')
    f.write('PairID,Pred_Score\n')
    with progressbar.ProgressBar(max_value=(df.shape[0]), redirect_stdout=True) as bar:
        for pair_id, text in zip(df['PairID'].values, df['text'].values):
            scores = cls_reg_inference(ffi, lib, cls_reg_model, device, [text])
            score = scores[0]
            f.write(f'{pair_id},{score[0]:.3f}\n')
    f.close()
