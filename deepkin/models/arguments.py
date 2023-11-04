from __future__ import print_function, division

import os
from argparse import ArgumentParser

from deepkin.utils.misc_functions import str2bool

def add_default_arguments(parser: ArgumentParser):
    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
    parser.add_argument("--morpho-dim-hidden", type=int, default=128)
    parser.add_argument("--stem-dim-hidden", type=int, default=256)

    parser.add_argument("--morpho-max-token-len", type=int, default=24)
    parser.add_argument("--morpho-rel-pos-bins", type=int, default=12)
    parser.add_argument("--morpho-max-rel-pos", type=int, default=12)

    parser.add_argument("--main-sequence-encoder-max-seq-len", type=int, default=512)
    parser.add_argument("--main-sequence-encoder-rel-pos-bins", type=int, default=256)
    parser.add_argument("--main-sequence-encoder-max-rel-pos", type=int, default=256)
    parser.add_argument("--dataset-max-seq-len", type=int, default=512)

    parser.add_argument("--morpho-dim-ffn", type=int, default=512)
    parser.add_argument("--main-sequence-encoder-dim-ffn", type=int, default=3072)

    parser.add_argument("--morpho-num-heads", type=int, default=4)
    parser.add_argument("--main-sequence-encoder-num-heads", type=int, default=12)

    parser.add_argument("--morpho-num-layers", type=int, default=4)
    parser.add_argument("--main-sequence-encoder-num-layers", type=int, default=12)

    parser.add_argument("--ft-reinit-layers", type=int, default=0)
    parser.add_argument("--ft-cwgnc", type=float, default=0.0)

    parser.add_argument("--morpho-dropout", type=float, default=0.1)
    parser.add_argument("--main-sequence-encoder-dropout", type=float, default=0.1)

    parser.add_argument("--layernorm-epsilon", type=float, default=1e-6)
    parser.add_argument("--pooler-dropout", type=float, default=0.3)

    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument("--load-saved-model", type=str2bool, default=True)
    parser.add_argument("--home-path", type=str, default=(KINLP_HOME + "/"))
    parser.add_argument("--train-parsed-corpus", type=str, default="data/plain/parsed_train_corpus_docs.txt")
    parser.add_argument("--dev-parsed-corpus", type=str, default="data/plain/parsed_dev_corpus_docs.txt")
    parser.add_argument("--train-unparsed-corpus", type=str, default="data/plain/train_corpus_docs.txt")
    parser.add_argument("--dev-unparsed-corpus", type=str, default="data/plain/dev_corpus_docs.txt")

    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-accumulation-steps", type=int, default=160)
    parser.add_argument("--bert-num-iters", type=int, default=200000)
    parser.add_argument("--bert-warmup-iters", type=int, default=2000)
    parser.add_argument("--bert-number-of-load-batches", type=int, default=16000)

    parser.add_argument("--gpt-batch-size", type=int, default=14)
    parser.add_argument("--gpt-accumulation-steps", type=int, default=36)
    parser.add_argument("--gpt-num-iters", type=int, default=320000)
    parser.add_argument("--gpt-warmup-iters", type=int, default=2000)
    parser.add_argument("--gpt-number-of-load-batches", type=int, default=18000)

    parser.add_argument("--syllabe-batch-size", type=int, default=16)
    parser.add_argument("--syllabe-accumulation-steps", type=int, default=32)
    parser.add_argument("--syllabe-num-iters", type=int, default=200000)
    parser.add_argument("--syllabe-warmup-iters", type=int, default=5000)
    parser.add_argument("--syllabe-number-of-load-batches", type=int, default=3200)
    parser.add_argument("--syllabe-max-seq-len", type=int, default=1024)
    parser.add_argument("--syllabe-peak-lr", type=float, default=3e-4)

    parser.add_argument("--char-batch-size", type=int, default=16)
    parser.add_argument("--char-accumulation-steps", type=int, default=32)
    parser.add_argument("--char-num-iters", type=int, default=200000)
    parser.add_argument("--char-warmup-iters", type=int, default=5000)
    parser.add_argument("--char-number-of-load-batches", type=int, default=3200)
    parser.add_argument("--char-max-seq-len", type=int, default=1024)
    parser.add_argument("--char-peak-lr", type=float, default=3e-4)

    parser.add_argument("--peak-lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--stop-grad-norm", type=float, default=1.0)
    parser.add_argument("--stop-loss", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=0.01)

    parser.add_argument("--corpus-id", type=int, default=1)

    parser.add_argument("--post-mlm-epochs", type=int, default=0)
    parser.add_argument("--pretrained-model-file", type=str,
                        default=(KINLP_HOME + "/models/kinyabert_base_2023-06-06.pt_160K.pt"))
    parser.add_argument("--head-trunk", type=str2bool, default=False)
    parser.add_argument("--encoder-fine-tune", type=str2bool, default=True)
    parser.add_argument("--max-input-lines", type=int, default=99999999)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--multi-task-weighting", type=str2bool, default=False)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--num-iters", type=int, default=200000)
    parser.add_argument("--warmup-iter", type=int, default=2000)


def py_trainer_args(list_args=None, silent=False):
    parser = ArgumentParser(description="Training/Inference arguments")
    add_default_arguments(parser)
    if list_args is not None:
        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()

    args.world_size = args.gpus

    if not silent:
        print('Call arguments:\n', args)

    return args

def finetune_args(list_args=None, silent=False):
    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
    parser = ArgumentParser(description="Training/Inference arguments")
    add_default_arguments(parser)

    # Add cls fine-tune arguments
    parser.add_argument("--cls-labels", type=str, default="0,1")
    parser.add_argument("--cls-train-input0", type=str, default=None)
    parser.add_argument("--cls-train-input1", type=str, default=None)
    parser.add_argument("--cls-train-label", type=str, default=None)
    parser.add_argument("--cls-dev-input0", type=str, default=None)
    parser.add_argument("--cls-dev-input1", type=str, default=None)
    parser.add_argument("--cls-dev-label", type=str, default=None)
    parser.add_argument("--cls-test-input0", type=str, default=None)
    parser.add_argument("--cls-test-input1", type=str, default=None)
    parser.add_argument("--cls-test-label", type=str, default=None)
    parser.add_argument("--devbest-cls-model-save-file-path", type=str, default=None)
    parser.add_argument("--final-cls-model-save-file-path", type=str, default=None)
    parser.add_argument("--devbest-cls-output-file", type=str, default=None)
    parser.add_argument("--final-cls-output-file", type=str, default=None)
    parser.add_argument("--regression-target", type=str2bool, default=False)
    parser.add_argument("--regression-scale-factor", type=float, default=5.0)
    parser.add_argument("--pretrained-roberta-model-dir", type=str, default=(KINLP_HOME+"models/"))
    parser.add_argument("--pretrained-roberta-checkpoint-file", type=str, default="roberta_checkpoint_best.pt")
    parser.add_argument("--xlmr", type=str2bool, default=False)
    parser.add_argument("--embed-dim", type=int, default=960)
    parser.add_argument("--inference-model-file", type=str, default=None)
    parser.add_argument("--model-keyword", type=str, default=None)
    parser.add_argument("--task-keyword", type=str, default=None)
    parser.add_argument("--input-format", type=str, default=None)

    if list_args is not None:
        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()

    args.world_size = args.gpus

    if not silent:
        print('Call arguments:\n', args)

    return args

