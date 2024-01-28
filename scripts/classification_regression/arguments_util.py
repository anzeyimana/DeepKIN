import os
from argparse import ArgumentParser
from deepkin.utils.misc_functions import str2bool

def add_base_model_arch_args(parser: ArgumentParser):
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
    parser.add_argument("--layernorm-epsilon", type=float, default=1e-6)
    parser.add_argument("--head-trunk", type=str2bool, default=False)
    parser.add_argument("--encoder-fine-tune", type=str2bool, default=True)
    return parser

def add_model_small_arch_args(parser: ArgumentParser):
    parser.add_argument("--morpho-dim-hidden", type=int, default=128)
    parser.add_argument("--stem-dim-hidden", type=int, default=256)
    parser.add_argument("--morpho-max-token-len", type=int, default=24)
    parser.add_argument("--morpho-rel-pos-bins", type=int, default=12)
    parser.add_argument("--morpho-max-rel-pos", type=int, default=12)
    parser.add_argument("--main-sequence-encoder-max-seq-len", type=int, default=512)
    parser.add_argument("--main-sequence-encoder-rel-pos-bins", type=int, default=256)
    parser.add_argument("--main-sequence-encoder-max-rel-pos", type=int, default=256)
    parser.add_argument("--morpho-dim-ffn", type=int, default=512)
    parser.add_argument("--main-sequence-encoder-dim-ffn", type=int, default=3072)
    parser.add_argument("--morpho-num-heads", type=int, default=4)
    parser.add_argument("--main-sequence-encoder-num-heads", type=int, default=12)
    parser.add_argument("--morpho-num-layers", type=int, default=4)
    parser.add_argument("--main-sequence-encoder-num-layers", type=int, default=12)
    parser.add_argument("--layernorm-epsilon", type=float, default=1e-6)
    parser.add_argument("--head-trunk", type=str2bool, default=False)
    parser.add_argument("--encoder-fine-tune", type=str2bool, default=True)
    return parser


def add_dropout_args(parser: ArgumentParser):
    parser.add_argument("--main-sequence-encoder-dropout", type=float, default=0.0)
    parser.add_argument("--morpho-dropout", type=float, default=0.1)
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    return parser

def add_finetune_and_inference_args(parser: ArgumentParser, list_args=None, silent=True):
    KINLP_HOME = (os.environ['KINLP_HOME']) if ('KINLP_HOME' in os.environ) else ('/opt/KINLP')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument("--load-saved-model", type=str2bool, default=False)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--num-iters", type=int, default=200000)
    parser.add_argument("--warmup-iter", type=int, default=2000)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--num-model-trials", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--stop-grad-norm", type=float, default=1.0)
    parser.add_argument("--stop-loss", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--max-input-lines", type=int, default=99999999)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--multi-task-weighting", type=str2bool, default=False)
    parser.add_argument("--ft-reinit-layers", type=int, default=0)
    parser.add_argument("--ft-cwgnc", type=float, default=0.0)

    parser.add_argument("--cls-labels", type=str, default="0,1")
    parser.add_argument("--train-dataset", type=str, default=None)
    parser.add_argument("--valid-dataset", type=str, default=None)
    parser.add_argument("--eval-dataset", type=str, default=None)
    parser.add_argument("--pretrained-model-file", type=str, default=(KINLP_HOME + "/models/kinyabert_base_2023-06-06.pt_160K.pt"))

    parser.add_argument("--devbest-model-file", type=str, default=None)
    parser.add_argument("--final-model-file", type=str, default=None)
    parser.add_argument("--devbest-model-id", type=str, default=None)
    parser.add_argument("--final-model-id", type=str, default=None)

    parser.add_argument("--regression-target", type=str2bool, default=False)
    parser.add_argument("--multi-label-cls", type=str2bool, default=False)
    parser.add_argument("--regression-scale-factor", type=float, default=5.0)

    parser.add_argument("--db-host", type=str, default=None)
    parser.add_argument("--db-user", type=str, default=None)
    parser.add_argument("--db-password", type=str, default=None)
    parser.add_argument("--db-database", type=str, default=None)

    parser.add_argument("--batch-task-id", type=str, default=None)
    parser.add_argument("--online-task-id", type=str, default=None)
    parser.add_argument("--api-host", type=str, default=None)
    parser.add_argument("--api-port", type=int, default=None)

    parser.add_argument("--default-device", type=int, default=0)
    parser.add_argument("--from_db_app", type=str2bool, default=True)
    parser.add_argument("--saved-cls-head", type=str, default=None)

    parser.add_argument("--models-save-dir", type=str, default=None)

    if list_args is not None:
        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()

    args.world_size = args.gpus

    if not silent:
        print('Call arguments:\n', args)

    if args.regression_target:
        args.cls_labels = '0'
    return args
