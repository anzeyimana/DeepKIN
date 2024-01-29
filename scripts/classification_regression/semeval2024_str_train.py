from __future__ import print_function, division

from argparse import ArgumentParser

import math
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from arguments_util import add_dropout_args, \
    add_finetune_and_inference_args, add_base_model_arch_args
from classification_regression_data_util import cls_reg_data_collate_wrapper
from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib
from deepkin.models.kinyabert import KinyaBERT_SequenceClassifier_from_pretrained
from deepkin.models.modules import BaseConfig
from deepkin.optim.adamw import mAdamW
from deepkin.optim.learning_rates import AnnealingLR
from classification_regression_train import TrainingEngine, update_task_state, \
    submit_valid_results, finalize_task, train_loop
from semeval2024_str_data_utils import Semeval2024StrRegDataset

training_engine = None

def semeval2024_do_train_main(_rank, args, cfg: BaseConfig):
    import time
    global training_engine

    device = torch.device('cuda:%d' % args.default_device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.default_device)

    build_kinlpy_lib()
    from kinlpy import ffi, lib
    lib.init_kinlp_socket()
    print('Morphokin library ready via Unix Socket!', flush=True)

    mydb = None

    task_id = args.batch_task_id

    training_engine = TrainingEngine(args, mydb, device, task_id)

    args.regression_scale_factor = 1.0
    args.cls_labels = '0'

    num_classes = 1
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps

    update_task_state(mydb, task_id, 'Pre-processing inputs ...', 0.0)
    train_dataset = Semeval2024StrRegDataset(ffi, lib, mydb, args.train_dataset,
                                  task_id=task_id,
                                  update_task_state_func=update_task_state)

    valid_dataset = Semeval2024StrRegDataset(ffi, lib, mydb, args.valid_dataset,
                                  task_id=task_id,
                                  update_task_state_func=update_task_state)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=cls_reg_data_collate_wrapper,
                                   shuffle=True, drop_last=True, num_workers=2, persistent_workers=True)

    cls_model = KinyaBERT_SequenceClassifier_from_pretrained(num_classes, device, args, cfg, args.pretrained_model_file)
    if args.saved_cls_head is not None:
        cls_head_state_dict = torch.load(args.saved_cls_head, map_location=device)
        cls_model.cls_head.load_state_dict(cls_head_state_dict['cls_head_state_dict'])

    peak_lr = args.peak_lr  # 1e-5
    wd = args.wd  # 0.1
    lr_decay_style = 'linear'
    init_step = 0

    num_epochs = args.num_epochs
    num_iters = math.ceil(num_epochs * len(train_data_loader) / accumulation_steps)
    warmup_iter = math.ceil(num_iters * args.warmup_ratio)  # warm-up for first 6% of iterations

    optimizer = mAdamW(cls_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd,
                       correct_bias=True,
                       local_normalization=False)

    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=lr_decay_style,
                               last_iter=init_step)

    best_accur = -99999.99
    best_F1 = -99999.99
    curr_epochs = 0
    if args.load_saved_model:
        kb_state_dict = torch.load(args.models_save_dir + '/' + args.devbest_model_file)
        best_accur = kb_state_dict['best_dev_accuracy']
        best_F1 = kb_state_dict['best_dev_F1']
        cls_model.load_state_dict(kb_state_dict['model_state_dict'])
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        curr_epochs = math.ceil(lr_scheduler.num_iters * accumulation_steps / len(train_data_loader))

    dev_accur, dev_F1 = 0.0, 0.0
    update_task_state(mydb, task_id, 'TRAINING ...', 0.0)
    start = time.time()
    epoch_loss = 99999.99
    for epoch in range(curr_epochs, num_epochs):
        best_accur, best_F1, dev_accur, dev_F1, epoch_loss = train_loop(args, cls_model, device, optimizer,
                                                                        lr_scheduler, train_data_loader,
                                                                        valid_dataset, best_accur, best_F1,
                                                                        accumulation_steps,
                                                                        args.models_save_dir + '/' + args.devbest_model_file)
        now = time.time()
        update_task_state(mydb, task_id, 'TRAINING ...', 100.0 * (epoch + 1) / num_epochs,
                          eta=((now - start) * (num_epochs - (epoch + 1)) / ((epoch + 1) - curr_epochs)))
        submit_valid_results(mydb, args.devbest_model_id, best_accur, best_F1, epoch_loss)
        submit_valid_results(mydb, args.final_model_id, dev_accur, dev_F1, epoch_loss)

    torch.save({'model_state_dict': cls_model.state_dict()}, args.models_save_dir + '/' + args.final_model_file)

    submit_valid_results(mydb, args.devbest_model_id, best_accur, best_F1, epoch_loss)
    submit_valid_results(mydb, args.final_model_id, dev_accur, dev_F1, epoch_loss)
    finalize_task(mydb, task_id, 'TRAINING COMPLETE', 100.0, True)

    training_engine = None
    return best_accur, best_F1


def cleanup():
    global training_engine
    try:
        if training_engine is not None:
            finalize_task(training_engine.mydb, training_engine.task_id, 'TRAINING STOPPED', 100.0, False)
            training_engine = None
    except Exception:
        pass


def semeval2024_train_main(rank, args, cfg: BaseConfig):
    semeval2024_do_train_main(rank, args, cfg)


def semeval2024_trainer_main(list_args=None, silent=True):
    import os
    import random
    parser = ArgumentParser(description="PyTorch Classification/Regression Trainer")
    parser = add_base_model_arch_args(parser)
    parser = add_dropout_args(parser)
    args = add_finetune_and_inference_args(parser,list_args=list_args, silent=silent)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '{}'.format(random.randint(8100, 8200))
    if args.gpus == 0:
        args.world_size = 1
    cfg = BaseConfig()

    mp.spawn(semeval2024_train_main, nprocs=args.world_size, args=(args, cfg))


if __name__ == '__main__':
    semeval2024_trainer_main()
