from __future__ import print_function, division

from argparse import ArgumentParser

import math
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelF1Score

from arguments_util import add_dropout_args, \
    add_finetune_and_inference_args, add_base_model_arch_args
from classification_regression_data_util import ClsRegDataset, cls_reg_data_collate_wrapper, \
    cls_reg_model_forward, cls_reg_model_predict
from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib
from deepkin.models.kinyabert import KinyaBERT_SequenceClassifier_from_pretrained
from deepkin.models.modules import BaseConfig
from deepkin.optim.adamw import mAdamW
from deepkin.optim.learning_rates import AnnealingLR
from deepkin.utils.misc_functions import time_now

training_engine = None


class TrainingEngine:
    def __init__(self, args, mydb, device, task_id):
        self.args = args
        self.mydb = mydb
        self.task_id = task_id
        self.device = device


def spearman_corr(r_x, r_y):
    return stats.spearmanr(r_x, r_y)[0]


def pearson_corr(r_x, r_y):
    return stats.pearsonr(r_x, r_y)[0]


def model_eval_classification(args, cls_model, device, eval_dataset: ClsRegDataset):
    import sklearn.metrics
    cls_model.eval()
    total = 0.0
    accurate = 0.0
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    labels = sorted(args.cls_labels.split(','))
    label_dict = {v: k for k, v in enumerate(labels)}
    inv_dict = {label_dict[k]: k for k in label_dict}
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data_item in eval_dataset.itemized_data:
            output_scores, predicted_label, true_label = cls_reg_model_predict(data_item, cls_model, device)
            total += 1.0
            if (predicted_label == true_label):
                accurate += 1.0
            if (predicted_label == true_label) and (predicted_label == 1):
                TP += 1.0
            if (predicted_label == true_label) and (predicted_label == 0):
                TN += 1.0
            if (predicted_label != true_label) and (predicted_label == 1):
                FP += 1.0
            if (predicted_label != true_label) and (predicted_label == 0):
                FN += 1.0
            y_pred.append(predicted_label)
            y_true.append(true_label)
    cls_labels = [i for i in range(len(inv_dict))]
    F1 = sklearn.metrics.f1_score(y_true, y_pred, labels=cls_labels, average='weighted')
    # return F1, F1
    if len(inv_dict) == 2:
        pos_label = 1
        if '1' in label_dict:
            pos_label = label_dict['1']
        neg = [x for x in label_dict if ('not' in x) or ('NOT' in x)]
        if len(neg) > 0:
            pos_label = int(1 - label_dict[neg[0]])
        F1 = sklearn.metrics.f1_score(y_true, y_pred, labels=cls_labels, pos_label=pos_label, average='binary')
    return accurate / total, F1


def model_eval_multi_label_classification(args, cls_model, device, eval_dataset: ClsRegDataset):
    cls_model.eval()
    labels = sorted(args.cls_labels.split(','))
    f1Metric = MultilabelF1Score(num_labels=len(labels), average='micro', threshold=0.5)
    targets = []
    predictions = []
    with torch.no_grad():
        for data_item in eval_dataset.itemized_data:
            output_scores, predicted_label, true_label_idx = cls_reg_model_predict(data_item, cls_model, device)
            prediction = F.sigmoid(output_scores).squeeze().cpu().tolist()
            predictions.append(prediction)
            target = [0 for _ in labels]
            for lidx in true_label_idx:
                target[lidx] = 1
            targets.append(target)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.int)
    F1 = f1Metric(predictions_tensor, targets_tensor)
    return F1, F1


def model_eval_regression(cls, cls_model, device, eval_dataset: ClsRegDataset, regression_scale_factor):
    cls_model.eval()
    true_vals = []
    hyp_vals = []
    with torch.no_grad():
        for data_item in eval_dataset.itemized_data:
            output_scores, predicted_label, true_label = cls_reg_model_predict(data_item, cls_model, device)
            hyp_vals.append(regression_scale_factor * output_scores.item())
            true_vals.append(regression_scale_factor * true_label)
    return spearman_corr(np.array(true_vals), np.array(hyp_vals)), pearson_corr(np.array(true_vals), np.array(hyp_vals))


def train_loop(args, cls_model, device, optimizer, lr_scheduler, train_data_loader, valid_dataset, best_accur,
               best_F1, accumulation_steps, save_file_path):
    cls_model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    epoch_avg_train_loss = 0.0
    epoch_iter_count = 0.0
    max_tot_grad_norm = 0.0
    max_cw_grad_norm = 0.0
    iter_loss = 0.0
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        output_scores, batch_labels = cls_reg_model_forward(batch_data_item, cls_model, device)
        if args.regression_target:
            output_scores = output_scores.view(-1).float()
            targets = torch.tensor(batch_labels, dtype=torch.float).to(device).float()
            loss = F.mse_loss(output_scores, targets)
        elif args.multi_label_cls:
            targets = torch.zeros(output_scores.size(0), output_scores.size(1), device=output_scores.device)
            for ii, lab in enumerate(batch_labels):
                targets[ii, lab] = 1.0
            loss = F.binary_cross_entropy_with_logits(output_scores, targets)
        else:
            output_scores = F.log_softmax(output_scores, dim=1)
            loss = F.nll_loss(output_scores, torch.tensor(batch_labels, dtype=torch.long).to(device))
        loss = loss / accumulation_steps
        loss.backward()
        iter_loss += loss.item()

        if ((batch_idx + 1) % accumulation_steps) == 0:
            total_norm = 0.0
            max_local_cw_grad_norm = 0.0
            for param in cls_model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    if param_norm > max_cw_grad_norm:
                        max_cw_grad_norm = param_norm
                    if param_norm > max_local_cw_grad_norm:
                        max_local_cw_grad_norm = param_norm
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            if total_norm > max_tot_grad_norm:
                max_tot_grad_norm = total_norm
            if args.ft_cwgnc > 0.0:
                for param in cls_model.parameters():
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_norm_(param, args.ft_cwgnc)
            else:
                torch.nn.utils.clip_grad_norm_(cls_model.parameters(), 100.0)
            if math.isfinite(iter_loss):
                lr_scheduler.step()
                optimizer.step()
                total_loss += iter_loss
                epoch_avg_train_loss += (iter_loss * accumulation_steps)
                epoch_iter_count += accumulation_steps
            optimizer.zero_grad()
            iter_loss = 0.0
            total_loss = 0.0
    if args.regression_target:
        dev_accur, dev_F1 = model_eval_regression(args, cls_model, device, valid_dataset, args.regression_scale_factor)
    elif args.multi_label_cls:
        dev_accur, dev_F1 = model_eval_multi_label_classification(args, cls_model, device, valid_dataset)
    else:
        dev_accur, dev_F1 = model_eval_classification(args, cls_model, device, valid_dataset)
    if (dev_accur > best_accur):  # or (dev_F1 > best_F1):
        best_accur = dev_accur
        best_F1 = dev_F1
        torch.save({'iter': iter,
                    'model_state_dict': cls_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_dev_accuracy': best_accur,
                    'best_dev_F1': best_F1},
                   save_file_path)
    epoch_loss = epoch_avg_train_loss / epoch_iter_count
    return best_accur, best_F1, dev_accur, dev_F1, epoch_loss


def update_task_state(mydb, task_id: str, state: str, progress: float, eta=None):
    print(time_now(), 'Task: {} ==> {:.1f}%: {} ETA: {:.0f} secs'.format(task_id, progress, state, eta if eta is not None else 0.0),
          flush=True)


def finalize_task(mydb, task_id: str, state: str, progress: float, success: bool):
    print(time_now(), 'Task: {} ==> {:.1f}%: {}'.format(task_id, progress, state), flush=True)


def submit_valid_results(mydb, model_id: str, main_metric, aux_metric, train_loss):
    print(time_now(), 'Validation result @ {} ==> Main metric: {:.4f} , Aux metric: {:.4f} , Train Loss: {:.8f}'.format(model_id, 100.0 * main_metric, 100.0 * aux_metric, train_loss), flush=True)



def cls_reg_do_train_main(_rank, args, cfg: BaseConfig, from_db_app):
    import time
    global training_engine

    device = torch.device('cuda:%d' % args.default_device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.default_device)

    kinlp_conf = '/home/nzeyi/KINLP/data/config_kinlp.conf'
    build_kinlpy_lib()
    from kinlpy import ffi, lib
    lib.init_kinlp_socket(kinlp_conf.encode('utf-8'))
    print('KINLPY Morpho-Analysis-Synthesis Lib Ready via Unix Socket!', flush=True)

    mydb = None

    task_id = args.batch_task_id

    training_engine = TrainingEngine(args, mydb, device, task_id)

    labels = sorted(args.cls_labels.split(','))
    label_dict = {v: k for k, v in enumerate(labels)}

    num_classes = len(label_dict)
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps

    update_task_state(mydb, task_id, 'Pre-processing inputs ...', 0.0)
    train_dataset = ClsRegDataset(ffi, lib, mydb, args.train_dataset,
                                  requires_morpho_analysis=True,
                                  label_dict=label_dict,
                                  regression_target=args.regression_target,
                                  regression_scale_factor=args.regression_scale_factor,
                                  multi_label_cls=args.multi_label_cls,
                                  max_seq_len=512,
                                  task_id=task_id,
                                  update_task_state_func=update_task_state)

    valid_dataset = ClsRegDataset(ffi, lib, mydb, args.valid_dataset,
                                  requires_morpho_analysis=True,
                                  label_dict=label_dict,
                                  regression_target=args.regression_target,
                                  regression_scale_factor=args.regression_scale_factor,
                                  multi_label_cls=args.multi_label_cls,
                                  max_seq_len=512,
                                  task_id=task_id,
                                  update_task_state_func=update_task_state)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=cls_reg_data_collate_wrapper,
                                   shuffle=True, drop_last=True, num_workers=2, persistent_workers=True)

    cls_model = KinyaBERT_SequenceClassifier_from_pretrained(num_classes, device, args, cfg,
                                                             args.home_path + 'data/' + args.pretrained_model_file)
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


def cls_reg_train_main(rank, args, cfg: BaseConfig, from_db_app):
    cls_reg_do_train_main(rank, args, cfg, from_db_app)


def cls_reg_trainer_main(list_args=None, silent=True):
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

    mp.spawn(cls_reg_train_main, nprocs=args.world_size, args=(args, cfg, args.from_db_app))


if __name__ == '__main__':
    cls_reg_trainer_main()
