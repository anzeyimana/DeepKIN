from __future__ import print_function, division

import math

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

from deepkin.models.arguments import finetune_args
from deepkin.models.cls_data_loaders import ClsDataset, cls_data_collate_wrapper
from deepkin.models.kinyabert import KinyaBERT_SequenceClassifier_from_pretrained, KinyaBERT_from_pretrained
from deepkin.models.modules import BaseConfig
from deepkin.optim.adamw import mAdamW
from deepkin.optim.learning_rates import AnnealingLR
from deepkin.utils.misc_functions import time_now, read_lines


def spearman_corr(r_x, r_y):
    return stats.spearmanr(r_x, r_y)[0]

def pearson_corr(r_x, r_y):
    return stats.pearsonr(r_x, r_y)[0]

def cls_model_eval_classification(args, cls_model, shared_encoder, device, eval_dataset:ClsDataset):
    import sklearn.metrics
    from cls_data_loaders import cls_model_predict
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
    y_true  = []
    y_pred = []
    with torch.no_grad():
        for data_item in eval_dataset.itemized_data:
            output_scores, predicted_label, true_label = cls_model_predict(data_item, cls_model, shared_encoder, device)
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
    Accuracy = accurate/total
    F1_weighted = sklearn.metrics.f1_score(y_true, y_pred, labels=cls_labels, average='weighted')
    F1_micro = sklearn.metrics.f1_score(y_true, y_pred, labels=cls_labels, average='micro')
    F1_Macro = sklearn.metrics.f1_score(y_true, y_pred, labels=cls_labels, average='macro')
    if len(labels) == 2:
        F1_Binary = sklearn.metrics.f1_score(y_true, y_pred, labels=cls_labels, average='binary')
        return (Accuracy, F1_Binary, F1_weighted, F1_micro, F1_Macro)
    else:
        return (Accuracy, F1_weighted, F1_micro, F1_Macro)

def cls_model_eval_regression(args, cls_model, shared_encoder, device, eval_dataset:ClsDataset, regression_scale_factor):
    from cls_data_loaders import cls_model_predict
    cls_model.eval()
    true_vals = []
    hyp_vals = []
    with torch.no_grad():
        for data_item in eval_dataset.itemized_data:
            output_scores, predicted_label, true_label = cls_model_predict(data_item, cls_model, shared_encoder, device)
            hyp_vals.append(regression_scale_factor * output_scores.item())
            true_vals.append(regression_scale_factor * true_label)
    return (pearson_corr(np.array(true_vals), np.array(hyp_vals)), spearman_corr(np.array(true_vals), np.array(hyp_vals)))

def train_loop(args, keyword, epoch, scaler, cls_model, shared_encoder, device, optimizer, lr_scheduler, train_data_loader, valid_dataset, best_results, accumulation_steps, save_file_path, regression_target, regression_scale_factor, bar):
    from cls_data_loaders import  cls_model_forward
    cls_model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    epoch_avg_train_loss = 0.0
    epoch_iter_count = 0.0
    max_tot_grad_norm = 0.0
    max_cw_grad_norm = 0.0
    iter_loss = 0.0
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        with torch.cuda.amp.autocast():
            output_scores, batch_labels = cls_model_forward(batch_data_item, cls_model, shared_encoder, device)
            if regression_target:
                output_scores = output_scores.view(-1).float()
                targets = torch.tensor(batch_labels, dtype=torch.float).to(device).float()
                loss = F.mse_loss(output_scores, targets)
            else:
                output_scores = F.log_softmax(output_scores, dim=1)
                loss = F.nll_loss(output_scores, torch.tensor(batch_labels, dtype=torch.long).to(device))
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        iter_loss += loss.item()

        if ((batch_idx+1) % accumulation_steps) == 0:
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
                scaler.step(optimizer)
                scaler.update()
                total_loss += iter_loss
                epoch_avg_train_loss += (iter_loss * accumulation_steps)
                epoch_iter_count += accumulation_steps
            optimizer.zero_grad()
            iter_loss = 0.0
            total_loss = 0.0

    if regression_target:
        dev_results = cls_model_eval_regression(args, cls_model, shared_encoder, device, valid_dataset, regression_scale_factor)
        if dev_results[1] >= best_results[1]:
            best_results = dev_results
            torch.save({'iter': iter,
                        'model_state_dict': cls_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict()},
                       save_file_path)
        print(keyword, 'After', (epoch + 1), 'epochs:',
              'Train Loss: {:.8f}, Max Grad Norm: {:.4f}/{:.4f}'.format(epoch_avg_train_loss/epoch_iter_count, max_cw_grad_norm, max_tot_grad_norm),
              '==> Validation set results (Pearson R, Spearman R) [%] = ('+(', '.join(['{:.2f}'.format(100.0 * r) for r in dev_results]))+')',
              '==> Best dev results so far (Pearson R, Spearman R) [%] = ('+(', '.join(['{:.2f}'.format(100.0 * r) for r in best_results]))+')')
    else:
        dev_results = cls_model_eval_classification(args, cls_model, shared_encoder, device, valid_dataset)
        if dev_results[1] >= best_results[1]:
            best_results = dev_results
            torch.save({'iter': iter,
                        'model_state_dict': cls_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict()},
                       save_file_path)
        print(keyword, 'After', (epoch + 1), 'epochs:',
              'Train Loss: {:.8f}, Max Grad Norm: {:.4f}/{:.4f}'.format(epoch_avg_train_loss/epoch_iter_count, max_cw_grad_norm, max_tot_grad_norm),
              '==> Validation set results (Accuracy [, F1_binary], F1_weighted, F1_micro, F1_Macro) [%] = (' + (', '.join(['{:.2f}'.format(100.0 * r) for r in dev_results]))+')',
              '==> Best dev results so far (Accuracy [, F1_binary], F1_weighted, F1_micro, F1_Macro) [%] = (' + (', '.join(['{:.2f}'.format(100.0 * r) for r in best_results]))+')')
    bar.update(lr_scheduler.num_iters)
    return best_results

def TEXT_CLS_train_main(rank, args, cfg: BaseConfig):
    import progressbar

    keyword=args.model_keyword

    #cls_labels = "0,1"
    labels = sorted(args.cls_labels.split(','))
    label_dict = {v:k for k,v in enumerate(labels)}

    USE_GPU = (args.gpus > 0)

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if USE_GPU and torch.cuda.is_available():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    print(time_now(), 'Called train_fn()', flush=True)
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)
    dist.barrier()
    print('Using device: ', device, "from", dist.get_world_size(), 'processes', flush=True)

    print(keyword, time_now(), 'Reading datasets ...', flush=True)

    train_lines_input0 = read_lines(args.cls_train_input0)
    train_lines_input1 = read_lines(args.cls_train_input1) if (args.cls_train_input1 is not None) else None
    train_label_lines = read_lines(args.cls_train_label)

    valid_lines_input0 = read_lines(args.cls_dev_input0)
    valid_lines_input1 = read_lines(args.cls_dev_input1) if (args.cls_dev_input1 is not None) else None
    valid_label_lines = read_lines(args.cls_dev_label)

    num_classes = len(label_dict)
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    log_each_batch_num = accumulation_steps

    start_line = 0
    max_input_lines = args.max_input_lines

    print(keyword, time_now(), 'Preparing dev set ...', flush=True)
    valid_dataset = ClsDataset(valid_lines_input0, lines_input1 = valid_lines_input1,
                               label_dict=label_dict, label_lines=valid_label_lines,
                               regression_target = args.regression_target,
                               regression_scale_factor=args.regression_scale_factor,
                               max_seq_len = args.main_sequence_encoder_max_seq_len)

    print(keyword, time_now(), 'Preparing training set ...', flush=True)
    train_dataset = ClsDataset(train_lines_input0, lines_input1 = train_lines_input1,
                               label_dict=label_dict, label_lines=train_label_lines,
                               regression_target = args.regression_target,
                               regression_scale_factor=args.regression_scale_factor,
                               max_seq_len = args.main_sequence_encoder_max_seq_len)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=cls_data_collate_wrapper,
                                   shuffle=True, drop_last=True, num_workers=2, persistent_workers=True)

    print(keyword, time_now(), 'Forming model ...', flush=True)

    # cls_model = KinyaBERT_SequenceClassifier(args, cfg, num_classes).to(device)
    cls_model = KinyaBERT_SequenceClassifier_from_pretrained(num_classes, device, args, cfg, args.pretrained_model_file)

    if args.encoder_fine_tune:
        shared_encoder = None
    else:
        # shared_encoder =  KinyaBERT_PretrainModel(args, cfg).to(device)
        shared_encoder = KinyaBERT_from_pretrained(device, args, cfg, args.pretrained_model_file).encoder

    peak_lr = args.peak_lr # 1e-5
    wd = args.wd # 0.1
    lr_decay_style = 'linear'
    init_step = 0

    num_epochs = args.num_epochs # 30 # ==> Corresponds to 10 epochs
    if len(train_lines_input0) > args.max_input_lines:
        num_loops = num_epochs * int(len(train_lines_input0) / args.max_input_lines)
        print(keyword, 'Adjusted training loops:', num_loops,'corresponding to ', num_epochs, 'epochs')
        num_epochs = num_loops

    num_iters = math.ceil(num_epochs * len(train_data_loader) / accumulation_steps)
    warmup_iter = math.ceil(num_iters * args.warmup_ratio) # warm-up for first 6% of iterations

    optimizer = mAdamW(cls_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd,
                       correct_bias=True,
                       local_normalization=False)

    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=lr_decay_style,
                               last_iter=init_step)

    scaler = torch.cuda.amp.GradScaler()

    best_dev_results = (-999.9, -999.9) if args.regression_target else (-999.9,-999.9,-999.9,-999.9)
    curr_epochs = 0
    if args.load_saved_model:
        kb_state_dict = torch.load(args.devbest_cls_model_save_file_path)
        cls_model.load_state_dict(kb_state_dict['model_state_dict'])
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        curr_epochs = math.ceil(lr_scheduler.num_iters * accumulation_steps / len(train_data_loader))

    is_nan = torch.stack([torch.isnan(p).any() for p in cls_model.parameters()]).any().item()
    print(keyword, time_now(), 'IS ANY MODEL PARAM NAN?', is_nan, flush=True)
    print(keyword, time_now(), 'Start training ...', flush=True)

    with progressbar.ProgressBar(initial_value=0, max_value=(lr_scheduler.end_iter + 1), redirect_stdout=True) as bar:
        bar.update(0)
        for epoch in range(curr_epochs,num_epochs):
            best_dev_results = train_loop(args, keyword, epoch, scaler, cls_model, shared_encoder, device, optimizer, lr_scheduler,
                                    train_data_loader, valid_dataset, best_dev_results,
                                    accumulation_steps, args.devbest_cls_model_save_file_path, args.regression_target, args.regression_scale_factor,bar)
            if max_input_lines < len(train_lines_input0):
                del train_dataset
                del train_data_loader
                if ((epoch+1) < num_epochs):
                    start_line = start_line + max_input_lines
                    print(keyword, time_now(), 'Preparing training set ...', flush=True)
                    train_dataset = ClsDataset(train_lines_input0, lines_input1=train_lines_input1,
                                               label_dict=label_dict, label_lines=train_label_lines,
                                               regression_target=args.regression_target,
                                               regression_scale_factor=args.regression_scale_factor,
                                               max_seq_len=args.main_sequence_encoder_max_seq_len)
                    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                   collate_fn=cls_data_collate_wrapper,
                                                   drop_last=True, shuffle=True)

    print(keyword, time_now(), keyword, 'Training complete!',  flush=True)
    if args.regression_target:
        print(keyword, '==> Final best dev set results:', '(Pearson R, Spearman R) [%] = ('+(', '.join(['{:.2f}'.format(100.0 * r) for r in best_dev_results]))+')')
    else:
        print(keyword, '==> Final best dev set results:', '(Accuracy [, F1_binary], F1_weighted, F1_micro, F1_Macro) [%] = (' + (', '.join(['{:.2f}'.format(100.0 * r) for r in best_dev_results]))+')')
    del valid_dataset
    if max_input_lines >= len(train_lines_input0):
        del train_dataset
        del train_data_loader

    if args.final_cls_model_save_file_path is not None:
        torch.save({'model_state_dict': cls_model.state_dict()}, args.final_cls_model_save_file_path)

    final_test_results, bestdev_test_results = None, None
    if args.cls_test_label is not None:
        print(keyword, time_now(), 'Preparing test set ...', flush=True)
        test_lines_input0 = read_lines(args.cls_test_input0)
        test_lines_input1 = read_lines(args.cls_test_input1) if (args.cls_test_input1 is not None) else None
        test_label_lines = read_lines(args.cls_test_label)
        test_dataset = ClsDataset(test_lines_input0, lines_input1=test_lines_input1,
                                   label_dict=label_dict, label_lines=test_label_lines,
                                   regression_target=args.regression_target,
                                   regression_scale_factor=args.regression_scale_factor,
                                   max_seq_len=args.main_sequence_encoder_max_seq_len)
        if args.regression_target:
            final_test_results = cls_model_eval_regression(args, cls_model, shared_encoder, device, test_dataset, args.regression_scale_factor)
            print(keyword, '==> Final test set results:', '(Pearson R, Spearman R) [%] = ('+(', '.join(['{:.2f}'.format(100.0 * r) for r in final_test_results]))+')')
        else:
            final_test_results = cls_model_eval_classification(args, cls_model, shared_encoder, device, test_dataset)
            print(keyword, '==> Final test set results:', '(Accuracy [, F1_binary], F1_weighted, F1_micro, F1_Macro) [%] = (' + (', '.join(['{:.2f}'.format(100.0 * r) for r in final_test_results]))+')')

        kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
        cls_model.load_state_dict(kb_state_dict['model_state_dict'])

        if args.regression_target:
            bestdev_test_results = cls_model_eval_regression(args, cls_model, shared_encoder, device, test_dataset, args.regression_scale_factor)
            print(keyword, '==> Final test set results (using best dev):', '(Pearson R, Spearman R) [%] = ('+(', '.join(['{:.2f}'.format(100.0 * r) for r in bestdev_test_results]))+')')
        else:
            bestdev_test_results = cls_model_eval_classification(args, cls_model, shared_encoder, device, test_dataset)
            print(keyword, '==> Final test set results (using best dev):', '(Accuracy [, F1_binary], F1_weighted, F1_micro, F1_Macro) [%] = (' + (', '.join(['{:.2f}'.format(100.0 * r) for r in bestdev_test_results]))+')')

    return best_dev_results, final_test_results, bestdev_test_results

def cls_trainer_main():
    import os
    import random
    args = finetune_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{random.randint(8181,9191)}'
    if args.gpus == 0:
        args.world_size = 1

    cfg = BaseConfig()
    print('BaseConfig: \n\ttot_num_stems: {}\n'.format(cfg.tot_num_stems),
          '\ttot_num_affixes: {}\n'.format(cfg.tot_num_affixes),
          '\ttot_num_lm_morphs: {}\n'.format(cfg.tot_num_lm_morphs),
          '\ttot_num_pos_tags: {}\n'.format(cfg.tot_num_pos_tags), flush=True)

    mp.spawn(TEXT_CLS_train_main, nprocs=args.world_size, args=(args,cfg))

if __name__ == '__main__':
    cls_trainer_main()
