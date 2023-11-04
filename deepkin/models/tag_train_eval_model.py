from __future__ import print_function, division

import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader

from deepkin.models.arguments import finetune_args
from deepkin.models.kinyabert import KinyaBERT_SequenceTagger_from_pretrained, KinyaBERT_from_pretrained
from deepkin.models.modules import BaseConfig
from deepkin.optim.adamw import mAdamW
from deepkin.optim.learning_rates import AnnealingLR
from deepkin.utils.misc_functions import time_now, read_lines


def tagger_model_eval(args, tagger_model, shared_encoder, device, eval_dataset, tag_dict):
    from tag_data_loaders import tag_model_predict
    tagger_model.eval()
    inv_dict = {tag_dict[k]:k for k in tag_dict}
    y_true  = []
    y_pred = []

    with torch.no_grad():
        for itr,data_item in enumerate(eval_dataset.itemized_data):
            output_scores, predicted_labels, true_labels = tag_model_predict(data_item, tagger_model, shared_encoder, device)
            y_pred.append([inv_dict[predicted_labels[i].item()] for i in range(len(true_labels))])
            y_true.append([inv_dict[true_label] for true_label in true_labels])

    micro_F1 = f1_score(y_true, y_pred, average= 'micro')
    macro_F1 = f1_score(y_true, y_pred, average= 'macro')
    weighted_F1 = f1_score(y_true, y_pred, average= 'weighted')
    return (micro_F1,macro_F1,weighted_F1)

def train_loop(args, keyword, epoch, scaler, tagger_model, shared_encoder, device, optimizer, lr_scheduler, train_data_loader, valid_dataset, best_results,  accumulation_steps, save_file_path, label_dict, bar):
    from tag_data_loaders import  tag_model_forward
    tagger_model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    epoch_avg_train_loss = 0.0
    epoch_iter_count = 0.0
    max_tot_grad_norm = 0.0
    max_cw_grad_norm = 0.0
    iter_loss = 0.0
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        with torch.cuda.amp.autocast():
            output_scores, batch_labels = tag_model_forward(batch_data_item, tagger_model, shared_encoder, device)
            output_scores = F.log_softmax(output_scores, dim=1)
            loss = F.nll_loss(output_scores, torch.tensor(batch_labels).to(device))
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        iter_loss += loss.item()

        if ((batch_idx+1) % accumulation_steps) == 0:
            total_norm = 0.0
            max_local_cw_grad_norm = 0.0
            for param in tagger_model.parameters():
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
                for param in tagger_model.parameters():
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_norm_(param, args.ft_cwgnc)
            else:
                torch.nn.utils.clip_grad_norm_(tagger_model.parameters(), 100.0)
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

    dev_results = tagger_model_eval(args, tagger_model, shared_encoder, device, valid_dataset, label_dict)
    if dev_results[0] > best_results[0]:
        best_results = dev_results
        torch.save({'iter': iter,
                    'model_state_dict': tagger_model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    # 'best_F1': best_F1
                    },
                   save_file_path)
    # print('@'+keyword+': After', (epoch+1), 'epochs:', '==> Validation set F1:','{:.2f}'.format(100.0 * dev_F1),  '==> Best valid F1:','{:.2f}'.format(100.0 * max(best_F1,dev_F1)))
    print(keyword, 'After', (epoch + 1), 'epochs:',
          'Train Loss: {:.6f}'.format(epoch_avg_train_loss / epoch_iter_count),
          '==> Validation set F1 %[micro, macro, weighted]:', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in dev_results]),
          '==> Best valid. F1 %[micro, macro, weighted] so far:', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in best_results]))
    bar.update(lr_scheduler.num_iters)
    return best_results

def KIN_NER_train_main(rank, args, cfg: BaseConfig):
    from tag_data_loaders import TagDataset, tag_data_collate_wrapper
    import progressbar

    keyword = args.model_keyword
    # @Workstation-PC
    home_path = args.home_path
    USE_GPU = (args.gpus > 0)

    label_dict = {}
    label_dict['B-PER'] = 0
    label_dict['I-PER'] = 1
    label_dict['B-ORG'] = 2
    label_dict['I-ORG'] = 3
    label_dict['B-LOC'] = 4
    label_dict['I-LOC'] = 5
    label_dict['B-DATE'] = 6
    label_dict['I-DATE'] = 7
    label_dict['O'] = 8

    print('Vocab ready!')

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

    print(time_now(), 'Reading datasets ...', flush=True)

    train_lines_input0 = read_lines("datasets/NER/parsed/train_parsed.txt") + read_lines("datasets/NER_V2/parsed/train_parsed.txt")
    train_label_lines = read_lines("datasets/NER/parsed/train_labels.txt") + read_lines("datasets/NER_V2/parsed/train_labels.txt")

    valid_lines_input0 = read_lines("datasets/NER/parsed/dev_parsed.txt") + read_lines("datasets/NER_V2/parsed/dev_parsed.txt")
    valid_label_lines = read_lines("datasets/NER/parsed/dev_labels.txt") + read_lines("datasets/NER_V2/parsed/dev_labels.txt")

    # train_lines_input0 = read_lines("datasets/NER/parsed/train_parsed.txt")
    # train_label_lines = read_lines("datasets/NER/parsed/train_labels.txt")
    #
    # valid_lines_input0 = read_lines("datasets/NER/parsed/dev_parsed.txt")
    # valid_label_lines = read_lines("datasets/NER/parsed/dev_labels.txt")

    # train_lines_input0 = read_lines("datasets/NER_V2/parsed/train_parsed.txt")
    # train_label_lines = read_lines("datasets/NER_V2/parsed/train_labels.txt")
    #
    # valid_lines_input0 = read_lines("datasets/NER_V2/parsed/dev_parsed.txt")
    # valid_label_lines = read_lines("datasets/NER_V2/parsed/dev_labels.txt")

    num_classes = len(label_dict)
    accumulation_steps = args.accumulation_steps # 16 batch size
    log_each_batch_num = accumulation_steps

    max_lines = args.max_input_lines

    print(time_now(), 'Preparing dev set ...', flush=True)
    valid_dataset = TagDataset(valid_lines_input0, valid_label_lines, label_dict)

    print(time_now(), 'Preparing training set ...', flush=True)
    train_dataset = TagDataset(train_lines_input0, train_label_lines, label_dict)

    print(time_now(), 'Forming model ...', flush=True)

    peak_lr = args.peak_lr #[1e-5, 3e-5, 5e-5, 8e-5]
    batch_size = args.batch_size #[16, 32]

    wd = args.wd
    lr_decay_style = 'linear'
    init_step = 0

    scaler = torch.cuda.amp.GradScaler()

    tag_model = KinyaBERT_SequenceTagger_from_pretrained(num_classes, device, args, cfg, args.pretrained_model_file)
    if args.encoder_fine_tune:
        shared_encoder = None
    else:
        shared_encoder = KinyaBERT_from_pretrained(device, args, cfg, args.pretrained_model_file).encoder

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=tag_data_collate_wrapper, drop_last=True, shuffle=True)

    num_iters = math.ceil(args.num_epochs * len(train_data_loader) / accumulation_steps)
    warmup_iter = math.ceil(num_iters * 0.06) # warm-up for first 6% of iterations

    optimizer = mAdamW(tag_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd, correct_bias=True,
                 local_normalization=False)

    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=lr_decay_style,
                               last_iter=init_step)
    # lr_scheduler = InverseSQRT_LRScheduler(optimizer, start_lr=peak_lr, warmup_iter=warmup_iter, num_iters=num_iters, warmup_init_lr=1e-8, last_iter=init_step)

    is_nan = torch.stack([torch.isnan(p).any() for p in tag_model.parameters()]).any().item()
    print(keyword, time_now(), 'IS ANY MODEL PARAM NAN?', is_nan, flush=True)

    # def nan_hook(self, inp, output):
    #     if not isinstance(output, tuple):
    #         outputs = [output]
    #     else:
    #         outputs = output
    #
    #     for i, out in enumerate(outputs):
    #         nan_mask = torch.isnan(out)
    #         if nan_mask.any():
    #             print("In", self.__class__.__name__)
    #             raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
    #                                out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
    #
    # for submodule in tag_model.modules():
    #     submodule.register_forward_hook(nan_hook)

    print(time_now(), 'Start training ...', flush=True)
    best_results = (0.0, 0.0, 0.0)
    with progressbar.ProgressBar(initial_value=0, max_value=(lr_scheduler.end_iter+1), redirect_stdout=True) as bar:
        bar.update(0)
        for epoch in range(args.num_epochs):
            best_results = train_loop(args, keyword, epoch, scaler, tag_model, shared_encoder, device, optimizer, lr_scheduler,
                                    train_data_loader, valid_dataset, best_results,
                                    accumulation_steps, args.devbest_cls_model_save_file_path, label_dict, bar)

    print(time_now(), 'Training complete!',  flush=True)
    print(keyword, f'==> Final dev test test-v1 test-v2 results using {args.pretrained_model_file}')
    print(keyword, '==> Final best dev set F1 %[micro, macro, weighted]:', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in best_results]))
    del valid_dataset
    del train_dataset
    del train_data_loader

    if args.final_cls_model_save_file_path is not None:
        torch.save({'model_state_dict': tag_model.state_dict()}, args.final_cls_model_save_file_path)

    print(time_now(), 'Preparing test-v1 set ...', flush=True)
    test_v1_lines_input0 = read_lines("datasets/NER/parsed/test_parsed.txt")
    test_v1_label_lines = read_lines("datasets/NER/parsed/test_labels.txt")
    test_v1_dataset = TagDataset(test_v1_lines_input0, test_v1_label_lines, label_dict)

    print(time_now(), 'Preparing test-v2 set ...', flush=True)
    test_v2_lines_input0 = read_lines("datasets/NER_V2/parsed/test_parsed.txt")
    test_v2_label_lines = read_lines("datasets/NER_V2/parsed/test_labels.txt")
    test_v2_dataset = TagDataset(test_v2_lines_input0, test_v2_label_lines, label_dict)

    print(time_now(), 'Preparing test-combined set ...', flush=True)
    test_combined_lines_input0 = read_lines("datasets/NER/parsed/test_parsed.txt") + read_lines("datasets/NER_V2/parsed/test_parsed.txt")
    test_combined_label_lines = read_lines("datasets/NER/parsed/test_labels.txt") + read_lines("datasets/NER_V2/parsed/test_labels.txt")
    test_combined_dataset = TagDataset(test_combined_lines_input0, test_combined_label_lines, label_dict)

    print('Test Set eval [final]:')
    final_test_v1_results = tagger_model_eval(args, tag_model, shared_encoder, device, test_v1_dataset, label_dict)
    final_test_v2_results = tagger_model_eval(args, tag_model, shared_encoder, device, test_v2_dataset, label_dict)
    final_test_combined_results = tagger_model_eval(args, tag_model, shared_encoder, device, test_combined_dataset, label_dict)
    print(keyword, '==> Final test-v1 set F1 %[micro, macro, weighted]:', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in final_test_v1_results]))
    print(keyword, '==> Final test-v2 set F1 %[micro, macro, weighted]:', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in final_test_v2_results]))
    print(keyword, '==> Final test-combined set F1 %[micro, macro, weighted]:', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in final_test_combined_results]))

    kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
    tag_model.load_state_dict(kb_state_dict['model_state_dict'])

    print('Test Set eval [dev best]:')
    best_dev_test_v1_results = tagger_model_eval(args, tag_model, shared_encoder, device, test_v1_dataset, label_dict)
    best_dev_test_v2_results = tagger_model_eval(args, tag_model, shared_encoder, device, test_v2_dataset, label_dict)
    best_dev_test_combined_results = tagger_model_eval(args, tag_model, shared_encoder, device, test_combined_dataset, label_dict)
    print(keyword, '==> Final test-v1 set F1 %[micro, macro, weighted] (using best dev):', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in best_dev_test_v1_results]))
    print(keyword, '==> Final test-v2 set F1 %[micro, macro, weighted] (using best dev):', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in best_dev_test_v2_results]))
    print(keyword, '==> Final test-combined set F1 %[micro, macro, weighted] (using best dev):', ' '.join(['{:.2f}'.format(100.0 * F1) for F1 in best_dev_test_combined_results]))


def tag_trainer_main():
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

    mp.spawn(KIN_NER_train_main, nprocs=args.world_size, args=(args,cfg))

if __name__ == '__main__':
    tag_trainer_main()
