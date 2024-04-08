from __future__ import print_function, division

import gc
import math
import os
import os.path
import sys
import time
from shutil import copyfile
from typing import Union

import apex
import progressbar
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from deepkin.models.arguments import py_trainer_args
from deepkin.models.data import mlm_data_collate_wrapper_dev, mlm_model_forward_dev
from deepkin.models.kinyabert import KinyaBERT_PretrainModel
from deepkin.models.modules import BaseConfig
from deepkin.optim.learning_rates import AnnealingLR
from deepkin.utils.misc_functions import time_now, date_now
from mlm_data import MyMLMDataset


def train_loop(cfg: BaseConfig, model: Union[DDP,KinyaBERT_PretrainModel], device, optimizer: apex.optimizers.FusedLAMB, lr_scheduler: AnnealingLR, data_loader,
               save_file_path, accumulation_steps, loop, num_loops, total_steps, bar):
    world_size = dist.get_world_size()
    model.train()
    model.zero_grad(set_to_none=True)

    loss_aggr = [torch.tensor(0.0, device=device) for _ in range(4)]
    loss_Z = len(data_loader) * world_size

    start_steps = total_steps
    start_time = time.time()
    count_items = 0

    # Train
    for batch_idx, batch_data_item in enumerate(data_loader):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            losses = mlm_model_forward_dev(batch_data_item, model, device, cfg)
            total_loss = sum(losses)/accumulation_steps
        total_loss.backward()
        for i in range(len(losses)):
            loss_aggr[i] += (losses[i].detach().clone().squeeze())
        total_steps += 1
        count_items += 1
        if int(total_steps % (accumulation_steps//world_size)) == 0:
            lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            current_time = time.time()
            torch.cuda.empty_cache()
            if (dist.get_rank() == 0):
                print(time_now(),
                      'Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                      'Warmup Iters: ', "{}".format(lr_scheduler.warmup_iter),
                      'OBJ:',
                      'STEM:', "{:.6f}".format(loss_aggr[0].item() / count_items),
                      'POS:', "{:.6f}".format(loss_aggr[1].item() / count_items),
                      'AFSET:', "{:.6f}".format(loss_aggr[2].item() / count_items),
                      'AFFIX:', "{:.6f}".format(loss_aggr[3].item() / count_items),
                      'LR: ', "{:.6f}/{}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
                      'Milli_Steps_Per_Second (MSS): ', "{:.3f}".format(
                        1000.0 * ((total_steps - start_steps) / (accumulation_steps // world_size)) / (
                                current_time - start_time)),
                      'Epochs:', '{}/{}'.format(loop + 1, num_loops), flush=True)
                bar.update(loop)
                bar.fd.flush()
                sys.stdout.flush()
                sys.stderr.flush()

    # Aggregate losses
    for i in range(4):
        dist.all_reduce(loss_aggr[i])

    # Logging & Checkpointing
    if (dist.get_rank() == 0):
        total_loss = sum([ls.item() for ls in loss_aggr]) / loss_Z
        print(time_now(),
              'After Iter: ', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              'LOSS:',
              'STEM:', "{:.6f}".format(loss_aggr[0].item()/loss_Z),
              'POS:', "{:.6f}".format(loss_aggr[1].item()/loss_Z),
              'AFSET:', "{:.6f}".format(loss_aggr[2].item()/loss_Z),
              'AFFIX:', "{:.6f}".format(loss_aggr[3].item()/loss_Z),
              'LR: ', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Loop:', '{}/{}'.format(loop, num_loops),
              'WarmupIters: ', "{}".format(lr_scheduler.warmup_iter), flush=True)

        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path+"_prev_checkpoint.pt")
            print(time_now(), 'Prev model file checkpointed!', flush=True)

        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            if math.isfinite(total_loss):
                torch.save({'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'loop': loop,
                            'num_loops': num_loops}, save_file_path+"_safe_checkpoint.pt")
            torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'loop': loop,
                        'num_loops': num_loops}, save_file_path)

    return total_steps

def post_mlm_train(rank, args, cfg:BaseConfig):
    print(time_now(), 'Called post_mlm_train()', flush=True)
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.backends.cudnn.benchmark = True
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)
    print('Using device: ', device, "from", dist.get_world_size(), 'processes', flush=True)

    if rank==0:
        print('Using device: ', device)

    home_path = args.home_path

    dataset = MyMLMDataset(args.train_unparsed_corpus)
    data_loader = DataLoader(dataset, batch_size=args.bert_batch_size,
                             collate_fn=mlm_data_collate_wrapper_dev,
                             drop_last=True, shuffle=True, pin_memory=True, num_workers=2,
                             persistent_workers=True)

    peak_lr = 2e-4
    wd = 0.01
    end_iters =  int(args.post_mlm_epochs * len(data_loader) / (args.bert_accumulation_steps//dist.get_world_size()))
    warmup_iter =  int(end_iters * 0.1)
    lr_decay_style = 'linear'

    if (dist.get_rank() == 0):
        print('Model Arguments:', args)
        print(time_now(), 'Forming model ...', flush=True)

    model = KinyaBERT_PretrainModel(args, cfg).to(device)
    model.float()
    model = DDP(model, device_ids=[rank])
    model.float()

    curr_save_file_path = home_path+(f"models/kinyabert_post_mlm_model_{date_now()}.pt")

    if (dist.get_rank() == 0):
        print('---------------------------------- KinyaBERT Model Size ----------------------------------------')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')

    optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-07, weight_decay=wd)
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=end_iters,
                               decay_style=lr_decay_style,
                               last_iter=-1)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # Load saved state
    if (dist.get_rank() == 0):
        print(time_now(), 'Loading model state...', flush=True)
    kb_state_dict = torch.load(args.pretrained_model_file, map_location=map_location)
    model.module.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()

    if (dist.get_rank() == 0):
        print('------------------ Train Config --------------------')
        print('accumulation_steps: ', args.bert_accumulation_steps)
        print('batch_size: ', args.bert_batch_size)
        print('effective_batch_size: ', args.bert_batch_size*args.bert_accumulation_steps)
        print('peak_lr: {:.8f}'.format(peak_lr))
        print('iters: ', lr_scheduler.num_iters)
        print('warmup_iter: ', warmup_iter)
        print('end_iters: ', end_iters)
        print('-----------------------------------------------------')

    total_steps = int(lr_scheduler.num_iters * args.bert_accumulation_steps)

    if (dist.get_rank() == 0):
        print(time_now(), 'Start training for', args.post_mlm_epochs, 'post-mlm epochs', flush=True)

    with progressbar.ProgressBar(initial_value=0, max_value=args.post_mlm_epochs, redirect_stdout=True) as bar:
        for mlm_epoch in range(args.post_mlm_epochs):
            bar.update(mlm_epoch)
            sys.stdout.flush()
            total_steps = train_loop(cfg, model, device, optimizer, lr_scheduler, data_loader,
                                     curr_save_file_path, args.bert_accumulation_steps, mlm_epoch, args.post_mlm_epochs,
                                     total_steps, bar)
    return model


def bert_trainer_main():
    args = py_trainer_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1

    cfg = BaseConfig()
    print('BaseConfig: \n\ttot_num_stems: {}\n'.format(cfg.tot_num_stems),
          '\ttot_num_affixes: {}\n'.format(cfg.tot_num_affixes),
          '\ttot_num_lm_morphs: {}\n'.format(cfg.tot_num_lm_morphs),
          '\ttot_num_pos_tags: {}\n'.format(cfg.tot_num_pos_tags), flush=True)

    mp.spawn(post_mlm_train, nprocs=args.world_size, args=(args, cfg,))

if __name__ == '__main__':
    from deepkin.clib.libkinlp.kinlpy import build_kinlpy_lib
    build_kinlpy_lib()
    bert_trainer_main()
