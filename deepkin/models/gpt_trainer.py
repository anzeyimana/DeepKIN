from __future__ import print_function, division

import gc
import math
import os
import os.path
import sys
import time
from shutil import copyfile

import apex
import progressbar
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from deepkin.data.base_data import read_corpus
from deepkin.models.arguments import py_trainer_args
from deepkin.models.data import GPTDataset, gpt_data_collate_wrapper, gpt_model_forward
from deepkin.models.kinyagpt import KinyaGPT
from deepkin.models.modules import BaseConfig
from deepkin.optim.gradvac import GradVac
from deepkin.optim.learning_rates import AnnealingLR
from deepkin.utils.misc_functions import time_now, date_now


def train_loop(cfg: BaseConfig, model: KinyaGPT, device, grad_optimizer: GradVac, lr_scheduler: AnnealingLR, data_loader,
               save_file_path, accumulation_steps, loop, num_loops, total_steps, bar):
    world_size = dist.get_world_size()
    model.train()
    model.zero_grad(set_to_none=True)

    loss_aggr = [torch.tensor(0.0, device=device) for _ in range(4)]
    nll_loss_aggr = [torch.tensor(0.0, device=device) for _ in range(3)]
    loss_Z = len(data_loader) * world_size

    start_steps = total_steps
    start_time = time.time()
    count_items = 0

    # Train
    for batch_idx, batch_data_item in enumerate(data_loader):
        with torch.cuda.amp.autocast():
            losses, nll_losses = gpt_model_forward(batch_data_item, model, device, cfg)
            mt_losses = [(loss/accumulation_steps) for loss in losses]
            grad_optimizer.backward(mt_losses)
        for i in range(len(mt_losses)):
            loss_aggr[i] += (mt_losses[i].detach().clone().squeeze() * accumulation_steps)
        for i in range(len(nll_losses)):
            nll_loss_aggr[i] += (nll_losses[i].detach().clone().squeeze())
        total_steps += 1
        count_items += 1
        if int(total_steps % (accumulation_steps//world_size)) == 0:
            lr_scheduler.step()
            grad_optimizer.step()
            current_time = time.time()
            torch.cuda.empty_cache()
            # if (dist.get_rank() == 0):
            #     print(time_now(),
            #           'Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
            #           'Warmup Iters: ', "{}".format(lr_scheduler.warmup_iter),
            #           'OBJ:',
            #           'STEM:', "{:.6f}".format(loss_aggr[0].item() / count_items),
            #           'POS:', "{:.6f}".format(loss_aggr[1].item() / count_items),
            #           'AFSET:', "{:.6f}".format(loss_aggr[2].item() / count_items),
            #           'AFFIX:', "{:.6f}".format(loss_aggr[3].item() / count_items),
            #           'NLL_OBJ:',
            #           'STEM:', "{:.6f}".format(nll_loss_aggr[0].item() / count_items),
            #           'POS:', "{:.6f}".format(nll_loss_aggr[1].item() / count_items),
            #           'AFSET:', "{:.6f}".format(nll_loss_aggr[2].item() / count_items),
            #           'LR: ', "{:.6f}/{}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
            #           'Milli_Steps_Per_Second (MSS): ', "{:.3f}".format(
            #             1000.0 * ((total_steps - start_steps) / (accumulation_steps // world_size)) / (
            #                         current_time - start_time)),
            #           'Epochs:', '{}/{}'.format(loop + 1, num_loops), flush=True)
            #     bar.update(loop)
            #     bar.fd.flush()
            #     sys.stdout.flush()
            #     sys.stderr.flush()

    # Aggregate losses
    for i in range(4):
        dist.all_reduce(loss_aggr[i])

    for i in range(3):
        dist.all_reduce(nll_loss_aggr[i])

    # Logging & Checkpointing
    if (dist.get_rank() == 0):
        print(time_now(),
              'After Iter: ', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              'LOSS:',
              'STEM:', "{:.6f}".format(loss_aggr[0].item()/loss_Z),
              'POS:', "{:.6f}".format(loss_aggr[1].item()/loss_Z),
              'AFSET:', "{:.6f}".format(loss_aggr[2].item()/loss_Z),
              'AFFIX:', "{:.6f}".format(loss_aggr[3].item()/loss_Z),
              'NLL_LOSS:',
              'STEM:', "{:.6f}".format(nll_loss_aggr[0].item()/loss_Z),
              'POS:', "{:.6f}".format(nll_loss_aggr[1].item()/loss_Z),
              'AFSET:', "{:.6f}".format(nll_loss_aggr[2].item()/loss_Z),
              'LR: ', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Loop:', '{}/{}'.format(loop, num_loops),
              'WarmupIters: ', "{}".format(lr_scheduler.warmup_iter), flush=True)

        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path+"_prev_checkpoint.pt")
            print(time_now(), 'Prev model file checkpointed!', flush=True)

        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            torch.save({'model_state_dict': model.state_dict(),
                        'grad_optimizer_state_dict': grad_optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'loop': loop,
                        'num_loops': num_loops}, save_file_path)

    return total_steps


def eval_loop(cfg: BaseConfig, model: KinyaGPT, device, accumulation_steps, eval_data_loader):
    world_size = dist.get_world_size()
    model.train()
    model.zero_grad(set_to_none=True)

    loss_aggr = [torch.tensor(0.0, device=device) for _ in range(4)]
    nll_loss_aggr = [torch.tensor(0.0, device=device) for _ in range(3)]
    loss_Z = len(eval_data_loader) * world_size

    # Train
    with progressbar.ProgressBar(max_value=len(eval_data_loader), redirect_stdout=True) as bar:
        for batch_idx, batch_data_item in enumerate(eval_data_loader):
            if (batch_idx % 10) == 0:
                bar.update(batch_idx)
            losses, nll_losses = gpt_model_forward(batch_data_item, model, device, cfg)
            mt_losses = [(loss/accumulation_steps) for loss in losses]
            for i in range(len(mt_losses)):
                loss_aggr[i] += (mt_losses[i].detach().clone().squeeze() * accumulation_steps)
            for i in range(len(nll_losses)):
                nll_loss_aggr[i] += (nll_losses[i].detach().clone().squeeze())

    # Aggregate losses
    for i in range(4):
        dist.all_reduce(loss_aggr[i])

    for i in range(3):
        dist.all_reduce(nll_loss_aggr[i])

    # Logging & Checkpointing
    if (dist.get_rank() == 0):
        print(time_now(),
              'LOSS:',
              'STEM:', "{:.6f}".format(loss_aggr[0].item()/loss_Z),
              'POS:', "{:.6f}".format(loss_aggr[1].item()/loss_Z),
              'AFSET:', "{:.6f}".format(loss_aggr[2].item()/loss_Z),
              'AFFIX:', "{:.6f}".format(loss_aggr[3].item()/loss_Z),
              'NLL_LOSS:',
              'STEM:', "{:.6f}".format(nll_loss_aggr[0].item()/loss_Z),
              'POS:', "{:.6f}".format(nll_loss_aggr[1].item()/loss_Z),
              'AFSET:', "{:.6f}".format(nll_loss_aggr[2].item()/loss_Z),
              'PERPLEXITY:'
              'STEM:', "{:.4f}".format(math.exp(nll_loss_aggr[0].item()/loss_Z)),
              'POS:', "{:.4f}".format(math.exp(nll_loss_aggr[1].item()/loss_Z)),
              'AFSET:', "{:.4f}".format(math.exp(nll_loss_aggr[2].item()/loss_Z)),
              'AFFIX:', "{:.4f}".format(math.exp(loss_aggr[3].item()/loss_Z)), flush=True)

def train_fn(rank, args, cfg:BaseConfig):
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(rank)
    scaler = torch.cuda.amp.GradScaler()

    if rank==0:
        print('Using device: ', device)

    home_path = args.home_path

    end_iters =  args.gpt_num_iters
    warmup_iter =  args.gpt_warmup_iters

    peak_lr = args.peak_lr #
    wd = args.wd # 0.01
    lr_decay_style = 'linear'

    if (dist.get_rank() == 0):
        print('Model Arguments:', args)
        print(time_now(), 'Forming model ...', flush=True)

    model = KinyaGPT(args, cfg).to(device)
    model.float()

    curr_save_file_path = home_path+(f"models/kinyagpt_model_{date_now()}.pt")

    if (dist.get_rank() == 0):
        print('---------------------------------- KinyaGPT Model Size ----------------------------------------')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')

    optmz = apex.optimizers.FusedLAMB(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-06, weight_decay=wd)
    grad_optimizer = GradVac(4, optmz, device, scaler=scaler, beta=1e-2, reduction='sum', cpu_offload = False)
    lr_scheduler = AnnealingLR(optmz,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=end_iters,
                               decay_style=lr_decay_style,
                               last_iter=-1)


    if (not args.load_saved_model) and (dist.get_world_size() > 1):
        if (dist.get_rank() == 0):
            model.eval()
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                torch.save({'model_state_dict': model.state_dict(),
                            'grad_optimizer_state_dict': grad_optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'loop': 0,
                            'num_loops': 20000}, curr_save_file_path)
        dist.barrier()
        args.load_saved_model = True

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if args.load_saved_model:
        # Load saved state
        if (dist.get_rank() == 0):
            print(time_now(), 'Loading model state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=map_location)
        model.load_state_dict(kb_state_dict['model_state_dict'])
        del kb_state_dict
        gc.collect()

        if (dist.get_rank() == 0):
            print(time_now(), 'Loading optimizer state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=torch.device('cpu'))
        grad_optimizer.load_state_dict(kb_state_dict['grad_optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        lr_scheduler.end_iter = end_iters
        del kb_state_dict
        gc.collect()

    # @TODO: Sync model parameters, how can params data be shared among processes while model load is from one process?

    num_train_loops = math.ceil(end_iters * args.gpt_accumulation_steps / args.gpt_number_of_load_batches)

    curr_loops = math.floor(lr_scheduler.num_iters * num_train_loops / end_iters)

    if (dist.get_rank() == 0):
        print('------------------ Train Config --------------------')
        print('curr_loops: ', curr_loops)
        print('num_train_loops: ', num_train_loops)
        print('number_of_load_batches: ', args.gpt_number_of_load_batches)
        print('accumulation_steps: ', args.gpt_accumulation_steps)
        print('batch_size: ', args.gpt_batch_size)
        print('effective_batch_size: ', args.gpt_batch_size*args.gpt_accumulation_steps)
        print('peak_lr: {:.8f}'.format(peak_lr))
        print('iters: ', lr_scheduler.num_iters)
        print('warmup_iter: ', warmup_iter)
        print('end_iters: ', end_iters)
        print('-----------------------------------------------------')

        print(time_now(), 'Reading corpus text ...', flush=True)

    # @TODO: See what can be optimized for multiprocessing, e.g. share corpus data without each process reading the file at the same time
    #
    parsed_corpus_file = (home_path+args.train_parsed_corpus)
    parsed_corpus_lines = read_corpus(parsed_corpus_file)
    parsed_corpus_doc_ends = [i for i in range(len(parsed_corpus_lines)) if (len(parsed_corpus_lines[i]) == 0)]

    if (dist.get_rank() == 0):
        print(time_now(), 'Corpus text read {} sentences {} docs!'.format(len(parsed_corpus_lines), len(parsed_corpus_doc_ends)), flush=True)

    total_steps = int(lr_scheduler.num_iters * args.gpt_accumulation_steps)

    if (dist.get_rank() == 0):
        print(time_now(), 'Start training for', num_train_loops, 'loops ({} iterations)'.format(args.gpt_num_iters), flush=True)

    with progressbar.ProgressBar(initial_value=curr_loops, max_value=num_train_loops, redirect_stdout=True) as bar:
        if (dist.get_rank() == 0):
            bar.update(curr_loops)
            sys.stdout.flush()
        loop = curr_loops
        while lr_scheduler.num_iters < lr_scheduler.end_iter:
            if (dist.get_rank() == 0):
                print(time_now(), 'Loading dataset...', flush=True)

            # @TODO: Reduce the amount of data read by each process

            gpt_dataset = GPTDataset(parsed_corpus_lines, parsed_corpus_doc_ends, cfg,
                                     args.gpt_number_of_load_batches * args.gpt_batch_size // dist.get_world_size(),
                                     max_seq_len=args.main_sequence_encoder_max_seq_len)

            data_loader = DataLoader(gpt_dataset, batch_size=args.gpt_batch_size, collate_fn=gpt_data_collate_wrapper,
                                     drop_last=False, shuffle=True, num_workers=1, persistent_workers=True)

            total_steps = train_loop(cfg, model, device, grad_optimizer, lr_scheduler, data_loader,
                                     curr_save_file_path, args.gpt_accumulation_steps, loop, num_train_loops,
                                     total_steps, bar)
            loop += 1
            if (dist.get_rank() == 0):
                print(time_now(), loop, 'TRAINING LOOPS COMPLETE!', flush=True)
                bar.update(loop)
                sys.stdout.flush()

            del data_loader
            del gpt_dataset
            gc.collect()

def eval_fn(rank, args, cfg:BaseConfig):
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(rank)
    if rank==0:
        print('Using device: ', device)

    home_path = args.home_path

    if (dist.get_rank() == 0):
        print('Model Arguments:', args)
        print(time_now(), 'Forming model ...', flush=True)

    model = KinyaGPT(args, cfg).to(device)
    model.float()

    curr_save_file_path = home_path+(f"models/kinyagpt_model_{date_now()}.pt")

    if (dist.get_rank() == 0):
        print('---------------------------------- KinyaGPT Model Size ----------------------------------------')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # Load saved state
    if (dist.get_rank() == 0):
        print(time_now(), 'Loading model state...', flush=True)
    kb_state_dict = torch.load(curr_save_file_path, map_location=map_location)
    model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()

    parsed_corpus_file = home_path + (args.dev_parsed_corpus)
    parsed_corpus_lines = read_corpus(parsed_corpus_file)
    parsed_corpus_doc_ends = [i for i in range(len(parsed_corpus_lines)) if (len(parsed_corpus_lines[i]) == 0)]

    gpt_dataset = GPTDataset(parsed_corpus_lines, parsed_corpus_doc_ends, cfg,
                             args.gpt_number_of_load_batches * args.gpt_batch_size // dist.get_world_size(),
                             max_seq_len=args.main_sequence_encoder_max_seq_len)

    eval_data_loader = DataLoader(gpt_dataset, batch_size=args.gpt_batch_size, collate_fn=gpt_data_collate_wrapper,
                             drop_last=False, shuffle=False, num_workers=1, persistent_workers=False)
    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        eval_loop(cfg, model, device, args.gpt_accumulation_steps, eval_data_loader)

def gpt_trainer_main():
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

    # mp.spawn(train_fn, nprocs=args.world_size, args=(args, cfg,))
    mp.spawn(eval_fn, nprocs=args.world_size, args=(args, cfg,))
    #train_fn(args,cfg)

if __name__ == '__main__':
    gpt_trainer_main()
