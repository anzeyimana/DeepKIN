from __future__ import print_function, division

import gc
import math
import os
import os.path
import random
import sys
import time
from datetime import datetime
from shutil import copyfile

import apex
import numpy as np
import progressbar
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from deepkin.data.base_data import read_corpus
from deepkin.models.arguments import py_trainer_args
from deepkin.models.syllabe_data import SyllabeGPTDataset, syllabe_gpt_data_collate_wrapper
from deepkin.models.syllabe_gpt_model import SyllabeGPT
from deepkin.optim.learning_rates import AnnealingLR
from deepkin.utils.misc_functions import time_now, date_now


def validation_loop(model: DDP, device,
               lr_scheduler: AnnealingLR, validation_data_loader,
               epoch, num_epochs):
    world_size = dist.get_world_size()
    model.eval()
    model.zero_grad(set_to_none=True)

    loss_aggr = torch.tensor(0.0, device=device)
    nll_loss_aggr = torch.tensor(0.0, device=device)

    # Train
    count_items = 0
    for batch_idx, batch_data_item in enumerate(validation_data_loader):
        (syllabe_ids, syllabe_id_lengths, input_masks_padded, decoder_mask) = batch_data_item
        loss, nll_loss = model(syllabe_ids, syllabe_id_lengths, input_masks_padded, decoder_mask)
        loss_aggr += (loss.detach().clone().squeeze())
        nll_loss_aggr += (nll_loss.detach().clone().squeeze())
        count_items += 1

    loss_Z = count_items * world_size
    # Aggregate losses
    dist.all_reduce(loss_aggr)
    dist.all_reduce(nll_loss_aggr)

    # Logging & Checkpointing
    if dist.get_rank() == 0:
        print(time_now(),
              'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              'VALIDATION LOSS:',
              'SYLLABE:', "{:.6f}".format(loss_aggr.item() / loss_Z),
              'NLL_LOSS:',
              'SYLLABE:', "{:.6f}".format(nll_loss_aggr.item() / loss_Z),
              'LR:', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Warmup Iters:', "{}".format(lr_scheduler.warmup_iter),
              'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
        sys.stdout.flush()

    return loss_aggr.item() / loss_Z


def eval_loop(model: DDP, device,
              validation_data_loader):
    world_size = dist.get_world_size()
    model.eval()
    model.zero_grad(set_to_none=True)

    loss_aggr = torch.tensor(0.0, device=device)
    nll_loss_aggr = torch.tensor(0.0, device=device)

    # Train
    count_items = 0
    with progressbar.ProgressBar(max_value=len(validation_data_loader), redirect_stdout=True) as bar:
        for batch_idx, batch_data_item in enumerate(validation_data_loader):
            if (batch_idx % 10) == 0:
                bar.update(batch_idx)
            (syllabe_ids, syllabe_id_lengths, input_masks_padded, decoder_mask) = batch_data_item
            loss, nll_loss = model(syllabe_ids, syllabe_id_lengths, input_masks_padded, decoder_mask)
            loss_aggr += (loss.detach().clone().squeeze())
            nll_loss_aggr += (nll_loss.detach().clone().squeeze())
            count_items += 1

    loss_Z = count_items * world_size
    # Aggregate losses
    dist.all_reduce(loss_aggr)
    dist.all_reduce(nll_loss_aggr)

    # Logging & Checkpointing
    if dist.get_rank() == 0:
        print(time_now(),
              'VALIDATION LOSS:',
              'SYLLABE:', "{:.6f}".format(loss_aggr.item() / loss_Z),
              'NLL_LOSS:',
              'SYLLABE:', "{:.6f}".format(nll_loss_aggr.item() / loss_Z), flush=True)
        sys.stdout.flush()

    return nll_loss_aggr.item() / loss_Z

def train_loop(model: DDP, device, scaler: torch.cuda.amp.GradScaler, optimizer: apex.optimizers.FusedAdam,
               lr_scheduler: AnnealingLR, training_data_loader, validation_data_loader,
               save_file_path, accumulation_steps, epoch, num_epochs, total_steps, bar, best_valid_loss):
    world_size = dist.get_world_size()
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    loss_aggr = torch.tensor(0.0, device=device)
    nll_loss_aggr = torch.tensor(0.0, device=device)

    # Train
    start_steps = total_steps
    start_time = time.time()
    count_items = 0
    total_items = len(training_data_loader)
    for batch_idx, batch_data_item in enumerate(training_data_loader):
        (syllabe_ids, syllabe_id_lengths, input_masks_padded, decoder_mask) = batch_data_item
        with torch.cuda.amp.autocast():
            loss, nll_loss = model(syllabe_ids, syllabe_id_lengths, input_masks_padded, decoder_mask)
            loss = (loss / accumulation_steps)
            scaler.scale(loss).backward()
        loss_aggr += (loss.detach().clone().squeeze() * accumulation_steps)
        nll_loss_aggr += (nll_loss.detach().clone().squeeze())
        total_steps += 1
        count_items += 1
        left_items = total_items - count_items
        if int(total_steps % (accumulation_steps // world_size)) == 0:
            lr_scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            current_time = time.time()
            # torch.cuda.empty_cache()
            if (dist.get_rank() == 0):
                print(time_now(),
                      'Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                      'Warmup Iters: ', "{}".format(lr_scheduler.warmup_iter),
                      'OBJ: SYLLABE:', "{:.6f}".format(loss_aggr.item() / count_items),
                      'NLL_OBJ: SYLLABE:', "{:.6f}".format(nll_loss_aggr.item() / count_items),
                      'LR: ', "{:.6f}/{}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
                      'Milli_Steps_Per_Second (MSS): ', "{:.3f}".format(
                        1000.0 * ((total_steps - start_steps) / (accumulation_steps // world_size)) / (
                                    current_time - start_time)),
                      'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
                bar.update(epoch)
                bar.fd.flush()
                sys.stdout.flush()
                sys.stderr.flush()
            if left_items < (accumulation_steps // world_size):
                break

    loss_Z = count_items * world_size
    # Aggregate losses
    dist.all_reduce(loss_aggr)
    dist.all_reduce(nll_loss_aggr)

    torch.cuda.empty_cache()

    # Logging & Checkpointing
    if dist.get_rank() == 0:
        print(time_now(),
              'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              'LOSS:',
              'SYLLABE:', "{:.6f}".format(loss_aggr.item() / loss_Z),
              'NLL_LOSS:',
              'SYLLABE:', "{:.6f}".format(nll_loss_aggr.item() / loss_Z),
              'LR:', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Warmup Iters:', "{}".format(lr_scheduler.warmup_iter),
              'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
        sys.stdout.flush()

        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path + "_prev_checkpoint.pt")
            print(time_now(), 'Prev model file checkpointed!', flush=True)

        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            torch.save({'model_state_dict': model.module.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'best_valid_loss': best_valid_loss,
                        'epoch': (epoch + 1),
                        'num_epochs': num_epochs}, save_file_path)

    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        valid_loss = validation_loop(model, device, lr_scheduler, validation_data_loader, epoch, num_epochs)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if dist.get_rank() == 0:
                torch.save({'model_state_dict': model.module.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'best_valid_loss': best_valid_loss,
                            'epoch': (epoch + 1),
                            'num_epochs': num_epochs}, save_file_path+'_best_valid_loss.pt')
    return total_steps, best_valid_loss


def train_fn(rank, args):
    print(time_now(), 'Called train_fn()', flush=True)
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)

    dist.barrier()
    print('Using device: ', device, "from", dist.get_world_size(), 'processes', flush=True)

    scaler = torch.cuda.amp.GradScaler()

    home_path = args.home_path
    end_iters =  args.syllabe_num_iters
    warmup_iter =  args.syllabe_warmup_iters
    peak_lr = args.syllabe_peak_lr #
    wd = args.wd # 0.01
    lr_decay_style = 'linear'

    if (dist.get_rank() == 0):
        print('Model Arguments:', args)
        print(time_now(), 'Forming model ...', flush=True)

    model = SyllabeGPT(max_seq_len=args.syllabe_max_seq_len).to(device)
    model.float()
    model = DDP(model, device_ids=[rank])
    model.float()

    curr_save_file_path = home_path  + (f"models/syllabe_gpt_model_{date_now()}.pt")

    best_valid_loss = 999999.9

    if (dist.get_rank() == 0):
        print('---------------------------------- SyllabeGPT Model Size ----------------------------------------')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')

    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-08, weight_decay=wd)
    lr_scheduler = AnnealingLR(optimizer,
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
                torch.save({'model_state_dict': model.module.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'best_valid_loss': best_valid_loss,
                            'epoch': 0,
                            'num_epochs': 200000}, curr_save_file_path)
        dist.barrier()
        args.load_saved_model = True

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if args.load_saved_model:
        # Load saved state
        if (dist.get_rank() == 0):
            print(time_now(), 'Loading model state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=map_location)
        model.module.load_state_dict(kb_state_dict['model_state_dict'])
        epoch = kb_state_dict['epoch']
        del kb_state_dict
        gc.collect()

        if (dist.get_rank() == 0):
            print(time_now(), 'Loading optimizer state...', flush=True)
        kb_state_dict = torch.load(curr_save_file_path, map_location=torch.device('cpu'))
        scaler.load_state_dict(kb_state_dict['scaler_state_dict'])
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        best_valid_loss = kb_state_dict['best_valid_loss']
        del kb_state_dict
        gc.collect()

    # @TODO: Sync model parameters, how can params data be shared among processes while model load is from one process?

    num_train_loops = math.ceil(end_iters * args.syllabe_accumulation_steps / args.syllabe_number_of_load_batches)

    curr_loops = math.floor(lr_scheduler.num_iters * num_train_loops / end_iters)

    if (dist.get_rank() == 0):
        print('------------------ Train Config --------------------')
        print('curr_loops: ', curr_loops)
        print('best_valid_loss: ', '{:.6f}'.format(best_valid_loss))
        print('num_train_loops: ', num_train_loops)
        print('number_of_load_batches: ', args.syllabe_number_of_load_batches)
        print('accumulation_steps: ', args.syllabe_accumulation_steps)
        print('batch_size: ', args.syllabe_batch_size, 'sequences')
        print('effective_batch_size: ', args.syllabe_batch_size * args.syllabe_accumulation_steps, 'sequences')
        print('peak_lr: {:.8f}'.format(peak_lr))
        print('iters: ', lr_scheduler.num_iters)
        print('warmup_iter: ', warmup_iter)
        print('end_iters: ', end_iters)
        print('-----------------------------------------------------')

        print(time_now(), 'Reading training corpus text ...', flush=True)

    # @TODO: See what can be optimized for multiprocessing, e.g. share corpus data without each process reading the file at the same time
    #
    unparsed_corpus_file = (home_path + args.train_unparsed_corpus)
    text_corpus_lines = read_corpus(unparsed_corpus_file)
    text_corpus_doc_ends = [i for i in range(len(text_corpus_lines)) if (len(text_corpus_lines[i]) == 0)]

    if (dist.get_rank() == 0):
        print(time_now(),
              'Corpus text read {} sentences {} docs!'.format(len(text_corpus_lines), len(text_corpus_doc_ends)),
              flush=True)

    dev_text = read_corpus(home_path + args.dev_unparsed_corpus)
    dev_text_doc_ends = [i for i in range(len(dev_text)) if (len(dev_text[i]) == 0)]

    # Static seed
    np.random.seed(123)
    random.seed(123)
    dev_gpt_dataset = SyllabeGPTDataset(dev_text, dev_text_doc_ends,
                                    2000,
                                    max_seq_len=args.syllabe_max_seq_len)
    validation_data_loader = DataLoader(dev_gpt_dataset, batch_size=1,
                                      collate_fn=syllabe_gpt_data_collate_wrapper,
                                      drop_last=False, shuffle=False, num_workers=1, persistent_workers=True)
    # Random seed
    seed_val = datetime.now().microsecond + (3737 * dist.get_rank())
    np.random.seed(seed_val)
    random.seed(seed_val)

    total_steps = int(lr_scheduler.num_iters * args.syllabe_accumulation_steps)

    if (dist.get_rank() == 0):
        print(time_now(), 'Start training for', num_train_loops, 'loops ({} iterations)'.format(args.syllabe_num_iters),
              flush=True)

    with progressbar.ProgressBar(initial_value=curr_loops, max_value=num_train_loops, redirect_stdout=True) as bar:
        if (dist.get_rank() == 0):
            bar.update(curr_loops)
            sys.stdout.flush()
        loop = curr_loops
        while lr_scheduler.num_iters < lr_scheduler.end_iter:
            if (dist.get_rank() == 0):
                print(time_now(), 'Loading dataset...', flush=True)

            # @TODO: Reduce the amount of data read by each process

            gpt_dataset = SyllabeGPTDataset(text_corpus_lines, text_corpus_doc_ends,
                                     args.syllabe_number_of_load_batches * args.syllabe_batch_size // dist.get_world_size(),
                                     max_seq_len=args.syllabe_max_seq_len)

            training_data_loader = DataLoader(gpt_dataset, batch_size=args.syllabe_batch_size, collate_fn=syllabe_gpt_data_collate_wrapper,
                                     drop_last=False, shuffle=True, num_workers=1, persistent_workers=True)

            total_steps, best_valid_loss = train_loop(model, device, scaler, optimizer,
                                                      lr_scheduler, training_data_loader, validation_data_loader,
                                                      curr_save_file_path, args.syllabe_accumulation_steps,
                                                      loop, num_train_loops, total_steps, bar, best_valid_loss)
            loop += 1
            if (dist.get_rank() == 0):
                print(time_now(), loop, 'TRAINING LOOPS COMPLETE!', flush=True)
                bar.update(loop)
                sys.stdout.flush()

            del training_data_loader
            del gpt_dataset
            gc.collect()

def eval_fn(rank, args):
    print(time_now(), 'Called eval_fn()', flush=True)
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)

    dist.barrier()
    print('Using device: ', device, "from", dist.get_world_size(), 'processes', flush=True)

    home_path = args.home_path

    if (dist.get_rank() == 0):
        print('Model Arguments:', args)
        print(time_now(), 'Forming model ...', flush=True)

    model = SyllabeGPT(max_seq_len=args.syllabe_max_seq_len).to(device)
    model.float()
    model = DDP(model, device_ids=[rank])
    model.float()

    curr_save_file_path = home_path  + (f"models/syllabe_gpt_model_{date_now()}.pt")

    if (dist.get_rank() == 0):
        print('---------------------------------- SyllabeGPT Model Size ----------------------------------------')
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
    model.module.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()

    dev_text = read_corpus(home_path + args.dev_unparsed_corpus)
    dev_text_doc_ends = [i for i in range(len(dev_text)) if (len(dev_text[i]) == 0)]

    # Static seed
    np.random.seed(123)
    random.seed(123)
    dev_gpt_dataset = SyllabeGPTDataset(dev_text, dev_text_doc_ends,
                                    2000,
                                    max_seq_len=args.syllabe_max_seq_len)
    validation_data_loader = DataLoader(dev_gpt_dataset, batch_size=1,
                                      collate_fn=syllabe_gpt_data_collate_wrapper,
                                      drop_last=False, shuffle=False, num_workers=1, persistent_workers=False)
    # Random seed
    seed_val = datetime.now().microsecond + (3737 * dist.get_rank())
    np.random.seed(seed_val)
    random.seed(seed_val)

    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        valid_loss = eval_loop(model, device, validation_data_loader)
        print('Eval Loss: Cross-entropy: {:.6f} -- Perplexity: {:.4f}'.format(valid_loss, math.exp(valid_loss)))


def syllabe_gpt_trainer_main():
    args = py_trainer_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1

    # mp.spawn(train_fn, nprocs=args.world_size, args=(args,))
    mp.spawn(eval_fn, nprocs=args.world_size, args=(args,))
    # train_fn(args,cfg)


if __name__ == '__main__':
    syllabe_gpt_trainer_main()
