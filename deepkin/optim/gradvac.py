import copy
import random
import numpy as np
import torch
import torch.distributed as dist

class GradVac():
    def __init__(self, num_tasks, optimizer: torch.optim.Optimizer, DEVICE, scaler: torch.cuda.amp.GradScaler = None, beta = 1e-2, reduction='sum', cpu_offload: bool = True):
        self.device = torch.device('cpu') if cpu_offload else DEVICE
        self.num_tasks = num_tasks
        self.cpu_offload = cpu_offload
        self.beta = beta
        self._scaler, self._optim, self._reduction = scaler, optimizer, reduction
        # Setup default accumulated gradient
        self.accum_grad = []
        for i in range(self.num_tasks):
            grad, shape, has_grad = self._retrieve_grad()
            self.accum_grad.append((grad, shape, has_grad))
        self.rho_T = torch.zeros(self.num_tasks, self.num_tasks, device=self.device)#.to(self.device)
        return

    def state_dict(self) -> dict:
        if self._scaler is not None:
            return {'scaler': self._scaler.state_dict(), 'optimizer': self._optim.state_dict(), 'rho_T': self.rho_T}
        else:
            return {'optimizer': self._optim.state_dict(), 'rho_T': self.rho_T}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict['scaler'])
            self._optim.load_state_dict(state_dict['optimizer'])
            self.rho_T.copy_(state_dict['rho_T'])
        else:
            self._optim.load_state_dict(state_dict['optimizer'])
            self.rho_T.copy_(state_dict['rho_T'])

    @property
    def optimizer(self):
        return self._optim

    @property
    def scaler(self):
        return self._scaler

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        ret = self._optim.zero_grad()
        # Setup zero accumulated gradient
        for i in range(self.num_tasks):
            self.accum_grad[i][0].zero_()
            self.accum_grad[i][2].zero_()
        return ret

    def step(self):
        '''
        update the parameters with the gradient
        '''
        # @TODO: @Parallel implementation: ===> perform reduction here
        # Note: processes should scale loss by args.accumulation_steps,
        # but perform gradient update[step()] on each args.accumulation_steps//torch.distributed.get_world_size()
        # e.g.
        # for i in range(len(grads)): # i.e. num_tasks
        #   torch.distributed.reduce(grads[i], 0, async_op=False)
        #   torch.distributed.reduce(has_grads[i], 0, async_op=False)
        #
        # if torch.distributed.get_rank()==0:
        #   pc_grad = self._apply_grad_vaccine(grads, has_grads)
        # else:
        #   pc_grad = torch.zeros(...)
        #
        # @TODO: @Parallel implementation: ===> perform broadcast of gradients here (i.e. after gradient vaccine)
        # torch.distributed.broadcast(pc_grad, 0, async_op=False)
        #
        # Note(optim.): pc_grad tensor can be pre_allocated!
        # ...

        grads, shapes, has_grads = self._pack_accum_grads()
        # Reduce
        for i in range(len(grads)):
            dist.reduce(grads[i], 0, async_op=False)
            dist.reduce(has_grads[i], 0, async_op=False)
        if dist.get_rank()==0:
            pc_grad = self._apply_grad_vaccine(grads, has_grads)
        else:
            pc_grad = torch.zeros_like(grads[0], device=grads[0].device) # .to(grads[0].device)
        # Broadcast
        dist.broadcast(pc_grad, 0, async_op=False)
        # Use locally
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

        if self._scaler is not None:
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            self._optim.step()

        return self.zero_grad()

    def backward(self, mt_losses):
        # Gradient accumulation
        for loss_id, loss in enumerate(mt_losses):
            self._optim.zero_grad()
            retain_graph = (loss_id < (self.num_tasks - 1))
            if self._scaler is not None:
                self._scaler.scale(loss).backward(retain_graph = retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)
            grad, shape, has_grad = self._retrieve_grad()
            acc_grad, acc_shape, acc_has_grad = self.accum_grad[loss_id]
            acc_grad += grad
            acc_has_grad = torch.logical_or(acc_has_grad, grad).to(dtype=acc_has_grad.dtype)
            self.accum_grad[loss_id] = (acc_grad, acc_shape, acc_has_grad)
        self._optim.zero_grad()

    def _apply_grad_vaccine(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grads, num_task = copy.deepcopy(grads), len(grads)
        for tn_i in range(num_task):
            task_index = list(range(num_task))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (pc_grads[tn_i].norm() * grads[tn_j].norm())
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = pc_grads[tn_i].norm() * (self.rho_T[tn_i, tn_j] * (1 - rho_ij ** 2).sqrt() - rho_ij * (
                                1 - self.rho_T[tn_i, tn_j] ** 2).sqrt()) / (
                                    grads[tn_j].norm() * (1 - self.rho_T[tn_i, tn_j] ** 2).sqrt())
                    pc_grads[tn_i] += grads[tn_j] * w
                    self.rho_T[tn_i, tn_j] = (1 - self.beta) * self.rho_T[tn_i, tn_j] + self.beta * rho_ij
        merged_grad = torch.zeros_like(grads[0], device=grads[0].device) #.to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grads]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grads]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grads]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx].to(p.device)
                idx += 1
        return

    def _pack_accum_grads(self):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for (grad, shape, has_grad) in self.accum_grad:
            grads.append(grad)
            has_grads.append(has_grad)
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    if self.cpu_offload:
                        grad.append(torch.zeros_like(p, device=torch.device('cpu')))
                        has_grad.append(torch.zeros_like(p, dtype=torch.int8, device=torch.device('cpu')))
                    else:
                        grad.append(torch.zeros_like(p, device=p.device))
                        has_grad.append(torch.zeros_like(p, dtype=torch.int8, device=p.device))
                else:
                    shape.append(p.grad.shape)
                    if self.cpu_offload:
                        grad.append(p.grad.detach().cpu())
                        has_grad.append(torch.ones_like(p, dtype=torch.int8, device=torch.device('cpu')))
                    else:
                        grad.append(p.grad.clone())
                        has_grad.append(torch.ones_like(p, dtype=torch.int8, device=p.device))
        grad_flatten = self._flatten_grad(grad, shape)
        has_grad_flatten = self._flatten_grad(has_grad, shape)
        return grad_flatten, shape, has_grad_flatten
