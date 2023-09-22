import os
import logging

from typing import Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import SphereLoss

# for typing
from configs import LAConfigs
from components import EncoderDecoder, LatentDynamics, Uncertainty
from dataset import LADataset

from common import DataPrefetcher

from plot import plot_comp


class EncoderCache(torch.nn.Module):

    def __init__(self, ncodes: int = None, shape: Tuple[int, Any] = None) -> None:
        super().__init__()
        if ncodes is None or shape is None:
            self.cache = None
        else:
            self.cache = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.zeros(shape))
                    for _ in range(ncodes)
                ]
            )

    def forward(self, seq_idxs: torch.Tensor, set_data: torch.Tensor = None):
        if self.cache is None:
            return None
        if set_data is not None:
            for i in range(len(seq_idxs)):
                self.cache[seq_idxs[i]] = set_data[i].detach().clone()
        return torch.stack([self.cache[idx] for idx in seq_idxs], dim=0)


class Evaluator:

    def __init__(self, logger: logging.Logger,
                 encoder_decoder: EncoderDecoder,
                 latent_dynamics: LatentDynamics,
                 uncertainty: Uncertainty,
                 encoder_cache: EncoderCache,
                 dataloader: DataLoader,
                 criterion: SphereLoss,
                 configs: LAConfigs,
                 rank: int,
                 ) -> None:

        self.logger = logger
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')

        self.configs = configs

        self.ed = encoder_decoder
        self.ld = latent_dynamics
        self.uq = uncertainty

        self.dataloader = dataloader

        self.encoder_cache = encoder_cache
        self.encoder_cache.requires_grad_(False)
        # self.optim_cd = torch.optim.Adam(self.encoder_cache, lr=configs.get("lr_cd", 1e-3))

        self.criterion = criterion

    def uncertainty_loss(self, xx: torch.Tensor, **kwargs) -> torch.Tensor:

        bs, nsteps, *features = xx.shape

        zz = self.ed(xx, operation='encode', **kwargs)

        zz_preds = self.ld(zz[:, :-1], nsteps=1, **kwargs)

        return self.uq(x=zz[:, :-1], x_pred=zz_preds, x_target=zz[:, 1:])

    def evaluate(self, pred_nsteps: int = 10, **kwargs):

        self.ed.eval()
        self.ed.requires_grad_(False)

        if self.ld is not None:
            self.ld.eval()
            self.ld.requires_grad_(False)
            self.uq.eval()
            self.uq.requires_grad_(False)

        loss_list = []

        if self.ld is None:
            pred_nsteps = 0

        for k, batch in enumerate(self.dataloader):

            batch: Dict[str, torch.Tensor]

            field_true = batch['data'].to(device=self.device, dtype=torch.float32)
            coords = batch['coords'].to(device=self.device, dtype=torch.float32)
            coords_ang = batch['coords_ang'].to(device=self.device, dtype=torch.float32)
            tt = batch['tt'][0].to(device=self.device, dtype=torch.float32)
            seq_idxs = batch['idxs'].to(device=self.device, dtype=torch.int)

            bs, nsteps, h, w, nstates = field_true.shape
            # (bs=4, nsteps=10, h=128, w=64, nstates=2)

            # coords.shape: (bs=4, h=128, w=64, coords_dim=3)

            # reconstruction

            # self.optim_cd.step()
            # self.optim_cd.zero_grad(set_to_none=True)
            z0 = self.encoder_cache(seq_idxs)

            field0 = field_true[:, 0]
            z0: torch.Tensor = self.ed(
                field_true[:, :z0.shape[1]],
                operation='encode',
                coords=coords[:, None],
                coords_ang=coords_ang[:, None],
                group='eval', z0=z0)
            self.encoder_cache(seq_idxs, set_data=z0)
            # (bs=4, ndim)

            zz = [z0.detach().clone()[:, 0]]

            if self.configs.ld_name == 'lstm':
                ht, ct = None, None
                for k in range(pred_nsteps):
                    if k == 0:
                        z_pred, (ht, ct) = self.ld(zz[-1])
                    else:
                        z_pred, (ht, ct) = self.ld(zz[-1], (ht, ct))
                    zz.append(z_pred.detach())
                # zz.shape: (bs=4, pred_steps=10, ndim)
            else:
                for _ in range(pred_nsteps):
                    zz.append(self.ld(zz[-1]))

            zz = torch.stack(zz, dim=1)
            # zz.shape: (bs=4, pred_steps=10, ndim)

            xx_pred = self.ed(zz, operation='decode',
                              coords=coords[:, None, ...],
                              coords_ang=coords_ang[:, None, ...],
                              group='eval')
            # (bs=4, pred_steps=10, h=128, w=64, nstates=2)

            loss_dyn = torch.stack([
                self.criterion(xx_pred[:, k], field_true[:, k], root=True).detach()
                for k in range(pred_nsteps + 1)
            ], dim=0)  # (pred_steps+1, - or nfeatures)

            loss_uq = self.uncertainty_loss(xx=field_true,
                                            coords=coords[:, None],
                                            coords_ang=coords_ang[:, None],
                                            z0=z0)
            # loss_list.append(loss_dyn)

            loss_list.append(loss_uq)

            if k == 0:
                plot_comp(
                    xx_pred[0, :pred_nsteps + 1, ..., 0],
                    field_true[0, :pred_nsteps + 1, ..., 0],
                    save_path=f'tmp/comp_evaluate_ed={self.configs.ed_name}_ld={self.configs.ld_name}_feature0.png')
                plot_comp(
                    xx_pred[0, :pred_nsteps + 1, ..., 1],
                    field_true[0, :pred_nsteps + 1, ..., 1],
                    save_path=f'tmp/comp_evaluate_ed={self.configs.ed_name}_ld={self.configs.ld_name}_feature1.png')

        loss_dyns = torch.stack(loss_list, dim=0)  # (nbs, pred_steps+1, - or nfeatures)
        avg_loss = torch.mean(loss_dyns, dim=0)  # (pred_steps+1, - or nfeatures)

        self.ed.requires_grad_(True)

        if self.ld is not None:
            self.ld.requires_grad_(True)
            self.uq.requires_grad_(True)

        return avg_loss


class Trainer:

    def __init__(self, logger: logging.Logger,
                 encoder_decoder: EncoderDecoder,
                 latent_dynamics: LatentDynamics,
                 uncertainty: Uncertainty,
                 encoder_cache: EncoderCache,
                 dataloader: DataLoader,
                 criterion: SphereLoss,
                 configs: LAConfigs,
                 rank: int,
                 ) -> None:

        self.logger = logger
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.configs = configs

        self.ed = encoder_decoder
        self.ld = latent_dynamics
        self.uq = uncertainty
        self.encoder_cache = encoder_cache

        if configs.ed_need_train:
            self.optim_ed = torch.optim.Adam(self.ed.parameters(), lr=configs.train["lr_ed"])

        if configs.ed_need_cache:
            self.encoder_cache = self.encoder_cache.to(self.device)
            self.optim_cd = torch.optim.Adam(self.encoder_cache.parameters(), lr=configs.train.get("lr_cd", 1e-3))

        if configs.ld_need_train:
            self.ld = self.ld.to(self.device)
            self.optim_ld = torch.optim.Adam(self.ld.parameters(), lr=configs.train["lr_ld"])
            self.uq = self.uq.to(self.device)
            self.optim_uq = torch.optim.Adam(self.uq.parameters(), lr=configs.train["lr_uq"])

        self.dataloader = dataloader

        self.exp_decay = configs.train.get('exp_decay', None)
        self.exp = self.exp_decay

        self.criterion = criterion
        self.mse = torch.nn.MSELoss()
        self.epoch = 0

    def ratio_preds(self, zz: torch.Tensor, ratio: float = 1., **kwargs) -> torch.Tensor:
        '''
        forward stepper

        Args:
            xx (torch.Tensor): shape=(bs, nsteps, *features)
            ratio (float): ratio of true states used for prediction
            ratio=0.: Only the first step is fed into the net and then roll-out
            ratio=1.: Each step is fed into the net for predicting the next step
            ratio\in(0,1): each is fed into the net with a probability ratio

        Returns:
            torch.Tensor: shape=(bs, nsteps, *features)
            The first step is the same as that of x, and the remaining steps are predictions from the previous steps
        '''
        bs, nsteps, *_ = zz.shape  # [0, nsteps-1]

        if ratio < 1e-3:
            obs_mask = torch.zeros(nsteps, dtype=torch.bool)
        elif ratio > 1 - 1e-3:
            obs_mask = torch.ones(nsteps, dtype=torch.bool)
        else:
            obs_mask = torch.rand(nsteps) < ratio
        obs_mask[0] = True
        obs_mask[-1] = True

        preds = [zz[:, 0]]

        start_step = 0
        end_step = 1

        while end_step < nsteps:

            while not obs_mask[end_step]:
                end_step += 1
            # increase end_step until it reaches the end or it is observed

            z = zz[:, start_step, ...]
            for step in range(start_step + 1, end_step + 1):
                # integrate from start_step to end_step
                z = self.ld(z)
                preds.append(z)

            start_step = end_step
            end_step += 1

        preds = torch.stack(preds, dim=1)

        return preds

    def ratio_preds2(self, zz: torch.Tensor, ratio: float = 1., **kwargs) -> torch.Tensor:
        '''
        forward stepper

        Args:
            xx (torch.Tensor): shape=(bs, nsteps, *features)
            ratio (float): ratio of true states used for prediction
            ratio=0.: Only the first step is fed into the net and then roll-out
            ratio=1.: Each step is fed into the net for predicting the next step
            ratio\in(0,1): each is fed into the net with a probability ratio

        Returns:
            torch.Tensor: shape=(bs, nsteps, *features)
            The first step is the same as that of x, and the remaining steps are predictions from the previous steps
        '''
        bs, nsteps, *_ = zz.shape  # [0, nsteps-1]

        if ratio < 1e-3:
            obs_mask = torch.zeros(nsteps, dtype=torch.bool)
        elif ratio > 1 - 1e-3:
            obs_mask = torch.ones(nsteps, dtype=torch.bool)
        else:
            obs_mask = torch.rand(nsteps) < ratio
        obs_mask[0] = True
        obs_mask[-1] = True

        cur_step = 1
        preds = [zz[:, 0]]
        prev_z = zz[:, 0]

        while cur_step < nsteps:
            preds.append(self.ld(prev_z))
            if obs_mask[cur_step]:
                prev_z = zz[:, cur_step]  # calibrate by the true z
            else:
                prev_z = preds[-1]

        preds = torch.stack(preds, dim=1)

        return preds

    def recovery_loss(self, xx: torch.Tensor, coords: torch.Tensor, coords_ang: torch.Tensor, root=False, **kwargs) -> torch.Tensor:

        zz = self.ed(xx, operation='encode', coords=coords, coords_ang=coords_ang, **kwargs)
        xx_rec = self.ed(zz, operation='decode', coords=coords, coords_ang=coords_ang, **kwargs)

        loss = self.criterion(xx_rec, xx, root=root)

        self.logger.debug(
            f'[rank={self.rank}]: {torch.cuda.memory_allocated() / 1e9} GB after calculating recovery loss')

        return loss

    def dynamics_loss(self, xx: torch.Tensor, pred_nsteps: int = 1,
                      exp: float = None, space='latent', **kwargs) -> torch.Tensor:

        bs, nsteps, *features = xx.shape

        zz = self.ed(xx, operation='encode', **kwargs)

        if self.configs.ld_name == 'lstm':
            xx = xx[:, 1:]
            zz_pred, _ = self.ld(zz[:, :-1])
            xx_pred = self.ed(zz_pred, operation='decode', **kwargs)

        elif exp is None:  # prediction with fixed pred_nsteps #! without recovery of the first step
            # zz_pred = zz[:, :nsteps - pred_nsteps]
            # for _ in range(pred_nsteps):
            #     zz_pred = self.ld(zz_pred, **kwargs)
            zz_pred = self.ld(zz[:, :nsteps - pred_nsteps], nsteps=pred_nsteps, **kwargs)

            if space == 'latent':
                zz = zz[:, pred_nsteps:]
                zz_pred = zz_pred
            else:
                xx = xx[:, pred_nsteps:]
                xx_pred = self.ed(zz_pred, operation='decode', **kwargs)

        else:  # prediction with exponential sampler
            zz_pred = self.ratio_preds(zz, ratio=exp, **kwargs)

            if space == 'latent':
                zz = zz[:, 1:]
                zz_pred = zz_pred[:, 1:]
            else:
                xx = xx[:, 1:]
                xx_pred = self.ed(zz_pred[:, 1:], operation='decode', **kwargs)

        if space == 'latent':
            loss = self.mse(zz_pred, zz)
        else:
            loss = self.criterion(xx_pred, xx)

        self.logger.debug(
            f'[rank={self.rank}]: {torch.cuda.memory_allocated() / 1e9} GB after calculating dynamics loss')

        return loss

    def multi_step_loss(self, xx: torch.Tensor, pred_nsteps: int = 1,
                        exp: float = None, root=False, **kwargs) -> torch.Tensor:

        bs, nsteps, *features = xx.shape

        zz = self.ed(xx, operation='encode', **kwargs)

        if exp is None:  # prediction with fixed pred_nsteps #! without recovery of the first step
            # z = zz[:, :nsteps - pred_nsteps]
            # for _ in range(pred_nsteps):
            #     z = self.ld(z, **kwargs)
            # xx_preds = self.ed(z, operation='decode', **kwargs)
            zz_preds = self.ld(zz[:, :nsteps - pred_nsteps], nsteps=nsteps, **kwargs)
            xx_preds = self.ed(zz_preds, operation='decode', **kwargs)
            return self.criterion(xx_preds, xx[:, pred_nsteps:], root=root)

        else:  # prediction with exponential sampler
            zz_preds = self.ratio_preds(zz, ratio=exp, **kwargs)
            xx_preds = self.ed(zz_preds, operation='decode', **kwargs)
            return self.criterion(xx_preds, xx, root=root)

    def uncertainty_loss(self, xx: torch.Tensor, **kwargs) -> torch.Tensor:

        bs, nsteps, *features = xx.shape

        zz = self.ed(xx, operation='encode', **kwargs)

        zz_preds = self.ld(zz[:, :-1], nsteps=1, **kwargs)

        return self.uq(x=zz[:, :-1], x_pred=zz_preds, x_target=zz[:, 1:])

    def train_one_epoch(self, epoch: int):

        # print(f'rank={self.rank}: exp={self.exp}')

        self.epoch = epoch

        running_rec_loss = 0.
        running_dyn_loss = 0.
        running_rec_rooted_loss = 0.
        running_uq_loss = 0.
        denomilator = 0

        self.ed.train()
        self.encoder_cache.train()
        if self.ld is not None:
            self.ld.train()
            self.uq.train()

        import time

        time_acc = 0.

        prefetcher = DataPrefetcher(self.dataloader, self.device)
        batch = prefetcher.next()
        while batch is not None:

            ts = time.time()
            batch: Dict[str, torch.Tensor]

            field_true = batch['data'].to(device=self.device, dtype=torch.float32)
            coords = batch['coords'].to(device=self.device, dtype=torch.float32)
            coords_ang = batch['coords_ang'].to(device=self.device, dtype=torch.float32)
            tt = batch['tt'][0].to(device=self.device, dtype=torch.float32)
            seq_idxs = batch['idxs'].to(device=self.device, dtype=torch.int)

            self.logger.debug(f'[rank={self.rank}]: {torch.cuda.memory_allocated() / 1e9} GB before training')
            # Forward pass through the models
            # Use the latent_states tensor as an initial state
            z0 = self.encoder_cache(seq_idxs)

            # recovery loss
            loss_rec = self.recovery_loss(xx=field_true,
                                          coords=coords[:, None],
                                          coords_ang=coords_ang[:, None],
                                          z0=z0)

            # step for latent states if exist
            if self.configs.ed_need_cache:
                self.optim_cd.zero_grad(set_to_none=True)
                loss_rec.backward()
                self.optim_cd.step()
            else:
                loss_rec.backward()
            # if self.encoder_cache is not None:
            #     self.optim_cd.zero_grad(set_to_none=True)
            #     loss_rec.backward()
            #     self.optim_cd.step()
            # else:
            #     loss_rec.backward()

            # dynamic loss if exists
            if self.ld is not None:
                if 'inr' in self.configs.ed_name:
                    loss_dyn = self.dynamics_loss(xx=field_true,
                                                  exp=self.exp,
                                                  space='latent',
                                                  coords=coords[:, None],
                                                  coords_ang=coords_ang[:, None],
                                                  z0=z0.detach())
                else:
                    loss_dyn = self.dynamics_loss(xx=field_true,
                                                  exp=self.exp,
                                                  space='physical',
                                                  coords=coords[:, None],
                                                  coords_ang=coords_ang[:, None],
                                                  z0=z0)
                loss_dyn.backward()
            else:
                loss_dyn = torch.tensor(0., device=self.device)

            # uncertainty loss
            # if self.ld is not None:
            #     loss_uq = self.uncertainty_loss(xx=field_true,
            #                                     coords=coords[:, None],
            #                                     coords_ang=coords_ang[:, None],
            #                                     z0=z0)
            #     loss_uq.backward()
            # else:
            #     loss_uq = torch.tensor(0., device=self.device)

            # accumulate loss
            running_rec_loss += loss_rec.detach() * field_true.shape[0]
            running_dyn_loss += loss_dyn.detach() * field_true.shape[0]
            # running_uq_loss += loss_uq.detach() * field_true.shape[0]

            with torch.no_grad():

                loss_rec_rooted = self.recovery_loss(xx=field_true, root=True,
                                                     coords=coords[:, None],
                                                     coords_ang=coords_ang[:, None],
                                                     z0=z0).detach()
                running_rec_rooted_loss += loss_rec_rooted * field_true.shape[0]

            # # Backward pass (gradient calculation)
            # loss.backward()

            # running_loss += loss.detach() * field_true.shape[0]
            denomilator += field_true.shape[0]

            self.logger.debug(f'[rank={self.rank}]: {torch.cuda.memory_allocated() / 1e9} GB after training')

            # Update the latent state
            # with torch.no_grad():
            #     latent_states[i] = encoded.detach()

            time_acc += time.time() - ts

            batch = prefetcher.next()
        # self.logger.debug(f'avg batch time: {time_acc / len(self.dataloader)}')

        # Update the weights and reset the gradients
        self.optim_ed.step()
        self.optim_ed.zero_grad()

        if self.ld is not None:
            self.optim_ld.step()
            self.optim_ld.zero_grad()
            # self.optim_uq.step()
            # self.optim_uq.zero_grad()

        return {
            'loss_rec': running_rec_loss / denomilator,
            'loss_dyn': running_dyn_loss / denomilator,
            'loss_rec_rooted': running_rec_rooted_loss / denomilator,
            'loss_uq': running_uq_loss / denomilator,
        }

    def _load_ckpt(self, ckpt_path: str, warm_start=False):

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.ed.load_state_dict(ckpt['ed'])

        if self.configs.ed_need_train:
            self.optim_ed.load_state_dict(ckpt['optim_ed'])

        if self.configs.ld_need_train:
            try:
                self.ld.load_state_dict(ckpt['ld'])
                self.optim_ld.load_state_dict(ckpt['optim_ld'])
                # self.uq.load_state_dict(ckpt['uq'])
                # self.optim_uq.load_state_dict(ckpt['optim_uq'])
            except Exception as e:
                # raise e
                assert warm_start
                self.logger.info('No ld in ckpt')

        if self.configs.ed_need_cache:
            self.optim_cd.load_state_dict(ckpt['optim_cd'])
            self.encoder_cache.load_state_dict(ckpt['encoder_cache'])

        # Load and synchronize epoch, exp, exp_decay
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for param_name in ['epoch', 'exp', 'exp_decay']:
                try:
                    param = ckpt[param_name]
                except:
                    assert warm_start
                    self.logger.info(f'No {param_name} in ckpt')

                if param is not None:
                    param = torch.tensor(param, device=self.device)
                    torch.distributed.broadcast(param, src=0)
                    setattr(self, param_name, param.item())
        else:
            self.epoch = ckpt['epoch']
            self.exp = ckpt['exp']
            self.exp_decay = ckpt['exp_decay']

        # set learning rate as configs
        if self.configs.ed_need_train:
            self.optim_ed.param_groups[0]['lr'] = self.configs.train['lr_ed']
        if self.configs.ld_need_train:
            self.optim_ld.param_groups[0]['lr'] = self.configs.train['lr_ld']
            self.optim_uq.param_groups[0]['lr'] = self.configs.train['lr_uq']
        if self.configs.ed_need_cache:
            self.optim_cd.param_groups[0]['lr'] = self.configs.train['lr_cd']

    def _save_ckpt(self, ckpt_path: str):

        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {
            'ed': self.ed.state_dict(),
            # 'optim_ed': self.optim_ed.state_dict(),
            # 'encoder_cache': self.encoder_cache,
            'epoch': self.epoch,
            'exp': self.exp,
            'exp_decay': self.exp_decay
        }

        if self.configs.ed_need_train:
            ckpt['optim_ed'] = self.optim_ed.state_dict()

        if self.configs.ld_need_train:
            ckpt['ld'] = self.ld.state_dict()
            ckpt['optim_ld'] = self.optim_ld.state_dict()
            ckpt['uq'] = self.uq.state_dict()
            ckpt['optim_uq'] = self.optim_uq.state_dict()

        if self.configs.ed_need_cache:
            ckpt['encoder_cache'] = self.encoder_cache.state_dict()
            ckpt['optim_cd'] = self.optim_cd.state_dict()

        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, evaluator=None) -> None:
        try:
            self._load_ckpt(self.configs.ckpt_path)
            self.logger.info(f'Checkpoint loaded from {self.configs.ckpt_path} successfully.')
        except Exception as e:
            # raise e
            self.logger.warning(
                f'loading ckpt from {self.configs.ckpt_path} with exception: {e}, training from scratch.')

            if 'pca' in self.configs.ed_name:
                self.ed.get_modes(self.dataloader)
                if evaluator is not None:
                    eval_loss = evaluator.evaluate()
                    self.logger.info(f'testing loss = {eval_loss.item():.4e}')
                else:
                    self.logger.warning(f'No evaluator, skip testing.')

                if not self.configs.ld_need_train:
                    return
            if self.configs.warm_start:
                self.logger.info(f'Trying warm-start from {self.configs.warm_start_ckpt_path}')
                try:
                    self._load_ckpt(self.configs.warm_start_ckpt_path, warm_start=True)
                except:
                    self.logger.info(
                        f'Warm-start from {self.configs.warm_start_ckpt_path} failed, training from scratch.')

    def train(self, evaluator: Evaluator):

        configs = self.configs

        nepochs = configs.train['nepochs']

        self.load_ckpt(evaluator=evaluator)

        total_time = 0.
        start_epoch = self.epoch + 1

        eval_best = 1e10

        self.logger.info(f'start with: {self.exp=}, {self.exp_decay=}')

        import time

        # Training loop
        for epoch in range(start_epoch, nepochs + 1):

            start_time = time.time()

            losses = self.train_one_epoch(epoch=epoch)
            loss_rec = losses['loss_rec']
            loss_dyn = losses['loss_dyn']
            loss_uq = losses['loss_uq']

            cur_loss = loss_dyn

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time

            # You might want to print your loss every epoch
            if self.rank == 0:
                self.logger.info(f'Epoch {epoch}, ' +
                                 ', '.join(
                                     [f'{k}={v:.6e}' for k, v in losses.items()]
                                 ) +
                                 f', time={elapsed_time:.2f} sec')

                if (epoch + 0) % 10 == 0:
                    # exponential decay
                    if self.exp_decay is not None:
                        if self.exp is None:
                            self.exp = 1.
                        self.exp *= self.exp_decay
                        self.logger.info(f'changing exp to {self.exp}')

                    # eval_loss = evaluator.evaluate()
                    # self.logger.info(f'Epoch {epoch}, eval_loss={eval_loss.cpu()}')
                    # if eval_loss.mean() < eval_best:
                    #     eval_best = eval_loss.mean()
                    #     self._save_ckpt(configs.ckpt_path)
                    #     self.logger.info(f'Epoch {epoch}, eval_best={eval_best.cpu()}, saved to ckpt.')

                    self.logger.info(f'Epoch {epoch}, train_loss={cur_loss.cpu()}')
                    if cur_loss < eval_best:
                        eval_best = cur_loss + 0.
                        self._save_ckpt(configs.ckpt_path)
                        self.logger.info(f'Epoch {epoch}, train_best={eval_best.cpu()}, saved to ckpt.')
