import os
import logging

from typing import *

import time
import hydra
import torch
from torch.utils.data import DataLoader
from torch import distributed as dist

from mlflow import log_params, log_metrics, log_artifacts, log_metric

from metrics import SphereLoss

# for typing
from components import EncoderDecoder, EncoderCache
from components import LatentDynamics

from configs.finetune.finetune_conf_schema import FineTuneConfig


from common import DataPrefetcher


class FineTuneer:

    def __init__(self, logger: logging.Logger,
                 encoder_decoder: EncoderDecoder,
                 latent_dynamics: LatentDynamics,
                 encoder_cache_tr: EncoderCache,
                 encoder_cache_va: EncoderCache,
                 dataloader_tr: DataLoader,
                 dataloader_va: DataLoader,
                 loss_fn_tr: SphereLoss,
                 loss_fn_va: SphereLoss,
                 cfg: FineTuneConfig,
                 rank: int) -> None:

        self.logger = logger
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.cfg = cfg

        self.ed = encoder_decoder
        self.ld = latent_dynamics
        self.encoder_cache_tr = encoder_cache_tr
        self.encoder_cache_va = encoder_cache_va

        if cfg.encoder_decoder.need_train:
            self.optim_ed = torch.optim.Adam(self.ed.parameters(),
                                             lr=cfg.encoder_decoder.training_params.lr_ed)

        if cfg.encoder_decoder.need_cache:
            self.encoder_cache_tr = self.encoder_cache_tr.to(self.device)
            self.encoder_cache_va = self.encoder_cache_va.to(self.device)
            self.optim_cd = torch.optim.Adam(self.encoder_cache_tr.parameters(),
                                             lr=cfg.encoder_decoder.training_params.lr_cd)

        if cfg.latent_dynamics.need_train:
            self.ld = self.ld.to(self.device)
            self.optim_ld = torch.optim.Adam(self.ld.parameters(),
                                             lr=cfg.latent_dynamics.training_params.lr_ld)

        self.dataloader_tr = dataloader_tr
        self.dataloader_va = dataloader_va

        self.loss_fn_tr = loss_fn_tr
        self.loss_fn_va = loss_fn_va

        self.mse = torch.nn.MSELoss()
        self.epoch = 0

        self.npreds = None
        self.exp = self.exp_decay = None

        if cfg.latent_dynamics.training_params.pred_ratio >= 1:
            self.npreds = cfg.latent_dynamics.training_params.pred_ratio
        else:
            self.exp = self.exp_decay = cfg.latent_dynamics.training_params.pred_ratio

    def set_train(self):
        self.ed.train()
        self.ed.requires_grad_(True)
        self.ld.train()
        self.ld.requires_grad_(True)
        self.encoder_cache_tr.train()
        self.encoder_cache_tr.requires_grad_(True)
        self.encoder_cache_va.train()
        self.encoder_cache_va.requires_grad_(True)

    def set_eval(self):
        self.ed.eval()
        self.ed.requires_grad_(False)
        self.ld.eval()
        self.ld.requires_grad_(False)
        self.encoder_cache_tr.eval()
        self.encoder_cache_tr.requires_grad_(False)
        self.encoder_cache_va.eval()
        self.encoder_cache_va.requires_grad_(False)

    def ratio_preds(self, zz: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        forward stepper

        Args:
            zz (torch.Tensor): shape=(bs, nsteps, *features)
            ratio (float): ratio of true states used for prediction
            ratio=0.: Only the first step is fed into the net and then roll-out
            ratio=1.: Each step is fed into the net for predicting the next step
            ratio\in(0,1): each is fed into the net with a probability ratio

        Returns:
            torch.Tensor: shape=(bs, nsteps, *features)
            The first step is the same as that of x, and the remaining steps are predictions from the previous steps
        '''
        bs, nsteps, *_ = zz.shape  # [0, nsteps-1]

        ratio = self.exp

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

    def latent_dynamics_loss(self, zz: torch.Tensor, **kwargs) -> torch.Tensor:

        bs, nsteps, *features = zz.shape

        if self.npreds is None:  # exponentional prediction
            zz_preds = self.ratio_preds(zz, ratio=self.exp)
            loss = self.mse(zz[:, 1:], zz_preds[:, 1:])
        else:
            assert nsteps > self.npreds
            zz_preds = self.ld(zz[:, :-self.npreds], nsteps=self.npreds)
            loss = self.mse(zz[:, self.npreds:], zz_preds)

        return loss

    def train_one_epoch(self, epoch: int):

        epoch_start = time.time()

        self.epoch = epoch

        accumulated_loss_rec = 0.
        accumulated_loss_dyn = 0.
        denomilator = 0

        self.set_train()

        prefetcher = DataPrefetcher(self.dataloader_tr, self.device)
        batch = prefetcher.next()
        while batch is not None:

            batch: Dict[str, torch.Tensor]

            windows = batch['window'].to(device=self.device, dtype=torch.float32)
            idxs = batch['idx'].to(device=self.device, dtype=torch.long)
            coord_cartes = batch['coord_cartes'].to(device=self.device, dtype=torch.float32)
            coord_latlon = batch['coord_latlon'].to(device=self.device, dtype=torch.float32)

            # windows.shape: (bs, win_length, h=128, w=64, nstates=2)
            # idxs.shape: (bs, win_length)] continuously
            # coord_cartes.shape: (bs, h=128, w=64, coord_dim=3)
            # coord_latlon.shape: (bs, h=128, w=64, coord_dim=2)

            self.logger.debug(f'{windows.shape=}')
            self.logger.debug(f'{idxs.shape=}')
            self.logger.debug(f'{coord_cartes.shape=}')
            self.logger.debug(f'{coord_latlon.shape=}')

            # recovery loss
            if self.cfg.encoder_decoder.need_cache:
                z_enc = self.encoder_cache_tr(idxs)  # directly read the cache
            else:
                z_enc = self.ed(windows, operation='encode')
            z_enc: torch.Tensor  # shape: (bs, win_length, latent_dim)
            x_rec = self.ed(z_enc, operation='decode',
                            coord_cartes=coord_cartes,
                            coord_latlon=coord_latlon)
            loss_rec = self.loss_fn_tr(x_rec, windows)

            # optimizer step
            # step for latent states if exist
            if self.cfg.encoder_decoder.need_cache:
                self.optim_cd.zero_grad(set_to_none=True)
                loss_rec.backward()
                self.optim_cd.step()
            else:
                loss_rec.backward()

            # dynamic loss
            loss_dyn = self.latent_dynamics_loss(zz=z_enc.detach().clone().requires_grad_(False),
                                                 coord_cartes=coord_cartes,
                                                 coord_latlon=coord_latlon)
            #! here we use the latent loss to train MLP, ReZero
            loss_dyn.backward()

            accumulated_loss_rec += loss_rec.detach() * windows.shape[0]
            accumulated_loss_dyn += loss_dyn.detach() * windows.shape[0]
            denomilator += windows.shape[0]

            batch = prefetcher.next()

        self.optim_ed.step()
        self.optim_ld.step()
        self.optim_ed.zero_grad()
        self.optim_ld.zero_grad()

        epoch_end = time.time()

        avg_loss_rec = (accumulated_loss_rec / denomilator).item()
        avg_loss_dyn = (accumulated_loss_dyn / denomilator).item()

        if self.rank == 0:
            self.logger.info(f'Epoch {epoch}, ' +
                             f'loss_rec={avg_loss_rec:.6e}; ' +
                             f'loss_dyn={avg_loss_dyn:.6e}; ' +
                             f'<fn={self.cfg.encoder_decoder.training_params.loss_fn_tr}>; ' +
                             f'Time elapsed {(epoch_end-epoch_start):.3f} (s)')

            log_metric('loss_rec', avg_loss_rec, step=epoch)
            log_metric('loss_dyn', avg_loss_dyn, step=epoch)
            log_metric('loss_rec_rooted', avg_loss_rec**.5, step=epoch)
            log_metric('loss_dyn_rooted', avg_loss_dyn**.5, step=epoch)

        return avg_loss_rec, avg_loss_dyn

    def _load_pretrain_ckpt(self, ckpt_path: str):

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.ed.load_state_dict(ckpt['ed'])

        if self.cfg.encoder_decoder.need_train:
            self.optim_ed.load_state_dict(ckpt['optim_ed'])

        if self.cfg.encoder_decoder.need_cache:
            self.optim_cd.load_state_dict(ckpt['optim_cd'])
            self.encoder_cache_tr.load_state_dict(ckpt['encoder_cache_tr'])
            self.encoder_cache_va.load_state_dict(ckpt['encoder_cache_va'])

        # disable syncing epoch
        # if dist.is_available() and dist.is_initialized():
        #     for param_name in ['epoch']:

        #         param = ckpt[param_name]

        #         if param is not None:
        #             param = torch.tensor(param, device=self.device)
        #             dist.broadcast(param, src=0)
        #             setattr(self, param_name, param.item())
        # else:
        #     self.epoch = ckpt['epoch']

        # set learning rate as configs
        if self.cfg.encoder_decoder.need_train:
            self.optim_ed.param_groups[0]['lr'] = self.cfg.encoder_decoder.training_params.lr_ed

        if self.cfg.encoder_decoder.need_cache:
            self.optim_cd.param_groups[0]['lr'] = self.cfg.encoder_decoder.training_params.lr_cd

    def _load_ckpt(self, ckpt_path: str):

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.ed.load_state_dict(ckpt['ed'])

        if self.cfg.encoder_decoder.need_train:
            self.optim_ed.load_state_dict(ckpt['optim_ed'])

        if self.cfg.latent_dynamics.need_train:
            self.ld.load_state_dict(ckpt['ld'])
            self.optim_ld.load_state_dict(ckpt['optim_ld'])

        if self.cfg.encoder_decoder.need_cache:
            self.optim_cd.load_state_dict(ckpt['optim_cd'])
            self.encoder_cache_tr.load_state_dict(ckpt['encoder_cache_tr'])
            self.encoder_cache_va.load_state_dict(ckpt['encoder_cache_va'])

        # Load and synchronize epoch, exp, exp_decay
        if dist.is_available() and dist.is_initialized():
            for param_name in ['epoch', 'npreds', 'exp', 'exp_decay']:

                param = ckpt[param_name]

                if param is not None:
                    param = torch.tensor(param, device=self.device)
                    dist.broadcast(param, src=0)
                    setattr(self, param_name, param.item())
        else:
            self.epoch = ckpt['epoch']
            self.npreds = ckpt['npreds']
            self.exp = ckpt['exp']
            self.exp_decay = ckpt['exp_decay']

        # set learning rate as configs
        if self.cfg.encoder_decoder.need_train:
            self.optim_ed.param_groups[0]['lr'] = self.cfg.encoder_decoder.training_params.lr_ed
        if self.cfg.latent_dynamics.need_train:
            self.optim_ld.param_groups[0]['lr'] = self.cfg.latent_dynamics.training_params.lr_ld
        if self.cfg.encoder_decoder.need_cache:
            self.optim_cd.param_groups[0]['lr'] = self.cfg.encoder_decoder.training_params.lr_cd

    def _save_ckpt(self, ckpt_path: str):

        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {
            'ed': self.ed.state_dict(),
            'epoch': self.epoch,
            'npreds': self.npreds,
            'exp': self.exp,
            'exp_decay': self.exp_decay,
        }

        if self.cfg.encoder_decoder.need_train:
            ckpt['optim_ed'] = self.optim_ed.state_dict()

        if self.cfg.latent_dynamics.need_train:
            ckpt['ld'] = self.ld.state_dict()
            ckpt['optim_ld'] = self.optim_ld.state_dict()

        if self.cfg.encoder_decoder.need_cache:
            ckpt['encoder_cache_tr'] = self.encoder_cache_tr.state_dict()
            ckpt['encoder_cache_va'] = self.encoder_cache_va.state_dict()
            ckpt['optim_cd'] = self.optim_cd.state_dict()

        torch.save(ckpt, ckpt_path)

    def load_ckpt(self) -> None:
        try:
            self._load_ckpt(self.cfg.ckpt_path)
            self.logger.info(f'Checkpoint loaded from {self.cfg.ckpt_path} successfully.')
        except Exception as e:
            # raise e
            self.logger.warning(f'loading ckpt from {self.cfg.ckpt_path} with exception: {e}. ' +
                                f'Try to load the pretrained model from {self.cfg.pretrain_ckpt_path}')
            self._load_pretrain_ckpt(self.cfg.pretrain_ckpt_path)

    def train(self):

        nepochs = self.cfg.nepochs
        eval_freq = self.cfg.eval_freq
        self.load_ckpt()

        start_epoch = self.epoch + 1

        eval_best = 1e10

        # Training loop
        for epoch in range(start_epoch, nepochs + 1):

            loss_rec, loss_dyn = self.train_one_epoch(epoch=epoch)

            # evaluate
            # if self.rank == 0:  # ! need this???

            if (epoch + 0) % eval_freq == 0:

                if self.exp is not None:
                    self.exp *= self.exp_decay
                    self.logger.info(f'changing exp to {self.exp}')
                    log_metric('exp', self.exp, step=epoch)

                self.logger.info(f'start evaluating...')
                loss_dyns_eval = self.evaluate(epoch=epoch)
                if loss_dyns_eval.mean() < eval_best:
                    eval_best = loss_dyns_eval.mean()
                    if self.rank == 0:
                        self._save_ckpt(self.cfg.ckpt_path)
                        self.logger.info(f'Epoch {epoch}, eval_best={eval_best:.6e}, saved to ckpt.')

    def evaluate(self, epoch: int):

        eval_start = time.time()

        accumulated_loss_dyns = 0.
        denomilator = 0

        self.set_eval()

        prefetcher = DataPrefetcher(self.dataloader_va, self.device)
        batch = prefetcher.next()
        while batch is not None:

            batch: Dict[str, torch.Tensor]

            windows = batch['window'].to(device=self.device, dtype=torch.float32)
            idxs = batch['idx'].to(device=self.device, dtype=torch.long)
            coord_cartes = batch['coord_cartes'].to(device=self.device, dtype=torch.float32)
            coord_latlon = batch['coord_latlon'].to(device=self.device, dtype=torch.float32)

            # windows.shape: (bs, win_length, h=128, w=64, nstates=2)
            # idxs.shape: (bs, win_length)] continuously
            # coord_cartes.shape: (bs, h=128, w=64, coord_dim=3)
            # coord_latlon.shape: (bs, h=128, w=64, coord_dim=2)

            self.logger.debug(f'{windows.shape=}')
            self.logger.debug(f'{idxs.shape=}')
            self.logger.debug(f'{coord_cartes.shape=}')
            self.logger.debug(f'{coord_latlon.shape=}')

            # encoding
            ts = time.time()
            #! the only difference from training is the updates for encoder_cache
            if self.cfg.encoder_decoder.need_cache:
                z0 = self.encoder_cache_va(idxs)
                z_enc = self.ed(windows, operation='encode',
                                coord_cartes=coord_cartes,
                                coord_latlon=coord_latlon,
                                z0=z0)
                self.encoder_cache_va(idxs, set_data=z_enc.detach().clone())  # update the cache
            else:
                z_enc = self.ed(windows, operation='encode')
            z_enc: torch.Tensor = z_enc.detach().clone().requires_grad_(False)
            # shape=(bs, win_length, latent_dim)
            te = time.time()
            self.logger.debug(f'encoding time: {te-ts:.3f} (s)')

            # dynamic loss
            zz = [z_enc.detach().clone()[:, 0]]  # only calculate for the first step
            for _ in range(z_enc.shape[1]):
                zz.append(self.ld(zz[-1]))
            zz = torch.stack(zz, dim=1)  # shape=(bs, win_length, latent_dim)
            xx_preds = self.ed(zz, operation='decode',
                               coord_cartes=coord_cartes,
                               coord_latlon=coord_latlon)

            loss_dyns = torch.stack([
                self.loss_fn_va(xx_pred, window).detach() for xx_pred, window in
                zip(torch.unbind(xx_preds, dim=1), torch.unbind(windows, dim=1))
            ], dim=0)  # (pred_length+1)

            accumulated_loss_dyns += loss_dyns.detach() * windows.shape[0]
            denomilator += windows.shape[0]

            batch = prefetcher.next()

        eval_end = time.time()

        avg_loss_dyns = accumulated_loss_dyns / denomilator

        loss_list = avg_loss_dyns.tolist()

        if self.rank == 0:
            self.logger.info(f'Evaluation, loss_dyns_eval={loss_list}; ' +
                             f'<fn={self.cfg.encoder_decoder.training_params.loss_fn_va}>; ' +
                             f'Time elapsed {(eval_end-eval_start):.3f} (s)')
            log_metrics({f'loss_dyn{k}_eval': loss for k, loss in enumerate(loss_list)}, step=epoch)
            log_metrics({f'loss_dyn{k}_eval_rooted': loss**.5 for k, loss in enumerate(loss_list)}, step=epoch)

        return avg_loss_dyns
