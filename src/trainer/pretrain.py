import os
import logging

from typing import *

import time
import hydra
from configs.pretrain.pretrain_conf_schema import PreTrainConfig

import torch
from torch.utils.data import DataLoader
from torch import distributed as dist

from mlflow import log_params, log_metrics, log_artifacts, log_metric

from metrics import SphereLoss

# for typing
from components import EncoderDecoder, EncoderCache

from common import DataPrefetcher


class PreTrainer:

    def __init__(self, logger: logging.Logger,
                 encoder_decoder: EncoderDecoder,
                 encoder_cache_tr: EncoderCache,
                 encoder_cache_va: EncoderCache,
                 dataloader_tr: DataLoader,
                 dataloader_va: DataLoader,
                 loss_fn_tr: SphereLoss,
                 loss_fn_va: SphereLoss,
                 cfg: PreTrainConfig,
                 rank: int) -> None:

        self.logger = logger
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.cfg = cfg

        self.ed = encoder_decoder
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

        self.dataloader_tr = dataloader_tr
        self.dataloader_va = dataloader_va

        self.loss_fn_tr = loss_fn_tr
        self.loss_fn_va = loss_fn_va

        self.mse = torch.nn.MSELoss()
        self.epoch = 0

    def set_train(self):
        self.ed.train()
        self.ed.requires_grad_(True)
        self.encoder_cache_tr.train()
        self.encoder_cache_tr.requires_grad_(True)
        self.encoder_cache_va.train()
        self.encoder_cache_va.requires_grad_(True)

    def set_eval(self):
        self.ed.eval()
        self.ed.requires_grad_(False)
        self.encoder_cache_tr.eval()
        self.encoder_cache_tr.requires_grad_(False)
        self.encoder_cache_va.eval()
        self.encoder_cache_va.requires_grad_(False)

    def train_one_epoch(self, epoch: int):

        epoch_start = time.time()

        self.epoch = epoch

        accumulated_loss_rec = 0.
        denomilator = 0

        self.set_train()

        prefetcher = DataPrefetcher(self.dataloader_tr, self.device)
        batch = prefetcher.next()
        while batch is not None:

            batch: Dict[str, torch.Tensor]

            snapshots = batch['snapshot'].to(device=self.device, dtype=torch.float32)
            idxs = batch['idx'].to(device=self.device, dtype=torch.long)
            coord_cartes = batch['coord_cartes'].to(device=self.device, dtype=torch.float32)
            coord_latlon = batch['coord_latlon'].to(device=self.device, dtype=torch.float32)

            # snapshots.shape: (bs, h=128, w=64, nstates=2)
            # idxs.shape: (bs,)
            # coord_cartes.shape: (bs, h=128, w=64, coord_dim=3)
            # coord_latlon.shape: (bs, h=128, w=64, coord_dim=2)

            self.logger.debug(f'{snapshots.shape=}')
            self.logger.debug(f'{idxs.shape=}')
            self.logger.debug(f'{coord_cartes.shape=}')
            self.logger.debug(f'{coord_latlon.shape=}')

            # recovery loss
            if self.cfg.encoder_decoder.need_cache:
                z_enc = self.encoder_cache_tr(idxs)  # directly read the cache
            else:
                z_enc = self.ed(snapshots, operation='encode')
            z_enc: torch.Tensor  # shape=(bs, latent_dim)

            x_rec = self.ed(z_enc, operation='decode',
                            coord_cartes=coord_cartes,
                            coord_latlon=coord_latlon)
            loss_rec = self.loss_fn_tr(x_rec, snapshots)

            # optimizer step
            # step for latent states if exist
            if self.cfg.encoder_decoder.need_cache:
                self.optim_cd.zero_grad(set_to_none=True)
                loss_rec.backward()
                self.optim_cd.step()
            else:
                loss_rec.backward()

            accumulated_loss_rec += loss_rec.detach() * snapshots.shape[0]
            denomilator += snapshots.shape[0]

            batch = prefetcher.next()

        self.optim_ed.step()
        self.optim_ed.zero_grad()

        epoch_end = time.time()

        avg_loss_rec = (accumulated_loss_rec / denomilator).item()

        if self.rank == 0:
            self.logger.info(f'Epoch {epoch}, loss={avg_loss_rec:.6e}; ' +
                             f'<fn={self.cfg.encoder_decoder.training_params.loss_fn_tr}>; ' +
                             f'Time elapsed {(epoch_end-epoch_start):.3f} (s)')

            log_metric('loss', avg_loss_rec, step=epoch)
            log_metric('loss_rooted', avg_loss_rec**0.5, step=epoch)

        return avg_loss_rec

    def _load_ckpt(self, ckpt_path: str):

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.ed.load_state_dict(ckpt['ed'])

        if self.cfg.encoder_decoder.need_train:
            self.optim_ed.load_state_dict(ckpt['optim_ed'])

        if self.cfg.encoder_decoder.need_cache:
            self.optim_cd.load_state_dict(ckpt['optim_cd'])
            self.encoder_cache_tr.load_state_dict(ckpt['encoder_cache_tr'])
            self.encoder_cache_va.load_state_dict(ckpt['encoder_cache_va'])

        # Load and synchronize epoch, exp, exp_decay
        if dist.is_available() and dist.is_initialized():
            for param_name in ['epoch']:

                param = ckpt[param_name]

                if param is not None:
                    param = torch.tensor(param, device=self.device)
                    dist.broadcast(param, src=0)
                    setattr(self, param_name, param.item())
        else:
            self.epoch = ckpt['epoch']

        # set learning rate as configs
        if self.cfg.encoder_decoder.need_train:
            self.optim_ed.param_groups[0]['lr'] = self.cfg.encoder_decoder.training_params.lr_ed

        if self.cfg.encoder_decoder.need_cache:
            self.optim_cd.param_groups[0]['lr'] = self.cfg.encoder_decoder.training_params.lr_cd

    def _save_ckpt(self, ckpt_path: str):

        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {
            'ed': self.ed.state_dict(),
            'epoch': self.epoch,
        }

        if self.cfg.encoder_decoder.need_train:
            ckpt['optim_ed'] = self.optim_ed.state_dict()

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
            self.logger.warning(
                f'loading ckpt from {self.cfg.ckpt_path} with exception: {e}, training from scratch.')

    def train(self):

        nepochs = self.cfg.nepochs
        eval_freq = self.cfg.eval_freq
        self.load_ckpt()

        start_epoch = self.epoch + 1

        eval_best = 1e10

        # Training loop
        for epoch in range(start_epoch, nepochs + 1):

            loss = self.train_one_epoch(epoch=epoch)

            # evaluate
            # if self.rank == 0:  # ! need this???

            if (epoch + 0) % eval_freq == 0:

                self.logger.info(f'start evaluating...')
                loss_eval = self.evaluate(epoch=epoch)
                if loss_eval < eval_best:
                    eval_best = loss_eval
                    if self.rank == 0:
                        self._save_ckpt(self.cfg.ckpt_path)
                        self.logger.info(f'Epoch {epoch}, eval_best={eval_best:.6e}, saved to ckpt.')

    def evaluate(self, epoch: int):

        eval_start = time.time()

        accumulated_loss_rec = 0.
        denomilator = 0

        self.set_eval()

        prefetcher = DataPrefetcher(self.dataloader_va, self.device)
        batch = prefetcher.next()
        while batch is not None:

            batch: Dict[str, torch.Tensor]

            snapshots = batch['snapshot'].to(device=self.device, dtype=torch.float32)
            idxs = batch['idx'].to(device=self.device, dtype=torch.long)
            coord_cartes = batch['coord_cartes'].to(device=self.device, dtype=torch.float32)
            coord_latlon = batch['coord_latlon'].to(device=self.device, dtype=torch.float32)

            # snapshots.shape: (bs, h=128, w=64, nstates=2)
            # idxs.shape: (bs,)
            # coord_cartes.shape: (bs, h=128, w=64, coord_dim=3)
            # coord_latlon.shape: (bs, h=128, w=64, coord_dim=2)

            self.logger.debug(f'{snapshots.shape=}')
            self.logger.debug(f'{idxs.shape=}')
            self.logger.debug(f'{coord_cartes.shape=}')
            self.logger.debug(f'{coord_latlon.shape=}')

            # encoding
            ts = time.time()
            #! the only difference from training is the updates for encoder_cache
            if self.cfg.encoder_decoder.need_cache:
                z0 = self.encoder_cache_va(idxs)
                z_enc = self.ed(snapshots, operation='encode',
                                coord_cartes=coord_cartes,
                                coord_latlon=coord_latlon,
                                z0=z0)
                self.encoder_cache_va(idxs, set_data=z_enc.detach().clone())  # update the cache
            else:
                z_enc = self.ed(snapshots, operation='encode')
            te = time.time()
            self.logger.debug(f'encoding time: {te-ts:.3f} (s)')

            # decoding
            x_rec = self.ed(z_enc, operation='decode',
                            coord_cartes=coord_cartes,
                            coord_latlon=coord_latlon)
            # recovery loss
            loss_rec = self.loss_fn_va(x_rec, snapshots)

            accumulated_loss_rec += loss_rec.detach() * snapshots.shape[0]
            denomilator += snapshots.shape[0]

            batch = prefetcher.next()

        eval_end = time.time()

        avg_loss_rec = (accumulated_loss_rec / denomilator).item()

        if self.rank == 0:
            self.logger.info(f'Evaluation, loss_eval={avg_loss_rec:.6e}; ' +
                             f'<fn={self.cfg.encoder_decoder.training_params.loss_fn_va}>; ' +
                             f'Time elapsed {(eval_end-eval_start):.3f} (s)')
            log_metric('loss_eval', avg_loss_rec, step=epoch)
            log_metric('loss_eval_rooted', avg_loss_rec**0.5, step=epoch)

        return avg_loss_rec
