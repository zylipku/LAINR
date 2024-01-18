import os
import logging

from typing import *

import time
import hydra
from configs.postproc.postproc_conf_schema import PostProcConfig

import torch
from torch.utils.data import DataLoader
from torch import distributed as dist

from mlflow import log_params, log_metrics, log_artifacts, log_metric

from metrics import SphereLoss

# for typing
from components import EncoderDecoder, EncoderCache, LatentDynamics, UncertaintyEst

from common import DataPrefetcher


class PostProcer:

    def __init__(self, logger: logging.Logger,
                 encoder_decoder: EncoderDecoder,
                 latent_dynamics: LatentDynamics,
                 uncertainty_est: UncertaintyEst,
                 encoder_cache_tr: EncoderCache,
                 encoder_cache_va: EncoderCache,
                 dataloader_tr: DataLoader,
                 dataloader_va: DataLoader,
                 cfg: PostProcConfig,
                 rank: int) -> None:

        self.logger = logger
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.cfg = cfg

        self.ed = encoder_decoder
        self.encoder_cache_tr = encoder_cache_tr
        self.encoder_cache_va = encoder_cache_va

        self.ld = latent_dynamics
        self.ue = uncertainty_est

        self.optim_ue = torch.optim.Adam(self.ue.parameters(),
                                         lr=cfg.uncertainty_est.training_params.lr_ue)

        self.dataloader_tr = dataloader_tr
        self.dataloader_va = dataloader_va

        self.mse = torch.nn.MSELoss()
        self.epoch = 0

        # set ed and ld to eval mode
        self.ed.eval()
        self.ed.requires_grad_(False)
        self.ld.eval()
        self.ld.requires_grad_(False)
        self.encoder_cache_tr.eval()
        self.encoder_cache_tr.requires_grad_(False)
        self.encoder_cache_va.eval()
        self.encoder_cache_va.requires_grad_(False)

    def set_train(self):
        self.ue.train()
        self.ue.requires_grad_(True)

    def set_eval(self):
        self.ue.eval()
        self.ue.requires_grad_(False)

    def uncertainty_loss(self, zz: torch.Tensor, **kwargs) -> torch.Tensor:

        bs, nsteps, *ndim = zz.shape

        zz_preds = self.ld(zz[:, :-1], nsteps=1, **kwargs)

        return self.ue(x=zz[:, :-1], x_pred=zz_preds, x_target=zz[:, 1:])

    def train_one_epoch(self, epoch: int):

        epoch_start = time.time()

        self.epoch = epoch

        accumulated_loss_ue = 0.
        accumulated_loss_ue_eval = 0.
        denomilator = 0
        denomilator_eval = 0

        self.set_train()

        nbatches_va = max(1, len(self.dataloader_tr) // 10)
        nbatches_tr = len(self.dataloader_tr) - nbatches_va

        for k, batch in enumerate(self.dataloader_tr):

            batch: Dict[str, torch.Tensor]

            snapshots = batch['window'].to(device=self.device, dtype=torch.float32)
            idxs = batch['idx'].to(device=self.device, dtype=torch.long)
            coord_cartes = batch['coord_cartes'].to(device=self.device, dtype=torch.float32)
            coord_latlon = batch['coord_latlon'].to(device=self.device, dtype=torch.float32)

            # snapshots.shape: (bs, nsteps=10, h=128, w=64, nstates=2)
            # idxs.shape: (bs,)
            # coord_cartes.shape: (bs, h=128, w=64, coord_dim=3)
            # coord_latlon.shape: (bs, h=128, w=64, coord_dim=2)

            coord_cartes.unsqueeze_(1)  # (bs, 1, h=128, w=64, coord_dim=3)
            coord_latlon.unsqueeze_(1)  # (bs, 1, h=128, w=64, coord_dim=2)

            self.logger.debug(f'{snapshots.shape=}')
            self.logger.debug(f'{idxs.shape=}')
            self.logger.debug(f'{coord_cartes.shape=}')
            self.logger.debug(f'{coord_latlon.shape=}')

            # recovery loss
            if self.cfg.encoder_decoder.need_cache:
                z_enc = self.encoder_cache_tr(idxs)  # directly read the cache
            else:
                z_enc = self.ed(snapshots, operation='encode')
            z_enc: torch.Tensor = z_enc.detach().clone()  # shape=(bs, latent_dim)

            # self.logger.debug(f'{k=}, {z_enc=}')
            if k < nbatches_tr:
                loss_ue = self.uncertainty_loss(z_enc).mean()  # (bs, nsteps-1)
                self.optim_ue.zero_grad()
                loss_ue.backward()
                self.optim_ue.step()
                accumulated_loss_ue += loss_ue.detach() * snapshots.shape[0]
                denomilator += snapshots.shape[0]
            else:
                with torch.no_grad():
                    loss_ue = self.uncertainty_loss(z_enc).mean()
                    accumulated_loss_ue_eval += loss_ue.detach() * snapshots.shape[0]
                    denomilator_eval += snapshots.shape[0]

        epoch_end = time.time()

        avg_loss_ue = (accumulated_loss_ue / denomilator).item()
        avg_loss_ue_eval = (accumulated_loss_ue_eval / denomilator_eval).item()

        if self.rank == 0:
            self.logger.info(f'Epoch {epoch}, loss={avg_loss_ue:.6e} ' +
                             f'loss_eval={avg_loss_ue_eval:.6e} ' +
                             f'on [{nbatches_va}/{len(self.dataloader_tr)}] batch(es). ' +
                             #  f'<fn=uncertainty loss>; ' +
                             f'Time elapsed {(epoch_end-epoch_start):.3f} (s)')
            self.logger.info('Estimator info: ' + self.ue(None, output='info'))

            log_metric('loss_ue', avg_loss_ue, step=epoch)
            log_metric('loss_ue_eval', avg_loss_ue_eval, step=epoch)

        return avg_loss_ue

    def _load_finetune_ckpt(self, ckpt_path: str):

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.ed.load_state_dict(ckpt['ed'])
        self.ld.load_state_dict(ckpt['ld'])

        if self.cfg.encoder_decoder.need_cache:
            # self.optim_cd.load_state_dict(ckpt['optim_cd'])
            self.encoder_cache_tr.load_state_dict(ckpt['encoder_cache_tr'])
            self.encoder_cache_va.load_state_dict(ckpt['encoder_cache_va'])

    def _load_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.ed.load_state_dict(ckpt['ed'])
        self.ld.load_state_dict(ckpt['ld'])
        self.ue.load_state_dict(ckpt['ue'])

        if self.cfg.encoder_decoder.need_cache:
            # self.optim_cd.load_state_dict(ckpt['optim_cd'])
            self.encoder_cache_tr.load_state_dict(ckpt['encoder_cache_tr'])
            self.encoder_cache_va.load_state_dict(ckpt['encoder_cache_va'])

        self.optim_ue.load_state_dict(ckpt['optim_ue'])

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

    def _save_ckpt(self, ckpt_path: str):

        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {
            'ed': self.ed.state_dict(),
            'ld': self.ld.state_dict(),
            'epoch': self.epoch,
        }

        if self.cfg.encoder_decoder.need_cache:
            ckpt['encoder_cache_tr'] = self.encoder_cache_tr.state_dict()
            ckpt['encoder_cache_va'] = self.encoder_cache_va.state_dict()

        ckpt['ue'] = self.ue.state_dict()
        ckpt['optim_ue'] = self.optim_ue.state_dict()

        torch.save(ckpt, ckpt_path)

    def load_ckpt(self) -> None:
        try:
            self._load_ckpt(self.cfg.ckpt_path)
            self.logger.info(f'Checkpoint loaded from {self.cfg.ckpt_path} successfully.')
        except Exception as e:
            # raise e
            self.logger.warning(
                f'loading ckpt from {self.cfg.ckpt_path} with exception: {e}, training from scratch.')
            self._load_finetune_ckpt(self.cfg.finetune_ckpt_path)

    def train(self):

        nepochs = self.cfg.nepochs
        eval_freq = self.cfg.eval_freq
        self.load_ckpt()

        start_epoch = self.epoch + 1

        eval_best = 1e10
        prev_loss = 1e10

        # Training loop
        for epoch in range(start_epoch, nepochs + 1):

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #              profile_memory=True, record_shapes=True, with_stack=True,
            #              on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler-log/debug')) as prof:
            loss = self.train_one_epoch(epoch=epoch)
            # if loss > prev_loss - abs(prev_loss) * 1e-6:
            #     self.logger.info(f'loss increase, stop training')
            #     break
            # prev_loss = loss

            # evaluate
            # if self.rank == 0:  # ! need this???

            # if (epoch + 0) % eval_freq == -1:

            #     self.logger.info(f'start evaluating...')
            #     loss_eval = self.evaluate(epoch=epoch)
            #     if loss_eval < eval_best:
            #         eval_best = loss_eval
            #         if self.rank == 0:
            #             self._save_ckpt(self.cfg.ckpt_path)
            #             self.logger.info(f'Epoch {epoch}, eval_best={eval_best:.6e}, saved to ckpt.')

        if self.rank == 0:
            self._save_ckpt(self.cfg.ckpt_path)
            self.logger.info(f'Succesfully saved to ckpt.')

    def evaluate(self, epoch: int):

        eval_start = time.time()

        accumulated_loss_ue = 0.
        denomilator = 0

        self.set_eval()

        prefetcher = DataPrefetcher(self.dataloader_va, self.device)
        batch = prefetcher.next()
        while batch is not None:

            batch: Dict[str, torch.Tensor]

            snapshots = batch['window'].to(device=self.device, dtype=torch.float32)
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

            snapshots = snapshots[:, :2]  # ! only encode the first 2 snapshots

            # encoding
            ts = time.time()
            #! no update for encoder_cache_va
            if self.cfg.encoder_decoder.need_cache:
                z_enc = self.encoder_cache_va(idxs)  # directly read the cache for the first snapshots
            else:
                z_enc = self.ed(snapshots, operation='encode')
            te = time.time()
            self.logger.debug(f'encoding time: {te-ts:.3f} (s)')

            # recovery loss
            loss_ue = self.uncertainty_loss(z_enc)

            accumulated_loss_ue += loss_ue.detach() * snapshots.shape[0]
            denomilator += snapshots.shape[0]

            batch = prefetcher.next()

        eval_end = time.time()

        avg_loss_ue = (accumulated_loss_ue / denomilator).item()

        if self.rank == 0:
            self.logger.info(f'Evaluation, loss_eval={avg_loss_ue:.6e}; ' +
                             f'<fn=uncertainty loss>; ' +
                             f'Time elapsed {(eval_end-eval_start):.3f} (s)')
            log_metric('loss_ue_eval', avg_loss_ue, step=epoch)
            # log_metric('loss_eval_rooted', avg_loss_rec**0.5, step=epoch)

        return avg_loss_ue
