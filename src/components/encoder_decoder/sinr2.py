import os

import logging
from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .abstract_ed import EncoderDecoder
from modules import SINR
from metrics import get_metrics, SphereLoss

from configs.conf_schema import EDConfig


class SINRED(EncoderDecoder):

    name = 'SINR'

    def __init__(self, logger: logging.Logger,
                 cfg: EDConfig,
                 nsnapshots_tr: int,
                 nsnapshots_va: int,
                 loss_fn_inner_loop: SphereLoss,
                 **kwargs) -> None:
        super().__init__(logger, **kwargs)

        self.cfg = cfg

        self.inr = SINR(**cfg.arch_params)

        self.latent_codes_tr = nn.ParameterList([torch.zeros(cfg.latent_dim) for _ in range(nsnapshots_tr)])
        self.latent_codes_va = nn.ParameterList([torch.zeros(cfg.latent_dim) for _ in range(nsnapshots_va)])

        self.loss_fn = loss_fn_inner_loop

    def inner_loop(self, x: torch.Tensor,
                   coord_latlon: torch.Tensor,
                   z0: torch.Tensor,
                   lr: float = None,
                   max_patience: int = None,
                   max_iters: int = None,
                   **kwargs) -> torch.Tensor:

        if lr is None:
            lr = self.cfg.arch_params.inner_loop_lr
        if max_patience is None:
            max_patience = self.cfg.arch_params.inner_loop_max_patience
        if max_iters is None:
            max_iters = self.cfg.arch_params.inner_loop_max_iters

        best_z = z0.detach().clone()
        loss_best = self.loss_fn(x, self.decode(z0, coord_latlon=coord_latlon), start_dim=-3)

        z = z0.detach().clone()
        z.requires_grad_(True)  # (bs, 400)

        optim_inner_loop = optim.Adam([z], lr=lr)

        patience = 0

        for k in range(max_iters):

            # z.shape: (4, 400)
            # coords.shape: (4, 1, 128, 64, 3)
            z_dec = self.decode(z, coord_latlon=coord_latlon)  # (4, 128, 64, 2) -> (4, 128, 64, 2)
            loss: torch.Tensor = self.loss_fn(x, z_dec, start_dim=-3)

            if loss < loss_best:
                loss_best = loss.item()
                best_z = z.detach().clone()
                patience = 0
            else:
                patience += 1

            optim_inner_loop.zero_grad()
            loss.backward()
            optim_inner_loop.step()

            if patience > max_patience:
                self.logger.debug(f"inner loops: break at iter {k}")
                break

        return best_z

    def encode(self, x: torch.Tensor,
               group: str, idxs: torch.Tensor,
               z0: torch.Tensor = None,
               **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            x (torch.Tensor): shape: (..., *state_shape)
            z0 (torch.Tensor): shape: (..., latent_dim) initial guess for the latent code
            max_patience (int): max number of patience for early stopping

        Returns:
            torch.Tensor: z, shape: (..., latent_dim)
        '''
        if group == 'tr':
            return torch.stack([self.latent_codes_tr[idx]for idx in idxs], dim=0)
        if group == 'va':
            z0 = torch.stack([self.latent_codes_tr[idx]for idx in idxs], dim=0)
            z_enc = self.inner_loop(x, z0=z0, **kwargs)
            for k, idx in enumerate(idxs):
                self.latent_codes_va[idx] = z_enc[k]  # * update for validation set
        else:  # ! used for testing
            z0 = torch.zeros(x.shape[0], self.cfg.latent_dim).to(x.device)
            z_enc = self.inner_loop(x, z0=z0, **kwargs)
        return z_enc.clone()

    def decode(self, z: torch.Tensor, coord_latlon: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (..., latent_dim)
            coord_latlon (torch.Tensor): shape: (..., h, w, 2)

        Returns:
            torch.Tensor: x, shape: (..., h, w, state_channels=2)
        '''
        assert z.ndim + 2 == coord_latlon.ndim

        z_unsqueezed = z.view(*z.shape[:-1], 1, 1, self.cfg.arch_params.state_channels, self.cfg.arch_params.code_dim)
        # shape: (..., 1, 1, [state_dim=2], [code_dim=200])
        coords_unsqueezed = coord_latlon.unsqueeze(-2)
        # shape: (..., h, w, 1, [coords_dim=2])
        dec_results = self.inr(coords_unsqueezed, z_unsqueezed)
        return dec_results
