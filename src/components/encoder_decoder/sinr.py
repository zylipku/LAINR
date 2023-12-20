import os

import logging
from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .abstract_ed import EncoderDecoder
from modules import SINR

from metrics import SphereLoss
from datasets import LADataset


class SINRED(EncoderDecoder):

    name = 'SINR'

    sinr_kwargs = {
        'depth': 5,
        'max_freq': 4,
        'hidden_dim': 128,
        'state_channels': 2,  # height & vorticity
        'code_dim': 200,
        'out_dim': 2,
    }

    optim_cod_kwargs = {
        'lr': 1e-3,
    }

    train_codes: torch.Tensor
    eval_codes: torch.Tensor

    def __init__(self, logger: logging.Logger,
                 criterion: SphereLoss,
                 **kwargs) -> None:
        super().__init__(logger, **kwargs)

        self.state_channels = 2

        self.code_dim = kwargs.get('code_dim', self.sinr_kwargs['code_dim'])
        self.latent_dim = self.code_dim * self.state_channels

        self.sinr_kwargs.update(kwargs)

        self.inr = SINR(**self.sinr_kwargs)

        self._ckpt_train_codes = None
        self._ckpt_eval_codes = None
        self._ckpt_optim_cod = None
        self._ckpt_sinr_state = None

        self.criterion = criterion

    def encode(self, x: torch.Tensor,
               coord_latlon: torch.Tensor,
               z0: torch.Tensor = None,
               max_patience: int = 10,
               optim_eval_max_inner_loops=100,
               **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            x (torch.Tensor): shape: (..., *state_shape)
            z0 (torch.Tensor): shape: (..., latent_dim) initial guess for the latent code
            max_patience (int): max number of patience for early stopping

        Returns:
            torch.Tensor: z, shape: (..., latent_dim)
        '''
        best_z = z0.detach().clone()
        loss_enc_best = self.criterion(x, self.decode(z0, coord_latlon=coord_latlon), start_dim=-3)
        z = z0.detach().clone()
        z.requires_grad_(True)  # (bs, 400)

        optim_eval = optim.Adam([z], **self.optim_cod_kwargs)

        patience = 0

        for k in range(optim_eval_max_inner_loops):

            # z.shape: (4, 400)
            # coords.shape: (4, 1, 128, 64, 3)
            z_dec = self.decode(z, coord_latlon=coord_latlon)  # (4, 128, 64, 2) -> (4, 128, 64, 2)
            loss_dec: torch.Tensor = self.criterion(x, z_dec, start_dim=-3)

            if loss_dec < loss_enc_best:
                loss_enc_best = loss_dec.item()
                best_z = z.detach().clone()
                patience = 0
            else:
                patience += 1

            optim_eval.zero_grad()
            loss_dec.backward()
            optim_eval.step()

            if patience > max_patience:
                self.logger.debug(f"inner loops: break at iter {k}")
                break

        return best_z

    def decode(self, z: torch.Tensor, coord_latlon: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (..., latent_dim)
            coord_latlon (torch.Tensor): shape: (..., h, w, 2)

        Returns:
            torch.Tensor: x, shape: (..., h, w, state_channels=2)
        '''
        assert z.ndim + 2 == coord_latlon.ndim

        z_unsqueezed = z.view(*z.shape[:-1], 1, 1, self.state_channels, self.code_dim)
        # shape: (..., 1, 1, [state_dim=2], [code_dim=200])
        coords_unsqueezed = coord_latlon.unsqueeze(-2)
        # shape: (..., h, w, 1, [coords_dim=2])
        dec_results = self.inr(coords_unsqueezed, z_unsqueezed)
        return dec_results

    @classmethod
    def calculate_latent_dim(cls, state_shape: Tuple[int, ...], **kwargs) -> int:
        h, w, c = state_shape
        return c * kwargs['code_dim']
