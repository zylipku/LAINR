import os

import logging
from typing import *

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .abstract_ed import EncoderDecoder
from modules import MyDINoINR

from metrics import SphereLoss


class FourierNetCartesED(EncoderDecoder):

    name = 'FourierNetCartes'

    inr_kwargs = {
        'coord_channels': 3,  # ! using the cartesian coordinates
        'code_dim': 200,
        'state_channels': 2,
        'hidden_dim': 256,
        'nlayers': 6,
    }

    optim_cod_kwargs = {
        'lr': 1e-3,
    }

    train_codes: torch.Tensor
    eval_codes: torch.Tensor

    def __init__(self, logger: logging.Logger,
                 loss_fn_inner_loop: SphereLoss,
                 **kwargs) -> None:
        super().__init__(logger, **kwargs)

        # self.state_size = kwargs.get('state_size', self.inr_kwargs['state_size'])
        self.state_channels = 2

        self.code_dim = kwargs.get('code_dim', self.inr_kwargs['code_dim'])
        self.latent_dim = self.state_channels * self.code_dim

        self.inr_kwargs.update(kwargs)

        self.inr = MyDINoINR(**self.inr_kwargs)
        # self.state_shape = (*self.state_size, self.state_channels)

        self._ckpt_train_codes = None
        self._ckpt_eval_codes = None
        self._ckpt_optim_cod = None
        self._ckpt_inr_state = None

        self.loss_fn = loss_fn_inner_loop

    def encode(self, x: torch.Tensor,
               coord_latlon: torch.Tensor,
               z0: torch.Tensor = None,
               max_patience: int = 10,
               optim_eval_max_inner_loops=100,
               **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            x (torch.Tensor): shape: (..., *state_shape)
            group (str): 'train' or 'eval'
            z0 (torch.Tensor): shape: (..., latent_dim), valid for group='eval'
            max_patience (int): max number of patience for early stopping, valid for group='eval'

        Returns:
            torch.Tensor: z, shape: (..., latent_dim)
        '''
        best_z = z0.detach().clone()
        loss_enc_best = self.loss_fn(x, self.decode(z0, coords=coord_latlon), start_dim=-3)
        z = z0.detach().clone()
        z.requires_grad_(True)  # (bs, 400)

        optim_eval = optim.Adam([z], **self.optim_cod_kwargs)

        patience = 0

        for k in range(optim_eval_max_inner_loops):

            # z.shape: (4, 400)
            # coords.shape: (4, 1, 128, 64, 3)
            z_dec = self.decode(z, coords=coord_latlon)  # (4, 128, 64, 2) -> (4, 128, 64, 2)
            loss_dec: torch.Tensor = self.loss_fn(x, z_dec, start_dim=-3)

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
            coords (torch.Tensor): shape: (..., h, w, coords_dim=3)

        Returns:
            torch.Tensor: x, shape: (..., h, w, state_channels=2)
        '''
        assert z.ndim + 2 == coord_latlon.ndim
        # z.shape: (..., 1, 1, [state_dim=2]x[code_dim=200])
        z_unsqueezed = z.view(*z.shape[:-1], 1, 1, self.state_channels, self.code_dim)
        # shape: (bs=4, nsteps=10, 1, 1, [state_dim=2], [code_dim=200])
        coords_unsqueezed = coord_latlon.unsqueeze(-2)
        # shape: (bs=4, nsteps=10, h, w, 1, [coords_dim=2])
        dec_results, _ = self.inr(coords_unsqueezed, z_unsqueezed)
        return dec_results

    @classmethod
    def calculate_latent_dim(cls, state_shape: Tuple[int, ...], **kwargs) -> int:
        h, w, c = state_shape
        return c * kwargs['params']['code_dim']
