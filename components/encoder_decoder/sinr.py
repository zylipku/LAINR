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

    def __init__(self,
                 logger: logging.Logger,
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

    def encode(self,
               x: torch.Tensor,
               coords_ang: torch.Tensor,
               group: str = 'train',
               seq_idxs: torch.Tensor = None,
               z0: torch.Tensor = None,
               max_patience: int = 10,
               optim_eval_max_inner_loops=500,
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
        if group == 'train':
            return z0

        # if not hasattr(self, 'eval_codes'):
        #     self.eval_codes = self._ckpt_eval_codes.to(x.device)
        # if not hasattr(self, 'optim_cod'):
        #     self.optim_cod = optim.Adam([self.eval_codes], **self.optim_cod_kwargs)

        if x.ndim == 5:
            x = x[:, ...]  # encoding only for the first time step
            coords_ang = coords_ang[:, ...]  # encoding only for the first time step

        best_z = z0.detach().clone()
        loss_enc_best = self.criterion(x, self.decode(z0, coords_ang=coords_ang), start_dim=-3)
        z = z0.detach().clone()
        z.requires_grad_(True)  # (bs, 400)

        optim_eval = optim.Adam([z], **self.optim_cod_kwargs)

        patience = 0

        for k in range(optim_eval_max_inner_loops):

            # z.shape: (4, 400)
            # coords.shape: (4, 1, 128, 64, 3)
            z_dec = self.decode(z, coords_ang=coords_ang)  # (4, 128, 64, 2) -> (4, 128, 64, 2)
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

    def decode(self, z: torch.Tensor, coords_ang: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (..., latent_dim)
            coords (torch.Tensor): shape: (..., h, w, coords_dim=3)

        Returns:
            torch.Tensor: x, shape: (..., h, w, state_channels=2)
        '''
        assert z.ndim + 2 == coords_ang.ndim
        # z.shape: (..., 1, 1, [state_dim=2]x[code_dim=200])
        z_unsqueezed = z.view(*z.shape[:-1], 1, 1, self.state_channels, self.code_dim)
        # shape: (bs=4, nsteps=10, 1, 1, [state_dim=2], [code_dim=200])
        coords_unsqueezed = coords_ang.unsqueeze(-2)
        # shape: (bs=4, nsteps=10, h, w, 1, [coords_dim=2])
        dec_results, _ = self.inr(coords_unsqueezed, z_unsqueezed)
        return dec_results

    # def init_before_train(self,
    #                       testloader: DataLoader,
    #                       criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    #                       trainloader: DataLoader = None,
    #                       **kwargs) -> None:
    #     '''initialize before training

    #     Args:
    #         trainloader (DataLoader): train data loader
    #         testloader (DataLoader): test data loader
    #     '''

    #     self.criterion = criterion
    #     # train codes

    #     if self._ckpt_train_codes is None and trainloader is not None:
    #         trainset: LADataset = trainloader.dataset
    #         self.nframes_train = trainset.train_width
    #         nseq = trainset.nwindows
    #         self._ckpt_train_codes = torch.zeros(
    #             nseq, self.nframes_train, self.state_channels * self.code_dim, **self.factory_kwargs)

    #     if self._ckpt_eval_codes is None:
    #         nseq = len(testloader.dataset)
    #         self._ckpt_eval_codes = torch.zeros(nseq, self.state_channels * self.code_dim, **self.factory_kwargs)

    #     self.train_codes = nn.Parameter(self._ckpt_train_codes.to(**self.factory_kwargs))
    #     self.optim_cod = optim.Adam([self.train_codes], **self.optim_cod_kwargs)
    #     if self._ckpt_optim_cod is not None:
    #         self.optim_cod.load_state_dict(self._ckpt_optim_cod)

    #     self.eval_codes = self._ckpt_eval_codes.detach().clone().to(**self.factory_kwargs)
    #     # encoding only for the first time step
    #     if self._ckpt_inr_state is not None:
    #         self.inr.load_state_dict(self._ckpt_inr_state)

    # @property
    # def model_state(self) -> dict:
    #     '''model state'''
    #     return {
    #         'name': 'fouriernet',
    #         'state_channels': self.state_channels,
    #         'code_dim': self.code_dim,
    #         'latent_dim': self.latent_dim,
    #         'train_codes': self.train_codes.detach().cpu(),
    #         'eval_codes': self.eval_codes.detach().cpu(),
    #         'optim_cod': self.optim_cod.state_dict(),
    #         'inr_state': self.inr.state_dict(),
    #     }

    # @model_state.setter
    # def model_state(self, model_state: dict):
    #     '''model state'''
    #     self.state_channels = model_state['state_channels']
    #     self.code_dim = model_state['code_dim']
    #     self.latent_dim = model_state['latent_dim']
    #     self._ckpt_train_codes = model_state['train_codes']
    #     self._ckpt_eval_codes = model_state['eval_codes']
    #     self._ckpt_optim_cod = model_state['optim_cod']
    #     self._ckpt_inr_state = model_state['inr_state']
    @classmethod
    def calculate_latent_dim(cls, state_shape: Tuple[int, ...], **kwargs) -> int:
        h, w, c = state_shape
        return c * kwargs['params']['code_dim']
