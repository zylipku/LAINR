import os

import logging
from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .abstract_ed import EncoderDecoder
from modules import CAE


class CAEED(EncoderDecoder):

    # test rmse=3.84e-2 for SW (7000 epochs)
    # test skip rmse=- for SW (50 epochs)
    # test weighted rmse=- for SW (50 epochs)

    name = 'CAE'

    cae_kwargs = {
        'state_size': (128, 64),
        'state_channels': 2,
        'hidden_channels': 16,
        'latent_channels': 8,
        'padding_type': ('circular', 'replicate'),
    }

    def __init__(self,
                 logger: logging.Logger,
                 **kwargs) -> None:
        super().__init__(logger, **kwargs)

        self.cae_kwargs.update(kwargs)

        self.state_size = self.cae_kwargs['state_size']
        self.state_channels = self.cae_kwargs['state_channels']

        self.cae = CAE(**self.cae_kwargs)
        self.state_shape = (*self.state_size, self.state_channels)

        # calculate latent dim
        with torch.no_grad():
            x_test = torch.randn(1, *self.state_shape).moveaxis(-1, -3)
            x_test = x_test.to(**self.factory_kwargs)
            self.cae.to(**self.factory_kwargs)
            z_test = self.cae.encode(x_test)
            bs, *latent_shape = z_test.shape
            self.latent_shape = latent_shape
            self.logger.info(f'latent_shape: {self.latent_shape}')
            self.latent_dim = int(torch.prod(torch.tensor(self.latent_shape)))
            del x_test
            del z_test

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            x (torch.Tensor): shape: (..., *state_shape)

        Returns:
            torch.Tensor: z, shape: (..., latent_dim)
        '''
        x_flat = x.contiguous().view(-1, *self.state_shape).moveaxis(-1, -3)
        z_flat = self.cae.encode(x_flat)
        z = z_flat.contiguous().view(*x.shape[:-len(self.state_shape)], -1)

        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (..., latent_dim)

        Returns:
            torch.Tensor: x, shape: (..., *state_shape)
        '''
        z_flat = z.contiguous().view(-1, *self.latent_shape)
        x_flat = self.cae.decode(z_flat)
        x = x_flat.contiguous().view(*z.shape[:-1], *x_flat.shape[1:]).moveaxis(-3, -1)

        return x

    @classmethod
    def calculate_latent_dim(cls, state_shape: Tuple[int, ...], **kwargs) -> int:
        h, w, c = state_shape
        latent_channels = kwargs.get('latent_channels', cls.cae_kwargs['latent_channels'])
        latent_h = h // 8
        latent_w = w // 8
        latent_dim = latent_h * latent_w * latent_channels
        return latent_dim
