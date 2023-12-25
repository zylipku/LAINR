import os

import logging
from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .abstract_ed import EncoderDecoder
from modules import AEflowV2


class AEflowED(EncoderDecoder):

    name = 'AEflow'

    aeflow_kwargs = {
        'state_size': (128, 64),
        'state_channels': 2,
        'hidden_channels': 4,
        'latent_channels': 8,
        'padding_type': ('circular', 'replicate'),
        'nresblocks': 12,
    }

    def __init__(self,
                 logger: logging.Logger,
                 **kwargs) -> None:
        super().__init__(logger, **kwargs)

        self.aeflow_kwargs.update(kwargs)

        # self.state_size = kwargs.get('state_size', self.aeflow_kwargs['state_size'])
        # self.state_channels = kwargs.get('state_channels', self.aeflow_kwargs['state_channels'])

        self.state_size = self.aeflow_kwargs['state_size']
        self.state_channels = self.aeflow_kwargs['state_channels']

        self.aeflow = AEflowV2(**self.aeflow_kwargs)
        self.state_shape = (*self.state_size, self.state_channels)

        # calculate latent dim
        with torch.no_grad():
            x_test = torch.randn(1, *self.state_shape).moveaxis(-1, -3)
            x_test = x_test.to(**self.factory_kwargs)
            self.aeflow.to(**self.factory_kwargs)
            z_test = self.aeflow.encode(x_test)
            bs, *latent_shape = z_test.shape
            self.latent_shape = latent_shape
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
        z_flat = self.aeflow.encode(x_flat)
        z = z_flat.contiguous().view(*x.shape[:-len(self.state_shape)], -1)

        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (..., latent_dim)

        Returns:
            torch.Tensor: x, shape: (..., *state_shape)
        '''
        z_flat = z.view(-1, *self.latent_shape)
        x_flat = self.aeflow.decode(z_flat)

        x = x_flat.view(*z.shape[:-1], *x_flat.shape[1:]).moveaxis(-3, -1)

        return x

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'aeflow',
            'aeflow': self.aeflow.state_dict(),
            'state_shape': self.state_shape,
            'latent_shape': self.latent_shape,
            'state_size': self.state_size,
            'state_channels': self.state_channels,
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.aeflow.load_state_dict(model_state['aeflow'])
        self.state_shape = model_state['state_shape']
        self.latent_shape = model_state['latent_shape']
        self.state_size = model_state['state_size']
        self.state_channels = model_state['state_channels']

    @classmethod
    def calculate_latent_dim(cls, state_shape: Tuple[int, ...], **kwargs) -> int:
        h, w, c = state_shape
        latent_channels = kwargs.get('latent_channels', cls.aeflow_kwargs['latent_channels'])
        latent_h = h // 8
        latent_w = w // 8
        latent_dim = latent_h * latent_w * latent_channels
        return latent_dim


if __name__ == '__main__':

    import sys

    sys.path.insert(0, '')
    sys.path.append('../')
    sys.path.append('../../')

    x = torch.randn(3, 128, 64, 2)
    model = AEflowED(None, state_size=(128, 64), kernel_size=5)
    y = model(x)
    print(f'{y.shape=}')
