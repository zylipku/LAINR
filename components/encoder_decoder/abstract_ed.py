import os
import logging

from typing import Tuple

import torch
from torch import nn


class EncoderDecoder(nn.Module):

    name = 'EncoderDecoder_abstract_class'

    def __init__(self, logger: logging.Logger, **kwargs) -> None:
        super().__init__()

        self.logger = logger
        self.cudaid = kwargs.get('cudaid', 0)

    @property
    def device(self) -> torch.DeviceObjType:
        return torch.device(f'cuda:{self.cudaid}' if torch.cuda.is_available() else 'cpu')

    @property
    def factory_kwargs(self):
        return {
            'device': self.device,
            'dtype': torch.float32,
        }

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            x (torch.Tensor): shape: (..., *state_shape)

        Returns:
            torch.Tensor: z, shape: (..., latent_dim)
        '''
        raise NotImplementedError

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (..., latent_dim)

        Returns:
            torch.Tensor: x, shape: (..., *state_shape)
        '''
        raise NotImplementedError

    def forward(self, xz: torch.Tensor, operation='encode', **kwargs):
        if operation == 'encode':
            return self.encode(xz, **kwargs)
        elif operation == 'decode':
            return self.decode(xz, **kwargs)
        else:
            raise NotImplementedError

    def init_before_train(*args, **kwargs) -> None:
        '''init before train'''
        pass

    @property
    def model_state(self):
        '''model state'''
        raise NotImplementedError

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        raise NotImplementedError

    def using_parallel(self) -> None:
        raise NotImplementedError

    @classmethod
    def calculate_latent_dim(cls, state_shape: Tuple[int, ...], **kwargs) -> int:
        raise NotImplementedError
