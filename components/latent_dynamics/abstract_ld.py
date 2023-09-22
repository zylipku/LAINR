import logging

import torch
from torch import nn


class LatentDynamics(nn.Module):

    name = 'LatentDyn_abstract_class'

    def __init__(self, logger: logging.Logger, ndim: int, **kwargs) -> None:
        super().__init__()

        self.logger = logger
        self.ndim = ndim

    def _forward_one_step(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (bs, nsteps, latent_dim)

        Returns:
            torch.Tensor: z_, shape: (bs, nsteps, latent_dim)
        '''
        raise NotImplementedError

    def forward(self, zz: torch.Tensor, nsteps: int = 1, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            zz (torch.Tensor): shape: (bs, nsteps, latent_dim)

        Returns:
            torch.Tensor: zz_, shape: (bs, nsteps, latent_dim)
        '''
        for _ in range(nsteps):
            zz = self._forward_one_step(zz)
        return zz

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'none',
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        pass

    def init_before_train(self, *args, **kwargs) -> None:
        '''init before train'''
        pass

    def using_parallel(self) -> None:
        raise NotImplementedError
