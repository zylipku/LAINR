from typing import Callable
import logging

import torch
from torch import nn

from torchdiffeq import odeint

from modules import MLP

from .abstract_ld import LatentDynamics


class MLPLD(LatentDynamics):

    name = 'MLP'

    def __init__(self, logger: logging.Logger,
                 ndim: int,
                 hidden_dim: int = 256,
                 nlayers: int = 4,
                 skip_connection: bool = True,
                 **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.net = MLP(
            mlp_list=[ndim] + [hidden_dim,] * (nlayers - 1) + [ndim],
            act_name='leaky',
        )
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.skip_connection = skip_connection

    def _forward_one_step(self, x0: torch.Tensor, **kwargs) -> torch.Tensor:

        if self.skip_connection:
            x01 = x0 + self.net(x0)
        else:
            x01 = self.net(x0)

        return x01

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'neuralode',
            'net': self.net.state_dict(),
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.net.load_state_dict(model_state['net'])
