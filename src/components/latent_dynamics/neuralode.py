from typing import Callable
import logging

import torch
from torch import nn

from torchdiffeq import odeint

from modules import MLP

from .abstract_ld import LatentDynamics


class NeuralODE(LatentDynamics):

    name = 'NeuralODE'

    def __init__(self, logger: logging.Logger, ndim: int, hidden_dim=800, **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.net = MLP(
            mlp_list=[
                ndim,
                hidden_dim, hidden_dim, hidden_dim,
                ndim,
            ],
            act_name='swish',
        )

    def f(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _forward_one_step(self, x0: torch.Tensor, **kwargs) -> torch.Tensor:

        tt = torch.tensor([0., 1.]).to(x0)

        x01 = odeint(self.f, y0=x0, t=tt, method='rk4')

        return x01[1]

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
