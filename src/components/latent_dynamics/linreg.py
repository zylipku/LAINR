from typing import Callable
import logging

import torch
from torch import nn

from modules import MLP

from .abstract_ld import LatentDynamics


class LinReg(LatentDynamics):

    def __init__(self, logger: logging.Logger, ndim: int, **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.net = MLP(
            mlp_list=[ndim, ndim,],
            act_name='swish',  # invalid
        )
        # linear

    def _forward_one_step(self, x0: torch.Tensor, **kwargs) -> torch.Tensor:

        x1 = self.net(x0)

        return x1

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'linreg',
            'net': self.net.state_dict(),
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.net.load_state_dict(model_state['net'])
