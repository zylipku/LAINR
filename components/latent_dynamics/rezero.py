from typing import Callable
import logging

import torch
from torch import nn

from torchdiffeq import odeint

from modules import ReZero

from .abstract_ld import LatentDynamics


class ReZeroDyn(LatentDynamics):

    name = 'ReZero'

    def __init__(self, logger: logging.Logger, ndim: int, **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.rezero = ReZero(logger=logger, ndim=ndim, **kwargs)

    def _forward_one_step(self, x0: torch.Tensor, **kwargs) -> torch.Tensor:

        x1 = self.rezero(x0)

        return x1

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'rezero',
            'rezero': self.rezero.state_dict(),
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.rezero.load_state_dict(model_state['rezero'])
