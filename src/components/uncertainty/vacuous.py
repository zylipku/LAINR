from typing import Callable
import logging

import torch
from torch import nn

from modules import MLP

from .abstract_uq import UncertaintyEst


class Vacuous(UncertaintyEst):

    def __init__(self, logger: logging.Logger, ndim: int, **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        output the logs of the variance

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''

        return x

    def log_det_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''log determinant of covariance matrix

        Args:
            x (torch.Tensor): shape: (bs, nsteps, latent_dim)

        Returns:
            torch.Tensor: log_det_cov, shape: (bs, nsteps)
        '''

        return 0.

    def regression_loss(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        _summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''
        return 0.
