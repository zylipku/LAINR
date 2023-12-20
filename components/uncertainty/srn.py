from typing import Callable
import logging

import torch
from torch import nn

from modules import MLP

from .abstract_uq import Uncertainty


class SRN(Uncertainty):

    def __init__(self, logger: logging.Logger, ndim: int, hidden_dim=1024, rk=5, **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.net = MLP(
            mlp_list=[
                ndim,
                hidden_dim,
                hidden_dim,
                hidden_dim,
                ndim * rk,
            ],
            act_name='tanh',
        )

        self.mu = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        output the logs of the variance

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''

        y = self.net(x)
        z = y.view(*y.shape[:-1], self.ndim, -1)

        return z

    def log_det_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''log determinant of covariance matrix

        Args:
            x (torch.Tensor): shape: (bs, nsteps, latent_dim)

        Returns:
            torch.Tensor: log_det_cov, shape: (bs, nsteps)
        '''

        log_covs = self.forward(x)

        return torch.sum(log_covs, dim=-1)

    def regression_loss(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        _summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''
        covs = torch.exp(self.forward(x))

        return lambda x1, x2: torch.sum((x1 - x2)**2 / covs, dim=-1) * (-.5)

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'srn',
            'net': self.net.state_dict(),
            'mu': self.mu,
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.net.load_state_dict(model_state['net'])
        self.mu = model_state['mu']
