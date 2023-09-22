from typing import Callable
import logging

import torch
from torch import nn

from modules import MLP

from .abstract_uq import Uncertainty


class Diagonal(Uncertainty):

    def __init__(self, logger: logging.Logger, ndim: int, hidden_dim=1024, **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.net = MLP(
            mlp_list=[
                ndim,
                hidden_dim,
                hidden_dim,
                ndim,
            ],
            act_name='tanh',
        )
        # zero initialization
        for p in self.net.parameters():
            nn.init.zeros_(p)

        # self.log_cov = nn.Parameter(torch.zeros(ndim))

    def get_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        output the logs of the variance

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''

        y = self.net(x)

        return y

    def get_cov_chol_mat(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        output the logs of the variance

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''

        log_cov = self.get_cov(x)

        deviation = torch.exp(log_cov / 2.)
        cov_chol_mat = torch.diag_embed(deviation)

        return cov_chol_mat

    def log_det_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''log determinant of covariance matrix

        Args:
            x (torch.Tensor): shape: (bs, nsteps, latent_dim)

        Returns:
            torch.Tensor: log_det_cov, shape: (bs, nsteps)
        '''

        log_covs = self.get_cov(x)

        return torch.sum(log_covs, dim=-1)

    def regression_loss(self, x: torch.Tensor,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        '''
        _summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''
        covs = torch.exp(self.get_cov(x))

        return torch.sum((x_pred - x_target)**2 / covs, dim=-1)

    def likelihood_loss(self, x: torch.Tensor,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        loss = self.log_det_cov(x, **kwargs) + self.regression_loss(x, x_pred, x_target, **kwargs)
        return torch.mean(loss)

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'diagonal',
            'net': self.net.state_dict(),
            # 'log_cov': self.log_cov,
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.net.load_state_dict(model_state['net'])
        # self.log_cov = model_state['log_cov']
