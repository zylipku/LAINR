from typing import Callable
import logging

import torch
from torch import nn

from modules import MLP

from .abstract_uq import Uncertainty


class Cholesky(Uncertainty):

    def __init__(self, logger: logging.Logger, ndim: int, hidden_dim=1024, **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.off_diag = nn.Parameter(torch.zeros(ndim, ndim))
        self.log_diag = nn.Parameter(torch.zeros(ndim))

    @property
    def chol_inv(self, **kwargs) -> torch.Tensor:
        '''
        \Sigma = LL^T
        \Sigma^{-1} = L^{-T}L^{-1}

        return the matrix L^{-1}, which is also a lower triangular matrix with positive diagonal elements
        '''
        chol_inv_mat = torch.tril(self.off_diag, diagonal=-1) + torch.diag(torch.exp(self.log_diag))
        # self.logger.info(f'{chol_inv_mat=}')
        return chol_inv_mat

    @property
    def chol(self, **kwargs) -> torch.Tensor:
        '''
        \Sigma = LL^T
        \Sigma^{-1} = L^{-T}L^{-1}

        return the matrix L
        '''
        chol_mat = torch.linalg.inv(self.chol_inv)
        return chol_mat

    def get_cov_chol_mat(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        return self.chol

    def regularization_loss(self, x: torch.Tensor,
                            x_pred: torch.Tensor = None,
                            x_target: torch.Tensor = None,
                            **kwargs) -> torch.Tensor:
        '''
        1/2 \log\det(\Sigma) = \log\det(L) = -\sum_i \log L^{-1}_{ii}
        '''
        return -torch.sum(self.log_diag)

    def regression_loss(self, x: torch.Tensor,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        '''
          1/2 (x-x')^T \Sigma^{-1} (x-x') 
        = 1/2 (x-x')^T L^{-T} L^{-1} (x-x') 
        = 1/2 \|L^{-1} (x-x')\|^2
        '''
        x_diff = x_pred - x_target
        chol_inv_x_diff = (self.chol_inv @ x_diff[..., None])[..., 0]  # matmul w.r.t. the last dimension

        return torch.mean(torch.sum(chol_inv_x_diff ** 2, dim=-1)) / 2

    def likelihood_loss(self, x: torch.Tensor,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        return self.regularization_loss(x, x_pred, x_target, **kwargs) +\
            self.regression_loss(x, x_pred, x_target, **kwargs)

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'diagonal',
            'off_diag': self.off_diag,
            'log_diag': self.log_diag,
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.off_diag = model_state['off_diag']
        self.log_diag = model_state['log_diag']
