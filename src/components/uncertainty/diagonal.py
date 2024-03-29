from typing import Callable
import logging

import torch
from torch import nn

from modules import MLP

from .abstract_uq import UncertaintyEst


class Diagonal(UncertaintyEst):

    name = 'diagonal uncertainty estimator'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.scalar_params = nn.Parameter(torch.zeros(self.ndim))

    @property
    def diag_rt(self, **kwargs) -> torch.Tensor:
        return self.positive_fn(self.scalar_params)

    def get_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.diag_rt**2 * (torch.eye(self.ndim).to(x))

    def get_cov_chol_mat(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.diag_rt * (torch.eye(self.ndim).to(x))

    def regularization_loss(self) -> torch.Tensor:
        '''
        .5\log\det\Sigma = n * \log\sigma
        '''
        return torch.sum(torch.log(self.diag_rt))

    def regression_loss(self, x: torch.Tensor,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        '''
        .5\log\det\Sigma = \sum_k\log\sigma_k
        '''
        return .5 * torch.sum((x_pred - x_target)**2 / self.diag_rt**2, dim=-1)

    def get_info(self, **kwargs) -> str:
        top_rt = ''
        for rt in self.diag_rt[:5]:
            top_rt += f'{rt.item():.2e} '
        return f"top rt diagonal: " + top_rt + f' | avg: {self.diag_rt.mean().item():.2e}'
