from typing import *
import logging

import torch
from torch import nn


from .abstract_uq import UncertaintyEst


class Scalar(UncertaintyEst):

    name = 'scalar uncertainty estimator'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.scalar_param = nn.Parameter(torch.zeros(1))

    @property
    def sigma(self, **kwargs) -> torch.Tensor:
        return self.positive_fn(self.scalar_param)

    def get_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        return self.sigma**2 * (torch.eye(self.ndim).to(x))

    def get_cov_chol_mat(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        return self.sigma * (torch.eye(self.ndim).to(x))

    def regularization_loss(self) -> torch.Tensor:
        '''
        .5\log\det\Sigma = n * \log\sigma
        '''
        return self.ndim * torch.log(self.sigma)

    def regression_loss(self, x: torch.Tensor = None,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        '''
        .5\|x_pred-x_target\|^2_{\Sigma}
        '''
        return .5 * torch.sum((x_pred - x_target)**2 / self.sigma**2, dim=-1)

    def get_info(self, **kwargs) -> str:
        return f"sigma: " + f'{self.sigma.item():.2e}'
