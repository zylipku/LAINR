from typing import Callable
import logging

import torch
from torch import nn

from modules import MLP

from .abstract_uq import UncertaintyEst


class Cholesky(UncertaintyEst):

    name = 'cholesky uncertainty estimator'

    def __init__(self, logger: logging.Logger, ndim: int,
                 positive_func: Callable[[torch.Tensor], torch.Tensor],
                 **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim)

        self.off_diag_params = nn.Parameter(torch.zeros(ndim, ndim))
        self.diag_params = nn.Parameter(torch.zeros(ndim))
        self.positive_func = positive_func

        self.register_buffer('mask', torch.tril(torch.ones(ndim, ndim), diagonal=-1))
        self.mask: torch.Tensor

    @property
    def off_diag(self, **kwargs) -> torch.Tensor:
        '''
        return the off-diagonal elements of the matrix L^{-T}L^{-1}
        '''
        off_diag = self.positive_func(self.off_diag_params) * self.mask
        return off_diag

    @property
    def diag_rt(self, **kwargs) -> torch.Tensor:
        return self.positive_func(self.diag_params)

    @property
    def chol_mat(self, **kwargs) -> torch.Tensor:
        '''
        return the lower triangular matrix L, \Sigma = LL^T
        '''
        return torch.linalg.inv(self.chol_inv)

    @property
    def chol_inv(self, **kwargs) -> torch.Tensor:
        '''
        return the matrix L^{-1}, which is also a lower triangular matrix with positive diagonal elements
        '''
        return self.off_diag + torch.diag_embed(self.diag_rt)

    def regularization_loss(self) -> torch.Tensor:
        '''
        .5\log\det\Sigma = n * \log\sigma
        '''
        #! the negative sign is used in that we parameterize L^{-1} rather than L
        return -torch.sum(torch.log(self.diag_rt))

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
        chol_inv_x_diff = (self.chol_inv @ x_diff[..., None])[..., 0]
        # matmul w.r.t. the last dimension

        return .5 * torch.sum(chol_inv_x_diff ** 2, dim=-1)
