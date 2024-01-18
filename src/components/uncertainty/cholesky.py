from typing import Callable
import logging

import torch
from torch import Tensor, nn

from modules import MLP

from .abstract_uq import UncertaintyEst


class Cholesky(UncertaintyEst):

    name = 'cholesky uncertainty estimator'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.off_diag_params = nn.Parameter(torch.zeros(self.ndim, self.ndim))
        self.diag_params = nn.Parameter(torch.zeros(self.ndim))

        regularization = kwargs.get('regularization', 0.)
        self.logger.info(f'{regularization=:.2e}')

        self.register_buffer('mask', torch.tril(torch.ones(self.ndim, self.ndim), diagonal=-1))
        self.mask: torch.Tensor
        self.register_buffer('regularization', torch.tensor(regularization))
        self.regularization: torch.Tensor

    @property
    def off_diag(self, **kwargs) -> torch.Tensor:
        '''
        return the off-diagonal elements of the matrix L^{-T}L^{-1}
        '''
        off_diag = self.off_diag_params * self.mask
        return off_diag

    @property
    def diag_rt(self, **kwargs) -> torch.Tensor:
        return 1. / self.positive_fn(self.diag_params)

    @property
    def chol_mat(self, **kwargs) -> torch.Tensor:
        '''
        return the lower triangular matrix L, \Sigma = LL^T
        '''
        return torch.linalg.inv(self.chol_inv)

    def get_cov_chol_mat(self, x: Tensor, **kwargs) -> Tensor:
        return self.chol_mat

    @property
    def chol_inv(self, **kwargs) -> torch.Tensor:
        '''
        return the matrix L^{-1}, which is also a lower triangular matrix with positive diagonal elements
        '''
        return self.off_diag + torch.diag_embed(self.diag_rt)

    def reg_off_diag_l1_loss(self) -> torch.Tensor:
        '''
        l1 pernalty for the off-diagonal elements of L
        '''
        reg_off = torch.sum(torch.abs(self.chol_mat * self.mask))
        # print(reg_off.requires_grad)
        # print(f'reg_off:', reg_off.item(), ';', f'{self.regularization.item()=:.2e}')
        return reg_off

    def regularization_loss(self) -> torch.Tensor:
        '''
        .5\log\det\Sigma = n * \log\sigma
        '''
        #! the negative sign is used in that we parameterize L^{-1} rather than L
        return -torch.sum(torch.log(self.diag_rt)) + self.reg_off_diag_l1_loss() * self.regularization

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

    def get_info(self, **kwargs) -> str:
        top_rt = ''
        for rt in self.diag_rt[:5]:
            top_rt += f'{rt.item():.2e} '
        return f'reg_off: {self.reg_off_diag_l1_loss():.2e} | avg diag_rt: {1/self.diag_rt.mean().item():.2e}'
