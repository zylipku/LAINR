import logging

import torch
from torch import nn


class UncertaintyEst(nn.Module):

    name = 'Uncertainty_abstract_class'

    def __init__(self, logger: logging.Logger, ndim: int, **kwargs) -> None:
        super().__init__()

        self.logger = logger
        self.ndim = ndim

    def get_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''get covariance matrix

        Args:
            x (torch.Tensor): shape: (bs, nsteps, latent_dim)

        Returns:
            torch.Tensor: cov, shape: (bs, nsteps, latent_dim, latent_dim)
        '''
        raise NotImplementedError

    def get_cov_chol_mat(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''get covariance matrix

        Args:
            x (torch.Tensor): shape: (bs, nsteps, latent_dim)

        Returns:
            torch.Tensor: cov_chol, shape: (bs, nsteps, latent_dim, latent_dim)
        '''
        raise NotImplementedError

    def regression_loss(self, x: torch.Tensor,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        '''
        return .5\|x_pred-x_target\|^2_{\Sigma}
        '''
        raise NotImplementedError

    def regularization_loss(self) -> torch.Tensor:
        '''
        return .5\log\det\Sigma
        '''
        raise NotImplementedError

    def log_likelihood_loss(self, x: torch.Tensor,
                            x_pred: torch.Tensor = None,
                            x_target: torch.Tensor = None,
                            **kwargs) -> torch.Tensor:
        return self.regularization_loss() + self.regression_loss(x, x_pred, x_target, **kwargs)

    def forward(self, x: torch.Tensor,
                x_pred: torch.Tensor = None,
                x_target: torch.Tensor = None,
                output='-llh', **kwargs) -> torch.Tensor:
        '''
        max the log likelihood of the model
        max log(1/det(cov)^{1/2}) - \|x-y\|^2/cov/2
        equivalent to
        min sum(log(cov)) + \|x-y\|^2/cov

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        '''

        if output == '-llh':
            loss = self.log_likelihood_loss(x, x_pred, x_target, **kwargs)
            return loss

        elif output == 'cov':
            return self.get_cov(x)

        elif output == 'cov_chol_mat':
            return self.get_cov_chol_mat(x)
