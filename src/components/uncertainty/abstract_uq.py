import logging

import torch
from torch import nn


class Uncertainty(nn.Module):

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

    # def log_det_cov(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    #     '''log determinant of covariance matrix

    #     Args:
    #         x (torch.Tensor): shape: (bs, nsteps, latent_dim)

    #     Returns:
    #         torch.Tensor: log_det_cov, shape: (bs, nsteps)
    #     '''
    #     raise NotImplementedError

    def likelihood_loss(self, x: torch.Tensor,
                        x_pred: torch.Tensor = None,
                        x_target: torch.Tensor = None,
                        **kwargs) -> torch.Tensor:
        '''get likelihood function of x: log det(\Sigma) + \|x-x'\|^2 (weighted by uncertainty)

        Args:
            x (torch.Tensor): shape: (bs, nsteps)

        Returns:
            torch.Tensor: function (x, x') -> float
        '''
        raise NotImplementedError

    def forward(self, x: torch.Tensor,
                x_pred: torch.Tensor = None,
                x_target: torch.Tensor = None,
                output='loss', **kwargs) -> torch.Tensor:
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

        if output == 'loss':
            loss = self.likelihood_loss(x, x_pred, x_target, **kwargs)
            return loss

        elif output == 'cov':
            return self.get_cov(x)

        elif output == 'cov_chol_mat':
            return self.get_cov_chol_mat(x)

    @property
    def model_state(self):
        '''model state'''
        return {
            'name': 'none',
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        pass

    def init_before_train(self, *args, **kwargs) -> None:
        '''init before train'''
        pass

    def using_parallel(self) -> None:
        raise NotImplementedError
