import os

import logging
from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .abstract_ed import EncoderDecoder
from modules import MLP

from metrics import SphereLoss
from plot import plot_comp


class PCAED(EncoderDecoder):

    name = 'PCA'

    def __init__(self,
                 logger: logging.Logger,
                 latent_dim: int = 1024,
                 criterion: SphereLoss = None,
                 **kwargs) -> None:
        super().__init__(logger, **kwargs)

        self.latent_dim = latent_dim

        self.initialized = False

        self.criterion = criterion

        self.register_buffer('pca_modes', torch.zeros(1, latent_dim))
        self.register_buffer('pca_mean', torch.zeros(1, latent_dim))

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            x (torch.Tensor): shape: (..., *state_shape)

        Returns:
            torch.Tensor: z, shape: (..., latent_dim)
        '''
        x_flat = x.reshape(-1, self.state_dim)
        z_flat = (x_flat - self.pca_mean) @ self.pca_modes
        z = z_flat.reshape(*x.shape[:-len(self.state_shape)], self.latent_dim)

        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        '''encode x into latent space

        Args:
            z (torch.Tensor): shape: (..., latent_dim)

        Returns:
            torch.Tensor: x, shape: (..., *state_shape)
        '''
        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = z_flat @ self.pca_modes.T + self.pca_mean
        x = x_flat.reshape(*z.shape[:-1], *self.state_shape)

        return x

    def get_modes(self, dataloader: DataLoader):
        # PCA
        self.logger.info('Computing PCA modes...')
        # get the empirical mean
        field_sum = 0.
        field_cnt = 0

        field_sample = next(iter(dataloader))['data']
        bs, nsteps, h, w, nstates = field_sample.shape
        self.state_shape = (h, w, nstates)
        self.state_dim = h * w * nstates

        for batch in dataloader:

            field_true: torch.Tensor = batch['data'].to(**self.factory_kwargs)
            # coords = batch['coords'].to(**self.factory_kwargs)
            # tt = batch['tt'][0].to(**self.factory_kwargs)
            # seq_idxs = batch['idxs'].to(**self.factory_int_kwargs)

            bs, nsteps, h, w, nstates = field_true.shape

            field_flatten = field_true.view(bs * nsteps, h * w * nstates)
            field_sum += field_flatten.sum(dim=0, keepdim=True)
            field_cnt += bs * nsteps
        field_mean = field_sum / field_cnt
        self.logger.info(f"field_mean obtained, shape: {field_mean.shape}")
        # (1, h*w*nstates)

        # get the empirical covariance
        field_cov = 0.

        for batch in dataloader:

            field_true: torch.Tensor = batch['data'].to(**self.factory_kwargs)
            # coords = batch['coords'].to(**self.factory_kwargs)
            # tt = batch['tt'][0].to(**self.factory_kwargs)
            # seq_idxs = batch['idxs'].to(**self.factory_int_kwargs)

            bs, nsteps, h, w, nstates = field_true.shape

            field_flatten = field_true.view(bs * nsteps, h * w * nstates)
            field_cov += (field_flatten - field_mean).T @ (field_flatten - field_mean)

        field_cov /= (field_cnt - 1)
        self.logger.info(f"field_cov obtained, shape: {field_cov.shape}")
        # (h*w*nstates, h*w*nstates)

        # get the eigenvectors and eigenvalues
        eigvals, eigvecs = torch.linalg.eigh(field_cov)
        sorted_eigvals, _ = torch.sort(eigvals, descending=True)
        # plot_eigval = True
        # if plot_eigval:
        #     import matplotlib
        #     import matplotlib.pyplot as plt
        #     import numpy as np
        #     matplotlib.use("pgf")
        #     matplotlib.rcParams.update({
        #         "pgf.texsystem": "pdflatex",
        #         'font.family': 'serif',
        #         'text.usetex': True,
        #         'pgf.rcfonts': True,
        #     })

        #     def latex_sci(num):
        #         exp = np.floor(np.log10(np.abs(num))).astype(int)
        #         base = round(num / 10**float(exp), 3)
        #         if abs(base - int(base)) < 0.01:
        #             base = int(base)
        #         return r"${0} \cdot 10^{{{1}}}$".format(base, exp)

        #     eigvals_plot = sorted_eigvals.cpu().numpy()
        #     # add markers for specific points
        #     plt.scatter([400, 1024], [eigvals_plot[400], eigvals_plot[1024]], s=10, color='red')

        #     plt.plot(eigvals_plot[:5000])
        #     # add line segments from points to axes
        #     plt.plot([400, 400], [0, eigvals_plot[400]], color='gray', linestyle='dotted')
        #     # plt.plot([0, 400], [eigvals_plot[400], eigvals_plot[400]], color='gray', linestyle='--')
        #     plt.plot([1024, 1024], [0, eigvals_plot[1024]], color='gray', linestyle='dotted')
        #     # plt.plot([0, 1024], [eigvals_plot[1024], eigvals_plot[1024]], color='gray', linestyle='--')

        #     plt.annotate(latex_sci(eigvals_plot[400]), (400 + 100, eigvals_plot[400]))
        #     plt.annotate(latex_sci(eigvals_plot[1024]), (1024 + 100, eigvals_plot[1024]))
        #     plt.grid(True, which="both", ls="--", color='0.65')
        #     # plt.gca().tick_params(axis='y', length=5, width=2, color='r')

        #     plt.yscale('log')
        #     plt.ylabel('eigenvalues')

        #     plt.xlim(left=0)
        #     plt.xlabel('index of modes')
        #     plt.savefig('./eigvals.pdf')
        #     # plt.show()
        #     plt.close()

        # (h*w*nstates,), (h*w*nstates, h*w*nstates)

        # sort eigenvectors and eigenvalues
        sorted_idxs = torch.argsort(eigvals, descending=True)
        modes: torch.Tensor = eigvecs[:, sorted_idxs][:, :self.latent_dim]

        self.initialized = True

        self.pca_modes = modes
        self.pca_mean = field_mean

        self.logger.info(f"modes obtained, {self.pca_modes.shape=}, {self.pca_mean.shape=}")

        train_loss = self.test(dataloader)
        self.logger.info(f'training loss = {train_loss:.4e}')

        return modes, field_mean, train_loss

    def test(self, dataloader: DataLoader, **kwargs) -> float:

        losses = []

        for k, batch in enumerate(dataloader):
            batch: Dict[str, torch.Tensor]

            field_true: torch.Tensor = batch['data'].to(**self.factory_kwargs)
            # coords = batch['coords'].to(**self.factory_kwargs)
            # tt = batch['tt'][0].to(**self.factory_kwargs)
            # seq_idxs = batch['idxs'].to(**self.factory_int_kwargs)

            field_rec = self.decode(self.encode(field_true))
            loss_rec = self.criterion(field_true, field_rec, root=True)

            losses.append(loss_rec.item())

            if k == 0:
                plot_comp(field_true[0, ..., 0], field_rec[0, ..., 0], 'tmp/comp_pca_debug_height.png')
                plot_comp(field_true[0, ..., 1], field_rec[0, ..., 1], 'tmp/comp_pca_debug_vorticity.png')

        avg_loss = sum(losses) / len(losses)

        return avg_loss

    @property
    def model_state(self) -> dict:
        '''model state'''
        return {
            'name': 'pca',
            'pca_modes': self.pca_modes,
            'pca_mean': self.pca_mean,
            'state_dim': self.state_dim,
            'state_shape': self.state_shape,
        }

    @model_state.setter
    def model_state(self, model_state: dict):
        '''model state'''
        self.pca_modes = model_state['pca_modes']
        self.pca_mean = model_state['pca_mean']
        self.state_dim = model_state['state_dim']
        self.state_shape = model_state['state_shape']

    @classmethod
    def calculate_latent_dim(cls, state_shape: Tuple[int, ...], **kwargs) -> int:
        return kwargs["params"]['latent_dim']
