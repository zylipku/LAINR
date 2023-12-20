from typing import Dict

import numpy as np
import torch

from matplotlib import pyplot as plt


def plot_comp(xx: torch.Tensor | np.ndarray, yy: torch.Tensor | np.ndarray, save_path: str, **plot_kwargs) -> None:
    '''
    plot comparison for xx and yy horizontally as well as their difference

    Args:
        xx (torch.Tensor): shape:(nsteps, h, w)
        yy (torch.Tensor): shape:(nsteps, h, w)
    '''
    plt.clf()

    if isinstance(xx, torch.Tensor):
        xx = xx.detach().cpu().numpy()

    if isinstance(yy, torch.Tensor):
        yy = yy.detach().cpu().numpy()

    diff = np.abs(yy - xx)

    nsteps, _, _ = xx.shape

    fig, axs = plt.subplots(nrows=3, ncols=nsteps, figsize=(5 * nsteps, 15))

    if nsteps == 1:
        im = axs[0].imshow(xx[0], **plot_kwargs)
        fig.colorbar(im, ax=axs[0])

        im = axs[1].imshow(yy[0], **plot_kwargs)
        fig.colorbar(im, ax=axs[1])

        im = axs[2].imshow(diff[0], **plot_kwargs)
        fig.colorbar(im, ax=axs[2])

    else:
        for step in range(nsteps):

            im = axs[0, step].imshow(xx[step], **plot_kwargs)
            fig.colorbar(im, ax=axs[0, step])

            im = axs[1, step].imshow(yy[step], **plot_kwargs)
            fig.colorbar(im, ax=axs[1, step])

            im = axs[2, step].imshow(diff[step], **plot_kwargs)
            fig.colorbar(im, ax=axs[2, step])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
