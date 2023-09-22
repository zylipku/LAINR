import logging
from typing import Callable, List

import torch
from torch import nn


class LatentStepper(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def dyn(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_stepper(self, xx: torch.Tensor, ratio: float = 1.) -> torch.Tensor:
        '''
        forward stepper

        Args:
            xx (torch.Tensor): shape=(bs, nsteps, *features)
            ratio (float): ratio of true states used for prediction
            ratio=0.: Only the first step is fed into the net and then roll-out
            ratio=1.: Each step is fed into the net for predicting the next step
            ratio\in(0,1): each is fed into the net with a probability ratio

        Returns:
            torch.Tensor: shape=(bs, nsteps, *features)
            The first step is the same as that of x, and the remaining steps are predictions from the previous steps
        '''
        bs, nsteps, *_ = xx.shape  # [0, nsteps-1]

        if ratio < 1e-3:
            obs_mask = torch.zeros(nsteps, dtype=torch.bool)
        elif ratio > 1 - 1e-3:
            obs_mask = torch.ones(nsteps, dtype=torch.bool)
        else:
            obs_mask = torch.rand(nsteps) < ratio
        obs_mask[0] = True
        obs_mask[-1] = True

        preds = [xx[:, 0]]

        start_step = 0
        end_step = 1

        while end_step < nsteps:

            while not obs_mask[end_step]:
                end_step += 1
            # increase end_step until it reaches the end or it is observed

            x = xx[:, start_step, ...]
            for step in range(start_step + 1, end_step + 1):
                # integrate from start_step to end_step
                x = self.dyn(x)
                preds.append(x)

            start_step = end_step
            end_step += 1

        preds = torch.stack(preds, dim=1)

        return preds
