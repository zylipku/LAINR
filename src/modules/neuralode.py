from typing import Callable
import logging

import torch
from torch import nn

from torchdiffeq import odeint

from .utils import MLP
from .latent_stepper import LatentStepper


class NeuralODE(LatentStepper):

    def __init__(self, ndim: int, hidden_dim=800) -> None:
        super().__init__()

        self.net = MLP(
            mlp_list=[
                ndim,
                hidden_dim, hidden_dim, hidden_dim,
                ndim,
            ],
            act_name='swish',
        )

    def f(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def dyn(self, x0: torch.Tensor) -> torch.Tensor:

        tt = torch.tensor([0., 1.]).to(x0)

        x01 = odeint(self.f, y0=x0, t=tt, method='rk4')

        return x01[1]

    def dyn_loss(self,
                 xx: torch.Tensor,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 ratio: float = 1.) -> torch.Tensor:

        xx_pred = self.forward_stepper(xx=xx, ratio=ratio)

        loss = criterion(xx[:, 1:, ...], xx_pred[:, 1:, ...])

        return loss
