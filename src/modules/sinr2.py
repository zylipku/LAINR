'''
modified spherical INR
using sinθ * sin(linear(cosφ)) * sin(linear(sinφ))
instead of spherical harmonics
'''


import math

from typing import Dict, Iterable, Tuple

import torch
from torch import nn


class SphericalFilter(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(1, hidden_dim)
        self.linear2 = nn.Linear(1, hidden_dim)
        self.linear3 = nn.Linear(1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        theta, phi = x[..., 0:1], x[..., 1:2]
        sin_linear_theta = torch.sin(self.linear1(theta))
        sin_linear_cos_phi = torch.sin(self.linear2(torch.cos(phi)))
        sin_linear_sin_phi = torch.sin(self.linear3(torch.sin(phi)))

        ftr_out = sin_linear_theta * sin_linear_cos_phi * sin_linear_sin_phi

        return ftr_out


class LayerStepper(nn.Module):

    def __init__(self, hidden_dim: int, code_dim: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.code_dim = code_dim

        self.forward_linear = nn.Linear(hidden_dim, hidden_dim)
        self.code_linear = nn.Linear(code_dim, hidden_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:

        z_next = self.forward_linear(z)
        amplitude_shift = self.code_linear(a)

        z = z_next + amplitude_shift

        return z


class SINR2(nn.Module):

    def __init__(self,
                 depth: int = 5,
                 hidden_dim: int = 256,
                 code_dim: int = 400,
                 **kwargs) -> None:
        super().__init__()

        self.depth = depth

        self.ftrs = nn.ModuleList(
            [
                SphericalFilter(hidden_dim=hidden_dim)
                for layer_idx in range(depth + 1)
            ]
        )
        self.layer_steppers = nn.ModuleList(
            [
                LayerStepper(hidden_dim=hidden_dim,
                             code_dim=code_dim)
                for layer_idx in range(1, depth + 1)
            ]
        )
        self.out_linears = nn.ModuleList(
            [
                nn.Linear(hidden_dim, 1)
                for layer_idx in range(depth + 1)
            ]
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:

        z = self.ftrs[0](x)
        out = self.out_linears[0](z)

        for layer_idx in range(self.depth):
            z = self.layer_steppers[layer_idx](z, a)
            z = z * self.ftrs[layer_idx + 1](x)
            out = out + self.out_linears[layer_idx + 1](z)

        return out[..., 0], x


if __name__ == '__main__':

    inr = SINR2().float()
    print(inr)

    x = torch.rand(8, 2, dtype=torch.float32)  # theta: [0, 2pi], phi: [0, pi]
    x[:, 0] *= 2 * math.pi
    x[:, 1] *= math.pi
    a = torch.rand(8, 400)

    print(inr(x, a))
