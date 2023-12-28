import math

import time

from typing import *

import torch
from torch import nn
from scipy.special import sph_harm

import hashlib


class SphericalFilter(nn.Module):

    def __init__(self, max_freq: int, hidden_dim: int, shift: int) -> None:
        super().__init__()

        self.max_freq = max_freq
        self.hidden_dim = hidden_dim
        self.shift = shift
        self.linear = nn.Linear(max_freq * 2 + 1, hidden_dim)

        self.cache: List[Tuple[torch.Tensor, Dict[Tuple[int, int], torch.Tensor]]] = []

    def get_Yreal(self, x: torch.Tensor, m: int, l: int) -> torch.Tensor:

        for cache_x, cache_value in self.cache:
            if x.shape == cache_x.shape and torch.allclose(x, cache_x):
                if (m, l) not in cache_value:
                    cache_value[(m, l)] = self.Yreal(x, m, l)
                return cache_value[(m, l)]

        self.cache.append((x.detach().clone(), {(m, l): self.Yreal(x, m, l)}))
        return self.cache[-1][1][(m, l)]

    def Yreal(self, x: torch.Tensor, m: int, l: int) -> torch.Tensor:
        '''
        x.shape=(..., h, w, 1, 2)
        output.shape=(..., h, w, 1)
        '''

        x_np = x.detach().cpu().numpy()

        Y_ml = sph_harm(m, l, x_np[..., 0], x_np[..., 1])
        Y_ml_ = sph_harm(-m, l, x_np[..., 0], x_np[..., 1])

        if m == 0:
            out = Y_ml
        elif m > 0:
            if m % 2 == 0:
                out = (Y_ml_ + Y_ml) / math.sqrt(2)
            else:
                out = (Y_ml_ - Y_ml) / math.sqrt(2)
        else:
            if m % 2 == 0:
                out = 1j * (Y_ml - Y_ml_) / math.sqrt(2)
            else:
                out = 1j * (Y_ml + Y_ml_) / math.sqrt(2)

        out = torch.from_numpy(out.real).to(x)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): coord_latlon, shape=(..., h, w, 1, 2)

        Returns:
            torch.Tensor: filter output, shape=(..., h, w, 1, hidden_dim)
        '''

        Ys = [self.get_Yreal(x, m, abs(m) + self.shift)
              for m in range(-self.max_freq, self.max_freq + 1)]  # 2 * max_freq + 1

        Y_stack = torch.stack(Ys, dim=-1)  # (..., h, w, 1, 2 * max_freq + 1)
        ftr_out = self.linear(Y_stack)  # (..., h, w, 1, hidden_dim)

        return ftr_out


class LayerStepper(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, code_dim: int) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim

        self.forward_linear = nn.Linear(hidden_dim, hidden_dim)
        self.code_linears = nn.ModuleList([nn.Linear(code_dim, hidden_dim) for _ in range(state_dim)])

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            z (torch.Tensor): coord_latlon, shape=(..., h, w, 1, hidden_dim)
            a (torch.Tensor): latent_code, shape=(..., 1, 1, [state_dim=2], [code_dim=200])

        Returns:
            output with amplitude shift, shape=(..., h, w, [state_dim=2], hidden_dim)
        '''

        z_next = self.forward_linear(z)
        amplitude_shift = torch.cat([linear(a1) for a1, linear in zip(
            torch.split(a, 1, dim=-2), self.code_linears)], dim=-2)

        z = z_next + amplitude_shift

        return z


class SINRv11(nn.Module):

    def __init__(self,
                 state_dim: int,
                 depth: int = 5,
                 max_freq: int = 4,
                 hidden_dim: int = 128,
                 code_dim: int = 400,
                 **kwargs) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.depth = depth
        self.max_freq = max_freq

        self.ftrs = nn.ModuleList(
            [
                SphericalFilter(max_freq=max_freq,
                                hidden_dim=hidden_dim,
                                shift=layer_idx)
                for layer_idx in range(depth + 1)
            ]
        )
        self.layer_steppers = nn.ModuleList(
            [
                LayerStepper(state_dim=state_dim,
                             hidden_dim=hidden_dim,
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
        '''

        Args:
            x (torch.Tensor): coord_latlon, shape=(..., h, w, 1, 2)
            a (torch.Tensor): latent_code, shape=(..., 1, 1, [state_dim=2], [code_dim=200])

        Returns:
            INR output, shape=(..., h, w, 1)
        '''

        z = self.ftrs[0](x)  # (..., h, w, 1, hidden_dim)
        out = self.out_linears[0](z)  # (..., h, w, 1, hidden_dim) -> (..., h, w, 1, 1)

        for layer_idx in range(self.depth):

            z = self.layer_steppers[layer_idx](z, a)
            # (..., h, w, [state_dim=2], hidden_dim)
            z = z * self.ftrs[layer_idx + 1](x)
            # (..., h, w, [state_dim=2], hidden_dim) * (..., h, w, 1, hidden_dim)
            out = out + self.out_linears[layer_idx + 1](z)
            # (..., h, w, [state_dim=2], hidden_dim) -> (..., h, w, [state_dim=2], 1)

        return out[..., 0]  # (..., h, w, [state_dim=2])


if __name__ == '__main__':

    inr = SINRv11().float()
    print(inr)

    x = torch.rand(8, 2, dtype=torch.float32)  # theta: [0, 2pi], phi: [0, pi]
    x[:, 0] *= 2 * math.pi
    x[:, 1] *= math.pi
    a = torch.rand(8, 400)

    print(inr(x, a))
