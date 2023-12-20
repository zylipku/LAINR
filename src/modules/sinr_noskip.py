import math

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from scipy.special import sph_harm


class SphericalFilter(nn.Module):

    def __init__(self, max_freq: int, hidden_dim: int, shift: int) -> None:
        super().__init__()

        self.max_freq = max_freq
        self.hidden_dim = hidden_dim
        self.shift = shift
        self.linear = nn.Linear(max_freq * 2 + 1, hidden_dim)

        self.cached_x1: torch.Tensor = None
        self.cached_x2: torch.Tensor = None
        self.cached_Yreal1: Dict[Tuple[int, int], torch.Tensor] = {}
        self.cached_Yreal2: Dict[Tuple[int, int], torch.Tensor] = {}

    def get_Yreal(self, x: torch.Tensor, m: int, l: int) -> torch.Tensor:

        if self.cached_x1 is None:  # x1 not initialized, cached on x1
            self.cached_x1 = x.clone()
            self.cached_Yreal1 = {}
            self.cached_Yreal1[(m, l)] = self.Yreal(x, m, l)
            return self.cached_Yreal1[(m, l)]

        elif self.cached_x1.shape != x.shape or not torch.allclose(x, self.cached_x1):  # unmatch x1
            if self.cached_x2 is None:  # x2 not initialized, cached on x2
                self.cached_x2 = x.clone()
                self.cached_Yreal2 = {}
                self.cached_Yreal2[(m, l)] = self.Yreal(x, m, l)
                return self.cached_Yreal2[(m, l)]
            elif self.cached_x2.shape != x.shape or not torch.allclose(x, self.cached_x1):  # unmatch x2
                self.cached_x1 = x.clone()  # overwrite x1
                self.cached_Yreal1 = {}
                self.cached_Yreal1[(m, l)] = self.Yreal(x, m, l)
                return self.cached_Yreal1[(m, l)]
            else:  # match x2
                if (m, l) not in self.cached_Yreal2:  # not cached on x2
                    self.cached_Yreal2[(m, l)] = self.Yreal(x, m, l)
                return self.cached_Yreal2[(m, l)]
        else:  # match x1
            if (m, l) not in self.cached_Yreal1:  # not cached on x1
                self.cached_Yreal1[(m, l)] = self.Yreal(x, m, l)
            return self.cached_Yreal1[(m, l)]

    def Yreal(self, x: torch.Tensor, m: int, l: int) -> torch.Tensor:

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

        Ys = [self.get_Yreal(x, m, abs(m) + self.shift)
              for m in range(-self.max_freq, self.max_freq + 1)]  # 2 * max_freq + 1
        Y_stack = torch.stack(Ys, dim=-1)
        ftr_out = self.linear(Y_stack)

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


class SINRNoSkip(nn.Module):

    def __init__(self,
                 depth: int = 5,
                 max_freq: int = 4,
                 hidden_dim: int = 128,
                 code_dim: int = 400,
                 **kwargs) -> None:
        super().__init__()

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
                LayerStepper(hidden_dim=hidden_dim,
                             code_dim=code_dim)
                for layer_idx in range(1, depth + 1)
            ]
        )
        self.out_linears = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:

        z = self.ftrs[0](x)

        for layer_idx in range(self.depth):
            z = self.layer_steppers[layer_idx](z, a)
            z = z * self.ftrs[layer_idx + 1](x)

        out = self.out_linears(z)

        return out[..., 0], x


if __name__ == '__main__':

    inr = SINRNoSkip().float()
    print(inr)

    x = torch.rand(8, 2, dtype=torch.float32)  # theta: [0, 2pi], phi: [0, pi]
    x[:, 0] *= 2 * math.pi
    x[:, 1] *= math.pi
    a = torch.rand(8, 400)

    print(inr(x, a))
