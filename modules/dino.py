'''
Adapted from https://github.com/mkirchmeyer/DINo/blob/main/network.py
'''

from typing import Iterable

import math

import torch
from torch import nn

from .utils import MLP


class CodeBilinear(nn.Module):
    '''
    Adapted from https://github.com/mkirchmeyer/DINo/blob/main/network.py
    '''
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, device=None, dtype=None) -> None:
        """
        linear1: x1 (..., in1_features) -> (..., out_features)
        linear2: x2 (..., in2_features) -> (..., out_features)
        outputs <- linear1(x1) + linear2(x2)
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weights1 = nn.Parameter(torch.empty(out_features, in1_features, **factory_kwargs))
        self.weights2 = nn.Parameter(torch.empty(out_features, in2_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in1_features)
        nn.init.kaiming_uniform_(self.weights1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights2, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # input1: b, ..., i
        # input2: b, ..., j
        # weights1: o, i
        # weights2: o, j
        # bias: o,

        y1 = nn.functional.linear(x1, self.weights1)
        y2 = nn.functional.linear(x2, self.weights2)

        return y1 + y2 + self.bias
        # aligned with respect to the last dimension

    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None)


class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """
    filters: Iterable[nn.Module]

    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers):
        super().__init__()
        self.first = 3
        self.bilinear = nn.ModuleList(
            [CodeBilinear(in_size, code_size, hidden_size)] +
            [CodeBilinear(hidden_size, code_size, hidden_size) for _ in range(int(n_layers))]
        )
        self.output_bilinear = nn.Linear(hidden_size, out_size)
        return

    def forward(self, x, code):

        out = self.filters[0](x) * self.bilinear[0](x * 0., code)

        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.bilinear[i](out, code)

        out: torch.Tensor = self.output_bilinear(out)

        return out[..., 0], x


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    """

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.cat(
            [torch.sin(nn.functional.linear(x, self.weight * self.weight_scale)),
             torch.cos(nn.functional.linear(x, self.weight * self.weight_scale))],
            dim=-1)


class FourierNet(MFNBase):
    """
    Taken from https://github.com/boschresearch/multiplicative-filter-networks
    """

    filters: Iterable[FourierLayer]

    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers=3, input_scale=256.0, **kwargs):
        super().__init__(in_size, hidden_size, code_size, out_size, n_layers)

        self.filters = nn.ModuleList(
            [FourierLayer(in_size, hidden_size // 2, input_scale / math.sqrt(n_layers + 1))
             for _ in range(n_layers + 1)])

    def get_filters_weight(self):
        weights = list()
        for ftr in self.filters:
            weights.append(ftr.weight)
        return torch.cat(weights)


class DINoINR(nn.Module):

    def __init__(self,
                 state_c: int,
                 hidden_c: int,
                 code_c: int,
                 coord_dim: int,
                 nlayers: int,
                 **kwargs) -> None:
        super().__init__()

        self.state_c = state_c
        self.hidden_c = hidden_c
        self.coord_dim = coord_dim
        self.out_dim = 1
        self.code_dim = code_c
        self.net = FourierNet(self.coord_dim, self.hidden_c, self.code_dim, self.out_dim, nlayers, input_scale=64)

    def forward(self, coord, codes=None):
        if codes is None:
            return self.net(coord)
        return self.net(coord, codes)


class Derivative(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        self.net = MLP([input_dim,] + [hidden_c,] * 3 + [input_dim,], act_name='swish')

    def forward(self, t, u):
        return self.net(u)
