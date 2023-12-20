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
        super().__init__()

        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}

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

    The network structure is as follows:
    input: x (bs, ..., state_dim=1, in_dim)
    y1 = filter0(x) * (B0 @ code) (bs, ..., state_dim, hidden_dim)
    y2 = filter1(x) * (A1 @ y1 + B1 @ code) (bs, ..., state_dim, hidden_dim)
    y3 = filter2(x) * (A2 @ y2 + B2 @ code) (bs, ..., state_dim, hidden_dim)
    ...
    yn = filtern(x) * (An @ yn-1 + Bn @ code) (bs, ..., state_dim, hidden_dim)
    z = output_linear(yn) (bs, ..., state_dim, out_dim=1)

    """
    filters: Iterable[nn.Module]

    def __init__(self,
                 in_dim: int,
                 code_dim: int,
                 hidden_dim: int,
                 nlayers: int) -> None:
        super().__init__()

        self.bilinears = nn.ModuleList(
            [CodeBilinear(in_dim, code_dim, hidden_dim)] +
            [CodeBilinear(hidden_dim, code_dim, hidden_dim) for _ in range(int(nlayers))]
        )
        out_dim = 1
        self.output_linear = nn.Linear(hidden_dim, out_dim)
        return

    def forward(self, x, code):

        out = x * 0.

        for ftr, bilinear in zip(self.filters, self.bilinears):
            out = ftr(x) * bilinear(out, code)

        out = self.output_linear(out)[..., 0]

        return out, x


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

    def __init__(self, in_dim, code_dim, hidden_dim, nlayers=3, input_scale=256.0, **kwargs):
        super().__init__(in_dim, code_dim, hidden_dim, nlayers)

        self.filters = nn.ModuleList(
            [FourierLayer(in_dim, hidden_dim // 2, input_scale / math.sqrt(nlayers + 1))
             for _ in range(nlayers + 1)])

    def get_filters_weight(self):

        return torch.cat([ftr.weight for ftr in self.filters], dim=0)


class MyDINoINR(nn.Module):
    '''
    Inputs:
    coord: (bs, nsteps(expanded), *domain_shape, state_dim(expanded), coord_dim)
    code: (bs, nsteps, *domain_shape(expanded), state_dim, code_dim)
    '''
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_NLAYERS = 6

    def __init__(self,
                 coord_channels: int,
                 code_dim: int,
                 state_channels: int,
                 **inr_kwargs) -> None:
        super().__init__()

        self.coord_dim = coord_channels
        self.state_dim = state_channels
        self.code_dim = code_dim

        # INR configurations
        self.hidden_dim = inr_kwargs.get('hidden_dim', self.DEFAULT_HIDDEN_DIM)
        self.nlayers = inr_kwargs.get('nlayers', self.DEFAULT_NLAYERS)
        self.net = FourierNet(in_dim=self.coord_dim,
                              code_dim=self.code_dim,
                              hidden_dim=self.hidden_dim,
                              nlayers=self.nlayers,
                              input_scale=64.)

    def forward(self, coord, codes):

        # coord: (bs, nsteps, h, w, state_dim=1, coord_dim)
        # code:  (bs, nsteps, h, w, state_dim, code_dim)
        # (bs, nsteps, h, w, state_dim, 1(squeezed))

        return self.net(coord, codes)


class MyDINoDyn(nn.Module):

    def __init__(self,
                 state_channels: int,
                 code_dim: int,
                 hidden_dim: int,
                 **kwargs):
        super().__init__()

        input_dim = code_dim * state_channels
        self.net = MLP([input_dim,] + [hidden_dim,] * 3 + [input_dim,], act_name='swish')

    def forward(self, t, u):
        return self.net(u)
