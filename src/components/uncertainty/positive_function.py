'''
implement the positive functions
'''
import torch
from torch import nn


class PositiveFunc:

    def __init__(self) -> None:
        pass

    @classmethod
    def get_func(cls, func_name):
        if func_name == 'exp':
            return ExpFunc()
        elif func_name == 'softplus':
            return SoftplusFunc()
        else:
            raise ValueError(f'func_name {func_name} is not supported')


class ExpFunc(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class SoftplusFunc(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(1 + torch.exp(x))
