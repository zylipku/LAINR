from collections import OrderedDict

import torch
from torch import nn

from .utils import Act


class ReZeroBlock(nn.Module):

    def __init__(self, ndim: int, act_name='leaky') -> None:
        super().__init__()

        self.ndim = ndim
        self.act_name = act_name

        self.transform = nn.Linear(ndim, ndim)
        self.act = Act(act_name=act_name)
        self.alpha = nn.Parameter(torch.zeros(1, ndim))

    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        # scale x with self.alpha along 1st dimension (channel)

        return x * self.alpha

    def forward(self, x: torch.Tensor):

        # x.shape: (bs, ndim=40)

        y = self.transform(x)
        y = self.act(y)
        y = self.scaling(y)
        y = y + x  # skip connection

        return y


class ReZero(nn.Module):

    def __init__(self, ndim=32, nblocks=5, **kwargs) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            OrderedDict(
                [(f'Block_{k}', ReZeroBlock(ndim, 'leaky')) for k in range(nblocks - 1)]
                + [(f'Block_{nblocks - 1}', ReZeroBlock(ndim, 'identity'))]
            )
        )

    def forward(self, x: torch.Tensor):

        return self.seq(x)


if __name__ == '__main__':

    x = torch.randn(8, 32, 15, 7)
    model = ReZero()
    print(model)
    print(model(x).shape)
