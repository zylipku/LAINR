import torch
from torch import nn

from .utils import MLP


class SIREN(nn.Module):

    name = 'siren'

    default_mlp_list = [2, ] + [16, ] * 8 + [1]

    def __init__(self, mlp_list=None) -> None:
        super().__init__()

        self.mlp_list = self.default_mlp_list if mlp_list is None else mlp_list
        self.act_name = 'sin'

        self.mlp = MLP(self.mlp_list, self.act_name)

    def forward(self, x: torch.Tensor):

        return self.mlp(x)
