from collections import OrderedDict

import torch
from torch import nn

from .activation import Act


class MLP(nn.Module):

    def __init__(self, mlp_list: list, act_name='relu') -> None:
        super().__init__()

        self.act_name = act_name
        self.nlayers = len(mlp_list) - 1

        layers_list = []

        for layer_idx in range(self.nlayers - 1):

            layers_list.append(
                (f'layer_{layer_idx+1}', nn.Linear(mlp_list[layer_idx], mlp_list[layer_idx + 1]))
            )
            layers_list.append(
                (f'activation_{layer_idx+1}({act_name})', Act(act_name))
            )

        layers_list.append(
            (f'layer_{self.nlayers}', nn.Linear(mlp_list[-2], mlp_list[-1]))
        )

        self.seq = nn.Sequential(OrderedDict(layers_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.seq(x)
