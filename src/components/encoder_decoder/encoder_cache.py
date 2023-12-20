from typing import *

import torch
from torch import nn


class EncoderCache(nn.Module):

    def __init__(self, ncodes: int = None, shape: Tuple[int, Any] = None) -> None:
        super().__init__()
        if ncodes is None or shape is None:
            self.cache = None
        else:
            self.cache = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.zeros(shape))
                    for _ in range(ncodes)
                ]
            )  # (ncodes, *shape)

    def forward(self, seq_idxs: torch.Tensor, set_data: torch.Tensor = None):
        if self.cache is None:
            return None
        if set_data is not None:
            for i in range(len(seq_idxs)):
                self.cache[seq_idxs[i]] = set_data[i].detach().clone()
        return torch.stack([self.cache[idx] for idx in seq_idxs], dim=0)
