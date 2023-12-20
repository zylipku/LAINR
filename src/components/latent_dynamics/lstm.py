from typing import Tuple
import logging

import torch
from torch import nn

from .abstract_ld import LatentDynamics


class LSTM(LatentDynamics):

    def __init__(self,
                 logger: logging.Logger,
                 ndim: int,
                 hidden_dim: int = 1024,
                 nlayers: int = 3,
                 **kwargs) -> None:
        super().__init__(logger=logger, ndim=ndim, **kwargs)

        self.hidden_dim = hidden_dim
        self.nlayers = nlayers

        proj_size = ndim if hidden_dim > ndim else 0

        self.lstm = nn.LSTM(input_size=ndim,
                            hidden_size=hidden_dim,
                            num_layers=nlayers,
                            batch_first=True,
                            proj_size=proj_size)  # auto-regressive

    def forward(self, x: torch.Tensor, h0_c0: Tuple[torch.Tensor, torch.Tensor] = None) -> torch.Tensor:
        '''
        x: (bs, seq_len, ndim) or (bs, ndim)
        '''

        if h0_c0 is None:
            out = self.lstm(x)
        else:
            out = self.lstm(x, h0_c0)

        return out
