from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class LatentAssimilation:

    name = 'LA_abstract_class'

    encoder: Callable[[torch.Tensor], torch.Tensor]
    decoder: Callable[[torch.Tensor], torch.Tensor]

    da_forecast: Callable[[torch.Tensor], torch.Tensor]
    da_analysis: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self) -> None:
        pass

    def train_one_epoch(self, train_loader: DataLoader,
                        optimizer: optim.Optimizer,
                        epoch: int,
                        **kwargs) -> None:
        raise NotImplementedError

    def validate(self, valid_loader: DataLoader,
                 epoch: int,
                 **kwargs) -> None:
        raise NotImplementedError

    def test(self, test_loader: DataLoader,
             epoch: int,
             **kwargs) -> None:
        raise NotImplementedError

    def train(self) -> None:

        raise NotImplementedError

    def assimilate(self, x0, yy_o: torch.Tensor, obs_tt_idx: torch.Tensor) -> None:

        raise NotImplementedError
