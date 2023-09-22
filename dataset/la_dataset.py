import logging
from typing import Dict

import numpy as np
import torch

from torch.utils.data import Dataset


class LADataset(Dataset):

    data: torch.Tensor  # shape=(ntrajs, Nsteps, *state_size, state_channels)
    coords: torch.Tensor  # shape=(*state_size, coord_dim)
    coords_ang: torch.Tensor  # shape=(*state_size, coord_dim)
    tt: torch.Tensor  # shape=(Nsteps,)

    ntrajs: int
    window_width: int
    train_width: int
    nwindows_per_traj: int

    def __init__(self, logger: logging.Logger, group: str) -> None:
        super().__init__()

        self.logger = logger

        assert group in ['train', 'test', 'train_eval']
        self.group = group

    @property
    def nwindows(self) -> int:
        return self.ntrajs * self.nwindows_per_traj

    def __len__(self) -> int:
        return self.nwindows

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        traj_idx = idx // self.nwindows_per_traj
        win_idx = idx % self.nwindows_per_traj

        start = win_idx * self.window_width
        if self.group == 'train':
            timestep_slice = slice(start, start + self.train_width)
        else:
            timestep_slice = slice(start, start + self.window_width)

        data = self.data[traj_idx, timestep_slice].float()

        return {
            'data': data,
            'tt': self.tt,
            'traj': traj_idx,
            'idxs': idx,
            'coords': self.coords,
        }


class LADatasetAug(Dataset):

    data: torch.Tensor  # shape=(ntrajs, Nsteps, *state_size, state_channels)
    coords: torch.Tensor  # shape=(*state_size, coord_dim)
    tt: torch.Tensor  # shape=(Nsteps,)

    ntrajs: int
    window_width: int
    train_width: int
    nwindows_per_traj: int

    def __init__(self, logger: logging.Logger, group: str) -> None:
        super().__init__()

        self.logger = logger

        assert group in ['train', 'test', 'train_eval']
        self.group = group

    @property
    def nwindows(self) -> int:
        return self.ntrajs * self.nwindows_per_traj

    def __len__(self) -> int:
        return self.nwindows * 4

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        traj_idx = idx // (self.nwindows_per_traj * 4)
        win_idx = (idx % (self.nwindows_per_traj * 4)) // 4
        shift = idx % 4

        start = win_idx * self.window_width
        if self.group == 'train':
            timestep_slice = slice(start, start + self.train_width)
        else:
            timestep_slice = slice(start, start + self.window_width)

        data = self.data[traj_idx, timestep_slice].float()
        data = torch.roll(data, shift=shift * 32, dims=2)  # 128 / 4 = 32

        return {
            'data': data,
            'tt': self.tt,
            'traj': traj_idx,
            'idxs': idx,
            'coords': self.coords,
        }
