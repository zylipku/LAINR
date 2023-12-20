import logging
from typing import *

import numpy as np
import torch

from torch.utils.data import Dataset


class PretrainDataset(Dataset):

    trajs: torch.Tensor

    def __init__(self, trajs: torch.Tensor,
                 coords: Dict[str, torch.Tensor] = dict(),
                 summary_info: str = None) -> None:
        super().__init__()

        self.trajs = trajs
        ntrajs, Nsteps, *state_size, state_channels = trajs.shape
        self.ntrajs = ntrajs
        self.Nsteps = Nsteps

        self.coords = coords
        self.summary_info = summary_info

    def __len__(self):
        return self.ntrajs * self.Nsteps

    def __getitem__(self, idx):
        traj_idx = idx // self.Nsteps
        step_idx = idx % self.Nsteps
        return {
            'snapshot': self.trajs[traj_idx, step_idx],
            'idx': idx,
        } | self.coords

    def get_summary(self):
        return self.summary_info


class DictDataset(Dataset):

    tensor_dict: Dict[str, torch.Tensor]

    def __init__(self, tensor_dict: Dict[str, torch.Tensor],
                 coords: Dict[str, torch.Tensor] = dict(), summary_info: str = None) -> None:
        super().__init__()

        key0 = list(tensor_dict.keys())[0]
        assert all([tensor_dict[key0].shape[0] == tensor_dict[k].shape[0] for k in tensor_dict.keys()])
        self.tensor_dict = tensor_dict

        self.coords = coords
        self.summary_info = summary_info

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.tensor_dict.items()} | self.coords

    def __len__(self):
        return self.tensor_dict[list(self.tensor_dict.keys())[0]].shape[0]

    def get_summary(self):
        return self.summary_info


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
