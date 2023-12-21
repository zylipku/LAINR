import logging
from typing import *

import numpy as np
import torch

from torch.utils.data import Dataset

from configs.pretrain.pretrain_conf_schema import DatasetConfig as DatasetConfigPT
from configs.finetune.finetune_conf_schema import DatasetConfig as DatasetConfigFT


class MyDataset(Dataset):

    trajs: torch.Tensor  # (ntrajs, Nsteps, *state_size, state_channels)

    def __init__(self, cfg: DatasetConfigPT | DatasetConfigFT,
                 trajs: torch.Tensor,
                 coords: Dict[str, torch.Tensor] = dict(),
                 summary_info: str = None) -> None:
        super().__init__()

        self.cfg = cfg
        self.trajs = trajs
        ntrajs, Nsteps, *state_size, state_channels = trajs.shape
        self.ntrajs = ntrajs
        self.Nsteps = Nsteps

        self.coords = coords
        self.summary_info = summary_info

    def get_summary(self):
        return self.summary_info


class PretrainDataset(MyDataset):

    def __init__(self, cfg: DatasetConfigPT,
                 trajs: torch.Tensor,
                 coords: Dict[str, torch.Tensor] = dict(),
                 summary_info: str = None) -> None:
        super().__init__(cfg, trajs, coords, summary_info)

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
        return (self.summary_info + ';\n' +
                f'{self.__len__()} batches with size' +
                str(self.trajs.shape[2:]))


class FineTuneDataset(MyDataset):

    def __init__(self, cfg: DatasetConfigFT,
                 trajs: torch.Tensor,
                 coords: Dict[str, torch.Tensor] = dict(),
                 summary_info: str = None) -> None:
        super().__init__(cfg, trajs, coords, summary_info)

        self.window_width = cfg.window_width

    def __len__(self):
        return self.ntrajs * (self.Nsteps - self.window_width)

    def __getitem__(self, idx):
        traj_idx = idx // (self.Nsteps - self.window_width)
        step_idx = idx % (self.Nsteps - self.window_width)
        snapshot_idx = step_idx + traj_idx * self.Nsteps
        return {
            'window': self.trajs[traj_idx, step_idx:step_idx + self.window_width],
            'idx': snapshot_idx,
        } | self.coords

    def get_summary(self):
        return (self.summary_info + ';\n' +
                f'{self.__len__()} batches with size' +
                str((self.window_width, *self.trajs.shape[2:])))
