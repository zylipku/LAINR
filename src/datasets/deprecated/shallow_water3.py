import os

from typing import Tuple
import logging

import torch

import h5py

from .la_dataset import LADataset, LADatasetAug


class ShallowWater3(LADataset):

    ROOT_PATH = '/home/lizhuoyuan/datasets/shallow_water/'

    '''
    ROOT/
        train_data.pt # shape=(45, 600, 128, 64, 2)
        test_data.pt # shape=(3, 600, 128, 64, 2)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_idx = list(range(1)) + list(range(11, 20)) + list(range(21, 30)) + list(range(31, 40)) + list(range(41, 50))
    test_idx = list(range(10, 20, 10))

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 window_width: int = 20,
                 train_width: int = 10,
                 normalize: bool = True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train' or group == 'train_eval':
            self.traj_idxs = self.train_idx
        elif group == 'test':
            self.traj_idxs = self.test_idx
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        if group == 'train':
            self.tt = torch.arange(0, train_width).float()
        else:
            self.tt = torch.arange(0, window_width).float()

        self.window_width = window_width
        self.train_width = train_width

        self.nwindows_per_traj = 401 // self.window_width
        logger.info(f'Set nwindows as the maximum: valid_timelength // window_width = {self.nwindows_per_traj}')

        self.data = self.load_traj_data()

        self.ntrajs = self.data.shape[0]

        self.coord_dim = self.coords.shape[-1]

        # if normalize:
        #     self.normalize()

        self.logger.info(f'data.shape={self.data.shape}, coords.shape={self.coords.shape}')
        # (18, 400, 128, 64, 2), (4, 128, 64, 3)
        # (3, 400, 128, 64, 2), (4, 128, 64, 3)

    def load_traj_data(self) -> torch.Tensor:

        data = torch.empty(len(self.traj_idxs), 401, 128, 64, 3)

        for k, idx in enumerate(self.traj_idxs):
            data_path = os.path.join(self.ROOT_PATH, f'traj_{idx}.pt')
            if not os.path.exists(data_path):
                raise ValueError(f'Cannot find data file: {data_path}')
            data[k] = torch.load(data_path)['data']
            if k == 0:
                self.coords = torch.load(data_path)['coords'].cpu()
                coords_ang = torch.load(data_path)['coords_ang'].cpu()
                phi = coords_ang[..., 1]  # [0, 2pi)
                theta = coords_ang[..., 0] + torch.pi / 2 - torch.pi / 128  # (pi/2, -pi/2) -> (pi, 0)
                self.coords_ang = torch.stack([phi, theta], dim=-1)

        self.logger.info(f'Successfully loaded data for group: {self.group}, {data.shape=}')

        return data

    def normalize(self) -> None:

        self.height_std, self.height_mean = torch.std_mean(self.data[..., 0])
        self.vorticity_std, self.vorticity_mean = torch.std_mean(self.data[..., 1])

        self.data[..., 0] = (self.data[..., 0] - self.height_mean) / self.height_std
        self.data[..., 1] = (self.data[..., 1] - self.vorticity_mean) / self.vorticity_std

        self.logger.info(f'height: mean={self.height_mean}, std={self.height_std}')
        self.logger.info(f'vorticity: mean={self.vorticity_mean}, std={self.vorticity_std}')
