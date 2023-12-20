import os

from typing import Tuple
import logging

import torch

import numpy as np

from .la_dataset import LADataset


class ERA5(LADataset):

    ROOT_PATH = '/home/lizhuoyuan/datasets/ERA5'

    '''
    ROOT/
        Z500_1979_128x64_2.8125deg.npy # shape=(8760, 64, 128)
        ...
        Z500_2019_128x64_2.8125deg.npy # shape=(8760, 64, 128)
        
        T850_1979_128x64_2.8125deg.npy # shape=(8760, 64, 128)
        ...
        T850_1979_128x64_2.8125deg.npy # shape=(8760, 64, 128)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''

    train_year = list(range(1979, 2017))
    test_year = [2017, 2018]

    valid_train_timesteps = (0, 8760)
    valid_test_timesteps = (0, 8760)

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 nwindows_per_traj: int = None,
                 window_width: int = 20,
                 train_width: int = 10,
                 normalize: bool = True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train' or group == 'train_eval':
            self.traj_ids = [str(year) for year in self.train_year]
            self.valid_timesteps = self.valid_train_timesteps
        elif group == 'test':
            self.traj_ids = [str(year) for year in self.test_year]
            self.valid_timesteps = self.valid_test_timesteps
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        self.phi_theta_path = os.path.join(self.ROOT_PATH, 'phi_theta.pt')

        if group == 'train':
            self.tt = torch.arange(0, train_width).float()
        else:
            self.tt = torch.arange(0, window_width).float()

        self.ntrajs = len(self.traj_ids)

        self.valid_timeslice = slice(*self.valid_timesteps)
        self.valid_timelength = self.valid_timesteps[1] - self.valid_timesteps[0]

        if window_width * nwindows_per_traj > self.valid_timelength:
            logger.warning(
                f'The dataset does not contain enough windows per trajectory.\n' +
                f'{self.valid_timelength} < {window_width} * {nwindows_per_traj} (width * nwindows)')
            logger.warning(
                f'Set nwindows as the maximum: valid_timelength // window_width = {self.valid_timelength // window_width}')
            nwindows_per_traj = self.valid_timelength // self.window_width

        self.nwindows_per_traj = nwindows_per_traj

        self.window_width = window_width
        self.train_width = train_width

        self.data, self.phi, self.theta = self.load_traj_data()

        # phi: [0, 2pi)
        # theta: (pi, 0)

        # spherical = torch.stack(torch.meshgrid(phi, theta,indexing='ij'), dim=-1)
        # phi_vert = spherical[..., 0]
        # theta_vert = spherical[..., 1]
        phi_vert, theta_vert = torch.meshgrid(self.phi, self.theta, indexing='ij')
        # phi_vert = [[phi[0], ..., phi[0]],
        #             ...,
        #             [phi[-1], ..., phi[-1]]]
        # theta_vert = [[theta[0], ..., theta[-1]],
        #                ...,
        #               [theta[0], ..., theta[-1]]]
        # spherical to cartesian
        r = 1.
        x = torch.cos(phi_vert) * torch.sin(theta_vert) * r  # x = cosϕsinθ
        y = torch.sin(phi_vert) * torch.sin(theta_vert) * r  # y = sinϕsinθ
        z = torch.cos(theta_vert) * r  # z = cosθ

        self.coords = torch.stack([x, y, z], dim=-1)
        self.coords_ang = torch.stack([phi_vert, theta_vert], dim=-1)
        self.coord_dim = self.coords.shape[-1]

        if normalize:
            self.normalize()

        self.logger.info(f'data.shape={self.data.shape}, coords.shape={self.coords.shape}')
        # (18, 400, 128, 64, 2), (4, 128, 64, 3)
        # (3, 400, 128, 64, 2), (4, 128, 64, 3)

    def load_traj_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data = torch.empty(self.ntrajs, self.valid_timelength, 128, 64, 2)

        for idx, year in enumerate(self.traj_ids):
            z500 = np.load(os.path.join(self.ROOT_PATH, f'Z500_{year}_128x64_2.8125deg.npy'))
            t850 = np.load(os.path.join(self.ROOT_PATH, f'T850_{year}_128x64_2.8125deg.npy'))
            data[idx] = torch.stack(
                [
                    torch.from_numpy(z500[self.valid_timeslice]).transpose(-2, -1),  # (8760, 128, 64)
                    torch.from_numpy(t850[self.valid_timeslice]).transpose(-2, -1),  # (8760, 128, 64)
                ],
                dim=-1
            )

        phi = torch.arange(0, 2 * torch.pi, 2 * torch.pi / 128)
        theta = torch.pi / 2 - torch.arange(-torch.pi / 2 + torch.pi / 128, torch.pi / 2, torch.pi / 64)

        # self.logger.info(f'number of NaNs: {torch.isnan(data).sum()}') # =0

        return data, phi, theta

    def normalize(self) -> None:

        self.z500_std, self.z500_mean = torch.std_mean(self.data[..., 0])
        self.t850_std, self.t850_mean = torch.std_mean(self.data[..., 1])

        self.data[..., 0] = (self.data[..., 0] - self.z500_mean) / self.z500_std
        self.data[..., 1] = (self.data[..., 1] - self.t850_mean) / self.t850_std

        self.logger.info(f'z500: mean={self.z500_mean}, std={self.z500_std}')
        self.logger.info(f't850: mean={self.t850_mean}, std={self.t850_std}')
