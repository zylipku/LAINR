import os

from typing import Tuple, Dict
import logging

import torch

import numpy as np

from .la_dataset import LADataset

ROOT_PATH = 'data/ERA5'


class ERA5(LADataset):

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

    train_year = list(range(2009, 2017))
    test_year = [2017, 2018]

    valid_train_timesteps = (0, 8760)
    valid_test_timesteps = (0, 8760)

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 read_cache: bool = False,
                 normalize: bool = True,
                 height_std_mean: Tuple[torch.Tensor, torch.Tensor] = None,
                 vorticity_std_mean: Tuple[torch.Tensor, torch.Tensor] = None,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train':
            self.traj_ids = [str(year) for year in self.train_year]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_train_data.pt')
            window_width = 10

        elif group == 'test':
            self.traj_ids = [str(year) for year in self.test_year]
            self.valid_timesteps = self.valid_test_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_train_data.pt')
            if normalize and (height_std_mean is None or vorticity_std_mean is None):
                raise ValueError('height_std_mean and vorticity_std_mean must be provided for test dataset')

            window_width = 20
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        self.phi_theta_path = os.path.join(ROOT_PATH, 'phi_theta.pt')

        self.tt = torch.arange(0, window_width).float()

        self.ntrajs = len(self.traj_ids)

        self.valid_timeslice = slice(*self.valid_timesteps)
        self.valid_timelength = self.valid_timesteps[1] - self.valid_timesteps[0]

        self.window_width = window_width

        self.nwindows_per_traj = self.valid_timelength // self.window_width
        logger.info(f'Set nwindows as the maximum: valid_timelength // window_width = {self.nwindows_per_traj}')

        self.read_cache = read_cache
        self.data, self.phi, self.theta = self.load_traj_data()

        self.normalize = normalize

        # normalize data
        if self.normalize:

            z500 = self.data[..., 0]
            t850 = self.data[..., 1]

            if height_std_mean is None:
                self.height_std, self.height_mean = torch.std_mean(z500)
            else:
                self.height_std, self.height_mean = height_std_mean

            if vorticity_std_mean is None:
                self.vorticity_std, self.vorticity_mean = torch.std_mean(t850)
            else:
                self.vorticity_std, self.vorticity_mean = vorticity_std_mean

            z500 = (z500 - self.height_mean) / self.height_std
            t850 = (t850 - self.vorticity_mean) / self.vorticity_std
            self.data = torch.stack([z500, t850], dim=-1)

            self.logger.info(f'data have been normalized: z500, t850')
            self.logger.info(f'z500_mean={self.height_mean}, z500_std={self.height_std}')
            self.logger.info(f't850_mean={self.vorticity_mean}, t850_std={self.vorticity_std}')

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

        # if normalize:
        #     self.normalize()

        self.logger.info(f'data.shape={self.data.shape}, coords.shape={self.coords.shape}')
        # (18, 400, 128, 64, 2), (4, 128, 64, 3)
        # (3, 400, 128, 64, 2), (4, 128, 64, 3)

    def load_traj_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data = torch.empty(self.ntrajs, self.valid_timelength, 128, 64, 2)

        for idx, year in enumerate(self.traj_ids):
            z500 = np.load(os.path.join(ROOT_PATH, f'Z500_{year}_128x64_2.8125deg.npy'))
            t850 = np.load(os.path.join(ROOT_PATH, f'T850_{year}_128x64_2.8125deg.npy'))
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

    # def normalize(self) -> None:

    #     # self.z500_std, self.z500_mean = torch.std_mean(self.data[..., 0])
    #     # self.t850_std, self.t850_mean = torch.std_mean(self.data[..., 1])

    #     self.z500_std, self.z500_mean = 3361., 54166.,
    #     self.t850_std, self.t850_mean = 15.5, 275.

    #     self.data[..., 0] = (self.data[..., 0] - self.z500_mean) / self.z500_std
    #     self.data[..., 1] = (self.data[..., 1] - self.t850_mean) / self.t850_std

    #     self.logger.info(f'z500: mean={self.z500_mean}, std={self.z500_std}')
    #     self.logger.info(f't850: mean={self.t850_mean}, std={self.t850_std}')

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        traj_idx = idx // self.nwindows_per_traj
        win_idx = idx % self.nwindows_per_traj

        start = win_idx * self.window_width
        timestep_slice = slice(start, start + self.window_width)

        data = self.data[traj_idx, timestep_slice].float()

        return {
            'data': data,
            'tt': self.tt,
            'traj': traj_idx,
            'idxs': idx,
            'coords': self.coords,
            'coords_ang': self.coords_ang,
        }
