import os

from typing import Tuple
import logging

import torch

import h5py

from .la_dataset import LADataset, LADatasetAug


class ShallowWater2(LADataset):

    ROOT_PATH = '/home/lizhuoyuan/datasets/ShallowWater2'

    '''
    ROOT/
        train_data.pt # shape=(45, 600, 128, 64, 2)
        test_data.pt # shape=(3, 600, 128, 64, 2)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_umax = [61, 62, 63, 64, 65, 66, 67, 68, 69,
                  71, 72, 73, 74, 75, 76, 77, 78, 79,]
    test_umax = [60, 70, 80]

    valid_train_timesteps = (200, 600)
    valid_test_timesteps = (200, 600)

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
            self.valid_timesteps = self.valid_train_timesteps
            self.data_path = os.path.join(self.ROOT_PATH, 'train_data.pt')
        elif group == 'test':
            self.valid_timesteps = self.valid_test_timesteps
            self.data_path = os.path.join(self.ROOT_PATH, 'test_data.pt')
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        self.phi_theta_path = os.path.join(self.ROOT_PATH, 'phi_theta.pt')

        if group == 'train':
            self.tt = torch.arange(0, train_width).float()
        else:
            self.tt = torch.arange(0, window_width).float()

        self.valid_timeslice = slice(*self.valid_timesteps)
        self.valid_timelength = self.valid_timesteps[1] - self.valid_timesteps[0]

        self.window_width = window_width
        self.train_width = train_width

        self.nwindows_per_traj = self.valid_timelength // self.window_width
        logger.info(f'Set nwindows as the maximum: valid_timelength // window_width = {self.nwindows_per_traj}')

        self.data, self.phi, self.theta = self.load_traj_data()

        self.ntrajs = self.data.shape[0]

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

        data = torch.load(self.data_path)
        phi_theta = torch.load(self.phi_theta_path)
        phi = phi_theta['phi']
        theta = phi_theta['theta']

        # reduce the resolution
        data = data[:, self.valid_timeslice, ::2, ::2, :]
        phi = phi[::2]
        theta = theta[::2]

        self.logger.info(f'Successfully loaded data for group: {self.group}')

        return data, phi, theta

    def normalize(self) -> None:

        self.height_std, self.height_mean = torch.std_mean(self.data[..., 0])
        self.vorticity_std, self.vorticity_mean = torch.std_mean(self.data[..., 1])

        self.data[..., 0] = (self.data[..., 0] - self.height_mean) / self.height_std
        self.data[..., 1] = (self.data[..., 1] - self.vorticity_mean) / self.vorticity_std

        self.logger.info(f'height: mean={self.height_mean}, std={self.height_std}')
        self.logger.info(f'vorticity: mean={self.vorticity_mean}, std={self.vorticity_std}')


class ShallowWater2Aug(LADatasetAug):

    ROOT_PATH = '/home/lizhuoyuan/datasets/ShallowWater2'

    '''
    ROOT/
        train_data.pt # shape=(45, 600, 128, 64, 2)
        test_data.pt # shape=(3, 600, 128, 64, 2)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_umax = [61, 62, 63, 64, 65, 66, 67, 68, 69,
                  71, 72, 73, 74, 75, 76, 77, 78, 79,]
    test_umax = [60, 70, 80]

    valid_train_timesteps = (200, 600)
    valid_test_timesteps = (200, 600)

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
            self.valid_timesteps = self.valid_train_timesteps
            self.data_path = os.path.join(self.ROOT_PATH, 'train_data.pt')
        elif group == 'test':
            self.valid_timesteps = self.valid_test_timesteps
            self.data_path = os.path.join(self.ROOT_PATH, 'test_data.pt')
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        self.phi_theta_path = os.path.join(self.ROOT_PATH, 'phi_theta.pt')

        if group == 'train':
            self.tt = torch.arange(0, train_width).float()
        else:
            self.tt = torch.arange(0, window_width).float()

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

        self.ntrajs = self.data.shape[0]

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

        data = torch.load(self.data_path)
        phi_theta = torch.load(self.phi_theta_path)
        phi = phi_theta['phi']
        theta = phi_theta['theta']

        # reduce the resolution
        data = data[:, self.valid_timeslice, ::2, ::2, :]
        phi = phi[::2]
        theta = theta[::2]

        self.logger.info(f'Successfully loaded data for group: {self.group}')

        return data, phi, theta

    def normalize(self) -> None:

        self.height_std, self.height_mean = torch.std_mean(self.data[..., 0])
        self.vorticity_std, self.vorticity_mean = torch.std_mean(self.data[..., 1])

        self.data[..., 0] = (self.data[..., 0] - self.height_mean) / self.height_std
        self.data[..., 1] = (self.data[..., 1] - self.vorticity_mean) / self.vorticity_std

        self.logger.info(f'height: mean={self.height_mean}, std={self.height_std}')
        self.logger.info(f'vorticity: mean={self.vorticity_mean}, std={self.vorticity_std}')
