import os

import logging

import numpy as np
import torch

from .la_dataset import LADataset


class ShallowWater4(LADataset):

    ROOT_PATH = '/home/lizhuoyuan/datasets/shallow_water'

    '''
    ROOT/
        traj_1_height.npy # shape=(600, 128, 64)
        traj_1_vorticity.npy # shape=(600, 128, 64)
        traj_1_phi.npy
        traj_1_theta.npy
        ...
        traj_50_height.npy
        traj_50_vorticity.npy
        traj_50_phi.npy
        traj_50_theta.npy
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_ids = list(range(1, 10)) + list(range(11, 20)) +\
        list(range(21, 30)) + list(range(31, 40)) + list(range(41, 50))
    test_ids = [10, 20, 30, 40, 50]

    height_mean = -3.e-5
    height_std = 8.e-5

    vorticity_mean = 0.
    vorticity_std = 7.e-2

    valid_train_timesteps = (200, 400)
    valid_test_timesteps = (200, 400)

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 window_width: int = 20,
                 train_width: int = 10,
                 read_cache=True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train' or group == 'train_eval':
            self.traj_ids = [str(umax) for umax in self.train_ids]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(self.ROOT_PATH, 'cached_train_data.pt')
        elif group == 'test':
            self.traj_ids = [str(umax) for umax in self.test_ids]
            self.valid_timesteps = self.valid_test_timesteps
            self.cached_data_path = os.path.join(self.ROOT_PATH, 'cached_test_data.pt')
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        if group == 'train':
            self.tt = torch.arange(0, train_width).float()
        else:
            self.tt = torch.arange(0, window_width).float()

        self.ntrajs = len(self.traj_ids)

        self.valid_timeslice = slice(*self.valid_timesteps)
        self.valid_timelength = self.valid_timesteps[1] - self.valid_timesteps[0]

        self.window_width = window_width
        self.train_width = train_width

        self.nwindows_per_traj = self.valid_timelength // self.window_width
        logger.info(f'Set nwindows as the maximum: valid_timelength // window_width = {self.nwindows_per_traj}')

        self.read_cache = read_cache

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

        self.logger.info(f'data.shape={self.data.shape}, coords.shape={self.coords.shape}')
        # (18, 400, 128, 64, 2), (4, 128, 64, 3)
        # (3, 400, 128, 64, 2), (4, 128, 64, 3)

    def load_traj_data(self):

        try:
            assert self.read_cache

            cached_data = torch.load(self.cached_data_path)
            data = cached_data['data']
            phi = cached_data['phi']
            theta = cached_data['theta']

        except:

            data = torch.empty(self.ntrajs, self.valid_timelength, 128, 64, 2)

            for idx, traj_id in enumerate(self.traj_ids):

                height = np.load(os.path.join(self.ROOT_PATH, f'traj_{traj_id}_height.npy'))
                vorticity = np.load(os.path.join(self.ROOT_PATH, f'traj_{traj_id}_vorticity.npy'))

                # truncate to valid time slice
                height = height[self.valid_timeslice]
                vorticity = vorticity[self.valid_timeslice]

                # normalize
                height = (height - self.height_mean) / self.height_std
                vorticity = (vorticity - self.vorticity_mean) / self.vorticity_std

                data[idx] = torch.stack([
                    torch.from_numpy(height),
                    torch.from_numpy(vorticity),
                ], dim=-1)  # (128, 64, 2)

                if idx == 0:
                    phi = np.load(os.path.join(self.ROOT_PATH, f'traj_{traj_id}_phi.npy'))
                    theta = np.load(os.path.join(self.ROOT_PATH, f'traj_{traj_id}_theta.npy'))
                    phi = torch.from_numpy(phi)
                    theta = torch.from_numpy(theta)

            torch.save(
                {
                    'data': data,
                    'phi': phi,
                    'theta': theta,
                },
                self.cached_data_path
            )

        self.logger.info(f'Successfully loaded data for group: {self.group}')

        return data, phi, theta