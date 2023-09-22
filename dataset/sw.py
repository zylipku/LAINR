'''
normalized DINo shallow water dataset
the time horizon has been extended to 10 days (240 hours)
the number of trajectories has been increased to 20
slightly harder case
steps from 360 to 600
'''

import os

import logging

from typing import Dict, Tuple

import numpy as np
import torch

from .la_dataset import LADataset

ROOT_PATH = 'data/sw'


class ShallowWater(LADataset):

    '''
    ROOT/
        traj_1_height.npy # shape=(600, 128, 64)
        traj_1_vorticity.npy # shape=(600, 128, 64)
        traj_1_phi.npy
        traj_1_theta.npy
        ...
        traj_20_height.npy
        traj_20_vorticity.npy
        traj_20_phi.npy
        traj_20_theta.npy

    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_ids = list(range(1, 19))
    test_ids = [19, 20]

    # height_mean = 0.
    # height_std = 1 / 3000.

    # vorticity_mean = 0.
    # vorticity_std = 1 / 2.

    valid_train_timesteps = (360, 600)
    valid_test_timesteps = (360, 600)

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 read_cache: bool = False,
                 normalize: bool = True,
                 height_std_mean: Tuple[torch.Tensor, torch.Tensor] = None,
                 vorticity_std_mean: Tuple[torch.Tensor, torch.Tensor] = None,
                 offgrid=False,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        read_cache = False
        self.offgrid = offgrid
        self.logger.info(f'offgrid={self.offgrid}')

        if group == 'train':
            self.traj_ids = [str(umax) for umax in self.train_ids]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_train_data.pt')
            window_width = 10
        elif group == 'test':
            self.traj_ids = [str(umax) for umax in self.test_ids]
            self.valid_timesteps = self.valid_test_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_test_data.pt')
            if normalize and (height_std_mean is None or vorticity_std_mean is None):
                raise ValueError('height_std_mean and vorticity_std_mean must be provided for test dataset')
            window_width = 20
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

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

            height = self.data[..., 0]
            vorticity = self.data[..., 1]

            if height_std_mean is None:
                self.height_std, self.height_mean = torch.std_mean(height)
            else:
                self.height_std, self.height_mean = height_std_mean

            if vorticity_std_mean is None:
                self.vorticity_std, self.vorticity_mean = torch.std_mean(vorticity)
            else:
                self.vorticity_std, self.vorticity_mean = vorticity_std_mean

            height = (height - self.height_mean) / self.height_std
            vorticity = (vorticity - self.vorticity_mean) / self.vorticity_std
            self.data = torch.stack([height, vorticity], dim=-1)

            self.logger.info(f'data have been normalized: height, vorticity')
            self.logger.info(f'height_mean={self.height_mean}, height_std={self.height_std}')
            self.logger.info(f'vorticity_mean={self.vorticity_mean}, vorticity_std={self.vorticity_std}')

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
            assert cached_data['data'].shape == (self.ntrajs, self.valid_timelength, 128, 64, 2)
            data = cached_data['data']
            phi = cached_data['phi']
            theta = cached_data['theta']

        except:

            data = torch.empty(self.ntrajs, self.valid_timelength, 128, 64, 2)

            for idx, traj_id in enumerate(self.traj_ids):

                if self.offgrid:
                    height = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_height_offgrid.npy'))
                    vorticity = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_vorticity_offgrid.npy'))
                else:
                    height = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_height.npy'))
                    vorticity = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_vorticity.npy'))

                # truncate to valid time slice
                height = height[self.valid_timeslice]
                vorticity = vorticity[self.valid_timeslice]

                # # normalize
                # height = (height - self.height_mean) / self.height_std
                # vorticity = (vorticity - self.vorticity_mean) / self.vorticity_std

                data[idx] = torch.stack([
                    torch.from_numpy(height),
                    torch.from_numpy(vorticity),
                ], dim=-1)  # (128, 64, 2)

                if idx == 0:
                    if self.offgrid:
                        phi = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_phi_offgrid.npy'))
                        theta = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_theta_offgrid.npy'))
                    else:
                        phi = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_phi.npy'))
                        theta = np.load(os.path.join(ROOT_PATH, f'traj_{traj_id}_theta.npy'))
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
