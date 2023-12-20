import os

import logging

import torch

import h5py

from .la_dataset import LADataset

if os.path.exists('/public2/share/lizhuoyuan'):
    ROOT_PATH = '/public2/share/lizhuoyuan/ShallowWater'
else:
    ROOT_PATH = '/home/lizhuoyuan/datasets/ShallowWater'


class ShallowWater(LADataset):

    '''
    ROOT/
        snapshots_umax=60/
            snapshots_umax=60_s1
            snapshots_umax=60_s1.h5
        snapshots_umax=61/
            snapshots_umax=61_s1
            snapshots_umax=61_s1.h5
        ...
        snapshots_umax=80/
            snapshots_umax=80_s1
            snapshots_umax=80_s1.h5 # shape=(1001, 256, 128)

    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_umax = [61, 62, 63, 64, 65, 66, 67, 68, 69,
                  71, 72, 73, 74, 75, 76, 77, 78, 79,]
    test_umax = [60, 70, 80]

    valid_train_timesteps = (300, 400)
    valid_test_timesteps = (300, 400)

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 nwindows_per_traj: int = None,
                 window_width: int = 20,
                 train_width: int = 10,
                 read_cache=True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train' or group == 'train_eval':
            self.traj_ids = [str(umax) for umax in self.train_umax]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_train_data.pt')
        elif group == 'test':
            self.traj_ids = [str(umax) for umax in self.test_umax]
            self.valid_timesteps = self.valid_test_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_test_data.pt')
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

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
                f = h5py.File(
                    os.path.join(ROOT_PATH,
                                 f'snapshots_umax={traj_id}',
                                 f'snapshots_umax={traj_id}_s1.h5'),
                    mode='r'
                )
                data[idx] = torch.stack(
                    [
                        torch.from_numpy(f['tasks/height'][self.valid_timeslice, ::2, ::2]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][self.valid_timeslice, ::2, ::2] * 2),
                    ],
                    dim=-1
                )
                if idx == 0:
                    phi = torch.tensor(f['tasks/vorticity'].dims[1][0][:].ravel()[::2])
                    theta = torch.tensor(f['tasks/vorticity'].dims[2][0][:].ravel()[::2])

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


class ShallowWaterNormalized(LADataset):

    ROOT_PATH = '/public2/share/lizhuoyuan/ShallowWater'

    '''
    ROOT/
        snapshots_umax=60/
            snapshots_umax=60_s1
            snapshots_umax=60_s1.h5
        snapshots_umax=61/
            snapshots_umax=61_s1
            snapshots_umax=61_s1.h5
        ...
        snapshots_umax=80/
            snapshots_umax=80_s1
            snapshots_umax=80_s1.h5 # shape=(1001, 256, 128)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_umax = [61, 62, 63, 64, 65, 66, 67, 68, 69,
                  71, 72, 73, 74, 75, 76, 77, 78, 79,]
    test_umax = [60, 70, 80]

    valid_train_timesteps = (300, 400)
    valid_test_timesteps = (300, 400)

    height_mean = -3.e-5
    height_std = 8.e-5

    vorticity_mean = 0.
    vorticity_std = 7.e-2

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 nwindows_per_traj: int = None,
                 window_width: int = 20,
                 train_width: int = 10,
                 read_cache=True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train' or group == 'train_eval':
            self.traj_ids = [str(umax) for umax in self.train_umax]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_normalized_train_data.pt')
        elif group == 'test':
            self.traj_ids = [str(umax) for umax in self.test_umax]
            self.valid_timesteps = self.valid_test_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_normalized_test_data.pt')
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        if group == 'train':
            self.tt = torch.arange(0, train_width).float()
        else:
            self.tt = torch.arange(0, window_width).float()

        self.ntrajs = len(self.traj_ids)

        self.valid_timeslice = slice(*self.valid_timesteps)
        self.valid_timelength = self.valid_timesteps[1] - self.valid_timesteps[0] + 1

        self.window_width = window_width
        self.train_width = train_width

        self.nwindows_per_traj = self.valid_timelength // self.window_width
        logger.info(f'Set nwindows as the maximum: valid_timelength // window_width = {self.nwindows_per_traj}')

        self.nwindows_per_traj = nwindows_per_traj

        self.window_width = window_width
        self.train_width = train_width

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
                f = h5py.File(
                    os.path.join(ROOT_PATH,
                                 f'snapshots_umax={traj_id}',
                                 f'snapshots_umax={traj_id}_s1.h5'),
                    mode='r'
                )
                data[idx] = torch.stack(
                    [torch.from_numpy(
                        (f['tasks/height'][self.valid_timeslice, :: 2, :: 2] - self.height_mean) / self.height_std),
                     torch.from_numpy(
                         (f['tasks/vorticity'][self.valid_timeslice, :: 2, :: 2] - self.vorticity_mean) / self.vorticity_std),],
                    dim=-1)
                if idx == 0:
                    phi = torch.tensor(f['tasks/vorticity'].dims[1][0][:].ravel()[::2])
                    theta = torch.tensor(f['tasks/vorticity'].dims[2][0][:].ravel()[::2])

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


class ShallowWaterPtNormalized(LADataset):

    ROOT_PATH = '/public2/share/lizhuoyuan/ShallowWater'

    '''
    ROOT/
        snapshots_umax=60/
            snapshots_umax=60_s1
            snapshots_umax=60_s1.h5
        snapshots_umax=61/
            snapshots_umax=61_s1
            snapshots_umax=61_s1.h5
        ...
        snapshots_umax=80/
            snapshots_umax=80_s1
            snapshots_umax=80_s1.h5 # shape=(1001, 256, 128)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_umax = [61, 62, 63, 64, 65, 66, 67, 68, 69,
                  71, 72, 73, 74, 75, 76, 77, 78, 79,]
    test_umax = [60, 70, 80]

    valid_train_timesteps = (300, 700)
    valid_test_timesteps = (300, 700)

    height_mean_file_name = 'mean_height.pt'
    height_std_file_name = 'std_height.pt'
    vorticity_mean_file_name = 'mean_vorticity.pt'
    vorticity_std_file_name = 'std_vorticity.pt'

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 nwindows_per_traj: int = None,
                 window_width: int = 20,
                 train_width: int = 10,
                 read_cache=True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train' or group == 'train_eval':
            self.traj_ids = [str(umax) for umax in self.train_umax]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_ptnormalized_train_data.pt')
        elif group == 'test':
            self.traj_ids = [str(umax) for umax in self.test_umax]
            self.valid_timesteps = self.valid_test_timesteps
            self.cached_data_path = os.path.join(ROOT_PATH, 'cached_ptnormalized_test_data.pt')
        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        self.height_mean = torch.load(os.path.join(ROOT_PATH, self.height_mean_file_name))[None, ::2, ::2]
        self.height_std = torch.load(os.path.join(ROOT_PATH, self.height_std_file_name))[None, ::2, ::2]
        self.vorticity_mean = torch.load(os.path.join(ROOT_PATH, self.vorticity_mean_file_name))[None, ::2, ::2]
        self.vorticity_std = torch.load(os.path.join(ROOT_PATH, self.vorticity_std_file_name))[None, ::2, ::2]

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
                f = h5py.File(
                    os.path.join(ROOT_PATH,
                                 f'snapshots_umax={traj_id}',
                                 f'snapshots_umax={traj_id}_s1.h5'),
                    mode='r'
                )
                data[idx] = torch.stack(
                    [(
                        torch.from_numpy(f['tasks/height'][self.valid_timeslice, :: 2, :: 2])
                        - self.height_mean
                    ) / self.height_std,
                        (
                        torch.from_numpy(f['tasks/vorticity'][self.valid_timeslice, :: 2, :: 2])
                        - self.vorticity_mean
                    ) / self.vorticity_std,],
                    dim=-1)
                if idx == 0:
                    phi = torch.tensor(f['tasks/vorticity'].dims[1][0][:].ravel()[::2])
                    theta = torch.tensor(f['tasks/vorticity'].dims[2][0][:].ravel()[::2])

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
