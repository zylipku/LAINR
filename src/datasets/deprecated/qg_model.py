import os

import logging

import torch

import h5py

from .la_dataset import LADataset


class QGModel(LADataset):

    ROOT_PATH = '/home/lizhuoyuan/datasets/QGModel'

    '''
    ROOT/
        rnd0_1h.h5 # shape=(12500, 193, 128, 2)
                                ny, nx
        rnd2_1h.h5 # shape=(12500, 193, 128, 2)
        rnd3_1h.h5 # shape=(12500, 193, 128, 2)
        rnd4_1h.h5 # shape=(12500, 193, 128, 2)
        rnd5_1h.h5 # shape=(12500, 193, 128, 2)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_idxs = [2, 3, 4, 5]
    test_idxs = [0]

    valid_train_timesteps = (2000, 12000)
    valid_test_timesteps = (9000, 10000)

    qpv0_mean_file_name = 'mean_qpv0.pt'
    qpv0_std_file_name = 'std_qpv0.pt'
    qpv1_mean_file_name = 'mean_qpv1.pt'
    qpv1_std_file_name = 'std_qpv1.pt'

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 nwindows_per_traj: int = 1,
                 window_width: int = 20,
                 train_width: int = 10,
                 read_cache=True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        if group == 'train' or group == 'train_eval':
            self.traj_ids = [str(idx) for idx in self.train_idxs]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(self.ROOT_PATH, 'cached_train_data.pt')

        elif group == 'test':
            self.traj_ids = [str(idx) for idx in self.test_idxs]
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

        # TODO modify trainset_params
        nwindows_per_traj = None

        if nwindows_per_traj is None:
            nwindows_per_traj = self.valid_timelength // window_width

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
                    os.path.join(self.ROOT_PATH,
                                 f'rnd{traj_id}_1h.h5'),
                    mode='r'
                )
                qpv = torch.from_numpy(f['QPV'][self.valid_timeslice]).transpose(-3, -2)
                data[idx] = qpv[:, :, 96 - 64 + 1:96 + 64 + 1:2, :]
                # 192 * dy truncated as 128, then downsampled to 64

                if idx == 0:

                    x = torch.from_numpy(f['x'][:, 0])
                    # (128, 1) -> (128,) \in [0, 46)
                    y = torch.from_numpy(f['y'][0, 96 - 64 + 1:96 + 64 + 1:2])
                    # (1, 193) -> (64,) \in (23, -23)
                    phi = 2 * torch.pi * x / 46.  # [0, 2pi)
                    theta = torch.pi * (y + 23.) / 46.  # (pi, 0)

            # f = h5py.File(
            #     os.path.join(self.ROOT_PATH, 'output.mat'),
            #     mode='r'
            # )
            # qpv = torch.from_numpy(f['QPV'][self.valid_timeslice]).transpose(-3, -2)
            # qpv_trunc = qpv[:, :, 96 - 64:96 + 64 + 1:2, :]
            # # 192 * dy truncated as 128, then downsampled to 64

            # x = torch.from_numpy(f['x'][:, 0])
            # # (128, 1) -> (128,) \in [0, 46)
            # y = torch.from_numpy(f['y'][0, 1::3])
            # # (1, 193) -> (64,) \in (23, -23)
            # data = qpv[None, ...]
            # phi = 2 * torch.pi * x / 46.  # [0, 2pi)
            # theta = torch.pi * (y + 23.) / 46.  # (pi, 0)
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


class QGModelPtNormalized(LADataset):

    ROOT_PATH = '/home/lizhuoyuan/datasets/QGModel'

    '''
    ROOT/
        rnd0_1h.h5 # shape=(12500, 193, 128, 2)
                                ny, nx
        rnd2_1h.h5 # shape=(12500, 193, 128, 2)
        rnd3_1h.h5 # shape=(12500, 193, 128, 2)
        rnd4_1h.h5 # shape=(12500, 193, 128, 2)
        rnd5_1h.h5 # shape=(12500, 193, 128, 2)
    
    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    train_idxs = [2, 3, 4, 5]
    test_idxs = [0]

    valid_train_timesteps = (2000, 12000)
    valid_test_timesteps = (9000, 10000)

    qpv0_mean_file_name = 'mean_qpv0.pt'
    qpv0_std_file_name = 'std_qpv0.pt'
    qpv1_mean_file_name = 'mean_qpv1.pt'
    qpv1_std_file_name = 'std_qpv1.pt'

    def __init__(self,
                 logger: logging.Logger,
                 group: str = 'train',
                 nwindows_per_traj: int = 1,
                 window_width: int = 20,
                 train_width: int = 10,
                 read_cache=True,
                 **kwargs,
                 ):
        super().__init__(logger=logger, group=group)

        self.qpv0_mean = torch.load(os.path.join(self.ROOT_PATH, self.qpv0_mean_file_name))
        self.qpv0_std = torch.load(os.path.join(self.ROOT_PATH, self.qpv0_std_file_name))
        self.qpv1_mean = torch.load(os.path.join(self.ROOT_PATH, self.qpv1_mean_file_name))
        self.qpv1_std = torch.load(os.path.join(self.ROOT_PATH, self.qpv1_std_file_name))

        if group == 'train' or group == 'train_eval':
            self.traj_ids = [str(idx) for idx in self.train_idxs]
            self.valid_timesteps = self.valid_train_timesteps
            self.cached_data_path = os.path.join(self.ROOT_PATH, 'cached_ptnormalized_train_data.pt')

        elif group == 'test':
            self.traj_ids = [str(idx) for idx in self.test_idxs]
            self.valid_timesteps = self.valid_test_timesteps
            self.cached_data_path = os.path.join(self.ROOT_PATH, 'cached_ptnormalized_test_data.pt')

        else:
            raise ValueError(f'group must be "train" or "test", but got {group}')

        if group == 'train':
            self.tt = torch.arange(0, train_width).float()
        else:
            self.tt = torch.arange(0, window_width).float()

        self.ntrajs = len(self.traj_ids)

        self.valid_timeslice = slice(*self.valid_timesteps)
        self.valid_timelength = self.valid_timesteps[1] - self.valid_timesteps[0]

        # TODO modify trainset_params
        nwindows_per_traj = None

        if nwindows_per_traj is None:
            nwindows_per_traj = self.valid_timelength // window_width

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
                    os.path.join(self.ROOT_PATH,
                                 f'rnd{traj_id}_1h.h5'),
                    mode='r'
                )
                qpv = torch.from_numpy(f['QPV'][self.valid_timeslice]).transpose(-3, -2)
                qpv0 = (qpv[..., 0] - self.qpv0_mean) / self.qpv0_std
                qpv1 = (qpv[..., 1] - self.qpv1_mean) / self.qpv1_std
                qpv = torch.stack([qpv0, qpv1], dim=-1)
                data[idx] = qpv[:, :, 96 - 64 + 1:96 + 64 + 1:2, :]
                # 192 * dy truncated as 128, then downsampled to 64

                if idx == 0:

                    x = torch.from_numpy(f['x'][:, 0])
                    # (128, 1) -> (128,) \in [0, 46)
                    y = torch.from_numpy(f['y'][0, 96 - 64 + 1:96 + 64 + 1:2])
                    # (1, 193) -> (64,) \in (23, -23)
                    phi = 2 * torch.pi * x / 46.  # [0, 2pi)
                    theta = torch.pi * (y + 23.) / 46.  # (pi, 0)

            # f = h5py.File(
            #     os.path.join(self.ROOT_PATH, 'output.mat'),
            #     mode='r'
            # )
            # qpv = torch.from_numpy(f['QPV'][self.valid_timeslice]).transpose(-3, -2)
            # qpv_trunc = qpv[:, :, 96 - 64:96 + 64 + 1:2, :]
            # # 192 * dy truncated as 128, then downsampled to 64

            # x = torch.from_numpy(f['x'][:, 0])
            # # (128, 1) -> (128,) \in [0, 46)
            # y = torch.from_numpy(f['y'][0, 1::3])
            # # (1, 193) -> (64,) \in (23, -23)
            # data = qpv[None, ...]
            # phi = 2 * torch.pi * x / 46.  # [0, 2pi)
            # theta = torch.pi * (y + 23.) / 46.  # (pi, 0)
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
