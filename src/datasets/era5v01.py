import os
import logging

from typing import *

from configs.pretrain.pretrain_conf_schema import DatasetConfig as DatasetConfigPT
from configs.finetune.finetune_conf_schema import DatasetConfig as DatasetConfigFT

import torch

import numpy as np

from .la_dataset import PreTrainDataset, FineTuneDataset, MyDataset
from .la_dataset import MetaData as MetaData


class ERA5v01:

    '''
    ROOT/
        Z500_1979_128x64_2.8125deg.npy # shape=(8760, 64, 128)
        ...
        Z500_2018_128x64_2.8125deg.npy # shape=(8760, 64, 128)

        T850_1979_128x64_2.8125deg.npy # shape=(8760, 64, 128)
        ...
        T850_1979_128x64_2.8125deg.npy # shape=(8760, 64, 128)

    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''

    year_ids_tr = list(range(1981, 2016))  # 1981 - 2015 (35 years)
    year_ids_va = [2017]
    year_ids_ts = [2018]

    timestep_slice = slice(0, 240 * 6, 6)

    dtype = torch.float32

    z500_mean: dtype
    z500_std: dtype
    t850_mean: dtype
    t850_std: dtype

    def __init__(self, logger: logging.Logger,
                 cfg: DatasetConfigPT | DatasetConfigFT, **kwargs):

        self.logger = logger
        self.cfg = cfg

        self.cached_meta_tr_path = os.path.join(self.cfg.root_path, 'cached_meta_tr.pt')
        self.cached_meta_va_path = os.path.join(self.cfg.root_path, 'cached_meta_va.pt')
        self.cached_meta_ts_path = os.path.join(self.cfg.root_path, 'cached_meta_ts.pt')

    def packing_from_raw(self, group: str) -> MetaData:

        self.logger.info(f'Packing {group} dataset from raw data...')

        year_ids = getattr(self, f'year_ids_{group}')

        traj_list = []
        for year_id in year_ids:
            z500_np = np.load(os.path.join(self.cfg.root_path, f'raw/Z500_{year_id}_128x64_2.8125deg.npy'))
            t850_np = np.load(os.path.join(self.cfg.root_path, f'raw/T850_{year_id}_128x64_2.8125deg.npy'))

            z500 = torch.from_numpy(z500_np[self.timestep_slice]).transpose(-2, -1).to(self.dtype)
            t850 = torch.from_numpy(t850_np[self.timestep_slice]).transpose(-2, -1).to(self.dtype)
            traj_field = torch.stack([z500, t850], dim=-1)  # (240, 128, 64, 2)
            traj_list.append(traj_field)  # +(240, 128, 64, 2)

        phi = torch.arange(0, 2 * torch.pi, 2 * torch.pi / 128)
        theta = torch.pi / 2 - torch.arange(-torch.pi / 2 + torch.pi / 128, torch.pi / 2, torch.pi / 64)

        trajs = torch.stack(traj_list, dim=0)  # (35, 240, 128, 64, 2)

        # normalize
        if self.cfg.normalize:

            if group == 'tr':
                try:
                    if self.cfg.normalize_mean is None:
                        self.z500_mean = torch.mean(trajs[..., 0])
                        self.t850_mean = torch.mean(trajs[..., 1])
                    else:
                        self.z500_mean = self.cfg.normalize_mean[0]
                        self.t850_mean = self.cfg.normalize_mean[1]

                    if self.cfg.normalize_std is None:
                        self.z500_std = torch.std(trajs[..., 0])
                        self.t850_std = torch.std(trajs[..., 1])
                    else:
                        self.z500_std = self.cfg.normalize_std[0]
                        self.t850_std = self.cfg.normalize_std[1]
                except Exception as e:
                    self.logger.error(f'Failed to calculate mean and std, with exception\n' + e)
                    self.logger.info('Possible reason: the va/ts dataset is loaded before tr dataset')
                    raise e

            trajs[..., 0] = (trajs[..., 0] - self.z500_mean) / self.z500_std
            trajs[..., 1] = (trajs[..., 1] - self.t850_mean) / self.t850_std

        # calculate the coordinates. phi: [0, 2pi); theta: (pi, 0).
        phi_mesh, theta_mesh = torch.meshgrid(phi, theta, indexing='ij')
        # phi_vert = [[phi[0], ..., phi[0]],
        #             ...,
        #             [phi[-1], ..., phi[-1]]]
        # theta_vert = [[theta[0], ..., theta[-1]],
        #                ...,
        #               [theta[0], ..., theta[-1]]]

        # spherical (128, 64, 2)
        coord_latlon = torch.stack([phi_mesh, theta_mesh], dim=-1)

        # cartesian (128, 64, 3)
        x = torch.cos(phi_mesh) * torch.sin(theta_mesh)  # x = cosϕsinθ
        y = torch.sin(phi_mesh) * torch.sin(theta_mesh)  # y = sinϕsinθ
        z = torch.cos(theta_mesh)  # z = cosθ
        coord_cartes = torch.stack([x, y, z], dim=-1)

        self.logger.info(f'Successfully packed {group} dataset from raw data.')
        self.logger.info(f'\n{trajs.shape=}\n{phi.shape=}\n{theta.shape=}' +
                         f'\n{coord_latlon.shape=}\n{coord_cartes.shape=}')

        metadata = MetaData(trajs=trajs,
                            coords={
                                'coord_latlon': coord_latlon,
                                'coord_cartes': coord_cartes,
                            },
                            summary_info=f'{group} dataset\n' +
                            f'{trajs.shape=}\n{phi.shape=}\n{theta.shape=}' +
                            f'\n{coord_latlon.shape=}\n{coord_cartes.shape=}',
                            )
        return metadata

    def _get_metadata(self, group: str) -> MetaData:
        if not hasattr(self, f'_{group}_meta'):
            if self.cfg.read_cache:

                cached_path = getattr(self, f'cached_meta_{group}_path')
                try:
                    setattr(self, f'_{group}_meta', torch.load(cached_path))
                    self.logger.info(f'Successfully loaded cached {group} metadata from {cached_path}')

                except Exception as e:
                    self.logger.warning(f'Failed to load cached {group} metadata from {cached_path}, ' +
                                        f'with exception\n' + str(e))
                    setattr(self, f'_{group}_meta', self.packing_from_raw(group))
            else:
                setattr(self, f'_{group}_meta', self.packing_from_raw(group))

        torch.save(getattr(self, f'_{group}_meta'), getattr(self, f'cached_meta_{group}_path'))

        metadata: MetaData = getattr(self, f'_{group}_meta')
        return metadata

    def _get_dataset(self, group: str, phase: str) -> PreTrainDataset | FineTuneDataset:

        metadata = self._get_metadata(group)

        if phase == 'pretrain':
            dataset = PreTrainDataset(cfg=self.cfg, metadata=metadata)
        if phase == 'finetune':
            dataset = FineTuneDataset(cfg=self.cfg, metadata=metadata)

        self.logger.info('\n' + dataset.get_summary())

        return dataset

    def get_datasets(self, phase: str) -> Tuple[MyDataset, MyDataset, MyDataset]:
        return self._get_dataset('tr', phase), self._get_dataset('va', phase), self._get_dataset('ts', phase)

    def get_metadata(self, group: str) -> MetaData:
        return self._get_metadata(group)
