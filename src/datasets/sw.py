'''
normalized DINo shallow water dataset
the time horizon has been extended to 10 days (240 hours)
the number of trajectories has been increased to 20
slightly harder case
steps from 360 to 600
'''

import os
import logging

from typing import *

from configs.pretrain.pretrain_conf_schema import DatasetConfig as DatasetConfigPT
from configs.finetune.finetune_conf_schema import DatasetConfig as DatasetConfigFT

import h5py

import numpy as np
import torch
from torch.utils.data import Dataset

from .la_dataset import PreTrainDataset, FineTuneDataset, MyDataset
from .la_dataset import MetaData as MetaData


class ShallowWater:

    '''
    ROOT/
        traj_1.h5 
        ..
        traj_20.h5

        keys:
            'height': height, # shape=(600, 256, 128)
            'vorticity': vorticity, # shape=(600, 256, 128)
            'phi': phi, # shape=(256,) : [0, 2pi)
            'theta': theta, # shape=(128,) : (pi, 0)


    self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
    self.coords.shape = (h=128, w=64, coord_dim=3)
    '''
    traj_ids_tr = list(range(1, 19))  # 16 trajs (80%) for training
    traj_ids_va = [19, 20]  # 2 trajs (10%) for validation
    traj_ids_ts = [19, 20]  # 2 trajs (10%) for testing

    trunc_timesteps = (360, 600)  # only keep the last 240 hours

    dtype = torch.float32

    height_mean: dtype
    height_std: dtype
    vorticity_mean: dtype
    vorticity_std: dtype

    def __init__(self, logger: logging.Logger,
                 cfg: DatasetConfigPT | DatasetConfigFT, **kwargs):

        self.logger = logger
        self.cfg = cfg

        self.cached_meta_tr_path = os.path.join(self.cfg.root_path, 'cached_meta_tr.pt')
        self.cached_meta_va_path = os.path.join(self.cfg.root_path, 'cached_meta_va.pt')
        self.cached_meta_ts_path = os.path.join(self.cfg.root_path, 'cached_meta_ts.pt')

        self.trun_timeslice = slice(*self.trunc_timesteps)
        self.trunc_timelength = self.trunc_timesteps[1] - self.trunc_timesteps[0]

    def packing_from_raw(self, group: str) -> MetaData:

        self.logger.info(f'Packing {group} dataset from raw data...')

        traj_ids = getattr(self, f'traj_ids_{group}')

        traj_list = []
        for idx, traj_id in enumerate(traj_ids):
            traj_file = h5py.File(os.path.join(self.cfg.root_path, f'raw/traj_{traj_id}.h5'), 'r')
            # field shape (600, 256, 128) -> (240, 128, 64)
            height_np = traj_file['height'][:][self.trun_timeslice, :: 2, :: 2]  # (240, 128, 64)
            vorticity_np = traj_file['vorticity'][:][self.trun_timeslice, :: 2, :: 2]  # (240, 128, 64)
            height = torch.from_numpy(height_np).to(self.dtype)
            vorticity = torch.from_numpy(vorticity_np).to(self.dtype)
            traj_field = torch.stack([height, vorticity], dim=-1)  # (240, 128, 64, 2)
            traj_list.append(traj_field)  # +(240, 128, 64, 2)

            if idx == 0:
                phi_np = traj_file['phi'][:][:: 2]  # (256,) -> (128,)
                theta_np = traj_file['theta'][:][:: 2]  # (128,) -> (64,)
                phi = torch.from_numpy(phi_np).to(self.dtype)  # (128,)
                theta = torch.from_numpy(theta_np).to(self.dtype)  # (64,)

        trajs = torch.stack(traj_list, dim=0)  # (ntrajs, 240, 128, 64, 2)

        # normalize
        if self.cfg.normalize:

            if group == 'tr':
                try:
                    if self.cfg.normalize_mean is None:
                        self.height_mean = torch.mean(trajs[..., 0])
                        self.vorticity_mean = torch.mean(trajs[..., 1])
                    else:
                        self.height_mean = self.cfg.normalize_mean[0]
                        self.vorticity_mean = self.cfg.normalize_mean[1]

                    if self.cfg.normalize_std is None:
                        self.height_std = torch.std(trajs[..., 0])
                        self.vorticity_std = torch.std(trajs[..., 1])
                    else:
                        self.height_std = self.cfg.normalize_std[0]
                        self.vorticity_std = self.cfg.normalize_std[1]
                except Exception as e:
                    self.logger.error(f'Failed to calculate mean and std, with exception\n' + e)
                    self.logger.info('Possible reason: the va/ts dataset is loaded before tr dataset')
                    raise e

            trajs[..., 0] = (trajs[..., 0] - self.height_mean) / self.height_std
            trajs[..., 1] = (trajs[..., 1] - self.vorticity_mean) / self.vorticity_std

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
        # if phase == 'pretrain-seq':
        #     dataset = PreTrainSeqDataset(cfg=self.cfg, metadata=metadata)

        self.logger.info('\n' + dataset.get_summary())

        return dataset

    def get_datasets(self, phase: str) -> Tuple[MyDataset, MyDataset, MyDataset]:
        return self._get_dataset('tr', phase), self._get_dataset('va', phase), self._get_dataset('ts', phase)

    def get_metadata(self, group: str) -> MetaData:
        return self._get_metadata(group)

# class ShallowWaterFineTune:

#     '''
#     ROOT/
#         traj_1.h5
#         ..
#         traj_20.h5

#         keys:
#             'height': height, # shape=(600, 256, 128)
#             'vorticity': vorticity, # shape=(600, 256, 128)
#             'phi': phi, # shape=(256,) : [0, 2pi)
#             'theta': theta, # shape=(128,) : (pi, 0)


#     self.data.shape = (ntrajs, Nsteps, h=128, w=64, nstates=2)
#     self.coords.shape = (h=128, w=64, coord_dim=3)
#     '''
#     traj_ids_tr = list(range(1, 17))  # 16 trajs (80%) for training
#     traj_ids_va = [17, 18]  # 2 trajs (10%) for validation
#     traj_ids_ts = [19, 20]  # 2 trajs (10%) for testing

#     trunc_timesteps = (360, 600)  # only keep the last 240 hours

#     dtype = torch.float32

#     height_mean: dtype
#     height_std: dtype
#     vorticity_mean: dtype
#     vorticity_std: dtype

#     def __init__(self, logger: logging.Logger,
#                  cfg: DatasetConfig, **kwargs):

#         self.logger = logger
#         self.cfg = cfg

#         self.cached_data_tr_path = os.path.join(self.cfg.root_path, 'cached_data_finetune_tr.pt')
#         self.cached_data_va_path = os.path.join(self.cfg.root_path, 'cached_data_finetune_va.pt')
#         self.cached_data_ts_path = os.path.join(self.cfg.root_path, 'cached_data_finetune_ts.pt')
#         self.cached_misc_path = os.path.join(self.cfg.root_path, 'cached_misc.pt')

#         self.trun_timeslice = slice(*self.trunc_timesteps)
#         self.trunc_timelength = self.trunc_timesteps[1] - self.trunc_timesteps[0]

#     def packing_from_raw(self, group: str) -> Dataset:

#         self.logger.info(f'Packing {group} dataset from raw data...')

#         traj_ids = getattr(self, f'traj_ids_{group}')

#         traj_list = []
#         for idx, traj_id in enumerate(traj_ids):
#             traj_file = h5py.File(os.path.join(self.cfg.root_path, f'raw/traj_{traj_id}.h5'), 'r')
#             # field shape (600, 256, 128) -> (240, 128, 64)
#             height_np = traj_file['height'][:][self.trun_timeslice, :: 2, :: 2]  # (240, 128, 64)
#             vorticity_np = traj_file['vorticity'][:][self.trun_timeslice, :: 2, :: 2]  # (240, 128, 64)
#             height = torch.from_numpy(height_np).to(self.dtype)
#             vorticity = torch.from_numpy(vorticity_np).to(self.dtype)
#             traj_field = torch.stack([height, vorticity], dim=-1)  # (240, 128, 64, 2)
#             traj_list.append(traj_field)  # +(240, 128, 64, 2)

#             if idx == 0:
#                 phi_np = traj_file['phi'][:][:: 2]  # (256,) -> (128,)
#                 theta_np = traj_file['theta'][:][:: 2]  # (128,) -> (64,)
#                 phi = torch.from_numpy(phi_np).to(self.dtype)  # (128,)
#                 theta = torch.from_numpy(theta_np).to(self.dtype)  # (64,)

#         trajs = torch.stack(traj_list, dim=0)  # (ntrajs, 240, 128, 64, 2)

#         # normalize
#         if self.cfg.normalize:

#             if group == 'tr':
#                 try:
#                     if self.cfg.normalize_mean is None:
#                         self.height_mean = torch.mean(trajs[..., 0])
#                         self.vorticity_mean = torch.mean(trajs[..., 1])
#                     else:
#                         self.height_mean = self.cfg.normalize_mean[0]
#                         self.vorticity_mean = self.cfg.normalize_mean[1]

#                     if self.cfg.normalize_std is None:
#                         self.height_std = torch.std(trajs[..., 0])
#                         self.vorticity_std = torch.std(trajs[..., 1])
#                     else:
#                         self.height_std = self.cfg.normalize_std[0]
#                         self.vorticity_std = self.cfg.normalize_std[1]
#                 except Exception as e:
#                     self.logger.error(f'Failed to calculate mean and std, with exception\n' + e)
#                     self.logger.info('Possible reason: the va/ts dataset is loaded before tr dataset')
#                     raise e

#             trajs[..., 0] = (trajs[..., 0] - self.height_mean) / self.height_std
#             trajs[..., 1] = (trajs[..., 1] - self.vorticity_mean) / self.vorticity_std

#         # calculate the coordinates. phi: [0, 2pi); theta: (pi, 0).
#         phi_mesh, theta_mesh = torch.meshgrid(phi, theta, indexing='ij')
#         # phi_vert = [[phi[0], ..., phi[0]],
#         #             ...,
#         #             [phi[-1], ..., phi[-1]]]
#         # theta_vert = [[theta[0], ..., theta[-1]],
#         #                ...,
#         #               [theta[0], ..., theta[-1]]]

#         # spherical (128, 64, 2)
#         coord_latlon = torch.stack([phi_mesh, theta_mesh], dim=-1)

#         # cartesian (128, 64, 3)
#         x = torch.cos(phi_mesh) * torch.sin(theta_mesh)  # x = cosϕsinθ
#         y = torch.sin(phi_mesh) * torch.sin(theta_mesh)  # y = sinϕsinθ
#         z = torch.cos(theta_mesh)  # z = cosθ
#         coord_cartes = torch.stack([x, y, z], dim=-1)

#         self.logger.info(f'Successfully packed {group} dataset from raw data.')
#         self.logger.info(f'\n{trajs.shape=}\n{phi.shape=}\n{theta.shape=}' +
#                          f'\n{coord_latlon.shape=}\n{coord_cartes.shape=}')

#         return FineTuneDataset(trajs=trajs,
#                                coords={
#                                    'coord_latlon': coord_latlon,
#                                    'coord_cartes': coord_cartes,
#                                },
#                                summary_info=f'{group} dataset\n' +
#                                f'{trajs.shape=}\n{phi.shape=}\n{theta.shape=}' +
#                                f'\n{coord_latlon.shape=}\n{coord_cartes.shape=}',
#                                )

#     def _get_dataset(self, group: str) -> DictDataset:

#         if not hasattr(self, f'_{group}_dataset'):
#             if self.cfg.read_cache:

#                 cached_path = getattr(self, f'cached_data_{group}_path')
#                 try:
#                     setattr(self, f'_{group}_dataset', torch.load(cached_path))
#                     self.logger.info(f'Successfully loaded cached {group} data from {cached_path}')

#                 except Exception as e:
#                     self.logger.warning(
#                         f'Failed to load cached {group} data from {cached_path}, with exception\n' + str(e))
#                     setattr(self, f'_{group}_dataset', self.packing_from_raw(group))
#             else:
#                 setattr(self, f'_{group}_dataset', self.packing_from_raw(group))

#         torch.save(getattr(self, f'_{group}_dataset'), getattr(self, f'cached_data_{group}_path'))

#         dataset: DictDataset = getattr(self, f'_{group}_dataset')
#         self.logger.info('\n' + dataset.get_summary())

#         return dataset

#     def get_datasets(self) -> Tuple[DictDataset, DictDataset, DictDataset]:
#         return self._get_dataset('tr'), self._get_dataset('va'), self._get_dataset('ts')
