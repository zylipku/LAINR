import os
import logging

from typing import *

from configs.pretrain.pretrain_conf_schema import DatasetConfig as DatasetConfigPT
from configs.finetune.finetune_conf_schema import DatasetConfig as DatasetConfigFT

import torch

import numpy as np

from .la_dataset import PreTrainDataset, MyDataset
from .la_dataset import MetaData as MetaData


class DatasetPacker:

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

    def __init__(self, logger: logging.Logger,
                 cfg: DatasetConfigPT | DatasetConfigFT, **kwargs):

        self.logger = logger
        self.cfg = cfg

        self.cached_meta_tr_path = os.path.join(self.cfg.root_path, 'cached_meta_tr.pt')
        self.cached_meta_va_path = os.path.join(self.cfg.root_path, 'cached_meta_va.pt')
        self.cached_meta_ts_path = os.path.join(self.cfg.root_path, 'cached_meta_ts.pt')

    def packing_from_raw(self, group: str) -> MetaData:

        raise NotImplementedError

    def _get_metadata(self, group: str) -> MetaData:
        if not hasattr(self, f'_{group}_meta'):
            if self.cfg.read_cache:

                cached_path = getattr(self, f'cached_meta_{group}_path')
                try:
                    metadata = torch.load(cached_path)
                    setattr(self, f'_{group}_meta', metadata)
                    self.logger.info(f'Successfully loaded cached {group} metadata from {cached_path}')
                    return metadata

                except Exception as e:
                    self.logger.warning(f'Failed to load cached {group} metadata from {cached_path}, ' +
                                        f'with exception\n' + str(e))
                    setattr(self, f'_{group}_meta', self.packing_from_raw(group))
            else:
                setattr(self, f'_{group}_meta', self.packing_from_raw(group))

        torch.save(getattr(self, f'_{group}_meta'), getattr(self, f'cached_meta_{group}_path'))

        metadata: MetaData = getattr(self, f'_{group}_meta')
        return metadata

    def _get_dataset(self, group: str, phase: str) -> PreTrainDataset:

        metadata = self._get_metadata(group)

        if phase == 'pretrain':
            dataset = PreTrainDataset(cfg=self.cfg, metadata=metadata)
        if phase == 'finetune':
            dataset = PreTrainDataset(cfg=self.cfg, metadata=metadata)

        self.logger.info('\n' + dataset.get_summary())

        return dataset

    def get_datasets(self, phase: str) -> Tuple[MyDataset, MyDataset, MyDataset]:
        return self._get_dataset('tr', phase), self._get_dataset('va', phase), self._get_dataset('ts', phase)

    def get_metadata(self, group: str) -> MetaData:
        return self._get_metadata(group)
