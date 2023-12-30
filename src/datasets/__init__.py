import logging

from .la_dataset import MetaData

from .sw import ShallowWater
from .era5 import ERA5
from .era5v01 import ERA5v01

from configs.conf_schema import DatasetConfig


def load_dataset(logger: logging.Logger, cfg: DatasetConfig, **kwargs):

    ds_name = cfg.name.lower()

    if ds_name == 'sw':
        return ShallowWater(logger=logger, cfg=cfg, **kwargs)  # alias with DINoShallowWater3
    elif ds_name == 'era5':
        return ERA5(logger=logger, cfg=cfg, **kwargs)
    elif ds_name == 'era5v01':
        return ERA5v01(logger=logger, cfg=cfg, **kwargs)
    else:
        raise ValueError(f"dataset_name '{ds_name}' not supported")
