import logging

from .la_dataset import MetaData

from .sw import ShallowWater
from .era5 import ERA5
from .era5v00 import ERA5v00
from .era5v01 import ERA5v01
from .era5v02 import ERA5v02
from .era5v03 import ERA5v03

from configs.conf_schema import DatasetConfig


def load_dataset(logger: logging.Logger, cfg: DatasetConfig, **kwargs):

    ds_name = cfg.name.lower()

    if ds_name == 'sw':
        return ShallowWater(logger=logger, cfg=cfg, **kwargs)  # alias with DINoShallowWater3
    
    elif ds_name == 'era5':
        return ERA5(logger=logger, cfg=cfg, **kwargs)
    
    elif ds_name == 'era5v00':
        return ERA5v00(logger=logger, cfg=cfg, **kwargs)
    
    elif ds_name == 'era5v01':
        return ERA5v01(logger=logger, cfg=cfg, **kwargs)
    
    elif ds_name == 'era5v02':
        return ERA5v02(logger=logger, cfg=cfg, **kwargs)
    
    elif ds_name == 'era5v03':
        return ERA5v03(logger=logger, cfg=cfg, **kwargs)
    else:
        raise ValueError(f"dataset_name '{ds_name}' not supported")
