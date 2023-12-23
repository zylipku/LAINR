import logging

from .sw import ShallowWater
# from .era5 import ERA5

from configs.conf_schema import DatasetConfig


def load_dataset(logger: logging.Logger, cfg: DatasetConfig, **kwargs):

    ds_name = cfg.name.lower()

    if ds_name == 'sw':
        return ShallowWater(logger=logger, cfg=cfg, **kwargs)  # alias with DINoShallowWater3
    # elif ds_name == 'era5':
    #     return ERA5(logger=logger, cfg=cfg, group=group, **kwargs)  # alias with ERA5v2
    else:
        raise ValueError(f"dataset_name '{ds_name}' not supported")
