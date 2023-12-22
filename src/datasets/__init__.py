import logging

from omegaconf import DictConfig

from .la_dataset import MyDataset

from .sw import ShallowWater
# from .era5 import ERA5


def load_dataset(logger: logging.Logger, cfg: DictConfig, **kwargs):
    # if dataset_name == 'shallow_water':
    #     return ShallowWater(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'qg_model':
    #     return QGModel(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'shallow_water_normalized':
    #     return ShallowWaterNormalized(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'shallow_water_ptnormalized':
    #     return ShallowWaterPtNormalized(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'shallow_water2':
    #     return ShallowWater2(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'shallow_water3':
    #     return ShallowWater3(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'shallow_water4':
    #     return ShallowWater4(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'dino_shallow_water':
    #     return DINoShallowWater(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'dino_shallow_water2':
    #     return DINoShallowWater2(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'dino_shallow_water3':
    #     return DINoShallowWater3(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'dino_shallow_water_extra':
    #     return DINoShallowWaterExtra(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'era5':
    #     return ERA5(logger=logger, group=group, **kwargs)

    # elif dataset_name == 'era5v2':
    #     return ERA5v2(logger=logger, group=group, **kwargs)

    ds_name = cfg.name.lower()

    if ds_name == 'sw':
        return ShallowWater(logger=logger, cfg=cfg, **kwargs)  # alias with DINoShallowWater3
    # elif ds_name == 'era5':
    #     return ERA5(logger=logger, cfg=cfg, group=group, **kwargs)  # alias with ERA5v2
    else:
        raise ValueError(f"dataset_name '{ds_name}' not supported")