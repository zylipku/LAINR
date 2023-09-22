import logging

from .la_dataset import LADataset

from .sw import ShallowWater
from .era5 import ERA5


def load_dataset(logger: logging.Logger, dataset_name, group: str, **kwargs):

    if dataset_name == 'shallow_water' or dataset_name == 'sw':
        return ShallowWater(logger=logger, group=group, **kwargs)

    elif dataset_name == 'era5':
        return ERA5(logger=logger, group=group, **kwargs)

    else:
        raise ValueError(f"dataset_name '{dataset_name}' not supported")
