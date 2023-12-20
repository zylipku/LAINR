import logging

from .la_dataset import LADataset

from .shallow_water import ShallowWater
from .shallow_water import ShallowWaterNormalized
from .shallow_water import ShallowWaterPtNormalized

from .qg_model import QGModel
from .qg_model import QGModelPtNormalized

from .shallow_water2 import ShallowWater2
from .shallow_water3 import ShallowWater3
from .shallow_water4 import ShallowWater4

from .dino_shallow_water import DINoShallowWater
from .dino_shallow_water2 import DINoShallowWater2
from .dino_shallow_water3 import DINoShallowWater3
from .dino_shallow_water_extra import DINoShallowWaterExtra

from .era5 import ERA5
from .era5v2 import ERA5v2

# trainset_params = {
#     'dataset_name': dataset_name,
#     "device": f"cuda:{cudaid}",
#     'buffer_shelve': None,
#     "n_seq": 64,
#     "n_seq_per_traj": 8,
#     "t_horizon": 20,
#     "dt": 1,
#     "size": (128, 64),
#     'n_frames_train': 20,
#     "group": "train",
#     'nwindows_per_traj': 10,
#     'window_width': 10,
#     'train_width': 10,
# }

# validset_params = {
#     'dataset_name': dataset_name,
#     "device": f"cuda:{cudaid}",
#     'buffer_shelve': None,
#     "n_seq": 64,
#     "n_seq_per_traj": 8,
#     "t_horizon": 20,
#     "dt": 1,
#     "size": (128, 64),
#     "group": "train_eval",
#     'n_frames_train': 10,
#     'nwindows_per_traj': 10,
#     'window_width': 20,
#     'train_width': 10,
# }

# testset_params = {
#     'dataset_name': dataset_name,
#     "device": f"cuda:{cudaid}",
#     'buffer_shelve': None,
#     "n_seq": 16,
#     "n_seq_per_traj": 8,
#     "t_horizon": 20,
#     "dt": 1,
#     "size": (128, 64),
#     "group": "test",
#     'n_frames_train': 10,
#     'nwindows_per_traj': 10,
#     'window_width': 20,
#     'train_width': 10,
# }


def load_dataset(logger: logging.Logger, dataset_name, group: str, **kwargs):
    if dataset_name == 'shallow_water':
        return ShallowWater(logger=logger, group=group, **kwargs)

    elif dataset_name == 'qg_model':
        return QGModel(logger=logger, group=group, **kwargs)

    elif dataset_name == 'shallow_water_normalized':
        return ShallowWaterNormalized(logger=logger, group=group, **kwargs)

    elif dataset_name == 'shallow_water_ptnormalized':
        return ShallowWaterPtNormalized(logger=logger, group=group, **kwargs)

    elif dataset_name == 'shallow_water2':
        return ShallowWater2(logger=logger, group=group, **kwargs)

    elif dataset_name == 'shallow_water3':
        return ShallowWater3(logger=logger, group=group, **kwargs)

    elif dataset_name == 'shallow_water4':
        return ShallowWater4(logger=logger, group=group, **kwargs)

    elif dataset_name == 'dino_shallow_water':
        return DINoShallowWater(logger=logger, group=group, **kwargs)

    elif dataset_name == 'dino_shallow_water2':
        return DINoShallowWater2(logger=logger, group=group, **kwargs)

    elif dataset_name == 'dino_shallow_water3':
        return DINoShallowWater3(logger=logger, group=group, **kwargs)

    elif dataset_name == 'dino_shallow_water_extra':
        return DINoShallowWaterExtra(logger=logger, group=group, **kwargs)

    elif dataset_name == 'era5':
        return ERA5(logger=logger, group=group, **kwargs)

    elif dataset_name == 'era5v2':
        return ERA5v2(logger=logger, group=group, **kwargs)

    else:
        raise ValueError(f"dataset_name '{dataset_name}' not supported")
