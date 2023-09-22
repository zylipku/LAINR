from .sphere_loss import SphereLoss

from .mse import MSELoss
from .skip_latlon import SkipLatLonLoss
from .weighted_latlon import WeightedLatLonLoss


def get_metrics(name: str, **kwargs):

    if name == 'mse':
        return MSELoss(**kwargs)
    elif name == 'skip':
        return SkipLatLonLoss(**kwargs)
    elif name == 'weighted':
        return WeightedLatLonLoss(**kwargs)
    else:
        raise ValueError(f'Unknown metric name: {name}')
