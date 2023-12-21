from dataclasses import dataclass, field
from typing import *

from omegaconf import MISSING

from ..conf_schema import CommonConfig, DatasetConfig, EDConfig


@dataclass
class PreTrainConfig(CommonConfig):

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder_decoder: EDConfig = field(default_factory=EDConfig)
