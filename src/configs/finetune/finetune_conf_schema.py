from dataclasses import dataclass, field
from typing import *

from omegaconf import MISSING

from ..conf_schema import CommonConfig, DatasetConfig, EDConfig, LDConfig


@dataclass
class FineTuneConfig(CommonConfig):

    phase: str = 'finetune'

    pretrain_name: str = MISSING
    pretrain_ckpt_path: str = MISSING

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder_decoder: EDConfig = field(default_factory=EDConfig)
    latent_dynamics: LDConfig = field(default_factory=LDConfig)
