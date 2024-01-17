from dataclasses import dataclass, field
from typing import *

from omegaconf import MISSING

from ..conf_schema import CommonConfig, DatasetConfig, EDConfig, LDConfig, UEConfig


@dataclass
class PostProcConfig(CommonConfig):

    phase: str = 'postproc'

    finetune_name: str = MISSING
    finetune_ckpt_path: str = MISSING

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder_decoder: EDConfig = field(default_factory=EDConfig)
    latent_dynamics: LDConfig = field(default_factory=LDConfig)
    uncertainty_est: UEConfig = field(default_factory=UEConfig)
