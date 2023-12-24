from dataclasses import dataclass, field
from typing import *

from omegaconf import MISSING

from ..conf_schema import DatasetConfig, EDConfig, LDConfig


@dataclass
class KalmanFilterConfig:

    name: str = MISSING

    ens_num: int = MISSING
    infl: float = MISSING


@dataclass
class AssimilateConfig:

    method_name: str = MISSING

    seed: int = MISSING
    cuda_id: int | None = None

    ckpt_path: str = MISSING

    ass_nsteps: int = MISSING

    sigma_x_b: float = MISSING
    sigma_z_b: float = MISSING

    sigma_m: float = MISSING
    sigma_o: float = MISSING

    n_obs: int = MISSING

    offgrid: bool = False

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder_decoder: EDConfig = field(default_factory=EDConfig)
    latent_dynamics: LDConfig = field(default_factory=LDConfig)

    save_dir: str = MISSING
