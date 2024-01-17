from dataclasses import dataclass, field
from typing import *

from omegaconf import MISSING

from ..conf_schema import DatasetConfig, EDConfig, LDConfig, UEConfig


@dataclass
class AssimilateConfig:

    name: str = MISSING
    finetune_name: str = MISSING
    postproc_name: str = MISSING

    seed: int = MISSING
    cuda_id: int | None = None

    finetune_ckpt_path: str = MISSING
    postproc_ckpt_path: str = MISSING

    ass_nsteps: int = MISSING

    sigma_x_b: float = MISSING
    sigma_z_b: Optional[float] = None

    sigma_m: Optional[float] = None
    sigma_o: float = MISSING

    n_obs: int = MISSING

    ens_num: int = MISSING
    infl: float = MISSING

    offgrid: bool = False

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder_decoder: EDConfig = field(default_factory=EDConfig)
    latent_dynamics: LDConfig = field(default_factory=LDConfig)
    uncertainty_est: UEConfig = field(default_factory=UEConfig)

    save_dir: str = MISSING
