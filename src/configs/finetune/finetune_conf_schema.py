from dataclasses import dataclass, field
from typing import *

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Configuration class for the datasets"""

    name: str = MISSING
    root_path: str = MISSING
    snapshot_shape: Tuple[int, ...] = MISSING

    read_cache: bool = True
    normalize: bool = True
    normalize_mean: Optional[Tuple[float, ...]] = None  # 'empirical' if set as None
    normalize_std: Optional[Tuple[float, ...]] = None

    window_width: int = 10


@dataclass
class EDArchConfig:

    hidden_channels: Optional[int] = None
    latent_channels: Optional[int] = None

    kernel_size: Optional[int] = None
    padding_type: Optional[Tuple[str, ...]] = None

    nresblocks: Optional[int] = None

    state_size: Optional[Tuple[int, ...]] = None
    state_channels: Optional[int] = None

    # for INR
    coord_channels: Optional[int] = None
    code_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    depth: Optional[int] = None
    max_freq: Optional[int] = None


@dataclass
class EDTrainingConfig:

    nepochs: int = MISSING
    bs: int = MISSING
    lr_ed: float = MISSING
    lr_cd: Optional[float] = None

    loss_fn: str = 'weighted'
    loss_fn_va: str = 'weighted_root'

    eval_freq: int = 10


@dataclass
class LDArchConfig:

    hidden_dim: Optional[int] = None
    nlayers: Optional[int] = None
    skip_connection: Optional[bool] = None


@dataclass
class LDTrainingConfig:

    lr_ld: float = MISSING

    loss_fn: str = 'weighted'
    loss_fn_va: str = 'weighted_root'

    pred_ratio: float | int = 1
    # if < 1 as float, using exponential sampling with probability (p=pred_ratio)
    # if >= 1 as int, using prediction with fixed steps (n=pred_ratio)


@dataclass
class EDConfig:
    """Configuration class for the models"""

    model_name: str = MISSING
    cfg_name: str = MISSING
    name: str = MISSING

    need_train: bool = True
    need_cache: bool = False

    latent_dim: Optional[int] = None

    arch_params: EDArchConfig = field(default_factory=EDArchConfig)
    training_params: EDTrainingConfig = field(default_factory=EDTrainingConfig)


@dataclass
class LDConfig:
    """Configuration class for the models"""

    model_name: str = MISSING
    cfg_name: str = MISSING
    name: str = MISSING

    need_train: bool = True

    latent_dim: Optional[int] = None

    arch_params: LDArchConfig = field(default_factory=LDArchConfig)
    training_params: LDTrainingConfig = field(default_factory=LDTrainingConfig)


@dataclass
class FineTuneConfig:

    phase: str = 'finetune'

    name: str = MISSING
    pretrain_name: str = MISSING

    seed: int = MISSING
    num_gpus: int = MISSING

    ckpt_path: str = MISSING
    pretrain_ckpt_path: str = MISSING

    nepochs: int = MISSING
    bs: int = MISSING

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder_decoder: EDConfig = field(default_factory=EDConfig)
    latent_dynamics: LDConfig = field(default_factory=LDConfig)
