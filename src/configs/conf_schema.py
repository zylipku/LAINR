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
class ArchEDConfig:

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

    # embedding
    inner_loop_loss_fn: Optional[str] = None
    inner_loop_lr: Optional[float] = None
    inner_loop_max_iters: Optional[int] = None
    inner_loop_max_patience: Optional[int] = None


@dataclass
class ArchLDConfig:

    hidden_dim: Optional[int] = None
    nlayers: Optional[int] = None
    skip_connection: Optional[bool] = None
    nblocks: Optional[int] = None


@dataclass
class ArchUEConfig:

    positive_fn: str = 'exp'

    regularization: float = 1.


@dataclass
class TrainEDConfig:

    bs: int = MISSING

    lr_ed: Optional[float] = None
    lr_cd: Optional[float] = None

    loss_fn_tr: str = 'weighted'
    loss_fn_va: str = 'weighted_root'


@dataclass
class TrainLDConfig:

    lr_ld: Optional[float] = None

    loss_fn_tr: str = 'weighted'
    loss_fn_va: str = 'weighted_root'

    pred_ratio: float | int = 1
    # if < 1 as float, using exponential sampling with probability (p=pred_ratio)
    # if >= 1 as int, using prediction with fixed steps (n=pred_ratio)


@dataclass
class TrainUEConfig:

    bs: int = MISSING
    lr_ue: Optional[float] = None


@dataclass
class EDConfig:

    model_name: str = MISSING
    cfg_name: str = MISSING
    name: str = MISSING

    latent_dim: Optional[int] = MISSING

    need_train: bool = True
    need_cache: bool = False

    arch_params: ArchEDConfig = field(default_factory=ArchEDConfig)
    training_params: TrainEDConfig = field(default_factory=TrainEDConfig)


@dataclass
class LDConfig:

    model_name: str = MISSING
    cfg_name: str = MISSING
    name: str = MISSING

    latent_dim: int = MISSING

    need_train: bool = True
    need_cache: bool = False

    arch_params: ArchLDConfig = field(default_factory=ArchLDConfig)
    training_params: TrainLDConfig = field(default_factory=TrainLDConfig)


@dataclass
class UEConfig:

    model_name: str = MISSING
    cfg_name: str = MISSING
    name: str = MISSING

    ndim: int = MISSING

    need_train: bool = False

    arch_params: ArchUEConfig = field(default_factory=ArchUEConfig)
    training_params: TrainUEConfig = field(default_factory=TrainUEConfig)


@dataclass
class CommonConfig:

    name: str = MISSING
    phase: str = MISSING

    seed: int = MISSING
    num_gpus: int = MISSING
    master_port: int = 23571

    ckpt_path: str = MISSING

    nepochs: int = MISSING
    bs: int = MISSING
    eval_freq: int = 10

    mix_precision: bool = False
