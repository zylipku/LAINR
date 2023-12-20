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


@dataclass
class ArchConfig:

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
class TrainingConfig:

    nepochs: int = MISSING
    bs: int = MISSING
    lr_ed: float = MISSING
    lr_cd: Optional[float] = None

    loss_fn: str = 'weighted'
    loss_fn_va: str = 'weighted'

    eval_freq: int = 10


@dataclass
class ModelConfig:
    """Configuration class for the models"""

    model_name: str = MISSING
    cfg_name: str = MISSING
    name: str = MISSING

    need_train: bool = True
    need_cache: bool = False

    latent_dim: Optional[int] = None

    arch_params: ArchConfig = field(default_factory=ArchConfig)
    training_params: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class CommonConfig:

    name: str = MISSING

    phase: str = MISSING

    seed: int = MISSING
    num_gpus: int = MISSING

    ckpt_path: str = MISSING


@dataclass
class PreTrainConfig(CommonConfig):

    name: str = MISSING
    phase: str = 'pretrain'

    seed: int = MISSING
    num_gpus: int = MISSING

    ckpt_path: str = MISSING

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder_decoder: ModelConfig = field(default_factory=ModelConfig)
