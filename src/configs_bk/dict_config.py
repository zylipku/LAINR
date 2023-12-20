import os
import logging
from dataclasses import dataclass, field
from typing import *

from omegaconf import DictConfig, MISSING


@dataclass
class DatasetRawConfig:
    """Configuration class for the datasets"""

    name: str = MISSING
    root_path: str = MISSING
    snapshot_shape: Tuple[int, ...] = MISSING


@dataclass
class ArchRawConfig:

    hidden_channels: int = MISSING
    latent_channels: int = MISSING

    kernel_size: int = MISSING
    padding_type: Tuple[str, str] = MISSING

    nresblocks: int = MISSING

    state_size: Tuple[int, ...] = MISSING
    state_channels: int = MISSING

    # for INR
    coord_channels: int = MISSING
    code_dim: int = MISSING
    hidden_dim: int = MISSING
    depth: int = MISSING
    max_freq: int = MISSING


@dataclass
class TrainingRawConfig:

    nepochs: int = MISSING
    bs: int = MISSING
    lr_ed: float = MISSING

    loss_fn: str = MISSING


@dataclass
class ModelRawConfig:
    """Configuration class for the models"""

    name: str = MISSING
    model_name: str = MISSING
    cfg_name: str = MISSING

    arch_params: ArchRawConfig = field(default_factory=ArchRawConfig)
    training_params: TrainingRawConfig = field(default_factory=TrainingRawConfig)


@dataclass
class CommonRawConfig:

    phase: str

    seed: int
    num_gpus: int


@dataclass
class PreTrainRawConfig:

    name: str = MISSING

    phase: str = 'pretrain'

    seed: int = MISSING
    num_gpus: int = MISSING

    dataset: DatasetRawConfig = field(default_factory=DatasetRawConfig)
    encoder_decoder: ModelRawConfig = field(default_factory=ModelRawConfig)


@dataclass
class FineTuneRawConfig(CommonRawConfig):

    encoder_decoder: ModelRawConfig
    latent_dynamics: ModelRawConfig


@dataclass
class PostProcRawConfig(CommonRawConfig):

    encoder_decoder: ModelRawConfig
    latent_dynamics: ModelRawConfig
    uncertainty_est: ModelRawConfig
