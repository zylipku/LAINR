import os
import logging
from dataclasses import dataclass, fields
from typing import *

from omegaconf import DictConfig, OmegaConf

from components import EncoderDecoder, LatentDynamics, UncertaintyEst
from components import ed_name2class, ld_name2class, uq_name2class

from configs.pretrain.pretrain_conf_schema import PreTrainConfig
from configs.conf_schema import CommonConfig, DatasetConfig, EDConfig, LDConfig

ALL_DATASET_NAME = ['sw', 'era5']
ALL_LOSS_NAME = ['mse', 'skip', 'weighted']
ALL_COMP_NAME = ['pca',
                 'cae', 'aeflow',
                 'fouriernet',
                 'sinr', 'sinr_noskip']
ALL_DYN_NAME = ['rezero', 'neuralode']
ALL_LOSS_NAME = ['mse', 'skip', 'weighted']
ALL_UQ_NAME = ['diagonal', 'srn', 'cholesky']


@dataclass
class DatasetConfig:
    """Configuration class for the datasets"""

    name: str
    root_path: str
    snapshot_shape: Tuple[int, ...]

    @classmethod
    def parse(cls, cfg: DictConfig):
        new_cls = cls(**cfg)

        return new_cls


@dataclass
class ModelConfig:
    """Configuration class for the models"""

    model_name: str
    cfg_name: str

    arch_params: DictConfig
    training_params: DictConfig

    @property
    def name(self):
        return f"{self.model_name}_{self.cfg_name}"

    @classmethod
    def parse(cls, cfg: ModelConfig):
        # new_cls = cls(model_name=cfg.model_name,
        #               cfg_name=cfg.cfg_name,
        #               arch_params=dict(cfg.arch_params),
        #               training_params=dict(cfg.training_params))
        new_cls = cls(**cfg)

        return new_cls


@dataclass
class CommonConfig:

    phase: str

    seed: int
    num_gpus: int


@dataclass
class FineTuneConfig(CommonConfig):

    encoder_decoder: ModelConfig
    latent_dynamics: ModelConfig


@dataclass
class PostProcConfig(CommonConfig):

    encoder_decoder: ModelConfig
    latent_dynamics: ModelConfig
    uncertainty_est: ModelConfig


class PreTrainConfig:

    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_CKPT_DIR = os.path.join(BASE_DIR, '../ckpts')

    @property
    def name(self) -> str:
        return self.ed.model_name.lower() + '_' + self.ed.cfg_name.lower()

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.BASE_CKPT_DIR, self.ds_name, self.ls_name, f'{self.name}.pt')

    def __init__(self, logger: logging.Logger, cfg: PreTrainConfig) -> None:
        self.logger = logger

        self.ds = DatasetConfig.parse(cfg.dataset)
        self.ed = ModelConfig.parse(cfg.encoder_decoder)

        # check if the dataset and the model are valid
        self.ds_name = self.ds.name.lower()
        self.ed_name = self.ed.model_name.lower()
        self.ls_name = self.ed.training_params.loss_fn.lower()
        assert self.ds_name in ALL_DATASET_NAME
        assert self.ed_name in ALL_COMP_NAME
        assert self.ls_name in ALL_LOSS_NAME

        self.ed_class: EncoderDecoder = ed_name2class[self.ed_name]

        self.ed_need_train = 'pca' not in self.ed_name
        self.ed_need_cache = 'fourier' in self.ed_name or 'inr' in self.ed_name

        print(self.ed.arch_params.__module__)

        self.ed.arch_params: DictConfig
        self.ed.arch_params.state_channels = self.ds.snapshot_shape[-1]

        # get latent dimension
        # self.state_shape = tuple(self.ds.snapshot_shape)
        # self.ed.arch_params.state_shape = self.state_shape
        # self.latent_dim = self.ed_class.calculate_latent_dim(cfg=self.ed)
        # self.ed.arch_params.latent_dim = self.latent_dim

        # add state_shape and latent_dim to arch_params

    # @property
    # def train(self) -> Dict[str, Any]:
    #     cfg = self.raw_config['training']
    #     cfg["ndim"] = self.latent_dim
    #     return cfg

    def print_summary(self) -> None:

        self.logger.info(f"{'='*20} configs {'='*20}")
        self.logger.info(f"{'='*20} dataset: {self.ds_name} {'='*20}")
        self.logger.info(f"{'='*20} encoder_decoder: {self.ed_name} {'='*20}")
        self.logger.info(f"{'='*20} ckpt_path: {self.ckpt_path} {'='*20}")
        self.logger.info(f"{'@'*5} ed: {self.ed_name}")
        # self.logger.info(f"{'@'*5} train: {self.train}")
