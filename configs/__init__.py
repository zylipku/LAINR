import os

from typing import Dict, Any
import logging

import json

from components import ed_name2class, ld_name2class, uq_name2class

# for typing
from components import EncoderDecoder, LatentDynamics, Uncertainty


class LAConfigs:

    ALL_DATASET_NAME = ['sw', 'era5']
    ALL_COMP_NAME = ['pca', 'cae', 'cae400', 'aeflow', 'fouriernet',
                     'inr2', 'sinr', 'sinr2', 'sinr3', 'sinr4', 'sinr_noskip']
    ALL_DYN_NAME = ['linreg', 'lstm', 'rezero', 'neuralode', 'none']
    ALL_LOSS_NAME = ['mse', 'skip', 'weighted']
    ALL_UQ_NAME = ['none', 'vacuous', 'diagonal', 'srn', 'cholesky']

    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_CKPT_DIR = os.path.join(BASE_DIR, '../ckpts2')

    @property
    def name(self) -> str:
        if self.custom_name is None:
            return self.ed_name + '_' + self.ld_name
        else:
            return self.custom_name

    @property
    def json_path(self) -> str:
        return os.path.join(self.BASE_DIR, self.ds_name, f'{self.name}.json')

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.BASE_CKPT_DIR, self.ds_name, self.ls_name, f'{self.name}.pt')

    @property
    def warm_start_ckpt_path(self) -> str:
        return os.path.join(self.BASE_CKPT_DIR, self.ds_name, self.ls_name, f'{self.ed_name}_none.pt')

    @staticmethod
    def load_json(json_path: str):
        with open(json_path, 'r') as f:
            config = json.load(f)
        return config

    def __init__(self, logger: logging.Logger,
                 ds_name: str,
                 ed_name: str,
                 ld_name: str,
                 ls_name: str,
                 uq_name: str,
                 custom_name: str = None) -> None:
        self.logger = logger

        assert ds_name.lower() in self.ALL_DATASET_NAME
        assert ed_name.lower() in self.ALL_COMP_NAME
        assert ld_name.lower() in self.ALL_DYN_NAME
        assert ls_name.lower() in self.ALL_LOSS_NAME
        assert uq_name.lower() in self.ALL_UQ_NAME

        self.ds_name = ds_name.lower()
        self.ed_name = ed_name.lower()
        self.ld_name = ld_name.lower()
        self.ls_name = ls_name.lower()
        self.uq_name = uq_name.lower()

        self.custom_name = custom_name

        if os.path.exists(self.json_path):
            self.raw_config = self.load_json(self.json_path)
            self.logger.info(f"load method_config:'{self.json_path}'successfully.")
        else:
            self.logger.warning(f"loading configs from\n'{self.json_path}'\nfailed, using configs for separate parts")

            ed_path = os.path.join(self.BASE_DIR, self.ds_name, f'{self.ed_name}.json')
            self.raw_config = self.load_json(ed_path)
            self.logger.info(f"load ed_config:'{ed_path}'successfully.")

            if self.ld_name != 'none':
                ld_path = os.path.join(self.BASE_DIR, self.ds_name, f'{self.ld_name}.json')
                self.logger.info(f"load ld_config:'{ld_path}'successfully.")
                self.raw_config = self.raw_config | self.load_json(ld_path)

        self.ed_class: EncoderDecoder = ed_name2class[self.ed_name.lower()]
        self.ld_class: LatentDynamics = ld_name2class[self.ld_name.lower()]
        self.uq_class: Uncertainty = uq_name2class[self.uq_name.lower()]

        self.state_shape = (128, 64, 2)

        self.latent_dim = self.ed_class.calculate_latent_dim(state_shape=self.state_shape, **self.ed)

        self.ed_need_train = not 'pca' in ed_name
        self.ld_need_train = not 'none' in ld_name

        self.ed_need_cache = 'fourier' in ed_name or 'inr' in ed_name

        self.warm_start = self.train.get('warm_start', False)

    @property
    def ed(self):
        return self.raw_config['encoder_decoder']

    @property
    def ld(self):
        try:
            cfg = self.raw_config['latent_dynamics']
            cfg["params"]["ndim"] = self.latent_dim
            return cfg
        except:
            return {"params": {}}

    @property
    def uq(self):
        try:
            cfg = self.raw_config['uncertainty']
            cfg["params"]["ndim"] = self.latent_dim
            return cfg
        except:
            return {"params": {'ndim': self.latent_dim}}

    @property
    def train(self) -> Dict[str, Any]:
        cfg = self.raw_config['training']
        cfg["ndim"] = self.latent_dim
        return cfg

    def print_summary(self) -> None:

        self.logger.info(f"{'='*20} configs {'='*20}")
        self.logger.info(f"{'='*20} dataset: {self.ds_name} {'='*20}")
        self.logger.info(f"{'='*20} encoder_decoder: {self.ed_name} {'='*20}")
        self.logger.info(f"{'='*20} latent_dynamics: {self.ld_name} {'='*20}")
        self.logger.info(f"{'='*20} uncertainty: {self.uq_name} {'='*20}")
        self.logger.info(f"{'='*20} loss: {self.ls_name} {'='*20}")
        self.logger.info(f"{'='*20} custom_name: {self.custom_name} {'='*20}")
        self.logger.info(f"{'='*20} ckpt_path: {self.ckpt_path} {'='*20}")
        self.logger.info(f"{'='*20} json_path: {self.json_path} {'='*20}")
        self.logger.info(f"{'@'*5} ed: {self.ed}")
        self.logger.info(f"{'@'*5} ld: {self.ld}")
        self.logger.info(f"{'@'*5} train: {self.train}")
