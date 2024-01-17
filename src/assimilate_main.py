import os
import logging
from datetime import datetime

from typing import *
from functools import partial

# mlflow
import mlflow
from mlflow import log_params, log_metrics, log_artifact, log_metric

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import create_logger, set_seed, transform_state_dict

from datasets import load_dataset
from components import EncoderDecoder, LatentDynamics, UncertaintyEst
from components import get_encoder_decoder, get_latent_dynamics, get_uncertainty_est
from components import EncoderCache
from metrics import get_metrics

# hmm
from hmm import Operator
from assimilation.xps import XPS
from assimilation.xps_uq import XPSuq

# hydra
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs.assimilate.assimilate_conf_schema import AssimilateConfig
from configs.conf_schema import EDConfig, LDConfig, UEConfig, DatasetConfig

# pandas
import pandas as pd


def conf_prepare(cfg: AssimilateConfig):

    cfg.encoder_decoder.arch_params.state_channels = cfg.dataset.snapshot_shape[-1]
    cfg.encoder_decoder.arch_params.state_size = cfg.dataset.snapshot_shape[:-1]
    cfg.latent_dynamics.latent_dim = cfg.encoder_decoder.latent_dim


cs = ConfigStore.instance()
cs.store(name="assimilate_schema", node=AssimilateConfig)
cs.store(group="encoder_decoder", name="encoder_decoder_schema", node=EDConfig)
cs.store(group="latent_dynamics", name="latent_dynamics_schema", node=LDConfig)
cs.store(group="uncertainty_est", name="uncertainty_est_schema", node=UEConfig)
cs.store(name="dataset_schema", group='dataset', node=DatasetConfig)


@hydra.main(config_path="configs/assimilate", config_name="config", version_base='1.2')
def main_assimilate(cfg: AssimilateConfig):

    conf_prepare(cfg)

    # dataset and model args
    ds_name = cfg.dataset.name
    # ['shallow_water', 'qg_model', 'shallow_water_normalized']
    ed_name = cfg.encoder_decoder.model_name
    # comp: ['linreg', 'cae', 'aeflow', 'aeflow2', 'aeflow-raw', 'fouriernet']
    ld_name = cfg.encoder_decoder.model_name
    # dyncomp: ['linreg', 'rezero', 'neuralode']

    # assimilation args
    sigma_x_b = cfg.sigma_x_b
    sigma_z_b = cfg.sigma_z_b
    sigma_m = cfg.sigma_m
    sigma_o = cfg.sigma_o
    n_obs = cfg.n_obs
    
    ens_num = cfg.ens_num
    infl = cfg.infl

    logger = logging.getLogger('assimilate')

    # set seed and device
    set_seed(cfg.seed)
    device = torch.device('cpu' if cfg.cuda_id is None else f'cuda:{cfg.cuda_id}')

    logger.info('Set device to: ' + str(device))
    logger.info('\n\033[91mUsing the following configurations:\033[0m\n' + str(OmegaConf.to_yaml(cfg)))

    # mlflow --------------------------------------------------------
    mlflow.set_experiment('assimilate' + '_' +
                          cfg.dataset.name + '_' +
                          cfg.encoder_decoder.model_name + '_' +
                          cfg.latent_dynamics.model_name + '_' +
                          cfg.uncertainty_est.model_name)
    mlflow.start_run(run_name=cfg.encoder_decoder.name + '_' +
                     cfg.latent_dynamics.name + '_' +
                     cfg.uncertainty_est.model_name + '_' +
                     str(sigma_z_b) + '_' +
                     str(sigma_m) + '_' +
                     datetime.now().strftime("%Y%m%d_%H%M%S"))
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    # ---------------------------------------------------------------

    ass_nsteps = cfg.ass_nsteps
    save_dir = cfg.save_dir

    # initialize dataset
    dataset_class = load_dataset(logger, cfg=cfg.dataset)
    dataset_tr = dataset_class.get_metadata('tr')
    dataset_va = dataset_class.get_metadata('va')
    dataset_ts = dataset_class.get_metadata('ts')

    # using the testing dataset
    dataset = dataset_ts

    loss_fn_va = get_metrics(name=cfg.encoder_decoder.training_params.loss_fn_va,
                             phi_theta=dataset_va.coords['coord_latlon'])

    if cfg.encoder_decoder.need_cache:
        loss_fn_inner_loop = get_metrics(name=cfg.encoder_decoder.arch_params.inner_loop_loss_fn,
                                         phi_theta=dataset_va.coords['coord_latlon'])
    else:
        loss_fn_inner_loop = None
    # Load the models
    encoder_decoder: EncoderDecoder = get_encoder_decoder(logger,
                                                          name=cfg.encoder_decoder.model_name,
                                                          loss_fn_inner_loop=loss_fn_inner_loop,
                                                          **cfg.encoder_decoder.arch_params)
    latent_dynamics: LatentDynamics = get_latent_dynamics(logger,
                                                          name=cfg.latent_dynamics.model_name,
                                                          ndim=cfg.latent_dynamics.latent_dim,
                                                          **cfg.latent_dynamics.arch_params)
    uncertainty_est: UncertaintyEst = get_uncertainty_est(logger,
                                                          name=cfg.uncertainty_est.model_name,
                                                          ndim=cfg.latent_dynamics.latent_dim,
                                                          **cfg.uncertainty_est.arch_params)

    # load encoder_decoder and latent_dynamics from ckpt
    encoder_decoder = encoder_decoder.to(device)
    latent_dynamics = latent_dynamics.to(device)

    # without uncertainty estimation
    if uncertainty_est is None:
        logger.info(f'loading ckpt from "{cfg.finetune_ckpt_path}')
        ckpt = torch.load(cfg.finetune_ckpt_path, map_location=device)
        encoder_decoder.load_state_dict(transform_state_dict(ckpt['ed']))
        latent_dynamics.load_state_dict(transform_state_dict(ckpt['ld']))
    else:
        uncertainty_est = uncertainty_est.to(device)
        logger.info(f'loading ckpt from "{cfg.postproc_ckpt_path}')
        ckpt = torch.load(cfg.postproc_ckpt_path, map_location=device)
        encoder_decoder.load_state_dict(transform_state_dict(ckpt['ed']))
        latent_dynamics.load_state_dict(transform_state_dict(ckpt['ld']))
        uncertainty_est.load_state_dict(transform_state_dict(ckpt['ue']))

    xx_t = dataset.trajs.data[0][:ass_nsteps]  # assimilate only for the first trajectory
    # shape: (nsteps, 128, 64, 2)
    coord_cartes = dataset.coords['coord_cartes'][None, ...]
    # shape: (1, 128, 64, 3)
    coord_latlon = dataset.coords['coord_latlon'][None, ...]
    # shape: (1, 128, 64, 3)

    factory_kwargs = {'device': device, 'dtype': torch.float32}

    xx_t = xx_t.to(**factory_kwargs)
    coord_cartes = coord_cartes.to(**factory_kwargs)
    coord_latlon = coord_latlon.to(**factory_kwargs)

    # latent state: (2x200,)
    # dynamic flow: (1, 1, 2x200) -> (1, 1, 2x200)
    # decoder: (1, 1, 2x200) -> (128, 64, 2) with coords: (128, 64, 1, 3)
    # observation: (128, 64, 2) -> flatten -> (n_obs=1024,) 6.25%

    nstates = xx_t[0].numel()  # 128 x 64 x 2
    obs_idx = torch.randperm(nstates)[:n_obs].to(device=device)
    logger.debug(f'{obs_idx.shape=}')
    logger.debug(f'{obs_idx[:10]=}')

    def rnd_obs(x: torch.Tensor) -> torch.Tensor:

        x_flatten = x.contiguous().view(*x.shape[:-3], -1)
        y_obs = x_flatten[..., obs_idx]

        return y_obs

    xps = XPS(logger=logger,
              mod_dim=latent_dynamics.ndim,
              obs_dim=n_obs,
              opM=Operator.compose(latent_dynamics.forward),
              # z.shape: (bs=4, [state_dim=2]x[code_dim=200])
              opH=Operator.compose(partial(encoder_decoder.decode,
                                           coord_cartes=coord_cartes,
                                           coord_latlon=coord_latlon), rnd_obs),
              # codes.shape: (1, 1, state_dim=2, code_dim=200)
              obs_sigma=sigma_o,
              mod_sigma=sigma_m,
              uq=uncertainty_est,
              )

    # xps.add_method('ExKF', infl=1.02, **model.factory_kwargs)
    xps.add_method('EnKF', infl=infl, ens_dim=ens_num, **factory_kwargs)
    xps.add_method('SEnKF', infl=infl, ens_dim=ens_num, **factory_kwargs)
    xps.add_method('DEnKF', infl=infl, ens_dim=ens_num, **factory_kwargs)
    # xps.add_method('EnSRKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('ETKF', infl=infl, ens_dim=ens_num, **factory_kwargs)
    xps.add_method('ETKF-Q', infl=infl, ens_dim=ens_num, **factory_kwargs)

    # xx_t.shape=(nsteps, h, w, nstates=2)
    x_t0 = xx_t[0]
    x_t0_noised = x_t0 + torch.randn_like(x_t0) * sigma_x_b
    z0 = torch.zeros(latent_dynamics.ndim, **factory_kwargs)
    z_t0 = encoder_decoder.encode(x_t0_noised, coord_latlon=coord_latlon[0],
                                  z0=z0, optim_eval_max_inner_loops=10000)
    z_b = z_t0
    z_b = z_b.detach().cpu().reshape(-1)

    # for uncertainty estimator, sigma_z_b is calculated via jacobian
    if uncertainty_est is None:
        logger.info(f'Using predefined sigma_z_b:{sigma_z_b}')
        covB = torch.eye(latent_dynamics.ndim) * sigma_z_b**2
    else:
        logger.info(f'Calculating the statistical estimation of z0... (pseudo-inverse of the Jacobian)')
        # calculate the pseudo-inverse of the Jacobian to get the background estimation of z_b

        def decoder4jac(z: torch.Tensor):
            return encoder_decoder.decode(z, coord_cartes=coord_cartes[0], coord_latlon=coord_latlon[0])

        logger.info(f'{z_t0.shape=}')
        logger.info(f'{coord_cartes.shape=}')
        logger.info(f'{coord_latlon.shape=}')

        jac = torch.autograd.functional.jacobian(decoder4jac, z_t0)  # shape=(features_dim, dim_z)
        jac = jac.reshape(-1, jac.shape[-1])  # shape=(dim_x, dim_z)
        jac_pinv = torch.linalg.pinv(jac)  # shape=(dim_z, dim_x)

        jac = jac.detach().cpu()
        jac_pinv = jac_pinv.detach().cpu()

        logger.info(f'{jac.shape=}')
        logger.info(f'{jac_pinv.shape=}')
        logger.info(f'{torch.diagonal(jac_pinv)=}')
        logger.info(f'{torch.diagonal(jac_pinv@ jac_pinv.T)=}')
        covB = (jac_pinv * sigma_x_b**2) @ jac_pinv.T
        logger.info(f'{torch.diagonal(covB)=}')

    # create observations
    yy_o = []
    for x_t in xx_t[1:]:
        y_o = x_t.reshape(-1)[obs_idx].detach().cpu()
        y_o = y_o + torch.randn_like(y_o) * sigma_o  # virtual observation noise
        yy_o.append(y_o)
    yy_o = torch.stack(yy_o, dim=0)
    obs_t_idxs = list(range(1, xx_t.shape[0]))

    with torch.no_grad():

        save_folder = f'./{save_dir}/{ds_name}/{cfg.name}/'
        prefix = f'{cfg.name}_{sigma_z_b=}_{sigma_m=}_{ens_num=}_{infl=}_'

        xps.run(save_folder=save_folder,
                ass_data={
                    'x_b': z_b,
                    'covB': covB,
                    'yy_o': yy_o,
                    'obs_t_idxs': obs_t_idxs,
                },
                prefix=prefix,
                )

        rmse_df = xps.rmse2dataframe(xx_t[obs_t_idxs],
                                     eval_fn=loss_fn_va,
                                     decoder=partial(encoder_decoder.decode,
                                                     coord_cartes=coord_cartes,
                                                     coord_latlon=coord_latlon),
                                     device=device)

        df = rmse_df.copy()
        df['ed_name'] = cfg.encoder_decoder.name
        df['ld_name'] = cfg.latent_dynamics.name
        df['ue_name'] = cfg.uncertainty_est.name
        df['ens_num'] = cfg.ens_num
        df['infl'] = cfg.infl
        
        if uncertainty_est is None:
            df['sigma_z_b'] = sigma_z_b
            df['sigma_m'] = sigma_m
        else:
            df['sigma_z_b'] = np.nan
            df['sigma_m'] = np.nan

        logger.info(f'assimilation rmse:\n{df}')
        df.to_pickle(os.path.join(save_folder, prefix + 'dataframe.pkl'))
        df.to_csv(os.path.join(save_folder, prefix + 'dataframe.csv'))
        log_artifact(os.path.join(save_folder, prefix + 'dataframe.csv'))

        # zeros = torch.zeros(nstates)
        # zeros[obs_idx] = 1.
        # mask = zeros.reshape(128, 64, 2)
        # xps.plot3(xx_t, mask, save_folder,
        #           decoder=partial(encoder_decoder.decode, coords=coords),  # ! not implemented yet!!!
        #           device=device,
        #           prefix=prefix,
        #           )
        # xps.plot_rmse(xx_t, save_folder,
        #               decoder=partial(encoder_decoder.decode,
        #                               coord_cartes=coord_cartes,
        #                               coord_latlon=coord_latlon),
        #               device=device,
        #               prefix=prefix)

    mlflow.end_run()


if __name__ == '__main__':

    main_assimilate()
