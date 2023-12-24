import os
import logging

from typing import *
from functools import partial


import torch
from torch.utils.data import DataLoader

from common import create_logger, set_seed, transform_state_dict

from datasets import load_dataset
from components import EncoderDecoder, LatentDynamics
from components import get_encoder_decoder, get_latent_dynamics
from components import EncoderCache
from metrics import get_metrics

# hmm
from hmm import Operator
from assimilation.xps import XPS

# hydra
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from omegaconf import OmegaConf

from configs.assimilate.assimilate_conf_schema import AssimilateConfig

cs = ConfigStore.instance()
cs.store(name="assimilate_schema", node=AssimilateConfig)


@hydra.main(config_path="configs/assimilate", config_name="config", version_base='1.2')
def main_assimilate(cfg: AssimilateConfig):

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

    logger = logging.getLogger('assimilate')

    # set seed and device
    set_seed(cfg.seed)
    device = torch.device('cpu' if cfg.cuda_id is None else f'cuda:{cfg.cuda_id}')

    ass_nsteps = cfg.ass_nsteps
    save_dir = cfg.save_dir

    # initialize dataset
    dataset_class = load_dataset(logger, cfg=cfg.dataset)
    dataset = dataset_class.get_metadata('test')

    # Load the models
    loss_fn_va = get_metrics(name=cfg.encoder_decoder.training_params.loss_fn_va,
                             phi_theta=dataset.coords['coord_latlon'])
    # Load the models
    encoder_decoder: EncoderDecoder = get_encoder_decoder(logger,
                                                          name=cfg.encoder_decoder.model_name,
                                                          criterion=loss_fn_va,
                                                          **cfg.encoder_decoder.arch_params)
    latent_dynamics: LatentDynamics = get_latent_dynamics(logger,
                                                          name=cfg.latent_dynamics.model_name,
                                                          ndim=cfg.latent_dynamics.latent_dim,
                                                          **cfg.latent_dynamics.arch_params)

    # load encoder_decoder and latent_dynamics from ckpt
    encoder_decoder = encoder_decoder.to(device)
    latent_dynamics = latent_dynamics.to(device)

    logger.info(f'loading ckpt from "{cfg.ckpt_path}')
    ckpt = torch.load(cfg.ckpt_path, map_location=device)

    encoder_decoder.load_state_dict(transform_state_dict(ckpt['ed']))
    latent_dynamics.load_state_dict(transform_state_dict(ckpt['ld']))

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

    xps = XPS(
        logger=logger,
        mod_dim=latent_dynamics.ndim,
        obs_dim=n_obs,
        opM=Operator.compose(latent_dynamics.forward),
        # z.shape: (bs=4, [state_dim=2]x[code_dim=200])
        opH=Operator.compose(partial(encoder_decoder.decode,
                                     coord_cartes=coord_cartes,
                                     coord_latlon=coord_latlon), rnd_obs),
        # codes.shape: (1, 1, state_dim=2, code_dim=200)
        mod_sigma=sigma_m,
        obs_sigma=sigma_o,
    )
    # xps.add_method('ExKF', infl=1.02, **model.factory_kwargs)
    xps.add_method('EnKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('SEnKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('DEnKF', infl=1.02, ens_dim=64, **factory_kwargs)
    # xps.add_method('EnSRKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('ETKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('ETKF-Q', infl=1.02, ens_dim=64, **factory_kwargs)

    # xx_t.shape=(nsteps, h, w, nstates=2)
    x_t0 = xx_t[0]
    x_t0_noised = x_t0 + torch.randn_like(x_t0) * sigma_x_b
    z0 = torch.zeros(latent_dynamics.ndim, **factory_kwargs)
    z_t0 = encoder_decoder.encode(x_t0_noised, coord_latlon=coord_latlon[0],
                                  z0=z0, optim_eval_max_inner_loops=10000)
    z_b = z_t0
    z_b = z_b.detach().cpu().reshape(-1)

    # create observations
    yy_o = []
    for x_t in xx_t[1:]:
        y_o = x_t.reshape(-1)[obs_idx].detach().cpu()
        y_o = y_o + torch.randn_like(y_o) * sigma_o  # virtual observation noise
        yy_o.append(y_o)
    yy_o = torch.stack(yy_o, dim=0)
    obs_t_idxs = list(range(1, xx_t.shape[0]))

    with torch.no_grad():

        save_folder = f'./{save_dir}/{ds_name}/{cfg.method_name}/'
        prefix = f'{cfg.method_name}_{sigma_x_b=}_{sigma_z_b=}_{sigma_m=}_{n_obs=}_'

        xps.run(save_folder=save_folder,
                ass_data={
                    'x_b': z_b,
                    'covB': torch.eye(latent_dynamics.ndim) * sigma_z_b**2,
                    'yy_o': yy_o,
                    'obs_t_idxs': obs_t_idxs,
                },
                prefix=prefix,
                )

        results = xps.evaluate(xx_t[obs_t_idxs],
                               decoder=partial(encoder_decoder.decode,
                                               coord_cartes=coord_cartes,
                                               coord_latlon=coord_latlon),
                               device=device)
        logger.info(f'RESULTS: {sigma_x_b=}, {sigma_z_b=}, {sigma_m=}, {n_obs=} | {results}')
        zeros = torch.zeros(nstates)
        zeros[obs_idx] = 1.
        mask = zeros.reshape(128, 64, 2)
        # xps.plot3(xx_t, mask, save_folder,
        #           decoder=partial(encoder_decoder.decode, coords=coords),  # ! not implemented yet!!!
        #           device=device,
        #           prefix=prefix,
        #           )
        xps.plot_rmse(xx_t, save_folder,
                      decoder=partial(encoder_decoder.decode, coords=coord_cartes, coords_ang=coord_latlon),
                      device=device,
                      prefix=prefix)


if __name__ == '__main__':

    main_assimilate()
