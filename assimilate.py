import os
import sys
import getopt

from typing import Dict
from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader

from common import create_logger, set_seed, transform_state_dict

from dataset import load_dataset
from components import get_encoder_decoder, get_latent_dynamics
from configs import LAConfigs
from metrics import get_metrics


def main(args):

    set_seed(2357)

    # dataset and model args
    ds_name = args.ds
    # ['shallow_water', 'qg_model', 'shallow_water_normalized']
    ed_name = args.ed
    # comp: ['linreg', 'cae', 'aeflow', 'aeflow2', 'aeflow-raw', 'fouriernet']
    ld_name = args.ld
    # dyncomp: ['linreg', 'rezero', 'neuralode']
    ls_name = args.ls

    # assimilation args
    sigma_x_b = args.sigma_x_b
    sigma_xz_b_ratio = args.sigma_xz_b_ratio
    mod_sigma = args.mod_sigma
    n_obs = args.n_obs

    # device args
    cudaid = args.cudaid
    device = torch.device(f'cuda:{cudaid}')

    ass_nsteps = 200
    save_dir = 'results2'

    if ld_name is None or ld_name.lower() == 'none':
        ld_name = 'none'
    method_name = ed_name + '_' + ld_name

    filename = os.path.split(__file__)[-1].split('.')[0]

    logger = create_logger(
        prefix=filename,
        sub_folder=f'{ds_name}/{method_name}',
        level=20,  # 20 for INFO, 10 for DEBUG
    )

    configs = LAConfigs(logger=logger,
                        ds_name=ds_name,
                        ed_name=ed_name,
                        ld_name=ld_name,
                        ls_name=ls_name,
                        uq_name='none'
                        )

    configs.print_summary()

    # initialize dataset
    trainset = load_dataset(logger, dataset_name=ds_name, group='train')
    testset = load_dataset(logger, dataset_name=ds_name, group='test',
                           height_std_mean=(trainset.height_std, trainset.height_mean),
                           vorticity_std_mean=(trainset.vorticity_std, trainset.vorticity_mean),
                           offgrid=args.offgrid
                           )

    testloader = DataLoader(testset, batch_size=configs.train["batch_size"])

    # Load the models
    criterion = get_metrics(name=ls_name, phi_theta=testset.coords_ang)
    encoder_decoder = get_encoder_decoder(logger, name=ed_name, criterion=criterion, **configs.ed["params"])
    latent_dynamics = get_latent_dynamics(logger, name=ld_name, **configs.ld["params"])
    encoder_decoder = encoder_decoder.to(device)
    latent_dynamics = latent_dynamics.to(device)

    logger.info(f'loading ckpt from "{configs.ckpt_path}')
    ckpt = torch.load(configs.ckpt_path, map_location=device)

    encoder_decoder.load_state_dict(transform_state_dict(ckpt['ed']))
    latent_dynamics.load_state_dict(transform_state_dict(ckpt['ld']))

    xx = testset.data[0]
    # shape: (400, 128, 64, 2)
    coords = testset.coords[None, ...]
    # shape: (1, 128, 64, 3)

    from hmm import HMM, Operator
    from assimilation.xps import XPS

    xx_t = testset.data[0][:ass_nsteps]
    # shape=(nsteps, h, w, nstates=2)
    coords = testset.coords[None, ...]  # without bs, nsteps dimensions
    coords_ang = testset.coords_ang[None, ...]
    # shape=(h, w, coord_dim=3)

    factory_kwargs = {'device': device, 'dtype': torch.float32}

    xx_t = xx_t.to(**factory_kwargs)
    coords = coords.to(**factory_kwargs)
    coords_ang = coords_ang.to(**factory_kwargs)

    # latent state: (2x200,)
    # dynamic flow: (1, 1, 2x200) -> (1, 1, 2x200)
    # decoder: (1, 1, 2x200) -> (128, 64, 2) with coords: (128, 64, 1, 3)
    # observation: (128, 64, 2) -> flatten -> (n_obs=1024,) 6.25%

    nstates = xx_t[0].numel()
    obs_idx = torch.randperm(nstates)[:n_obs].to(device=device)
    logger.info(f'{obs_idx.shape=}')
    logger.info(f'{obs_idx[:10]=}')

    def rnd_obs(x: torch.Tensor) -> torch.Tensor:

        x_flatten = x.contiguous().view(*x.shape[:-3], -1)
        y_obs = x_flatten[..., obs_idx]

        return y_obs

    obs_sigma = 1e-1

    # import torchsummary as ts

    # logging.info(ts.summary(model.inr, [(128, 64, 1, 3), (1, 1, 2, 200)], batch_size=64))

    xps = XPS(
        logger=logger,
        mod_dim=latent_dynamics.ndim,
        obs_dim=n_obs,
        opM=Operator.compose(latent_dynamics.forward),
        # z.shape: (bs=4, [state_dim=2]x[code_dim=200])
        opH=Operator.compose(partial(encoder_decoder.decode, coords=coords, coords_ang=coords_ang), rnd_obs),
        # codes.shape: (1, 1, state_dim=2, code_dim=200)
        mod_sigma=mod_sigma,
        obs_sigma=obs_sigma,
    )
    # xps.add_method('ExKF', infl=1.02, **model.factory_kwargs)
    xps.add_method('EnKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('SEnKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('DEnKF', infl=1.02, ens_dim=64, **factory_kwargs)
    # xps.add_method('EnSRKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('ETKF', infl=1.02, ens_dim=64, **factory_kwargs)
    xps.add_method('ETKF-Q', infl=1.02, ens_dim=64, **factory_kwargs)

    sigma_z_b = sigma_x_b * sigma_xz_b_ratio

    # xx_t.shape=(nsteps, h, w, nstates=2)
    x_t0 = xx_t[0]
    x_t0_noised = x_t0 + torch.randn_like(x_t0) * sigma_x_b
    z0 = torch.zeros(latent_dynamics.ndim, **factory_kwargs)
    z_t0 = encoder_decoder.encode(x_t0_noised, z0=z0,
                                  coords=coords[0], coords_ang=coords_ang[0],
                                  group='eval', optim_eval_max_inner_loops=10000)
    z_b = z_t0
    z_b = z_b.detach().cpu().reshape(-1)

    yy_o = []
    for x_t in xx_t[1:]:
        y_o = x_t.reshape(-1)[obs_idx].detach().cpu()
        y_o = y_o + torch.randn_like(y_o) * obs_sigma  # virtual observation noise
        yy_o.append(y_o)
    yy_o = torch.stack(yy_o, dim=0)
    obs_t_idxs = list(range(1, xx_t.shape[0]))

    with torch.no_grad():

        save_folder = f'./{save_dir}/{ds_name}/{method_name}/'
        prefix = f'{method_name}_{sigma_x_b=}_{sigma_z_b=}_{mod_sigma=}_{n_obs=}_'

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
                               decoder=partial(encoder_decoder.decode, coords=coords, coords_ang=coords_ang),
                               device=device)
        logger.info(f'RESULTS: {sigma_x_b=}, {sigma_z_b=}, {mod_sigma=}, {n_obs=} | {results}')
        zeros = torch.zeros(nstates)
        zeros[obs_idx] = 1.
        mask = zeros.reshape(128, 64, 2)
        # xps.plot3(xx_t, mask, save_folder,
        #           decoder=partial(encoder_decoder.decode, coords=coords),  # ! not implemented yet!!!
        #           device=device,
        #           prefix=prefix,
        #           )
        xps.plot_rmse(xx_t, save_folder,
                      decoder=partial(encoder_decoder.decode, coords=coords, coords_ang=coords_ang),
                      device=device,
                      prefix=prefix)


if __name__ == '__main__':

    import argparse
    # Specify the defaults
    ds = 'dino_shallow_water3'
    ed = 'cae'
    ld = 'rezero'
    ls = 'weighted'
    offgrid = False
    # default_ed = 'aeflow'
    # default_ld = 'rezero'

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--ds', type=str, default=ds, help='Dataset name.')
    parser.add_argument('--ed', type=str, default=ed, help='Encoder-decoder model name.')
    parser.add_argument('--ld', type=str, default=ld, help='Latent dynamics model name.')
    parser.add_argument('--ls', type=str, default=ls, help='loss name.')
    parser.add_argument('--sigma-x-b', type=float, default=1e-1, help='sigma_x_b')
    parser.add_argument('--sigma-xz-b-ratio', type=float, default=1e-0, help='sigma_xz_b_ratio')
    parser.add_argument('--mod-sigma', type=float, default=3e-1, help='mod_sigma')
    parser.add_argument('--n-obs', type=int, default=1024, help='n_obs')
    parser.add_argument('--cudaid', type=int, default=2, help='cuda id')
    parser.add_argument('--offgrid', type=bool, default=offgrid, help='assimilate offgrid points')

    args = parser.parse_args()
    main(args)
