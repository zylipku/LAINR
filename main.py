import os
import logging

import time

from typing import Dict

import argparse
from configs import LAConfigs
from components import get_encoder_decoder, get_latent_dynamics, get_uncertainty
from dataset import load_dataset
from train import Trainer, Evaluator, EncoderCache


import torch
from torch.utils.data import DataLoader

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from common import create_logger, set_seed
from metrics import get_metrics
from configs import LAConfigs


def main_worker(rank, num_gpus: int, args: argparse.Namespace):

    filename = os.path.split(__file__)[-1].split('.')[0]
    method_name = f'{args.ed}_{args.ld}'if args.cfgn is None else args.cfgn
    logger = create_logger(prefix=filename,
                           sub_folder=f'{args.ds}/{method_name}/{args.ls}',
                           level=logging.INFO)

    os.environ['MASTER_ADDR'] = 'localhost'

    os.environ['MASTER_PORT'] = '14918'

    # Initialize the disctrbuted environment
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)

    # Set the seed for reproducibility
    set_seed(2357)

    # set the device for current process
    device = torch.device(f'cuda:{rank}')

    # Load the configuration for the selected model and dataset
    configs = LAConfigs(logger=logger,
                        ds_name=args.ds,
                        ed_name=args.ed,
                        ld_name=args.ld,
                        ls_name=args.ls,
                        uq_name=args.uq,
                        custom_name=args.cfgn)

    if rank == 0:
        configs.print_summary()

    # Load your dataset
    trainset = load_dataset(logger, dataset_name=args.ds, group='train')
    if hasattr(trainset, 'height_std'):
        testset = load_dataset(logger, dataset_name=args.ds, group='test',
                               height_std_mean=(trainset.height_std, trainset.height_mean),
                               vorticity_std_mean=(trainset.vorticity_std, trainset.vorticity_mean),)
    else:
        testset = load_dataset(logger, dataset_name=args.ds, group='test')
    # Use distributed sampler to split the dataset into chunks for each process
    sampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=configs.train["batch_size"], sampler=sampler)
    # Use distributed sampler to split the dataset into chunks for each process
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=configs.train["batch_size"], sampler=sampler)

    criterion = get_metrics(name=args.ls, phi_theta=testset.coords_ang)

    # Load the models
    encoder_decoder = get_encoder_decoder(logger, name=args.ed, criterion=criterion, **configs.ed["params"])
    latent_dynamics = get_latent_dynamics(logger, name=args.ld, **configs.ld["params"])
    uncertainty = get_uncertainty(logger, name=args.uq, **configs.uq["params"])

    # Convert to distributed models

    # encoder_decoder
    encoder_decoder = encoder_decoder.to(device)
    if configs.ed_need_train:
        encoder_decoder = DDP(encoder_decoder, device_ids=[rank])

    # latent_dynamics
    if configs.ld_need_train:
        latent_dynamics = latent_dynamics.to(device)
        latent_dynamics = DDP(latent_dynamics, device_ids=[rank])
        uncertainty = uncertainty.to(device)
        uncertainty = DDP(uncertainty, device_ids=[rank])

    # encoder_cache
    if configs.ed_need_cache:

        ncodes = len(trainset)
        nsteps = next(iter(trainloader))['data'].shape[1]
        ndim = configs.latent_dim
        encoder_cache_tr = EncoderCache(ncodes=ncodes, shape=(nsteps, ndim))
        ncodes = len(testset)
        encoder_cache_ts = EncoderCache(ncodes=ncodes, shape=(nsteps, ndim))

        encoder_cache_tr = encoder_cache_tr.to(device)
        encoder_cache_tr = DDP(encoder_cache_tr, device_ids=[rank], find_unused_parameters=True)
        encoder_cache_ts = encoder_cache_ts.to(device)
        encoder_cache_ts = DDP(encoder_cache_ts, device_ids=[rank], find_unused_parameters=True)

    else:
        encoder_cache_tr = EncoderCache()
        encoder_cache_ts = EncoderCache()

    # Run your training loop
    trainer = Trainer(logger=logger,
                      encoder_decoder=encoder_decoder,
                      latent_dynamics=latent_dynamics,
                      uncertainty=uncertainty,
                      encoder_cache=encoder_cache_tr,
                      dataloader=trainloader,
                      criterion=criterion,
                      configs=configs,
                      rank=rank)

    evaluator = Evaluator(logger=logger,
                          encoder_decoder=encoder_decoder,
                          latent_dynamics=latent_dynamics,
                          uncertainty=uncertainty,
                          encoder_cache=encoder_cache_ts,
                          dataloader=testloader,
                          criterion=criterion,
                          configs=configs,
                          rank=rank)

    trainer.train(evaluator=evaluator)
    # trainer.load_ckpt(configs.ckpt_path)
    # evaluator.evaluate()


def main(args: argparse.Namespace):

    # Specify the number of GPUs to use
    num_gpus = args.num_gpus
    # num_gpus = 1

    # Run the training loop
    mp.spawn(main_worker, args=(num_gpus, args), nprocs=num_gpus, join=True)


if __name__ == '__main__':

    # Specify the defaults
    default_ds = 'sw'
    default_ed = 'sinr'
    default_ld = 'neuralode'
    default_ls = 'weighted'
    default_uq = 'none'

    default_cfgn = None

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--ds', type=str, default=default_ds, help='Dataset name.')
    parser.add_argument('--ed', type=str, default=default_ed, help='Encoder-decoder model name.')
    parser.add_argument('--ld', type=str, default=default_ld, help='Latent dynamics model name.')
    parser.add_argument('--ls', type=str, default=default_ls, help='loss name.')
    parser.add_argument('--uq', type=str, default=default_uq, help='uncertainty model name.')
    parser.add_argument('--cfgn', type=str, default=default_cfgn, help='configs filename.')
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use.')

    args = parser.parse_args()
    main(args)
