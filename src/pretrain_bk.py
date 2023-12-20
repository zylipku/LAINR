import sys
import os
import logging

import argparse

# hydra
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from config import PreTrainConfig
from configs.pretrain.pretrain_conf_schema import PreTrainConfig, ModelConfig, DatasetConfig, ArchConfig, TrainingConfig

# torch
import torch
from torch.utils.data import DataLoader

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# components
from components import get_encoder_decoder, get_latent_dynamics, get_uncertainty
from components import EncoderCache
# datasets
from datasets import load_dataset

# metrics
from metrics import get_metrics

# misc
from common import set_seed


def main_worker(rank, num_gpus: int, logger: logging.Logger, cfg: DictConfig):

    # initialize logger ---------------------------------------------
    # ---------------------------------------------------------------

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '13412'

    # Initialize the disctrbuted environment
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)

    # Set the seed for reproducibility
    set_seed(2357)

    # set the device for current process
    device = torch.device(f'cuda:{rank}')

    # Load the configuration for the selected model and dataset
    configs = PreTrainConfig(logger=logger, cfg=cfg)

    if rank == 0:
        configs.print_summary()

    # Load your dataset
    trainset = load_dataset(logger, cfg=configs.ds, group='train')
    if hasattr(trainset, 'height_std'):
        testset = load_dataset(logger, cfg=configs.ds, group='test',
                               height_std_mean=(trainset.height_std, trainset.height_mean),
                               vorticity_std_mean=(trainset.vorticity_std, trainset.vorticity_mean),)
    else:
        testset = load_dataset(logger, cfg=configs.ds, group='test')
    # Use distributed sampler to split the dataset into chunks for each process
    sampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=configs.ed.training_params.bs, sampler=sampler)
    # Use distributed sampler to split the dataset into chunks for each process
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=configs.ed.training_params.bs, sampler=sampler)

    criterion = get_metrics(name=configs.ls_name, phi_theta=testset.coords_ang)

    # Load the models
    encoder_decoder = get_encoder_decoder(logger, name=configs.ed_name, cfg=configs.ed)

    # Convert to distributed models

    # encoder_decoder
    encoder_decoder = encoder_decoder.to(device)
    if configs.ed_need_train:
        encoder_decoder = DDP(encoder_decoder, device_ids=[rank])

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

    exit(0)

    # Run your training loop
    trainer = Trainer(logger=logger,
                      encoder_decoder=encoder_decoder,
                      encoder_cache=encoder_cache_tr,
                      dataloader=trainloader,
                      criterion=criterion,
                      configs=configs,
                      rank=rank)

    evaluator = Evaluator(logger=logger,
                          encoder_decoder=encoder_decoder,
                          encoder_cache=encoder_cache_ts,
                          dataloader=testloader,
                          criterion=criterion,
                          configs=configs,
                          rank=rank)

    trainer.train(evaluator=evaluator)
    # trainer.load_ckpt(configs.ckpt_path)
    # evaluator.evaluate()


cs = ConfigStore.instance()
cs.store(name="pretrain_schema", node=PreTrainConfig)
cs.store(name="encoder_decoder_schema", group='encoder_decoder', node=ModelConfig)


@hydra.main(config_path="configs/pretrain", config_name="pretrain", version_base='1.2')
def main_pretrain(cfg: PreTrainConfig):

    logger = logging.getLogger(__name__)

    print(f'{cfg.name=}')
    print(cfg.encoder_decoder.__module__)

    # Set the seed for reproducibility
    seed = cfg.seed
    set_seed(seed)

    # configurations for the number of GPUs
    num_gpus = cfg.num_gpus
    max_num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = max_num_gpus
        cfg.num_gpus = num_gpus
        logging.info(f'num_gpus is set to {num_gpus} as max available GPUs.')
    else:
        num_gpus = min(num_gpus, max_num_gpus)
        logging.info(f'num_gpus is set to {num_gpus} [max num_gpus={max_num_gpus}].')

    logging.info(f'Seed: {seed}, start...')

    logger.info('\n\033[91mUsing the following configurations:\033[0m\n' + str(OmegaConf.to_yaml(cfg)))

    exit(0)
    # Specify the number of GPUs to use
    # num_gpus = 1

    # Run the training loop
    mp.spawn(main_worker, args=(num_gpus, logger, cfg), nprocs=num_gpus, join=True)


if __name__ == '__main__':

    main_pretrain()
