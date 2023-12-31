import sys
import os
import logging

from datetime import datetime

import cloudpickle as pkl

# hydra
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from omegaconf import OmegaConf

from configs.conf_schema import EDConfig
from configs.pretrain.pretrain_conf_schema import PreTrainConfig

# mlflow
import mlflow

# torch
import torch
from torch.utils.data import DataLoader

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# components
from components import EncoderDecoder
from components import get_encoder_decoder
from components import EncoderCache
# datasets
from datasets import load_dataset

# metrics
from metrics import get_metrics

# misc
from common import set_seed

from trainer.pretrain_prev import PreTrainer


def conf_prepare(cfg: PreTrainConfig):

    cfg.encoder_decoder.arch_params.state_channels = cfg.dataset.snapshot_shape[-1]
    cfg.encoder_decoder.arch_params.state_size = cfg.dataset.snapshot_shape[:-1]


def main_worker(rank, num_gpus: int, cfg: PreTrainConfig):

    conf_prepare(cfg)

    # initialize logger ---------------------------------------------
    if rank == 0:
        singleton_state = pkl.load(open('.hydra_state.pkl', 'rb'))
        HydraConfig.set_state(singleton_state)
        hc = HydraConfig.get()
        # logging.config.dictConfig(OmegaConf.to_container(hc.hydra_logging, resolve=True))
        configure_log(hc.job_logging, hc.verbose)

    logger = logging.getLogger('pretrain_prev')

    logger.info(f'{cfg.name=}')
    logger.info(f'{cfg.ckpt_path=}')

    # configurations for the number of GPUs
    num_gpus = cfg.num_gpus
    logger.info(f'num_gpus is set to {num_gpus}.')

    # Set the seed for reproducibility
    seed = cfg.seed
    set_seed(seed)
    logger.info(f'Seed: {seed}, start...')

    cfg.phase = 'pretrain-prev'
    cfg.bs = cfg.bs // 10
    cfg.encoder_decoder.arch_params.inner_loop_max_iters = 1000

    # print summary
    logger.info('\n\033[91mUsing the following configurations:\033[0m\n' + str(OmegaConf.to_yaml(cfg)))
    # ---------------------------------------------------------------

    # DDP initialization---------------------------------------------
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '21972'

    # Initialize the disctrbuted environment
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)

    # set the device for current process
    device = torch.device(f'cuda:{rank}')
    # ---------------------------------------------------------------

    # mlflow --------------------------------------------------------
    if rank == 0:
        mlflow.set_experiment('pretrain_prev_' + cfg.dataset.name + '_' + cfg.encoder_decoder.model_name)
        mlflow.start_run(run_name=cfg.encoder_decoder.name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    # ---------------------------------------------------------------

    # Load the dataset
    if rank == 0:
        logger.info('Loading dataset...')
        dataset = load_dataset(logger, cfg=cfg.dataset)
        dataset_tr, dataset_va, dataset_ts = dataset.get_datasets('pretrain-seq')
        dist.broadcast_object_list([dataset_tr, dataset_va, dataset_ts], src=0)
    else:
        object_list = [None, None, None]
        dist.broadcast_object_list(object_list, src=0)
        dataset_tr, dataset_va, dataset_ts = object_list

    # Use distributed sampler to split the dataset into chunks for each process
    sampler_tr = DistributedSampler(dataset_tr)
    sampler_va = DistributedSampler(dataset_va)
    sampler_ts = DistributedSampler(dataset_ts)
    dataloader_tr = DataLoader(dataset_tr, batch_size=cfg.bs, sampler=sampler_tr)
    dataloader_va = DataLoader(dataset_va, batch_size=cfg.bs, sampler=sampler_va)
    dataloader_ts = DataLoader(dataset_ts, batch_size=cfg.bs, sampler=sampler_ts)

    loss_fn_tr = get_metrics(name=cfg.encoder_decoder.training_params.loss_fn_tr,
                             phi_theta=dataset_tr.coords['coord_latlon'])
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
    latent_dim = encoder_decoder.calculate_latent_dim(
        state_shape=cfg.dataset.snapshot_shape, **cfg.encoder_decoder.arch_params)

    # Convert to distributed models

    # encoder_decoder
    encoder_decoder = encoder_decoder.to(device)

    if cfg.encoder_decoder.need_train:
        encoder_decoder = DDP(encoder_decoder, device_ids=[rank])

    # encoder_cache
    if cfg.encoder_decoder.need_cache:

        ndim = latent_dim

        ncodes_tr = len(dataset_tr)
        ncodes_va = len(dataset_va)
        ncodes_ts = len(dataset_ts)

        encoder_cache_tr = EncoderCache(ncodes=ncodes_tr, shape=(cfg.dataset.window_width, ndim,)).to(device)
        encoder_cache_va = EncoderCache(ncodes=ncodes_va, shape=(ndim,)).to(device)
        encoder_cache_ts = EncoderCache(ncodes=ncodes_ts, shape=(ndim,)).to(device)

        encoder_cache_tr = DDP(encoder_cache_tr, device_ids=[rank], find_unused_parameters=True)
        encoder_cache_va = DDP(encoder_cache_va, device_ids=[rank], find_unused_parameters=True)
        encoder_cache_ts = DDP(encoder_cache_ts, device_ids=[rank], find_unused_parameters=True)

    else:
        encoder_cache_tr = EncoderCache()
        encoder_cache_va = EncoderCache()
        encoder_cache_ts = EncoderCache()

    # Run your training loop
    trainer = PreTrainer(logger=logger,
                         encoder_decoder=encoder_decoder,
                         encoder_cache_tr=encoder_cache_tr,
                         encoder_cache_va=encoder_cache_va,
                         dataloader_tr=dataloader_tr,
                         dataloader_va=dataloader_va,
                         loss_fn_tr=loss_fn_tr,
                         loss_fn_va=loss_fn_va,
                         cfg=cfg,
                         rank=rank)

    trainer.train()


cs = ConfigStore.instance()
cs.store(name="pretrain_schema", node=PreTrainConfig)
cs.store(name="encoder_decoder_schema", group='encoder_decoder', node=EDConfig)


@hydra.main(config_path="configs/pretrain", config_name="config", version_base='1.2')
def main_pretrain(cfg: PreTrainConfig):

    # Specify the number of GPUs to use
    # num_gpus = 1
    num_gpus = cfg.num_gpus
    max_num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = max_num_gpus
        cfg.num_gpus = num_gpus
    else:
        num_gpus = min(num_gpus, max_num_gpus)

    # Save the state of the config to be able to restore it with the same hydra init
    # Inspired by https://github.com/facebookresearch/hydra/issues/1126 [spent a whole night for debugging :(]
    singleton_state = HydraConfig.get_state()
    pkl.dump(singleton_state, open('.hydra_state.pkl', 'wb'))

    # Run the training loop
    mp.spawn(main_worker, args=(num_gpus, cfg), nprocs=num_gpus, join=True)


if __name__ == '__main__':

    main_pretrain()
