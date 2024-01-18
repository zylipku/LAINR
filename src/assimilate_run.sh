#!/bin/bash

sigma_z_b_list="1e-1,3e-2,1e-2"
sigma_m_list="1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4"
infl_list="1.02,1.05,1.1"
ens_num_list="32,64"

python main.py --phase=assimilate --multirun \
sigma_z_b=$sigma_z_b_list sigma_m=$sigma_m_list \
infl=$infl_list ens_num=$ens_num_list \
encoder_decoder=default_sinr_v11 latent_dynamics=ablation/neuralode/no_exp uncertainty_est=default_diagonal

python main.py --phase=assimilate --multirun \
sigma_z_b=$sigma_z_b_list sigma_m=$sigma_m_list \
infl=$infl_list ens_num=$ens_num_list \
encoder_decoder=ablation/cae/h32_k5 latent_dynamics=default_rezero

python main.py --phase=assimilate --multirun \
sigma_z_b=$sigma_z_b_list sigma_m=$sigma_m_list \
infl=$infl_list ens_num=$ens_num_list \
encoder_decoder=ablation/aeflow/b4_k5 latent_dynamics=default_rezero

python main.py --phase=assimilate --multirun \
sigma_z_b=$sigma_z_b_list sigma_m=$sigma_m_list \
infl=$infl_list ens_num=$ens_num_list \
encoder_decoder=default_sinr_v11 latent_dynamics=ablation/neuralode/no_exp
