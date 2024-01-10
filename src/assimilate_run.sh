#!/bin/bash
python assimilate_main.py --multirun sigma_z_b=0.01,0.001 sigma_m=0.1,0.03,0.01,0.003,0.001 \
encoder_decoder=ablation/cae/h32_k5 latent_dynamics=default_rezero
python assimilate_main.py --multirun sigma_z_b=0.01,0.001 sigma_m=0.1,0.03,0.01,0.003,0.001 \
encoder_decoder=ablation/aeflow/b4_k5 latent_dynamics=default_rezero
python assimilate_main.py --multirun sigma_z_b=0.01,0.001 sigma_m=0.01,0.003,0.001,0.0003,0.0001 \
encoder_decoder=default_sinr_v11 latent_dynamics=default_neuralode