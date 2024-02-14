#!/bin/bash
python main.py --phase=pretrain encoder_decoder=cae nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/aeflow/b12_k5 nepochs=30000
python main.py --phase=pretrain dataset=era5v00 encoder_decoder=sinr
