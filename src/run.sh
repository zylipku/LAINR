#!/bin/bash
python main.py --phase=pretrain encoder_decoder=ablation/aeflow/b12_k3 nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/aeflow/b12_k5 nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/aeflow/b12_k7 nepochs=30000
