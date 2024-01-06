#!/bin/bash
python main.py --phase=pretrain encoder_decoder=ablation/cae/h16_k3 nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/cae/h16_k5 nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/cae/h16_k7 nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/cae/h32_k3 nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/cae/h32_k5 nepochs=30000
python main.py --phase=pretrain encoder_decoder=ablation/cae/h32_k7 nepochs=30000