import sys
import os
import logging
import subprocess
import argparse

# hydra
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING, DictConfig

# from config import PreTrainConfig


def main():

    assert len(sys.argv) >= 2, \
        "Please specify the phase. Usage: python main.py --phase=[pretrain|fintune] <optional args>"

    assert sys.argv[1] in ['--phase=pretrain', '--phase=finetune'], \
        "Please specify the phase. Usage: python main.py --phase=[pretrain|fintune] <optional args>"
    phase = sys.argv[1]

    if phase == '--phase=pretrain':
        # print in red text
        print("\033[93m" + "-" * 30 + "\033[0m")
        print("\033[94mPretraining\033[0m with the following arguments:")
        if len(sys.argv) == 2:
            print("<None>")
        else:
            print(', '.join(sys.argv[2:]))
        print("\033[93m" + "-" * 30 + "\033[0m")
        subprocess.run([sys.executable, 'pretrain_main.py'] + sys.argv[2:])

    if phase == '--phase=finetune':
        # print in red text
        print("\033[93m" + "-" * 30 + "\033[0m")
        print("\033[94mFinetuning\033[0m with the following arguments:")
        if len(sys.argv) == 2:
            print("<None>")
        else:
            print(', '.join(sys.argv[2:]))
        print("\033[93m" + "-" * 30 + "\033[0m")
        subprocess.run([sys.executable, 'finetune_main.py'] + sys.argv[2:])

    if phase == '--phase=assimilate':
        # print in red text
        print("\033[93m" + "-" * 30 + "\033[0m")
        print("\033[94mAssimilate\033[0m with the following arguments:")
        if len(sys.argv) == 2:
            print("<None>")
        else:
            print(', '.join(sys.argv[2:]))
        print("\033[93m" + "-" * 30 + "\033[0m")
        subprocess.run([sys.executable, 'assimilate_main.py'] + sys.argv[2:])


if __name__ == '__main__':

    main()
