import os
from typing import Tuple, Any, List, Dict

import torch
from torch import nn

from .random_variable import RandomVariable
from .operator import Operator


class HMM:
    '''
    hidden Markov models
    x_k = Mk(x_{k-1}) + eps_Mk
    y_k = Hk(x_k) + eps_Hk
    '''

    name = 'HMM_abstract_class'

    ndim: int = NotImplemented
    obs_dim: int = NotImplemented

    bs: int = NotImplemented
    x0: RandomVariable = NotImplemented

    delta_t: float = NotImplemented

    Mk: Operator = NotImplemented
    Hk: Operator = NotImplemented

    mod_sigma: float = NotImplemented
    obs_sigma: float = NotImplemented

    sim_length: int = NotImplemented
    obs_start_tk: int = 0  # default to 0

    obs_t_idxs: List[int] = NotImplemented

    SAVE_DIR = './data/'

    tqdm = False

    def __init__(self) -> None:

        # check if the observation time indices are strictly ascending
        assert all(p < q for p, q in zip(self.obs_t_idxs[:-1], self.obs_t_idxs[1:]))

    @property
    def Mk_for_ass(self):
        return self.Mk

    @property
    def Hk_for_ass(self):
        return self.Hk

    def simulate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        simulate the hidden Markov model with observation operator

        Returns:
            xx (nsteps=sim_length+1), yy (nsteps=len(obs_idxs))
        '''
        x_init = self.x0.sample(bs=self.bs)
        y_init = self.Hk(x_init, 0)

        xx_t = torch.empty(self.sim_length + 1 - self.obs_start_tk, *x_init.shape)  # placeholder
        yy_o = torch.empty(len(self.obs_t_idxs), *y_init.shape)  # placeholder

        xx_t_cnt = 0
        yy_o_cnt = 0
        x_t = x_init

        if self.tqdm:

            from tqdm import tqdm

            # warm-up for time length obs_start_tk
            tqdm_bar = tqdm(range(self.sim_length + 1))
            for k in tqdm_bar:

                if k == 0:
                    x_t = x_init  # initial state
                else:
                    x_t = self.Mk(x_t, k * self.delta_t)  # otherwise, use the transition operator

                # set tqdm description
                if k < self.obs_start_tk:
                    tqdm_bar.set_description('Warming up...')
                else:
                    tqdm_bar.set_description('Observing...')
                    # observation starts at obs_start_tk
                    xx_t[xx_t_cnt] = x_t
                    xx_t_cnt += 1
                    if k - self.obs_start_tk in self.obs_t_idxs:
                        yy_o[yy_o_cnt] = self.Hk(x_t, k * self.delta_t)
                        yy_o_cnt += 1

        else:
            # warm-up for time length obs_start_tk
            for k in range(self.sim_length + 1):

                if k == 0:
                    x_t = x_init  # initial state
                else:
                    x_t = self.Mk(x_t, k * self.delta_t)  # otherwise, use the transition operator

                # set tqdm description
                if k < self.obs_start_tk:
                    pass
                else:
                    # observation starts at obs_start_tk
                    xx_t[xx_t_cnt] = x_t
                    xx_t_cnt += 1
                    if k - self.obs_start_tk in self.obs_t_idxs:
                        yy_o[yy_o_cnt] = self.Hk(x_t, k * self.delta_t)
                        yy_o_cnt += 1

        return xx_t, yy_o

    def get_io_path(self, filename=None):
        if filename is None:
            filename = self.name + '.pt'
        return os.path.join(self.SAVE_DIR, filename)

    def save_data(self, xx_t: torch.Tensor, yy_o: torch.Tensor, filename=None):

        save_path = self.get_io_path(filename)
        saved_data = {
            'hmm_model': self,
            'xx_t': xx_t,
            'yy_o': yy_o,
            'obs_t_idxs': self.obs_t_idxs,
        }
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        torch.save(saved_data, save_path)

        return saved_data

    def load_data(self, filename=None, read_cache=True) -> Dict[str, Any]:

        load_path = self.get_io_path(filename)

        if read_cache and os.path.isfile(load_path):  # load from cache
            return torch.load(load_path)

        else:  # simulate and save
            xx_t, yy_o = self.run()
            data = {
                'hmm_model': self,
                'xx_t': xx_t,
                'yy_o': yy_o,
                'obs_t_idxs': self.obs_t_idxs,
            }
            os.makedirs(os.path.split(load_path)[0], exist_ok=True)
            torch.save(data, load_path)
            return data

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        return xx_t and yy_o
        '''
        raise NotImplementedError
