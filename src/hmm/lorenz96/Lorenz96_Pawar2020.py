from typing import Tuple

import torch

from ..random_variable import Gaussian
from ..operator import Operator

from .Lorenz96 import Lorenz96


class Lorenz96_Pawar2020(Lorenz96):

    name = 'Lorenz96-Pawar2020'

    ndim = 40
    obs_dim = 4  # 4 (10%), 8 (20%), 20 (50%)

    # initialized in __init__
    bs: int = NotImplemented
    x0: Gaussian = NotImplemented

    delta_t = 5e-3

    # initialized in __init__
    Mk: Operator = NotImplemented
    Hk: Operator = NotImplemented
    max_dt = 5e-3

    mod_sigma = 1e-2
    obs_sigma = 1e-1

    sim_length = 3000  # [0, 10] dt=5e-3
    obs_start_tk = 1000  # [-5, 0] dt=5e-3

    obs_t_idxs = [10 * k for k in range(1, 201)]

    def __init__(self, obs_dim=20, bs=1) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.name = f'{self.name}_m={self.obs_dim}'

        F = 10.

        self.bs = bs
        x0_mean = torch.zeros(self.ndim) + F
        x0_mean[19] += .01
        self.x0 = Gaussian(mean=x0_mean, sigma=0.)
        # it is feasible to set sigma=0. since the model itself is noisy,
        # and different model noise leads to different trajectories.

        self.Mk = self.Mk_class(delta_t=self.delta_t,
                                max_dt=self.max_dt,
                                add_noise=True,
                                sigma=self.mod_sigma,
                                F=F)

        self.obs_idxs = [(self.ndim // self.obs_dim) * k for k in range(self.obs_dim)]

        self.Hk = self.Hk_class(obs_idxs=self.obs_idxs,
                                add_noise=True,
                                sigma=self.obs_sigma)

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:

        xx_t, yy_o = self.simulate()

        return xx_t, yy_o
