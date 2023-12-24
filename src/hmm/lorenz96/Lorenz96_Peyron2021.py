from typing import Tuple

import torch
from torch import nn

from ..random_variable import Gaussian
from ..operator import Operator

from .Lorenz96 import Lorenz96


class Low2HighTrans(Operator):

    def __init__(self, linear: torch.Tensor, sigma=0.) -> None:
        super().__init__(sigma=sigma)

        self.linear = linear

    def f(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        linear = self.linear.to(x)

        y = (linear @ x[..., None])[..., 0]
        z = .01 * y**3

        return z

    def jac(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        linear = self.linear.to(x)

        y = (linear @ x[..., None])[..., 0]
        z = .01 * y**3

        jac_xy = linear
        jac_yz = torch.diag_embed(.03 * y**2)

        jac = jac_yz @ jac_xy

        return jac


class High2LowTrans(Operator):

    def __init__(self, linear: torch.Tensor, sigma=0.) -> None:
        super().__init__(sigma=sigma)

        self.linear = linear

    def f(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        linear = self.linear.to(x)

        y = (linear @ x[..., None])[..., 0]
        z = (torch.abs(y) / .01)**(1 / 3) * torch.sign(y)  # avoid negative base

        return z

    def jac(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        linear = self.linear.to(x)
        y = (linear @ x[..., None])[..., 0]
        z = (torch.abs(y) / .01)**(1 / 3) * torch.sign(y)

        jac_xy = linear
        jac_yz = torch.diag_embed(z / (3 * y))

        jac = jac_yz @ jac_xy

        return jac


class Lorenz96_Peyron2021(Lorenz96):

    name = 'Lorenz96-Peyron2021'

    ndim = 400
    obs_dim = 400  # the observation operator is the identity

    lorenz_dim = 40

    # initialized in __init__
    bs: int = NotImplemented
    x0: Gaussian = NotImplemented

    delta_t = 1e-2

    # initialized in __init__
    Mk: Operator = NotImplemented
    Hk: Operator = NotImplemented
    max_dt = 1e-2

    mod_sigma = .3
    obs_sigma = 1.

    sim_length = 1000  # [0, 5] dt=1e-2
    obs_start_tk = 500  # not specified in the paper, we borrow the configuration from Pawar2020

    obs_t_idxs = [k for k in range(1, 501)]  # observed interval: 1e-2, for 500 steps

    tqdm = True

    def __init__(self, bs=1000) -> None:
        super().__init__()

        self.name = f'{self.name}_m={self.obs_dim}'

        F = 8.

        self.bs = bs
        # x0_mean = torch.randn(self.ndim) * .01 + F
        # self.x0 = Gaussian(mean=x0_mean, sigma=.3)
        self.x0 = Gaussian(torch.zeros(self.lorenz_dim), sigma=.01)

        self.Mk = self.Mk_class(delta_t=self.delta_t,
                                max_dt=self.max_dt,
                                add_noise=True,
                                sigma=self.mod_sigma,
                                F=F)

        self.obs_idxs = [k for k in range(self.lorenz_dim)]
        # all indices are observed

        self.Hk = self.Hk_class(obs_idxs=self.obs_idxs,
                                add_noise=True,
                                sigma=0.)

        self.ortho: torch.Tensor = torch.linalg.svd(torch.randn(400, 40), full_matrices=False)[0]

        self.cubic = Low2HighTrans(self.ortho)
        self.cubic_inv = High2LowTrans(self.ortho.transpose(-2, -1))

        self.alpha = .01

    @property
    def Mk_for_ass(self):
        return Operator.compose(self.cubic_inv, self.Mk, self.cubic)

    @property
    def Hk_for_ass(self):
        return Operator.compose()  # identity

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:

        zz_t, yy_t_z = self.simulate()  # Lorenz96_m=40 data
        # xx_t.shape: (nsteps=sim_length+1, bs, ndim)
        # yy_t_z.shape: (nsteps=len(obs_idxs), bs, obs_dim)
        #! yy_t_z does not contain any noise
        xx_t = self.cubic(zz_t)
        yy_t = self.cubic(yy_t_z)

        yy_o = yy_t + torch.randn_like(yy_t) * self.obs_sigma

        return xx_t, yy_o
