from typing import List, Tuple

import torch
from torch import nn

from ..hmm import HMM
from ..operator import Operator, OdeOperator
from ..utils.rk import RKdec


class Model(OdeOperator):

    default_force = 8.
    default_h = 1.
    default_model_noise = .1

    def __init__(self,
                 delta_t: float, max_dt: float,
                 add_noise=True, sigma=.1,
                 **kwargs) -> None:
        super().__init__(delta_t, max_dt, add_noise)

        self.force = kwargs.get('force', self.default_force)
        self.h = kwargs.get('h', self.default_h)

        self.delta_t = delta_t
        self.max_dt = max_dt
        self.sigma = sigma

    @staticmethod
    def nonlinear(u: torch.Tensor) -> torch.Tensor:

        u_p = torch.roll(u, -1, dims=-1)
        u_n = torch.roll(u, 1, dims=-1)
        u_nn = torch.roll(u, 2, dims=-1)
        return (u_p - u_nn) * u_n  # (u_{j+1} - u_{j-2}) * u_{j-1}

    @RKdec(method='rk45')
    def rhs(self, x: torch.Tensor, t: float = None, dt=1.) -> torch.Tensor:

        return self.h * self.nonlinear(x) - x + self.force

    def jac(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        x_p = torch.roll(x, -1, dims=-1)
        x_n = torch.roll(x, 1, dims=-1)
        x_nn = torch.roll(x, 2, dims=-1)

        offset0 = torch.diag_embed(torch.zeros_like(x) - 1.)  # dy_j/dx_j
        offsetp = torch.diag_embed(self.h * x_n)  # dy_j/dx_{j+1}
        offsetn = torch.diag_embed(self.h * (x_p - x_nn))  # dy_j/dx_{j-1}
        offsetnn = torch.diag_embed(self.h * (-x_n))  # dy_j/dx_{j-2}

        offsetp = torch.cat([offsetp[..., -1:], offsetp[..., :-1]], dim=-1)
        offsetn = torch.cat([offsetn[..., 1:], offsetn[..., :1]], dim=-1)
        offsetnn = torch.cat([offsetnn[..., 2:], offsetnn[..., :2]], dim=-1)

        jf = offset0 + offsetp + offsetn + offsetnn

        jf2 = jf @ jf
        jf3 = jf2 @ jf
        jf4 = jf3 @ jf

        jac = torch.diag_embed(torch.ones_like(x)) +\
            1 / 1. * jf * self.max_dt +\
            1 / 2. * jf2 * self.max_dt**2 +\
            1 / 6. * jf3 * self.max_dt**3 +\
            1 / 24. * jf4 * self.max_dt**4

        return jac


class Observ(Operator):

    def __init__(self, obs_idxs: List[int],
                 add_noise=True, sigma=.15) -> None:
        super().__init__(add_noise)

        self.obs_idxs = torch.tensor(obs_idxs, dtype=torch.int32)
        self.sigma = sigma

        self.obs_dim = len(obs_idxs)

    def f(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        return torch.index_select(x, -1, self.obs_idxs.to(x.device))

    def jac(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        xy = [list(range(self.obs_dim)),
              self.obs_idxs]
        values = [1., ] * self.obs_dim

        jac_sparse = torch.sparse_coo_tensor(xy, values, size=(self.obs_dim, x.shape[-1]))
        jac = jac_sparse.to_dense().to(x)

        return jac


class Lorenz96(HMM):

    name = 'Lorenz96'

    def __init__(self) -> None:
        super().__init__()

        # it is feasible to set sigma=0. since the model itself is noisy,
        # and different model noise leads to different trajectories.

        self.Mk_class = Model
        self.Hk_class = Observ
