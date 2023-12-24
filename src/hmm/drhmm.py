'''
Data-driven Hidden Markov Model
'''

import os
from typing import Tuple, Callable, List, Dict

import torch
from torch import nn

from .hmm import HMM


class Operator(nn.Module):

    '''
    f(x): (bs, in_shape) -> (bs, out_shape)
    '''

    def __init__(self, sigma=0.) -> None:
        super().__init__()

        self.sigma = sigma

    def f(self, x: torch.Tensor, t: float = None) -> torch.Tensor:
        '''
        define the operation

        Args:
            x (torch.Tensor): shape=(bs, in_shape)

        Returns:
            torch.Tensor: f(x), shape=(bs, out_shape)
        '''
        raise NotImplementedError

    def jac(self, x: torch.Tensor, t: float = None) -> torch.Tensor:
        '''
        define the operation

        Args:
            x (torch.Tensor): shape=(bs, in_shape)

        Returns:
            torch.Tensor: df/dx, shape=(bs, out_shape, in_shape)
        '''
        from torch.autograd.functional import jacobian

        def f_without_t(x: torch.Tensor):
            return self.f(x, t)

        jac = jacobian(f_without_t, x, create_graph=False)
        return jac.detach()

    def add_noise(self, fx: torch.Tensor, t: float, sigma: float = None) -> torch.Tensor:
        '''
        define the operation

        Args:
            fx (torch.Tensor): shape=(bs, out_shape)

        Returns:
            torch.Tensor: df/dx, shape=(bs, out_shape)
        '''
        if sigma is None:
            sigma = self.sigma

        return fx + sigma * torch.randn_like(fx)

    def forward(self, x: torch.Tensor, t: float = None, sigma: float = None) -> torch.Tensor:

        fx = self.f(x, t)
        fx_noise = self.add_noise(fx, t, sigma=sigma)

        return fx_noise


class OdeOperator(Operator):

    def __init__(self, delta_t: float, max_dt: float, add_noise=True) -> None:
        super().__init__(add_noise)

        self.delta_t = delta_t
        self.max_dt = max_dt

    def rhs(self, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        raise NotImplementedError

    def f(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        cur_t = 0.
        while cur_t < self.delta_t:
            cur_dt = min(self.delta_t - cur_t, self.max_dt)
            x = x + self.rhs(x, cur_t, dt=cur_dt)
            cur_t += cur_dt

        return x


class RKdec:

    def __init__(self, method='rk23') -> None:
        '''
        See 
        https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        for details
        '''

        self.method = method

        if method == 'rk23':
            # Ralston's third-order method
            # https://github.com/scipy/scipy/blob/v1.10.0/scipy/integrate/_ivp/rk.py#L183-L277
            self.N = 3
            self.B = torch.tensor([2 / 9, 1 / 3, 4 / 9])
            self.C = torch.tensor([0, 1 / 2, 3 / 4])
            self.A = torch.tensor([[0, 0, 0],
                                  [1 / 2, 0, 0],
                                  [0, 3 / 4, 0]])

        elif method == 'rk45':
            self.N = 6
            self.B = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
            self.C = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
            self.A = torch.tensor([[0, 0, 0, 0, 0],
                                   [1 / 5, 0, 0, 0, 0],
                                   [3 / 40, 9 / 40, 0, 0, 0],
                                   [44 / 45, -56 / 15, 32 / 9, 0, 0],
                                   [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
                                   [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]])

        else:
            raise NotImplementedError

    def __call__(self, func: Callable[..., torch.Tensor]):

        def rk_func(u: torch.Tensor, t: float, dt: float = 1.):

            A = self.A.to(u)
            B = self.B.to(u)
            C = self.C.to(u)

            k0 = func(u, t)
            K = torch.empty(*k0.shape, self.N + 1, dtype=k0.dtype, device=k0.device)
            K[..., 0] = k0
            for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
                du = (K[..., :s] @ a[:s]) * dt
                K[..., s] = func(u + du, t + dt * c)

            return (K[..., :-1] @ B) * dt
        return rk_func


class DRHMM:
    '''
    data-driven hidden Markov models
    x_k = Mk(x_{k-1}) + eps_Mk
    y_k = Hk(x_k) + eps_Hk
    '''

    name = 'DRHMM_abstract_class'

    ndim: int = NotImplemented
    obs_dim: int = NotImplemented

    true_hmm: HMM = NotImplemented
    surrogate_hmm: HMM = NotImplemented

    SAVE_DIR = './data/training/'

    def __init__(self) -> None:
        pass

    def get_true_hmm(self):

        # load the true HMM, if not exists, the hmm model will generate a new one
        self.true_hmm.load_data()

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
