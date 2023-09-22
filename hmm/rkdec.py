from typing import Callable

import torch


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
