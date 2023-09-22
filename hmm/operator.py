from typing import Callable, Any

import torch
from torch import nn


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

    def __call__(self, x: torch.Tensor, t: float = None, sigma: float = None) -> torch.Tensor:

        fx = self.f(x, t)
        fx_noise = self.add_noise(fx, t, sigma=sigma)

        return fx_noise

    @classmethod
    def from_func(cls, f: Callable[[torch.Tensor], torch.Tensor], sigma=0.) -> 'Operator':
        '''
        create an operator from a function
        '''
        return FuncOperator(f, sigma=sigma)

    @classmethod
    def compose(cls, *operators: 'Operator') -> 'Operator':
        '''
        compose operators without parameter t
        '''
        return ComposedOperator(*operators)


class FuncOperator(Operator):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], sigma=0.) -> None:
        super().__init__(sigma)
        self._f = f

    def f(self, x: torch.Tensor, t: float = None) -> torch.Tensor:
        return self._f(x)


class ComposedOperator(Operator):
    def __init__(self, *operators: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()

        self.operators = [op if isinstance(op, Operator) else Operator.from_func(op) for op in operators]

    def f(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        for op in self.operators:

            x = op(x)

        return x

    def jac(self, x: torch.Tensor, t: float = None) -> torch.Tensor:

        jac_mat = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)

        for op in self.operators:

            jac_mat = op.jac(x) @ jac_mat
            x = op.f(x)

        return jac_mat


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
