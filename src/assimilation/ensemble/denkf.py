import logging
from typing import Callable

import math

import torch

from .en_method import EnMethod


class DEnKF(EnMethod):
    '''
    Deterministic Ensemble Kalman Filter

    https://epubs.siam.org/doi/epdf/10.1137/1.9781611974546.ch6
    '''

    name = 'DEnKF'

    def __init__(
        self, logger: logging.Logger,
        mod_dim: int, obs_dim: int, ens_dim: int,
        default_opM: Callable[[torch.Tensor], torch.Tensor] = None,
        default_opH: Callable[[torch.Tensor], torch.Tensor] = None,
        default_covM: torch.Tensor = None,
        default_covH: torch.Tensor = None,
        infl: float = 1.,
        dtype=None, device=None,
            **kwargs) -> None:
        super().__init__(logger, mod_dim, obs_dim, ens_dim, default_opM, default_opH,
                         default_covM, default_covH, infl, dtype, device)

    def analysis(self,
                 x_b_info: torch.Tensor,
                 y_o: torch.Tensor,
                 opH: Callable[[torch.Tensor], torch.Tensor],
                 covH_chol: torch.Tensor = None) -> torch.Tensor:
        '''
        analysis phase

        x_b_ens: (x1, x2, ... xN): (n, N) 
        ensembles background estimates for the current step with the following
        opH: (m,) -> (m,): observation operator
        opH need to support batch operation w.r.t. the last dimension
        y_o: (m,): observation
        covH: (m, m): model error covariance matrix 

        Returns:
            torch.Tensor: x_a_ens: the analysis states for the ensembles in the current step
        '''
        # 1. ensemble mean and anomaly matrix
        x_b_mean, x_b_ano = self._ens_to_mean_ano(x_b_info)

        # 2. innovation vectors
        d = y_o[..., None] - self.batch_op(opH.f, x_b_mean, dim=-2)

        # 3. Kalman gain matrix
        linear_assumption = False
        if linear_assumption:
            # linearity assumption
            Hjac: torch.Tensor = opH.jac(x_b_mean[..., 0])
            Hx_b_ano = Hjac @ x_b_ano
        else:
            # non-linear case
            Hx_b_ens = self.batch_op(opH.f, x_b_info, dim=-2)  # (n, N) -> (m, N)
            Hx_b_mean, Hx_b_ano = self._ens_to_mean_ano(Hx_b_ens)

        Pxz = x_b_ano @ Hx_b_ano.transpose(-2, -1)
        Pzz = Hx_b_ano @ Hx_b_ano.transpose(-2, -1)

        Kmat = Pxz @ torch.linalg.pinv(Pzz + covH_chol @ covH_chol.T, hermitian=True)

        # 4. assimilate the forecast state
        x_a = x_b_mean[..., 0] + (Kmat @ d)[..., 0]

        # 5. analyzed anomalies
        x_a_ano = x_b_ano - .5 * Kmat @ Hx_b_ano

        # 6. analyzed ensembles
        x_a_ens = x_a[..., None] + math.sqrt(self.N - 1) * x_a_ano

        return x_a_ens
