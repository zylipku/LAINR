import logging
from typing import Callable, Any

import math
import torch

from .en_method import EnMethod

from hmm import Operator


class ETKF(EnMethod):
    '''
    Ensemble Transformed Kalman Filter

    https://epubs.siam.org/doi/epdf/10.1137/1.9781611974546.ch6
    '''

    name = 'ETKF'

    def __init__(self, logger: logging.Logger,
                 mod_dim: int, obs_dim: int, ens_dim: int,
                 default_opM: Operator = None,
                 default_opH: Operator = None,
                 default_covM: torch.Tensor = None,
                 default_covH: torch.Tensor = None,
                 infl: float = 1,
                 dtype=None, device=None,
                 **kwargs) -> None:
        super().__init__(logger, mod_dim, obs_dim, ens_dim, default_opM,
                         default_opH, default_covM, default_covH, infl, dtype, device)

        self.eye_N = torch.eye(self.N, dtype=self.dtype, device=self.device)
        self.one_1N = torch.ones(1, self.N, dtype=self.dtype, device=self.device)

        self.U = self.eye_N
        # (N, N) identity matrix for the ensemble space

        # Note: the physical balance of the ensemble can be affected by different choices of $U$.
        # and the identity matrix minimizes the displacement between the prior perturbation and the updated perturbation.
        # See page 165 of the reference.

    def analysis(self,
                 x_b_info: torch.Tensor,
                 y_o: torch.Tensor,
                 opH: Operator,
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

        # 2. compute the ensemble means/anomalies of the observations
        Hx_b_ens = self.batch_op(opH.f, x_b_info, dim=-2)  # (n, N) -> (m, N)
        Hx_b_mean, Hx_b_ano = self._ens_to_mean_ano(Hx_b_ens)

        # # 3. transforming matrix
        # covH_sqrt_inv = self.sqrtm_inv(covH_chol @ covH_chol.transpose(-2, -1))
        # Skmat = covH_sqrt_inv @ Hx_b_ano
        # SkT_Sk = Skmat.transpose(-2, -1) @ Skmat
        # Tkmat = torch.linalg.pinv(self.eye_N + SkT_Sk, hermitian=True)

        # # 4. normalized innovation vector
        # delta_k = covH_sqrt_inv @ (y_o[..., None] - Hx_b_mean)

        # # 5. update the analysis ensembles
        # w_a = Tkmat @ Skmat.transpose(-2, -1) @ delta_k
        # x_a_ens = x_b_mean @ self.one_1N + x_b_ano @ (
        #     w_a @ self.one_1N + math.sqrt(self.N - 1) * self.sqrtm(Tkmat) @ self.U
        # )

        # 3. transforming matrix
        covH = covH_chol @ covH_chol.transpose(-2, -1)
        covH_inv = torch.linalg.pinv(covH, hermitian=True)
        SkT_Sk = Hx_b_ano.transpose(-2, -1) @ covH_inv @ Hx_b_ano
        Tkmat = torch.linalg.pinv(self.eye_N + SkT_Sk, hermitian=True)

        # 4. innovation vector
        d = y_o[..., None] - Hx_b_mean

        # 5. update the analysis ensembles
        w_a = Tkmat @ (Hx_b_ano.transpose(-2, -1) @ (covH_inv @ d))
        x_a_ens = x_b_mean @ self.one_1N + x_b_ano @ (
            w_a @ self.one_1N + math.sqrt(self.N - 1) * self.sqrtm(Tkmat) @ self.U
        )

        return x_a_ens
