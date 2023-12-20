import logging
from typing import Callable, Any

import math
import torch

from .en_method import EnMethod

from hmm import Operator


class EnSRKF(EnMethod):
    '''
    Ensemble Square-Root Kalman Filter

    https://epubs.siam.org/doi/epdf/10.1137/1.9781611974546.ch6
    '''

    name = 'EnSRKF'

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
        self.eye_m = torch.eye(self.m, dtype=self.dtype, device=self.device)

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

        # 3. transforming matrix
        covH = covH_chol @ covH_chol.transpose(-2, -1)
        covH_sqrt_inv = self.sqrtm_inv(covH)
        Skmat = covH_sqrt_inv @ Hx_b_ano
        SkT_Sk = Skmat.transpose(-2, -1) @ Skmat

        try:
            Tkmat = torch.linalg.pinv(self.eye_N + SkT_Sk, hermitian=True)
        except:
            Tkmat = torch.linalg.inv(self.eye_N + SkT_Sk)
        # 4. normalized innovation vector
        delta_k = covH_sqrt_inv @ (y_o[..., None] - Hx_b_mean)

        # 5. the original Kalman gain matrix
        linear_assumption = False
        if linear_assumption:
            # linearity assumption
            Hjac: torch.Tensor = opH.jac(x_b_mean[..., 0])
            Hx_b_ano = Hjac @ x_b_ano
        else:
            # non-linear case
            Hx_b_mean, Hx_b_ano = self._ens_to_mean_ano(Hx_b_ens)

        Pxz = x_b_ano @ Hx_b_ano.transpose(-2, -1)
        Pzz = Hx_b_ano @ Hx_b_ano.transpose(-2, -1)

        # Kmat = Pxz @ torch.linalg.pinv(Pzz + covH_chol @ covH_chol.T, hermitian=True)

        # # 6. the modified Kalman gain matrix
        # covH = covH_chol @ covH_chol.transpose(-2, -1)
        # Kmat_modified = Kmat @ (
        #     torch.linalg.pinv(
        #         self.eye_m + torch.linalg.pinv(
        #             self.eye_m + Pzz @ torch.linalg.inv(covH),
        #             hermitian=True,
        #         ),
        #         hermitian=True,
        #     )
        # )  # 3 times slower

        Pzz_covH = Pzz + covH
        Pzz_covH_invT = torch.linalg.inv(Pzz_covH).transpose(-2, -1)
        Pzz_covH_sqrt = self.sqrtm(Pzz_covH)
        Kmat = Pxz @ Pzz_covH_invT @ torch.linalg.inv(Pzz_covH_sqrt + covH_chol)

        # 7. update the analysis ensembles
        w_a = Tkmat @ Skmat.transpose(-2, -1) @ delta_k
        x_a_mean = x_b_mean + x_b_ano @ w_a
        x_a_ano = (x_b_ano - Kmat @ Hx_b_ano) @ self.U
        x_a_ens = x_a_mean @ self.one_1N + math.sqrt(self.N - 1) * x_a_ano

        return x_a_ens
