import logging
from typing import Callable

import torch

from .en_method import EnMethod

from hmm import Operator


class EnKF(EnMethod):
    '''
    Classical Ensemble Kalman Filter

    https://github.com/surajp92/LSTM_Nudging/blob/master/EnKF/lorenz96_enkf_sparse.py
    '''

    name = 'EnKF'

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

    def analysis(self,
                 x_b_info: torch.Tensor,
                 y_o: torch.Tensor,
                 opH: Operator,
                 covH_chol: torch.Tensor = None) -> torch.Tensor:
        '''
        analysis phase

        x_b_info: (x1, x2, ... xN): (n, N) 
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

        # 2. perturbed observations
        obs_pert = self.randn_from_cov_chol(covH_chol, *y_o.shape[:-1], self.N).transpose(-2, -1)  # (..., m, N)
        z_o_ens = y_o[..., None] + obs_pert

        # 3. innovation vectors
        Hx_b_ens = self.batch_op(opH.f, x_b_info, dim=-2)  # (n, N) -> (m, N)
        d_ens = z_o_ens - Hx_b_ens

        # 4. Kalman gain matrix
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

        Kmat = Pxz @ torch.linalg.pinv(Pzz + covH_chol @ covH_chol.T, hermitian=True)

        # 5. update the analyzed emsenbles

        x_a_ens = x_b_info + Kmat @ d_ens

        return x_a_ens
