import logging
from typing import Callable, Any

import torch

from .da import DA

from hmm import Operator


class ExKF(DA):

    name = 'ExKF'

    def __init__(self,
                 logger: logging.Logger,
                 mod_dim: int, obs_dim: int,
                 default_opM: Operator = None,
                 default_opH: Operator = None,
                 default_covM: torch.Tensor = None,
                 default_covH: torch.Tensor = None,
                 infl: float = 1,
                 dtype=None, device=None,
                 **kwargs) -> None:
        super().__init__(logger,mod_dim, obs_dim,
                         default_opM, default_opH,
                         default_covM, default_covH,
                         infl, dtype, device)

    def _get_init_info(self, x_b: torch.Tensor, covB: torch.Tensor) -> Any:
        return (x_b, covB)

    def _x_info_to_a(self, x_info: Any) -> torch.Tensor:
        '''
        x_info = (x, cov) -> x

        Args:
            x_info (Any): (x, cov)

        Returns:
            torch.Tensor: x
        '''
        return x_info[0]

    def _forecast(self,
                  x_a_prev_info: Any,
                  opM: Operator,
                  covM_chol: torch.Tensor) -> Any:

        x_a_prev, x_a_prev_cov = x_a_prev_info

        Mjac: torch.Tensor = opM.jac(x_a_prev)
        MjacT = Mjac.transpose(-2, -1)

        # 1. forward propogation
        x_b = opM.f(x_a_prev)

        # 2. background covariance matrix
        x_b_cov = Mjac @ x_a_prev_cov @ MjacT + covM_chol @ covM_chol.T

        return (x_b, x_b_cov)

    def _analysis(self,
                  x_b_info: Any,
                  y_o: torch.Tensor,
                  opH: Operator,
                  covH_chol: torch.Tensor) -> Any:

        x_b, x_b_cov = x_b_info

        Hjac: torch.Tensor = opH.jac(x_b)
        HjacT = Hjac.transpose(-2, -1)

        # 1. innovation vector
        d = y_o - opH.f(x_b)

        # 2. Kalman gain matrix
        Smat = Hjac @ x_b_cov @ HjacT + covH_chol @ covH_chol.T
        Kmat = x_b_cov @ HjacT @ torch.linalg.inv(Smat)

        # 3. update the analyzed state
        x_a = x_b + (Kmat @ d[..., None])[..., 0]

        # 4. update the analyzed convariance matrix
        x_a_cov = x_b_cov - Kmat @ Hjac @ x_b_cov

        return (x_a, x_a_cov)
