import logging
from typing import Callable, Tuple, Any

import math

import torch

from ..da import DA

from hmm import Operator


class EnMethod(DA):

    '''
    Summary:
    abstract class for ensemble methods

    Inputs:
    Observation vector y0: (m,)
    Observation operator opH: (n,) -> (m,)
    observation error covariance matrix R: (m, m)
    ensembles: (x1, x2, ... xN): (n, N)
    model operator: opM: (n,) -> (n,)
    model error covariance matrix Q: (n, n)
    inflation paramater: infl

    '''

    def __init__(self,
                 logger: logging.Logger,
                 mod_dim: int, obs_dim: int, Nens: int,
                 default_opM: Operator = None,
                 default_opH: Operator = None,
                 default_covM: torch.Tensor = None,
                 default_covH: torch.Tensor = None,
                 infl: float = 1.,
                 dtype=None, device=None) -> None:
        super().__init__(logger, mod_dim, obs_dim,
                         default_opM, default_opH,
                         default_covM, default_covH,
                         infl, dtype, device)

        self.N = self.Nens = Nens

    @staticmethod
    def sqrtm(x: torch.Tensor) -> torch.Tensor:

        # x = eig_vec @ diag(eig_val) @ eig_vec.T
        eig_val, eig_vec = torch.linalg.eigh(x)
        eig_val_sqrt = torch.sqrt(eig_val)
        x_sqrt = (eig_vec * eig_val_sqrt) @ eig_vec.transpose(-2, -1)
        return x_sqrt

    @staticmethod
    def sqrtm_inv(x: torch.Tensor) -> torch.Tensor:

        # x = eig_vec @ diag(eig_val) @ eig_vec.T
        eig_val, eig_vec = torch.linalg.eigh(x)
        eig_val_sqrt_inv = 1. / torch.sqrt(eig_val)
        x_sqrt_inv = (eig_vec * eig_val_sqrt_inv) @ eig_vec.transpose(-2, -1)
        return x_sqrt_inv

    @staticmethod
    def batch_op(op: Operator, x: torch.Tensor, dim=-1, mini_bs=4) -> torch.Tensor:
        '''
        Take batch operation on x w.r.t. the specific dimension

        Args:
            op (Callable[[torch.Tensor],torch.Tensor]): operator
            x (torch.Tensor): variables
            dim (int, optional): the dimension that needs to be taken operation on. Defaults to -1.

        Returns:
            torch.Tensor: result
        '''
        x_trans = x.transpose(dim, -1)
        x_trans_squeezed = x_trans.reshape(-1, x_trans.shape[-1])

        start_idx = 0
        end_idx = min(mini_bs, x_trans_squeezed.shape[0])

        opx_trans_squeezed_list = []

        while start_idx < x_trans_squeezed.shape[0]:
            opx_trans_squeezed_list.append(op(x_trans_squeezed[start_idx:end_idx]))
            start_idx = end_idx
            end_idx = min(end_idx + mini_bs, x_trans_squeezed.shape[0])

        opx_trans_squeezed = torch.cat(opx_trans_squeezed_list, dim=0)

        opx_trans = opx_trans_squeezed.reshape(*x_trans.shape[:-1], opx_trans_squeezed.shape[-1])
        opx = opx_trans.transpose(dim, -1)

        return opx

    @staticmethod
    def randn_from_cov_chol(cov_chol: torch.Tensor, *pre_shape) -> torch.Tensor:
        '''
        generate a random tensor according to the specific covariance matrix

        Args:
            cov_chol (torch.Tensor): L.shape=(N, N), cholesky decomposition of the covariance matrix = LL^T

        Returns:
            torch.Tensor: the generated random tensor shape=(*shape, N)
        '''
        randn = torch.randn(*pre_shape, cov_chol.size(0), 1).to(cov_chol)
        cov_randn = cov_chol @ randn
        cov_randn = cov_randn[..., 0]  # squeeze the last dimension

        return cov_randn

    def _create_ensembles(self, x: torch.Tensor, cov_chol: torch.Tensor, Nens: int) -> torch.Tensor:
        '''
        create inital ensembles for x_b

        Args:
            x (torch.Tensor): background estimation (n,)
            cov: (torch.Tensor): covariance matrix for the background estimation (n, n)

        Returns:
            torch.Tensor: ensembles (n, N)
        '''
        ens_mean = x[..., None]  # (n, 1)
        ens_pert = self.randn_from_cov_chol(cov_chol, *x.shape[:-1], Nens)
        # (..., N, n)

        x_ens = ens_mean + ens_pert.transpose(-2, -1)

        return x_ens

    def _ens_to_mean_ano(self, ens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        transform ensembles to empirical mean and anomaly matrix

        Args:
            ens (torch.Tensor): (n, N)

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: mean=(n, 1), ano=(n, N)
        '''
        N = ens.shape[-1]
        mean = torch.mean(ens, dim=-1, keepdim=True)
        ano = (ens - mean) / math.sqrt(N - 1.)  # divided by sqrt(N - 1)

        return mean, ano

    def _get_init_info(self, x_b: torch.Tensor, covB: torch.Tensor) -> Any:

        x_ens = self._create_ensembles(x_b, covB, Nens=self.Nens)

        return x_ens

    def _x_info_to_a(self, x_info: Any) -> torch.Tensor:
        '''
        x_info -> x_a

        Args:
            x_info (Any): (x_a, x_a_cov) for ExKF, x_a_ens for EnMethod

        Returns:
            torch.Tensor: x_a
        '''

        return self._ens_to_mean_ano(x_info)[0][..., 0]

    def _inflation(self, x_a_ens: torch.Tensor) -> torch.Tensor:
        '''
        inflation for x_a_ens

        Args:
            x_a_ens (torch.Tensor): (n, N)

        Returns:
            torch.Tensor: (n, N)
        '''
        N = x_a_ens.shape[-1]
        x_a_mean, x_a_ano = self._ens_to_mean_ano(x_a_ens)
        x_a_ens_infl = x_a_mean + self.infl * math.sqrt(N - 1.) * x_a_ano

        return x_a_ens_infl

    def _forecast(self,
                  x_a_prev_info: torch.Tensor,
                  opM: Operator,
                  covM_chol: torch.Tensor = None) -> torch.Tensor:
        '''
        forecast phase

        x_a_prev_ens: (x1, x2, ... xN): (n, N) analysis states for the ensembles for the previous step
        opM: (n,) -> (n,): model operator
        opM need to support batch operation w.r.t. the last dimension
        covM: (n, n): model error covariance matrix

        Returns:
            torch.Tensor: x_b_ens: the forecast/background states for the ensembles in the current step
        '''
        import time

        # x_b_ens = M(x_a_prev_ens) + model error
        N = x_a_prev_info.shape[-1]  # ensemble size
        ts = time.time()
        x_b_ens = self.batch_op(opM.f, x_a_prev_info, dim=-2)  # (n, N)
        te = time.time()
        # print(f'time cost: {te-ts=} s')
        mod_pert = self.randn_from_cov_chol(covM_chol, *x_b_ens.shape[:-2], N).transpose(-2, -1)  # (n, N)
        x_b_info = x_b_ens + mod_pert

        return x_b_info

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
        raise NotImplementedError

    def _analysis(self,
                  x_b_info: torch.Tensor,
                  y_o: torch.Tensor,
                  opH: Operator,
                  covH_chol: torch.Tensor = None) -> torch.Tensor:

        x_a_ens = self.analysis(x_b_info, y_o, opH, covH_chol)
        x_a_ens_infl = self._inflation(x_a_ens)

        return x_a_ens_infl
