import logging
from typing import Callable, Tuple, Any

import math
import torch

from .en_method import EnMethod

from hmm import Operator


class ETKF_Q(EnMethod):
    '''
    Ensemble Transformed Kalman Filter - Q

    Latent space data assimilation by using deep learning, Peyron 2021
    '''

    name = 'ETKF-Q'

    DEFAULT_DEVICE = torch.device('cpu')

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
        self.eye_N1 = torch.eye(self.N - 1, dtype=self.dtype, device=self.device)
        self.one_1N = torch.ones(1, self.N, dtype=self.dtype, device=self.device)
        self.one_N1 = torch.ones(self.N, 1, dtype=self.dtype, device=self.device)

        self.U = self.eye_N

        # initialization for the U matrix
        e1 = torch.zeros(self.N, dtype=self.dtype, device=self.device)
        e1[0] = 1.
        # (I - 2ww^T)e_1 = 1/\sqrt N => w // (e_1 - 1/\sqrt N)
        w = e1 - 1. / math.sqrt(self.N)
        w /= torch.norm(w)
        self.I_2wwT = self.eye_N - 2. * torch.outer(w, w)  # (I - 2ww^T)
        # construct matU (corrsponding to the U_m matrix in the paper)
        self.matU = self.I_2wwT[:, 1:]  # (N, N-1)
        self.scrU = torch.cat([self.one_N1 / self.N, self.matU / math.sqrt(self.N - 1)], dim=1)
        self.scrU_inv = torch.cat([self.one_N1, self.matU * math.sqrt(self.N - 1)], dim=1).T

    def _ens_to_mean_dev(self, x_ens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mean_dev = x_ens @ self.scrU
        mean = mean_dev[..., :1]
        dev = mean_dev[..., 1:]

        return (mean, dev)

    def _mean_dev_to_ens(self, mean: torch.Tensor, dev: torch.Tensor) -> torch.Tensor:

        mean_dev = torch.cat([mean, dev], dim=-1)
        x_ens = mean_dev @ self.scrU_inv

        return x_ens

    def _forecast(self,
                  x_a_prev_info: torch.Tensor,
                  opM: Operator,
                  covM_chol: torch.Tensor = None) -> torch.Tensor:

        # x_b_ens = M(x_a_prev_ens) + model error
        x_f_ens = self.batch_op(opM.f, x_a_prev_info, dim=-2)  # (n, N)
        x_f_mean, x_f_dev = self._ens_to_mean_dev(x_f_ens)

        x_f_cov = x_f_dev @ x_f_dev.transpose(-2, -1)
        x_f_cov_covM = x_f_cov + covM_chol @ covM_chol.transpose(-2, -1)

        eig_val, eig_vec = torch.linalg.eigh(x_f_cov_covM)
        eig_val_trunc = eig_val[..., -(self.N - 1):]  # the largest N-1 eigenvalues (N-1,)
        eig_vec_trunc = eig_vec[..., -(self.N - 1):]  # the largest N-1 eigenvectors (n,N-1)

        # update x_b_dev and x_b_ens
        x_b_dev = eig_vec_trunc * torch.sqrt(eig_val_trunc)[..., None, :]  # replace mat-vec by mat-mul
        additional_n_zeros = x_f_dev.shape[-1] - x_b_dev.shape[-1]
        if additional_n_zeros > 0:
            x_b_dev = torch.cat([x_b_dev, torch.zeros(*x_b_dev.shape[:-1], additional_n_zeros).to(x_b_dev)], dim=-1)
        x_b_ens = self._mean_dev_to_ens(x_f_mean, x_b_dev)

        return x_b_ens

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
        # 1. ensemble mean and dev matrix
        x_b_mean, x_b_dev = self._ens_to_mean_dev(x_b_info)

        # 2. ensemble mean and dev matrix of the observations
        Hx_b_ens = self.batch_op(opH.f, x_b_info, dim=-2)  # (n, N) -> (m, N)
        Hx_b_mean, Hx_b_dev = self._ens_to_mean_dev(Hx_b_ens)

        # 3. transforming matrix
        covH_sqrt_inv = self.sqrtm_inv(covH_chol @ covH_chol.transpose(-2, -1))
        Skmat = covH_sqrt_inv @ Hx_b_dev
        SkT_Sk = Skmat.transpose(-2, -1) @ Skmat
        Tkmat = torch.linalg.pinv(self.eye_N1 + SkT_Sk, hermitian=True)

        # 4. normalized innovation vector
        delta_k = covH_sqrt_inv @ (y_o[..., None] - Hx_b_mean)

        # 5. update the analysis ensembles
        w_a = Tkmat @ Skmat.transpose(-2, -1) @ delta_k
        x_a_mean = x_b_mean + x_b_dev @ w_a
        x_a_dev = x_b_dev @ self.sqrtm(Tkmat)
        x_a_ens = self._mean_dev_to_ens(x_a_mean, x_a_dev)

        return x_a_ens
