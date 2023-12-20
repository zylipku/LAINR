import logging
from typing import Callable, List, Any

import torch
from torch import nn

from hmm import Operator


class DAuq(nn.Module):

    '''
    Summary:
    Data assimilation

    Inputs:
    Observation vector y0: (m,)
    Observation operator opH: (n,) -> (m,)
    observation error covariance matrix R: (m, m)
    model operator: opM: (n,) -> (n,)
    model error covariance matrix Q: (n, n)
    inflation paramater: infl

    '''
    name = 'DA_abstract_class'

    DEFAULT_DTYPE = torch.float64
    DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self,
                 logger: logging.Logger,
                 mod_dim: int, obs_dim: int,
                 default_opM: Operator = None,
                 default_opH: Operator = None,
                 default_uq: Callable[[torch.Tensor], torch.Tensor] = None,
                 default_covH: torch.Tensor = None,
                 infl: float = 1.,
                 dtype=None, device=None) -> None:
        super().__init__()

        self.logger = logger

        self.n = self.mod_dim = mod_dim
        self.m = self.obs_dim = obs_dim

        self.infl = infl

        self.opM = default_opM
        self.opH = default_opH
        self.uq = default_uq
        self.covH = default_covH

        self.dtype = self.DEFAULT_DTYPE if dtype is None else dtype
        self.device = self.DEFAULT_DEVICE if device is None else device

    @property
    def tqdm_title(self):
        return f'{self.name}_m={self.m} [{torch.cuda.memory_allocated(self.device)/1024**2:.2f} MB]'

    def _get_covM(self, x_a_prev_info: Any) -> torch.Tensor:
        '''
        get covariance matrix for the model error

        Args:
            x_a_prev_info (Any): (x_a, x_a_cov)

        Returns:
            torch.Tensor: covM
        '''
        raise NotImplementedError

    def _forecast(self,
                  x_a_prev_info: Any,
                  opM: Operator,
                  covM_chol: torch.Tensor) -> Any:
        '''
        forecast phase

        x_a_prev_info: analysis states for the previous step
        opM: (n,) -> (n,): model operator
        opM need to support batch operation w.r.t. the last dimension
        covM: (n, n): model error covariance matrix

        Returns:
            torch.Tensor: x_b_info: the forecast/background states for the current step
        '''

        raise NotImplementedError

    def _analysis(self,
                  x_b_info: Any,
                  y_o: torch.Tensor,
                  opH: Operator,
                  covH_chol: torch.Tensor) -> Any:
        '''
        analysis phase

        x_b_info: (x1, x2, ... xN): background estimates for the current step with the following
        opH: (m,) -> (m,): observation operator
        opH need to support batch operation w.r.t. the last dimension
        y_o: (m,): observation
        covH: (m, m): model error covariance matrix 

        Returns:
            torch.Tensor: x_a_info: the analysis states for the the current step
        '''

        raise NotImplementedError

    def assimilate_cycle(self,
                         x_a_prev_info: Any,
                         y_o: torch.Tensor,
                         opM: Operator,
                         opH: Operator,
                         covM_chol: torch.Tensor,
                         covH_chol: torch.Tensor) -> Any:
        '''
        assimilate for single cycle

        Args:
            x_a_prev_info (Any): analysis states for the previous step.
            y_o (torch.Tensor): observation for the current step.
            opM (Operator): forward propagation operator. Defaults to None.
            opH (Operator): observation operator. Defaults to None.
            covM (torch.Tensor): covariance matrix for the operator M. Defaults to None.
            covH (torch.Tensor): covariance matrix for the operator H. Defaults to None.

        Returns:
            torch.Tensor: x_a_info: the analysis states for the current step
        '''
        # forecast phase
        x_b_info = self._forecast(x_a_prev_info=x_a_prev_info, opM=opM, covM_chol=covM_chol)

        # analysis phase
        x_a_info = self._analysis(x_b_info=x_b_info, y_o=y_o, opH=opH, covH_chol=covH_chol)
        return x_a_info

    def _get_init_info(self, x_b: torch.Tensor, covB: torch.Tensor) -> Any:

        raise NotImplementedError

    def _x_info_to_a(self, x_info: Any) -> torch.Tensor:
        '''
        x_info -> x_a

        Args:
            x_info (Any): (x_a, x_a_cov) for ExKF, x_a_ens for EnMethod

        Returns:
            torch.Tensor: x_a
        '''

        raise NotImplementedError

    def assimilate(self,
                   x_b: torch.Tensor,
                   covB: torch.Tensor,
                   yy_o: torch.Tensor,
                   obs_t_idxs: List[int],
                   opM: Operator = None, opH: Operator = None,
                   uq: Callable[[torch.Tensor], torch.Tensor] = None,
                   covH: torch.Tensor = None,
                   assim_length: int = None) -> torch.Tensor:
        '''
        main assimilation routine

        Args:
            x_b (torch.Tensor): background estimation
            covB (torch.Tensor): covariance matrix for the background estimation
            yy_o (torch.Tensor): sequence of observations shape=(nsteps, m)
            obs_t_idxs (List[int]): time indices for observations yy_o (ascending order), x_b is assumed to cooresponds to index 0
            opM (Callable[[torch.Tensor], torch.Tensor], optional): forecast model. Defaults to None.
            opH (Callable[[torch.Tensor], torch.Tensor], optional): observation model. Defaults to None.
            covM (torch.Tensor, optional): covariance matrix for the error of M. Defaults to None.
            covH (torch.Tensor, optional): covariance matrix for the error of M. Defaults to None.
            assim_length (int, optional): the total length for the assimilation ptocess, default as the last element in obs_t_idxs

        Returns:
            xx_a: sequence of analyzed states for x, starting from time index 0, [0, ..., assim_length]
        '''
        if opM is None:
            opM = self.opM
        if opH is None:
            opH = self.opH
        if uq is None:
            uq = self.uq
        if covH is None:
            covH = self.covH

        covH_chol = torch.linalg.cholesky(covH).to(self.device, self.dtype)

        x_b = x_b.to(self.device, self.dtype)
        covB = covB.to(self.device, self.dtype)
        yy_o = yy_o.to(self.device, self.dtype)

        assert len(obs_t_idxs) == yy_o.shape[0]  # yy_o.shape: (nsteps, bs, *features)
        assert all(p < q for p, q in zip(obs_t_idxs[:-1], obs_t_idxs[1:]))  # strictly ascending
        if assim_length is None:
            assim_length = obs_t_idxs[-1]

        t_idxs_y_o = {
            t_idx: y_o for t_idx, y_o in zip(obs_t_idxs, yy_o)
        }

        x_info = self._get_init_info(x_b, covB)
        xx_a = []

        from tqdm import tqdm
        for assim_step in tqdm(range(assim_length + 1), desc=self.tqdm_title):  # range(assim_length + 1):

            # for assim_step in tqdm(range(assim_length + 1), desc=self.tqdm_title):
            # self.logger.info(f'memory consuming: {torch.cuda.memory_allocated(torch.device("cuda:1")) / 1024 ** 3:.2f} GB')
            cur_y_o = t_idxs_y_o.get(assim_step)
            if cur_y_o is not None:  # observation data are available for current step, do analysis process
                x_info = self._analysis(x_b_info=x_info,
                                        y_o=cur_y_o,
                                        opH=opH,
                                        covH_chol=covH_chol)
                xx_a.append(self._x_info_to_a(x_info))

            x_info = self._forecast(x_a_prev_info=x_info,
                                    opM=opM,
                                    covM_chol=covM_chol)
            # forward propagation towards next time step

        xx_a = torch.stack(xx_a, dim=0)
        return xx_a.cpu()
