import os
import logging

from typing import List, Dict, Tuple, Callable

import torch

from .__init__ import ExKF, EnKF, SEnKF, DEnKF, ETKF, EnSRKF, ETKF_Q
from .da import DA
from hmm import Operator


class XPS:

    non_ensemble_methods = ['ExKF']
    ensemble_methods = ['EnKF', 'SEnKF', 'DEnKF', 'ETKF', 'EnSRKF', 'ETKF-Q']

    methods = non_ensemble_methods + ensemble_methods

    method_name_to_class = {
        'ExKF': ExKF,
        'EnKF': EnKF,
        'SEnKF': SEnKF,
        'DEnKF': DEnKF,
        'ETKF': ETKF,
        'EnSRKF': EnSRKF,
        'ETKF-Q': ETKF_Q,
        'ETKF_Q': ETKF_Q,
    }

    methods: List[DA]

    def __init__(self,
                 logger: logging.Logger,
                 mod_dim: int,
                 obs_dim: int,
                 opM: Operator,
                 opH: Operator,
                 obs_sigma: float,
                 mod_sigma: float = None,
                 uq: Callable[[torch.Tensor], torch.Tensor] = None,
                 **kwargs):

        self.logger = logger
        self.methods = []

        self.mod_dim = mod_dim
        self.obs_dim = obs_dim
        self.opM = opM
        self.opH = opH
        self.obs_sigma = obs_sigma

        self.covH = torch.eye(self.obs_dim) * self.obs_sigma**2
        if uq is None:
            self.covM = torch.eye(self.mod_dim) * mod_sigma**2
        else:
            self.covM = uq

        self.method_base_kwargs = {
            'mod_dim': self.mod_dim,
            'obs_dim': self.obs_dim,
            'default_opM': self.opM,
            'default_opH': self.opH,
            'default_covM': self.covM,
            'default_covH': self.covH,
        }

    def add_method(self, method_name: str, **kwargs):

        # kwargs include infl, ens_dim, etc.
        method_kwargs = self.method_base_kwargs | kwargs
        method_class = self.method_name_to_class.get(method_name, None)

        if method_class is None:
            raise NotImplementedError(f'DA method {method_name} not implemented!')
        else:
            method = method_class(logger=self.logger, **method_kwargs)

        self.methods.append(method)

    def _run_one_method(self, method: DA, **kwargs) -> torch.Tensor:

        try:
            xx_a = method.assimilate(x_b=self.ass_data['x_b'],
                                     covB=self.ass_data['covB'],
                                     yy_o=self.ass_data['yy_o'],
                                     obs_t_idxs=self.ass_data['obs_t_idxs'],
                                     **kwargs,
                                     )
        except Exception as e:
            print('diverge! with exception', e)
            # raise e
            xx_a = None

        return {
            'method': method.name,
            'ass_data': self.ass_data,
            'xx_a': xx_a,
        }

    def run(self,
            save_folder: str,
            ass_data: Dict[str, torch.Tensor],
            prefix: str = '',
            **kwargs,
            ):

        os.makedirs(save_folder, exist_ok=True)
        self.ass_data = ass_data

        self.xx_a_list = []

        for method in self.methods:

            results = self._run_one_method(method, **kwargs)
            self.xx_a_list.append(results['xx_a'])

            save_filename = prefix + f'method={method.name}_assimilated.pt'
            save_path = os.path.join(save_folder, save_filename)

            torch.save(results, save_path)

    def evaluate(self,
                 xx_t: torch.Tensor,
                 decoder: Callable[[torch.Tensor], torch.Tensor] = None,
                 device=None) -> Dict[str, Tuple[float, float]]:

        named_rmse = {}

        for method, xx_a in zip(self.methods, self.xx_a_list):

            method: DA
            xx_a: torch.Tensor

            if xx_a is None:
                rmse = torch.nan

                named_rmse[method.name] = (rmse, -1)

            else:

                if device is not None:
                    xx_a = xx_a.to(device)

                if decoder is not None:
                    mini_bs = 16

                    start_idx = 0
                    end_idx = min(mini_bs, xx_a.shape[0])

                    opxx_a_list = []

                    while start_idx < xx_a.shape[0]:
                        opxx_a_list.append(decoder(xx_a[start_idx:end_idx]))
                        start_idx = end_idx
                        end_idx = min(end_idx + mini_bs, xx_a.shape[0])

                    xx_a = torch.cat(opxx_a_list, dim=0)

                xx_diff = xx_a - xx_t  # (nsteps, bs, *features)
                xx_diff_nan_mask = torch.isnan(xx_diff).transpose(0, 1)
                xx_diff_nan_mask = xx_diff_nan_mask.reshape(xx_diff_nan_mask.shape[0], -1)
                without_nan_idxs = ~torch.any(xx_diff_nan_mask, dim=1)
                without_nan_nidx = torch.sum(without_nan_idxs).item()
                ratio = without_nan_nidx / xx_diff.shape[1]

                if without_nan_nidx == 0:
                    rmse = torch.nan
                else:
                    xx_diff_without_nan = xx_diff[:, without_nan_idxs]
                    rmse = torch.sqrt(torch.mean(xx_diff_without_nan**2)).item()

                named_rmse[method.name] = (rmse, ratio)

        return named_rmse

    def strong_evaluate(self,
                        xx_t: torch.Tensor,
                        decoder: Callable[[torch.Tensor], torch.Tensor] = None,
                        device=None) -> List[Tuple[str, float]]:
        '''
        return None if any NaN in xx_a
        '''

        named_rmse = {}

        for method, xx_a in zip(self.methods, self.xx_a_list):

            method: DA
            xx_a: torch.Tensor

            if xx_a is None:
                rmse = torch.nan

                named_rmse[method.name] = (rmse, 'N/A')

            else:

                if device is not None:
                    xx_a = xx_a.to(device)

                if decoder is not None:
                    mini_bs = 16

                    start_idx = 0
                    end_idx = min(mini_bs, xx_a.shape[0])

                    opxx_a_list = []

                    while start_idx < xx_a.shape[0]:
                        opxx_a_list.append(decoder(xx_a[start_idx:end_idx]))
                        start_idx = end_idx
                        end_idx = min(end_idx + mini_bs, xx_a.shape[0])

                    xx_a = torch.cat(opxx_a_list, dim=0)

                xx_diff = xx_a - xx_t  # (nsteps, bs, *features)
                xx_diff_nan_mask = torch.isnan(xx_diff).transpose(0, 1)
                xx_diff_nan_mask = xx_diff_nan_mask.reshape(xx_diff_nan_mask.shape[0], -1)
                without_nan_idxs = ~torch.any(xx_diff_nan_mask, dim=1)
                without_nan_nidx = torch.sum(without_nan_idxs).item()
                ratio = without_nan_nidx / xx_diff.shape[1]

                if without_nan_nidx == 0:
                    rmse = torch.nan
                else:
                    xx_diff_without_nan = xx_diff[:, without_nan_idxs]
                    rmse = torch.sqrt(torch.mean(xx_diff_without_nan**2)).item()

                named_rmse[method.name] = (rmse, ratio)

        return named_rmse

    def plot(self, xx_t: torch.Tensor, save_dir: str):

        from matplotlib import pyplot as plt

        for method, xx_a in zip(self.methods, self.xx_a_list):

            try:

                xx_a_plot = xx_a[:, 0]
                xx_t_plot = xx_t[:, 0]

                xx_diff = xx_a_plot - xx_t_plot

                plt.clf()
                fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(6, 5))

                xx_range = xx_t_plot.max() - xx_t_plot.min()
                clip_min = xx_t_plot.min() - xx_range * .1
                clip_max = xx_t_plot.max() + xx_range * .1
                clip_min, clip_max = -10., 10.

                im0 = ax0.imshow(xx_a_plot.clamp_(clip_min, clip_max).T, origin='lower',
                                 cmap='coolwarm', extent=[0, 10., 0, 40], aspect='auto')
                ax0.set_ylabel('xx_a')
                plt.colorbar(im0, ax=ax0)
                # ax0.set_xlim([0., 10.])
                im1 = ax1.imshow(xx_t_plot.clamp_(clip_min, clip_max).T, origin='lower',
                                 cmap='coolwarm', extent=[0, 10., 0, 40], aspect='auto')
                ax1.set_ylabel('xx_t')
                plt.colorbar(im1, ax=ax1)
                # ax1.set_xlim([0., 10.])
                im2 = ax2.imshow(xx_diff.clamp_(clip_min, clip_max).T, origin='lower',
                                 cmap='coolwarm', extent=[0, 10., 0, 40], aspect='auto')
                ax2.set_ylabel('xx_a - xx_t')
                cb2 = plt.colorbar(im2, ax=ax2)
                # ax2.set_xlim([0., 10.])
                fig.tight_layout()

                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, method.name + '.png'), dpi=300)
                plt.close()

            except:

                print('plot failed for method', method.name)

    def plot2(self, xx_t: torch.Tensor, mask: torch.Tensor, ass_dir: str,
              decoder: Callable[[torch.Tensor], torch.Tensor] = None,
              device=None,
              prefix=''):

        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        for method in self.methods:
            ass_filename = f'method={method.name}_assimilated.pt'
            ass_path = os.path.join(ass_dir, ass_filename)
            results = torch.load(ass_path)  # 'method', 'ass_data', 'xx_a'
            xx_a: torch.Tensor = results['xx_a']

            if xx_a is None:
                continue

            if device is not None:
                xx_a = xx_a.to(device)

            if decoder is not None:
                mini_bs = 16

                start_idx = 0
                end_idx = min(mini_bs, xx_a.shape[0])

                opxx_a_list = []

                while start_idx < xx_a.shape[0]:
                    opxx_a_list.append(decoder(xx_a[start_idx:end_idx]))
                    start_idx = end_idx
                    end_idx = min(end_idx + mini_bs, xx_a.shape[0])

                xx_a = torch.cat(opxx_a_list, dim=0)

            try:

                xx_a_plot = xx_a[::40].detach().cpu().numpy()
                xx_t_plot = xx_t[::40].detach().cpu().numpy()
                # 400 nsteps in total, plot 10 frames

                xx_diff = xx_a_plot - xx_t_plot

                plt.clf()
                fig = plt.figure(figsize=(12, 11))
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(6, 11),
                                 axes_pad=.05
                                 )

                grid[11 * 1 + 0].imshow(mask[0])
                grid[11 * 4 + 0].imshow(mask[1])

                if xx_t_plot.shape[1] == 2:
                    to_enumerate = enumerate(zip(xx_t_plot[:, 0], xx_a_plot[:, 0]))
                elif xx_t_plot.shape[-1] == 2:
                    to_enumerate = enumerate(zip(xx_t_plot[..., 0], xx_a_plot[..., 0]))

                for k, (frame_t, frame_a) in to_enumerate:
                    frame_diff = frame_a - frame_t
                    grid[11 * 0 + 1 + k].imshow(frame_t)
                    grid[11 * 1 + 1 + k].imshow(frame_a)
                    grid[11 * 2 + 1 + k].imshow(frame_diff)
                    grid[11 * 0 + 1 + k].set_axis_off()
                    grid[11 * 1 + 1 + k].set_axis_off()
                    grid[11 * 2 + 1 + k].set_axis_off()

                if xx_t_plot.shape[1] == 2:
                    to_enumerate = enumerate(zip(xx_t_plot[:, 1], xx_a_plot[:, 1]))
                elif xx_t_plot.shape[-1] == 2:
                    to_enumerate = enumerate(zip(xx_t_plot[..., 1], xx_a_plot[..., 1]))

                for k, (frame_t, frame_a) in to_enumerate:
                    frame_diff = frame_a - frame_t
                    grid[11 * 3 + 1 + k].imshow(frame_t)
                    grid[11 * 4 + 1 + k].imshow(frame_a)
                    grid[11 * 5 + 1 + k].imshow(frame_diff)
                    grid[11 * 3 + 1 + k].set_axis_off()
                    grid[11 * 4 + 1 + k].set_axis_off()
                    grid[11 * 2 + 1 + k].set_axis_off()

                plt.savefig(os.path.join(ass_dir, prefix + f'method={method.name}_plot.png'),
                            dpi=72, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            except Exception as e:

                # raise e

                print('plot failed for method', method.name)

    def plot3(self, xx_t: torch.Tensor, mask: torch.Tensor, ass_dir: str,
              decoder: Callable[[torch.Tensor], torch.Tensor] = None,
              device=None,
              prefix=''):
        '''
        xx_t.shape=(nsteps, h, w, 2)
        '''

        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        for method in self.methods:
            ass_filename = prefix + f'method={method.name}_assimilated.pt'
            ass_path = os.path.join(ass_dir, ass_filename)
            results = torch.load(ass_path)  # 'method', 'ass_data', 'xx_a'
            xx_a: torch.Tensor = results['xx_a']

            if xx_a is None:
                continue

            if device is not None:
                xx_a = xx_a.to(device)

            if decoder is not None:
                mini_bs = 16

                start_idx = 0
                end_idx = min(mini_bs, xx_a.shape[0])

                opxx_a_list = []

                while start_idx < xx_a.shape[0]:
                    opxx_a_list.append(decoder(xx_a[start_idx:end_idx]))
                    start_idx = end_idx
                    end_idx = min(end_idx + mini_bs, xx_a.shape[0])

                xx_a = torch.cat(opxx_a_list, dim=0)

            try:

                xx_a_plot = xx_a[::40].detach().cpu().numpy()
                xx_t_plot = xx_t[::40].detach().cpu().numpy()
                # 400 nsteps in total, plot 10 frames

                xx_diff = xx_a_plot - xx_t_plot

                plt.clf()
                fig = plt.figure(figsize=(12, 11))
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(6, 11),
                                 axes_pad=.05
                                 )

                grid[11 * 1 + 0].imshow(mask[0])
                grid[11 * 4 + 0].imshow(mask[1])

                for k, (frame_t, frame_a) in enumerate(zip(xx_t_plot[..., 0], xx_a_plot[..., 0])):
                    frame_diff = frame_a - frame_t
                    grid[11 * 0 + 1 + k].imshow(frame_t)
                    grid[11 * 1 + 1 + k].imshow(frame_a)
                    grid[11 * 2 + 1 + k].imshow(frame_diff)
                    grid[11 * 0 + 1 + k].set_axis_off()
                    grid[11 * 1 + 1 + k].set_axis_off()
                    grid[11 * 2 + 1 + k].set_axis_off()

                for k, (frame_t, frame_a) in enumerate(zip(xx_t_plot[..., 1], xx_a_plot[..., 1])):
                    frame_diff = frame_a - frame_t
                    grid[11 * 3 + 1 + k].imshow(frame_t)
                    grid[11 * 4 + 1 + k].imshow(frame_a)
                    grid[11 * 5 + 1 + k].imshow(frame_diff)
                    grid[11 * 3 + 1 + k].set_axis_off()
                    grid[11 * 4 + 1 + k].set_axis_off()
                    grid[11 * 2 + 1 + k].set_axis_off()

                plt.savefig(os.path.join(ass_dir, prefix + f'method={method.name}_plot.png'),
                            dpi=72, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            except Exception as e:

                # raise e

                print('plot failed for method', method.name)

    def plot_rmse(self, xx_t: torch.Tensor, ass_dir: str,
                  decoder: Callable[[torch.Tensor], torch.Tensor] = None,
                  device=None,
                  prefix=''):

        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        for method in self.methods:
            ass_filename = prefix + f'method={method.name}_assimilated.pt'
            ass_path = os.path.join(ass_dir, ass_filename)
            results = torch.load(ass_path)  # 'method', 'ass_data', 'xx_a'
            xx_a: torch.Tensor = results['xx_a']

            if xx_a is None:
                continue

            if device is not None:
                xx_a = xx_a.to(device)

            if decoder is not None:
                mini_bs = 16

                start_idx = 0
                end_idx = min(mini_bs, xx_a.shape[0])

                opxx_a_list = []

                while start_idx < xx_a.shape[0]:
                    opxx_a_list.append(decoder(xx_a[start_idx:end_idx]))
                    start_idx = end_idx
                    end_idx = min(end_idx + mini_bs, xx_a.shape[0])

                xx_a = torch.cat(opxx_a_list, dim=0)

            try:

                xx_a_plot = xx_a.detach().cpu()
                xx_t_plot = xx_t.detach().cpu()
                # 400 nsteps in total, plot 10 frames

                xx_diff = xx_a_plot - xx_t_plot[1:]
                xx_rmse = torch.sqrt(torch.mean(xx_diff * xx_diff, dim=(-3, -2), keepdim=False))

                plt.clf()
                plt.plot(xx_rmse[..., 0].numpy()[:-20], label='height')
                plt.plot(xx_rmse[..., 1].numpy()[:-20], label='vorticity')
                plt.yscale('log')
                plt.xlabel('time steps')
                plt.legend()
                plt.title(f'RMSE of {method.name}')
                plt.savefig(os.path.join(ass_dir, prefix + f'method={method.name}_rmse.pdf'), dpi=180)

            except Exception as e:

                raise e

                print('plot failed for method', method.name)
