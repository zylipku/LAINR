import os

import logging

import math
import numpy as np

import torch

from torch_harmonics.examples import ShallowWaterSolver

from typing import Tuple


from common import set_seed


class SymShallowWaterSolver(ShallowWaterSolver):

    def __init__(
            self, nlat, nlon, dt, lmax=None, mmax=None, grid='legendre-gauss', radius=6371220, omega=0.00007292,
            gravity=9.80616, havg=10000, hamp=120):
        super().__init__(nlat, nlon, dt, lmax, mmax, grid, radius, omega, gravity, havg, hamp)

        # self.hyperdiff = torch.exp(self.dt / 2 * torch.as_tensor(1e5 * 3600 / radius**2 / 32**2, dtype=torch.float64))
        # self.hyperdiff = torch.exp(-self.dt / 2 * torch.as_tensor(1e5 / 3600 / 32**4, dtype=torch.float64))
        self.hyperdiff = torch.exp(torch.asarray(1e2 * (-self.dt / 2 / 3600.) * (self.lap / self.lap[-1, 0])**4))

    def galewsky_initial_condition(self, seedno: int = 0):
        """
        Overload the original galewsky_initial_condition() function from the original torch-harmonics package.
        Random 
        umax: [60., 80.]
        1/alpha: [2., 4.]
        1/beta: [10., 20.]

        Initializes non-linear barotropically unstable shallow water test case of Galewsky et al. (2004, Tellus, 56A, 429-440).

        [1] Galewsky; An initial-value problem for testing numerical models of the global shallow-water equations;
            DOI: 10.1111/j.1600-0870.2004.00071.x; http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
        """
        device = self.lap.device

        np.random.seed(seedno)

        umax = np.random.uniform(60., 80.)
        phi0 = torch.asarray(torch.pi / 7., device=device)
        phi1 = torch.asarray(0.5 * torch.pi - phi0, device=device)
        phi2 = 0.25 * torch.pi
        en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0)**2, device=device))
        alpha = 1. / np.random.uniform(2., 4.)
        beta = 1. / np.random.uniform(10., 20.)

        # self.lats = (pi/2, -pi/2)
        # self.lons = [0, 2pi)

        lats, lons = torch.meshgrid(self.lats, self.lons, indexing='ij')

        u1 = (umax / en) * torch.exp(1. / ((torch.abs(lats) - phi0) * (torch.abs(lats) - phi1)))
        ugrid = torch.where(
            torch.logical_and(torch.abs(lats) < phi1, torch.abs(lats) > phi0),
            u1,
            torch.zeros(self.nlat, self.nlon, device=device))
        vgrid = torch.zeros((self.nlat, self.nlon), device=device)
        hbump = self.hamp * torch.cos(lats) * torch.exp(-((lons - torch.pi) / alpha)
                                                        ** 2) * torch.exp(-(phi2 - lats)**2 / beta)

        hbump += self.hamp * torch.cos(lats) * torch.exp(-((lons) / alpha)
                                                         ** 2) * torch.exp(-(phi2 + lats)**2 / beta)
        # intial velocity field
        ugrid = torch.stack((ugrid, vgrid))
        # intial vorticity/divergence field
        vrtdivspec = self.vrtdivspec(ugrid)
        vrtdivgrid = self.spec2grid(vrtdivspec)

        # solve balance eqn to get initial zonal geopotential with a localized bump (not balanced).
        tmp = ugrid * (vrtdivgrid + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid**2, dim=0))
        phispec = self.invlap * tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity * (self.havg + hbump))

        # assemble solution
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=vrtdivspec.dtype, device=device)
        uspec[0] = phispec
        uspec[1:] = vrtdivspec

        return torch.tril(uspec)


class ShallowWaterGenerator:

    '''
    random galewsky initial condition
    '''

    data_dir = '/home/lizhuoyuan/datasets/shallow_water'

    default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    empirical_std = torch.tensor([5.5e3, 2.4e-5, 2.2e-7])
    empirical_mean = torch.tensor([9.5e4, 3.3e-7, 2.6e-9])

    def __init__(self,
                 logger: logging.Logger,
                 ntrajs: int = 1,
                 trunc_intv: Tuple[int, int] = (200, 600),
                 shape: Tuple[int, int] = (512, 256),
                 device: torch.device = None) -> None:

        self.logger = logger

        hour = 3600
        dt_cfl = 256 / shape[1] * 150
        dt_cfl = 300 * 64 / shape[1]
        self.nsteps = int(math.ceil(hour / dt_cfl))  # record every nsteps

        self.ntrajs = ntrajs
        self.trunc_slice = slice(*trunc_intv)

        self.nlon, self.nlat = shape

        self.device = self.default_device if device is None else device

        lmax = math.ceil(self.nlat / 3)
        mmax = lmax
        dt_solver = 1. * hour / self.nsteps

        self.solver = SymShallowWaterSolver(
            self.nlat, self.nlon, dt_solver,
            lmax=lmax, mmax=mmax, grid='equiangular'
        ).to(self.device).float()

    def generate(self, data_dir: str = None) -> None:

        data_dir = self.data_dir if data_dir is None else data_dir

        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        for k in range(self.ntrajs):

            ic_spec = self.solver.galewsky_initial_condition(seedno=k)
            spec0 = self.solver.timestep(ic_spec, self.nsteps * self.trunc_slice.start)
            grid0 = self.solver.spec2grid(spec0)

            traj_data = torch.empty(self.trunc_slice.stop - self.trunc_slice.start + 1,
                                    *grid0.shape, device=self.device)
            traj_data[0] = grid0.detach()

            spec = spec0.detach()

            import tqdm

            for step in tqdm.tqdm(range(self.trunc_slice.stop - self.trunc_slice.start)):
                spec = self.solver.timestep(spec, self.nsteps).detach()
                traj_data[step + 1] = self.solver.spec2grid(spec)

            lats, lons = torch.meshgrid(self.solver.lats, self.solver.lons, indexing='ij')
            coords_ang = torch.stack([lats, lons], dim=-1)  # lats = (pi/2, -pi/2) # self.lons = [0, 2pi)
            r = 1.
            x = torch.cos(lons) * torch.cos(lats) * r  # x = cosϕsinθ
            y = torch.sin(lons) * torch.cos(lats) * r  # y = sinϕsinθ
            z = torch.sin(lats) * r  # z = cosθ
            coords_cart = torch.stack([x, y, z], dim=-1)

            self.coords = coords_cart.transpose(0, 1)
            self.coords_ang = coords_ang.transpose(0, 1)
            self.coords_dim = coords_cart.shape[-1]

            traj_data = traj_data.permute(0, 3, 2, 1)

            traj_data -= self.empirical_mean.to(traj_data)
            traj_data /= self.empirical_std.to(traj_data)

            # downsample
            traj_data = traj_data[:, ::4, ::4, :]
            coords = self.coords[::4, ::4, :]
            coords_ang = self.coords_ang[::4, ::4, :]

            torch.save({
                'data': traj_data,
                'coords': coords,
                'coords_ang': coords_ang,
            }, os.path.join(data_dir, f'traj_{k}.pt'))

            self.logger.info(
                f'Generated {k+1}-th trajectory with {traj_data.shape=}, {coords.shape=}, {coords_ang.shape=}')


if __name__ == '__main__':

    from common import create_logger
    logger = create_logger(prefix='generate_shallow_water')

    generator = ShallowWaterGenerator(logger)
    generator.generate()
