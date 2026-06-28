"""
Arithmetic Brownian motion reference model: dS = sigma*dW.
"""

import numpy as np
import scipy.stats as st
from scipy.sparse import eye
from scipy.sparse.linalg import factorized

from .base import ReferenceModel, _fd2


class BrownianMotion(ReferenceModel):
    """
    Arithmetic Brownian motion: dS = sigma*dW.

    sigma is the dollar volatility - same units as S.
    The transition kernel is a Gaussian with std = sigma*sqrt(dt), centered at 0.
    """

    def __init__(self, s0: float, sigma: float):
        self._s0   = float(s0)
        self.sigma = float(sigma)

    @property
    def s0(self) -> float:
        return self._s0

    def variance(self, T: float) -> float:
        return self.sigma**2 * T

    def kernel(self, dt: float, x: np.ndarray) -> np.ndarray:
        return st.norm.pdf(x, loc=0, scale=self.sigma * np.sqrt(dt))

    def cdf(self, t: float, x: np.ndarray) -> np.ndarray:
        return st.norm.cdf(x, loc=self._s0, scale=self.sigma * np.sqrt(t))

    def qf(self, t: float, u) -> np.ndarray:
        return st.norm.ppf(u, loc=self._s0, scale=self.sigma * np.sqrt(t))

    def simulate(self, n_paths: int, T: float, n_steps: int,
                 seed=None) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        dt   = T / n_steps
        inc  = rng.standard_normal((n_paths, n_steps)) * self.sigma * np.sqrt(dt)
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0]  = self._s0
        paths[:, 1:] = self._s0 + np.cumsum(inc, axis=1)
        return paths

    def simulate_with_increments(self, n_paths: int, T: float, n_steps: int,
                                 seed=None):
        """
        Returns (Y, dW) for Ito transport.

        Y  : (n_paths, n_steps+1)  price paths
        dW : (n_paths, n_steps)    Brownian increments
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps
        dW  = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
        Y   = np.empty((n_paths, n_steps + 1))
        Y[:, 0]  = self._s0
        Y[:, 1:] = self._s0 + np.cumsum(dW * self.sigma, axis=1)
        return Y, dW

    def solve_pde(self, g0, state0, T: float,
                  Nx: int = 501, Nt: int = 250, x_width: float = 7.0,
                  fast: bool = False) -> dict:
        """
        1-D backward heat equation: f_t + 0.5*sigma^2*f_xx = 0, f(T, x) = g0(x).

        Parameters
        ----------
        g0     : callable  terminal condition (Brenier map phi)
        state0 : float     initial spot y0
        T      : float     time horizon
        Nx     : int       number of spatial grid points
        Nt     : int       number of time steps (Crank-Nicolson)
        x_width: float     grid half-width in units of sigma*sqrt(T)
        fast   : bool      if True halves Nx and Nt for quick tests

        Returns
        -------
        dict with keys: t_grid, x_grid, f_time, fy_time
            f_time  : (Nt+1, Nx)  solution indexed forward in physical time
            fy_time : (Nt+1, Nx)  df/dx, used for dX = sigma*f_y*dW
        """
        if fast:
            Nx = max(Nx // 2, 51)
            Nt = max(Nt // 2, 50)

        y0     = float(state0)
        x_std  = self.sigma * np.sqrt(T)
        x_grid = np.linspace(y0 - x_width * x_std, y0 + x_width * x_std, Nx)
        hx     = x_grid[1] - x_grid[0]
        dt     = T / Nt

        L  = 0.5 * self.sigma**2 * _fd2(Nx, hx, bc='neumann')
        I  = eye(Nx, format='csr')

        b_idx = np.array([0, Nx - 1])
        b_val = g0(x_grid[[0, -1]])

        u       = g0(x_grid).copy()
        f_tau   = np.empty((Nt + 1, Nx))
        fy_tau  = np.empty_like(f_tau)
        f_tau[0]  = u
        fy_tau[0] = np.gradient(u, x_grid, edge_order=2)

        ML = (I - 0.5 * dt * L).tolil()
        ML.rows[0]    = [0];    ML.data[0]    = [1.0]
        ML.rows[Nx-1] = [Nx-1]; ML.data[Nx-1] = [1.0]
        MR    = I + 0.5 * dt * L
        solve = factorized(ML.tocsr().tocsc())

        for n in range(Nt):
            rhs         = MR @ u
            rhs[b_idx]  = b_val
            u           = solve(rhs)
            f_tau[n+1]  = u
            fy_tau[n+1] = np.gradient(u, x_grid, edge_order=2)

        return {
            't_grid':  np.linspace(0, T, Nt + 1),
            'x_grid':  x_grid,
            'f_time':  f_tau[::-1].copy(),
            'fy_time': fy_tau[::-1].copy(),
        }
