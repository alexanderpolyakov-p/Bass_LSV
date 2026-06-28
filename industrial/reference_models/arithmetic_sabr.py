"""
Arithmetic (beta=0) SABR reference model for Bass SV.

    dY = A*dW,    d(log A) = nu*dZ - 0.5*nu^2*dt,    corr(dW, dZ) = rho

Transition kernel via Hagan (2002) implied vol + Breeden-Litsenberger.
PDE solver: Crank-Nicolson on the (y, log alpha) grid.
"""

import numpy as np
from functools import lru_cache
from scipy.interpolate import make_smoothing_spline
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized

from .base import ReferenceModel, _fd1, _fd2, _dirichlet_boundary_2d, _impose_dirichlet

# utils.py lives one level up (industrial/); support both package and direct-run contexts.
try:
    from utils import bs_call_price
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from utils import bs_call_price


class ArithmeticSABR(ReferenceModel):
    """
    Arithmetic (beta=0) SABR:
        dY = A*dW,    d(log A) = nu*dZ - 0.5*nu^2*dt,    corr(dW, dZ) = rho.

    Parameters
    ----------
    s0     : float  initial spot
    alpha  : float  initial Bachelier vol (dollar units)
    rho    : float  dW-dZ correlation
    nu     : float  vol-of-vol
    N      : int    grid size for kernel / CDF computation (default 20 000)
    nsigma : float  grid half-width in units of std (default 8)
    """

    def __init__(self, s0: float, alpha: float, rho: float, nu: float,
                 N: int = 20_000, nsigma: float = 8.0):
        self._s0   = float(s0)
        self.alpha = float(alpha)
        self.rho   = float(rho)
        self.nu    = float(nu)
        self._N    = int(N)
        self._nsig = float(nsigma)

    @property
    def s0(self) -> float:
        return self._s0

    # Variance

    def variance(self, T: float) -> float:
        if self.nu > 1e-10:
            return self.alpha**2 * (np.exp(self.nu**2 * T) - 1) / self.nu**2
        return self.alpha**2 * T

    # Hagan (2002) implied vol

    @staticmethod
    def _iv(s0: float, alpha: float, rho: float, nu: float,
            t: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Lognormal implied volatility via Hagan (2002) 4th-order expansion."""
        eps = 1e-12
        k   = np.maximum(np.asarray(k, float), eps)
        log = np.log(s0 / k)
        z   = nu / alpha * np.sqrt(s0 * k) * log
        chi = np.log(np.maximum(
            (np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho), eps))
        ratio = np.where(
            np.abs(k - s0) > 1e-8,
            np.where(np.abs(chi) > eps, log * z / ((s0 - k) * chi), 1/s0),
            1/s0)
        return alpha * ratio * (
            1 + t * (alpha**2 / (24 * s0 * k) + (2 - 3*rho**2) * nu**2 / 24))

    def implied_vol(self, t, k) -> np.ndarray:
        return self._iv(self._s0, self.alpha, self.rho, self.nu, t, k)

    def call_price(self, t, k) -> np.ndarray:
        return bs_call_price(self._s0, self.implied_vol(t, k), t,
                             np.asarray(k, float))

    # Kernel / CDF / QF via Breeden-Litsenberger

    @lru_cache(maxsize=64)
    def _dist(self, t_round: float) -> dict:
        """Compute and cache the full distribution at time t_round."""
        t   = float(t_round)
        std = np.sqrt(self.variance(t))
        w   = min(self._nsig * std, 0.9999 * self._s0)
        inc = np.linspace(-w, w, 2 * self._N + 3)
        K   = np.maximum(self._s0 + inc, 1e-8)
        dx  = inc[1] - inc[0]
        C   = self.call_price(t, K)
        pdf = np.maximum(np.diff(C, n=2), 0) / dx**2
        spl = make_smoothing_spline(inc[1:-1], pdf, lam=0.01)
        pdf = np.maximum(spl(inc[1:-1]), 0)
        pdf /= (pdf * dx).sum()
        cdf = np.zeros_like(pdf)
        cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(K[1:-1]))
        cdf /= max(cdf[-1], 1e-12)
        cdf  = np.clip(np.maximum.accumulate(cdf), 0, 1)
        cu, ui = np.unique(cdf, return_index=True)
        ks = K[1:-1][ui]
        if cu[0]  > 0: cu = np.insert(cu, 0, 0.0);  ks = np.insert(ks, 0, K[0])
        if cu[-1] < 1: cu = np.append(cu,  1.0);    ks = np.append(ks,  K[-1])
        return {'inc': inc[1:-1], 'pdf': pdf, 'spot': K[1:-1],
                'cdf': cdf, 'cu': cu, 'ks': ks}

    def _get(self, t: float) -> dict:
        return self._dist(round(float(t), 10))

    def kernel(self, dt: float, x: np.ndarray) -> np.ndarray:
        d = self._get(dt)
        return np.interp(x, d['inc'], d['pdf'], left=0, right=0)

    def cdf(self, t: float, x: np.ndarray) -> np.ndarray:
        d = self._get(t)
        return np.clip(np.interp(x, d['spot'], d['cdf'], left=0, right=1), 0, 1)

    def qf(self, t: float, u) -> np.ndarray:
        d = self._get(t)
        return np.interp(u, d['cu'], d['ks'])

    # Simulation

    def simulate(self, n_paths: int, T: float, n_steps: int,
                 seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt  = T / n_steps
        Z   = rng.standard_normal((2, n_steps, n_paths))
        dW  = Z[0] * np.sqrt(dt)
        dZ  = (self.rho * Z[0] + np.sqrt(1 - self.rho**2) * Z[1]) * np.sqrt(dt)
        logA = np.zeros((n_paths, n_steps + 1))
        logA[:, 0] = np.log(self.alpha)
        for i in range(n_steps):
            logA[:, i+1] = logA[:, i] + self.nu * dZ[i] - 0.5 * self.nu**2 * dt
        A = np.exp(logA)
        S = np.zeros((n_paths, n_steps + 1));  S[:, 0] = self._s0
        for i in range(n_steps):
            S[:, i+1] = S[:, i] + A[:, i] * dW[i]
        return S

    def simulate_with_increments(self, n_paths: int, T: float, n_steps: int,
                                 seed=None):
        """
        Returns (S, A, dW, dZ) for Ito transport.

        S, A : (n_paths, n_steps+1)  spot and vol paths
        dW   : (n_paths, n_steps)    price Brownian increments
        dZ   : (n_paths, n_steps)    vol Brownian increments
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps
        sq  = np.sqrt(dt)
        Z1  = rng.standard_normal((n_paths, n_steps))
        Z2  = rng.standard_normal((n_paths, n_steps))
        dW  = Z1 * sq
        dZ  = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * sq
        logA = np.zeros((n_paths, n_steps + 1))
        logA[:, 0] = np.log(self.alpha)
        for i in range(n_steps):
            logA[:, i+1] = logA[:, i] + self.nu * dZ[:, i] - 0.5 * self.nu**2 * dt
        A = np.exp(logA)
        S = np.zeros_like(A);  S[:, 0] = self._s0
        for i in range(n_steps):
            S[:, i+1] = S[:, i] + A[:, i] * dW[:, i]
        return S, A, dW, dZ

    # PDE solver: backward Kolmogorov on (y, log alpha)

    def solve_pde(self, g0, state0, T: float,
                  Ny: int = 181, Nu: int = 111, Nt: int = 100,
                  y_width: float = 10.0, u_width: float = 5.5,
                  fast: bool = False) -> dict:
        """
        Solve f_t + L*f = 0, f(T, .) = g0(.), where:
            L*f = 0.5*alpha^2*f_yy + rho*nu*alpha*f_{yu} + 0.5*nu^2*f_{uu} - 0.5*nu^2*f_u

        Parameters
        ----------
        g0     : callable  terminal condition g0(y) = phi(y)
        state0 : (y0, alpha0)  initial (spot, vol)
        T      : float  interval length
        Ny, Nu : int    grid sizes in y and u=log(alpha) directions
        Nt     : int    number of Crank-Nicolson time steps
        y_width: float  half-width in units of alpha_0*sqrt(T)
        u_width: float  half-width in units of nu*sqrt(T)
        fast   : bool   if True halves all grid sizes for quick tests

        Returns
        -------
        dict with keys: t_grid, y_grid, u_grid, a_grid, f_time, fy_time, fa_time
            fy_time, fa_time : df/dy and (1/alpha)*df/d(log alpha)
                              used for dX = A*f_y*dW + nu*A*f_alpha*dZ
        """
        if fast:
            Ny = max(Ny // 2, 51)
            Nu = max(Nu // 2, 31)
            Nt = max(Nt // 2, 50)

        y0, a0 = float(state0[0]), float(state0[1])
        ys     = a0 * np.sqrt(T)
        y_grid = np.linspace(y0 - y_width * ys, y0 + y_width * ys, Ny)
        u0     = np.log(a0)
        u_hw   = max(u_width * self.nu * np.sqrt(T), 1e-3)
        u_grid = np.linspace(u0 - u_hw, u0 + u_hw, Nu)
        hy, hu = y_grid[1] - y_grid[0], u_grid[1] - u_grid[0]
        a_grid = np.exp(u_grid)

        A_c = 0.5 * a_grid**2
        B_c = self.rho * self.nu * a_grid
        C_c = 0.5 * self.nu**2
        D_c = -0.5 * self.nu**2

        D1y = _fd1(Ny, hy, bc='one-sided')
        D2y = _fd2(Ny, hy, bc='plain')
        D1u = _fd1(Nu, hu)
        D2u = _fd2(Nu, hu)
        Iy  = eye(Ny, format='csr');  Iu = eye(Nu, format='csr')

        L = (diags(np.tile(A_c, Ny)) @ kron(D2y, Iu, format='csr')
             + diags(np.tile(B_c, Ny)) @ kron(D1y, D1u, format='csr')
             + C_c * kron(Iy, D2u, format='csr')
             + D_c * kron(Iy, D1u, format='csr')).tocsr()

        N  = Ny * Nu
        I  = eye(N, format='csr')
        Y_mesh, _ = np.meshgrid(y_grid, u_grid, indexing='ij')
        U  = g0(Y_mesh).ravel(order='C')

        b_idx, b_vec = _dirichlet_boundary_2d(Ny, Nu, y_grid, g0)
        U[b_idx] = b_vec[b_idx]
        dt = T / Nt

        ML = I - 0.5 * dt * L;  MR = I + 0.5 * dt * L
        ML, _ = _impose_dirichlet(ML, np.zeros(N), b_idx, b_vec)
        solve  = factorized(ML.tocsc())

        f_t  = np.zeros((Nt + 1, Ny, Nu))
        fy_t = np.zeros_like(f_t)
        fa_t = np.zeros_like(f_t)

        def _grads(F):
            fy = np.gradient(F, y_grid, axis=0, edge_order=2)
            fu = np.gradient(F, u_grid, axis=1, edge_order=2)
            return fy, fu / a_grid[np.newaxis, :]

        F0 = U.reshape((Ny, Nu), order='C')
        f_t[0], fy_t[0], fa_t[0] = (F0, *_grads(F0))

        for n in range(Nt):
            rhs = MR @ U;  rhs[b_idx] = b_vec[b_idx]
            U   = solve(rhs)
            F   = U.reshape((Ny, Nu), order='C')
            f_t[n+1], fy_t[n+1], fa_t[n+1] = (F, *_grads(F))

        return {
            't_grid':  np.linspace(0, T, Nt + 1),
            'y_grid':  y_grid,
            'u_grid':  u_grid,
            'a_grid':  a_grid,
            'f_time':  f_t[::-1].copy(),
            'fy_time': fy_t[::-1].copy(),
            'fa_time': fa_t[::-1].copy(),
        }

    # Parameter calibration

    @classmethod
    def calibrate(cls, s0: float, t, k, iv,
                  bounds=((1e-3, 5e3), (-0.999, 0.999), (1e-3, 10.0)),
                  seed: int = 42) -> 'ArithmeticSABR':
        """
        Fit (alpha, rho, nu) to market implied vols by minimising relative RMSE.

        Parameters
        ----------
        s0     : float    spot price
        t, k   : arrays   maturities and strikes
        iv     : array    market lognormal implied vols
        bounds : tuple    search bounds for (alpha, rho, nu)
        seed   : int      random seed for differential_evolution

        Returns
        -------
        ArithmeticSABR instance with calibrated parameters
        """
        from scipy.optimize import differential_evolution
        t  = np.atleast_1d(t).astype(float)
        k  = np.atleast_1d(k).astype(float)
        iv = np.atleast_1d(iv).astype(float)
        if t.size == 1:
            t = np.full_like(k, t[0])
        mask = (np.isfinite(t) & np.isfinite(k) & np.isfinite(iv)
                & (t > 0) & (k > 0) & (iv > 0))
        t, k, iv = t[mask], k[mask], iv[mask]

        def obj(p):
            alpha, rho, nu = p
            if alpha <= 0 or nu < 0 or abs(rho) >= 1:
                return 1e12
            try:
                vm = cls._iv(s0, alpha, rho, nu, t, k)
                return float(np.mean(((vm - iv) / np.maximum(iv, 1e-6))**2))
            except Exception:
                return 1e12

        res = differential_evolution(obj, bounds, seed=seed, polish=True)
        return cls(s0, *res.x)
