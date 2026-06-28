"""
BassSV - Bass stochastic-volatility calibration model.

Takes a ReferenceModel and a set of market marginals (one Density per
maturity), calibrates the Bass transport maps, and provides simulation:

    simulate_smile(interval_idx, n_paths)
        - fast terminal simulation for smile calibration checks

    simulate_direct(n_paths, n_steps)
        - direct simulation via time-dependent stretching functions

    simulate_ito(n_paths, n_steps)
        - Ito-formula simulation via backward Kolmogorov PDE
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import numpy as np
import scipy.stats as st
from scipy.interpolate import interp1d
import torch

from utils import Density, CalibrationResult, mc_iv
from reference_models import BrownianMotion, ArithmeticSABR


@dataclass
class _IntervalData:
    t:           float
    density:     Density
    brenier_map: Optional[Callable] = None  # reference spot -> market spot
    alpha_qf:    Optional[Callable] = None  # uniform -> reference spot (alpha-measure QF)


class BassSV:
    """
    Bass SV calibration and simulation.

    Parameters
    ----------
    s0               : float  initial spot
    reference_model  : BrownianMotion | ArithmeticSABR
    market_marginals : dict[float, Density]  maturity -> marginal density
    """

    def __init__(self, s0: float, reference_model, market_marginals: Dict[float, Density]):
        self.s0    = float(s0)
        self.ref   = reference_model
        self._data = [_IntervalData(t=t, density=d)
                      for t, d in sorted(market_marginals.items())]
        self._grid = None

    # Calibration

    def calibrate(self, N: int = 20_000, nsigma: float = 6.0,
                  tol: float = 1e-4, max_iter: int = 100,
                  verbose: bool = True) -> List[CalibrationResult]:
        """
        Compute Brenier maps and alpha-quantile functions for each interval.

        Returns a list of CalibrationResult (one per maturity).
        After calibration, brenier_map and alpha_qf are stored in each
        _IntervalData entry and can be accessed via self._data[i].
        """
        T   = np.array([d.t for d in self._data])
        std = np.sqrt(self.ref.variance(T[-1]))
        grid = np.linspace(-nsigma * std, nsigma * std, 2 * N + 1)
        dx   = grid[1] - grid[0]
        self._grid = grid
        results = []

        # Interval 0: Brenier map = QF_{mu_1} o F_{ref}(T_1), no iteration needed
        bm_vals = self._data[0].density.qf(self.ref.cdf(T[0], grid + self.s0))
        self._data[0].brenier_map = interp1d(grid + self.s0, bm_vals, fill_value='extrapolate')
        if verbose:
            print(f'  T={T[0]:.4f}: interval 0 - trivial Brenier map (no iteration)')
        results.append(CalibrationResult(
            maturity=T[0], error_evolution=[0.0], iterations=0, final_error=0.0))

        # Intervals 1, 2, ...: Banach fixed-point iteration
        for i in range(1, len(T)):
            qf2  = self._data[i].density.qf
            cdf1 = self._data[i - 1].density.cdf
            dt   = T[i] - T[i - 1]
            kern = self.ref.kernel(dt, grid) * dx
            b0   = qf2(1e-8)
            b1   = qf2(1 - 1e-8)

            def A(F, _kern=kern, _qf2=qf2, _cdf1=cdf1, _b0=b0, _b1=b1):
                # forward conv: pushes CDF forward under the kernel
                cf = np.clip(self._conv(F, _kern, 0.0, 1.0, 'forward'), 1e-6, 1 - 1e-6)
                q  = _qf2(cf)
                # backward conv: conditional expectation
                return _cdf1(self._conv(q, _kern, _b0, _b1, 'backward'))

            F    = st.norm.cdf(grid, scale=std * 0.5)
            errs = []
            for it in range(max_iter):
                F1   = A(F)
                err  = float(np.max(np.abs(F1 - F)))
                errs.append(err)
                F    = F1
                if err < tol:
                    break

            if verbose:
                print(f'  T={T[i]:.4f}: {len(errs)} iters, ||Delta F||_inf = {errs[-1]:.2e}')

            # Brenier map phi_i: forward-convolve F_alpha to get CDF of X_{T_i}
            bm_vals = qf2(np.clip(self._conv(F, kern, 0.0, 1.0, 'forward'), 0, 1))
            self._data[i].brenier_map = interp1d(
                grid + self.s0, bm_vals, fill_value='extrapolate')
            # alpha_qf: inverse CDF of the alpha-measure (maps uniform -> spot increment)
            self._data[i].alpha_qf = interp1d(
                F, grid + self.s0, fill_value='extrapolate')
            results.append(CalibrationResult(
                maturity=T[i], error_evolution=errs,
                iterations=len(errs), final_error=errs[-1]))

        return results

    # Fast terminal simulation (for smile checks)

    def simulate_smile(self, interval_idx: int, n_paths: int,
                       n_steps: int = 200, seed=None) -> np.ndarray:
        """
        Return n_paths terminal spot values X_{T_i}.

        Uses direct (non-Ito) transport: sample xi ~ F_alpha, simulate one
        batch of reference paths, apply the Brenier map. Works correctly
        for all arithmetic reference models (translation-invariant kernels).
        Requires calibrate() to have been called first.

        Parameters
        ----------
        interval_idx : maturity index (0, 1, ...)
        n_paths      : number of Monte Carlo paths
        n_steps      : number of time steps for the reference simulation
        seed         : numpy random seed

        Returns
        -------
        X_terminal : (n_paths,) array of market spot values at T_i
        """
        rng = np.random.default_rng(seed)
        T   = np.array([d.t for d in self._data])
        T_prev = 0.0 if interval_idx == 0 else T[interval_idx - 1]
        dt  = T[interval_idx] - T_prev

        Y   = self.ref.simulate(n_paths, dt, n_steps,
                                seed=int(rng.integers(1 << 31)))[:, -1]
        bm  = self._data[interval_idx].brenier_map

        if interval_idx == 0:
            return bm(Y)

        xi  = self._data[interval_idx].alpha_qf(rng.uniform(size=n_paths))
        # Translation invariance: shift reference paths to start from xi
        return bm(xi + Y - self.s0)

    # Direct simulation (full paths)

    def simulate_direct(self, n_paths: int, n_steps: int):
        """
        Simulate Bass SV paths using time-dependent stretching functions.

        For each time step t in interval [T_{i-1}, T_i], applies:
            X_t = E[phi_i(Y_{T_i}) | Y_t]  (backward convolution of Brenier map)

        Returns a list of stretched path arrays, one per interval.
        """
        T      = np.array([d.t for d in self._data])
        splits = self._distribute(n_steps, T)
        s0     = self.s0
        final  = []

        for iv_idx, (T_prev, T_cur) in enumerate(
                zip(np.concatenate([[0.0], T[:-1]]), T)):
            dt_iv = T_cur - T_prev
            n_iv  = splits[iv_idx]
            paths = self.ref.simulate(n_paths, dt_iv, n_iv)

            if iv_idx > 0:
                # Invert the full-interval stretching to recover alpha-measure start
                f_inv = self._stretch(iv_idx, dt_iv, 'inverse')
                start = f_inv(final[-1][:, -1])
                # Translate reference paths to start from the recovered xi
                paths = paths + (start - s0)[:, np.newaxis]

            stretched = np.zeros_like(paths)
            for step in range(n_iv + 1):
                delta = T_cur - (T_prev + (step / n_iv) * dt_iv)
                f     = self._stretch(iv_idx, delta, 'direct')
                stretched[:, step] = f(paths[:, step])

            final.append(stretched)

        return final

    # Ito simulation

    def simulate_ito(self, n_paths: int, n_steps: int, pde_kw: dict = None):
        """
        Ito-formula simulation via backward Kolmogorov PDE.

        Solves the backward Kolmogorov PDE for each interval and applies
        dX = grad(f) * noise. Supports BrownianMotion and ArithmeticSABR.
        Returns list of Bass path arrays, one per interval.
        """
        from scipy.interpolate import RegularGridInterpolator
        T      = np.array([d.t for d in self._data])
        splits = self._distribute(n_steps, T)
        pde_kw = pde_kw or {}
        s0     = self.s0
        results = []

        for iv_idx, (T_prev, T_cur) in enumerate(
                zip(np.concatenate([[0.0], T[:-1]]), T)):
            dt_iv = T_cur - T_prev
            n_iv  = splits[iv_idx]
            bm    = self._data[iv_idx].brenier_map

            if isinstance(self.ref, BrownianMotion):
                state0 = s0
                rng = np.random.default_rng()
                dt  = dt_iv / n_iv
                dW  = rng.standard_normal((n_paths, n_iv)) * np.sqrt(dt)
                Y   = np.zeros((n_paths, n_iv + 1))
                Y[:, 0] = s0
                Y[:, 1:] = s0 + np.cumsum(dW, axis=1)
                pde = self.ref.solve_pde(bm, state0, dt_iv, **pde_kw)
                X   = np.zeros((n_paths, n_iv + 1))
                interp0 = RegularGridInterpolator(
                    (pde['x_grid'],), pde['f_time'][0],
                    bounds_error=False, fill_value=None)
                X[:, 0] = interp0(Y[:, 0:1])[:, 0]
                for k in range(n_iv):
                    fy = RegularGridInterpolator(
                        (pde['x_grid'],), pde['fy_time'][k],
                        bounds_error=False, fill_value=None)(Y[:, k:k+1])[:, 0]
                    X[:, k+1] = X[:, k] + self.ref.sigma * fy * dW[:, k]

            elif isinstance(self.ref, ArithmeticSABR):
                state0 = (s0, self.ref.alpha)
                S, A, dW, dZ = self.ref.simulate_with_increments(
                    n_paths, dt_iv, n_iv)
                pde = self.ref.solve_pde(bm, state0, dt_iv, **pde_kw)
                X   = np.zeros((n_paths, n_iv + 1))
                interp0 = RegularGridInterpolator(
                    (pde['y_grid'], pde['u_grid']), pde['f_time'][0],
                    bounds_error=False, fill_value=None)
                pts0 = np.column_stack([S[:, 0], np.log(A[:, 0])])
                X[:, 0] = interp0(pts0)
                for k in range(n_iv):
                    pts = np.column_stack([S[:, k], np.log(A[:, k])])
                    fy  = RegularGridInterpolator(
                        (pde['y_grid'], pde['u_grid']), pde['fy_time'][k],
                        bounds_error=False, fill_value=None)(pts)
                    fa  = RegularGridInterpolator(
                        (pde['y_grid'], pde['u_grid']), pde['fa_time'][k],
                        bounds_error=False, fill_value=None)(pts)
                    X[:, k+1] = (X[:, k]
                                 + A[:, k] * fy * dW[:, k]
                                 + self.ref.nu * A[:, k] * fa * dZ[:, k])

            else:
                raise TypeError(f'Unsupported reference model: {type(self.ref)}')

            results.append(X)

        return results

    # Internal helpers

    @staticmethod
    def _conv(x: np.ndarray, kernel: np.ndarray,
              fill_left: float, fill_right: float, mode: str = 'forward') -> np.ndarray:
        """
        1-D convolution/cross-correlation via PyTorch FFT - O(N log N).

        mode='forward'  : true convolution  integral f(x-z)*k(z) dz  (pushes CDF forward)
        mode='backward' : cross-correlation integral f(x+z)*k(z) dz  (conditional expectation)

        torch.fft.rfft computes true convolution, so:
          forward  -> kernel as-is
          backward -> kernel flipped (cross-corr = true conv with flipped kernel)
        For symmetric kernels both modes are identical; they differ when rho != 0.
        """
        pad = len(kernel) // 2
        xp  = np.concatenate([[fill_left] * pad, x, [fill_right] * pad])
        k   = kernel if mode == 'forward' else kernel[::-1]

        n_out = len(xp) + len(k) - 1
        nfft  = int(2 ** np.ceil(np.log2(n_out)))

        x_t = torch.tensor(xp,       dtype=torch.float64)
        k_t = torch.tensor(k.copy(), dtype=torch.float64)

        out = torch.fft.irfft(
            torch.fft.rfft(x_t, n=nfft) * torch.fft.rfft(k_t, n=nfft),
            n=nfft
        )[:n_out]

        return out[2 * pad: 2 * pad + len(x)].numpy()

    def _stretch(self, interval, delta_t, mode):
        """
        Time-dependent stretching E[phi_i(Y_{T_i}) | Y_t].

        Computed as the backward convolution of the Brenier map phi_i with the
        transition kernel over the remaining time delta_t = T_i - t.
        """
        if delta_t == 0:
            return self._data[interval].brenier_map
        grid = self._grid
        dx   = grid[1] - grid[0]
        kern = self.ref.kernel(delta_t, grid) * dx
        bm   = self._data[interval].brenier_map
        d    = self._data[interval].density
        b0, b1 = d.qf(1e-8), d.qf(1 - 1e-8)
        # backward conv: E[phi(x + Z_{delta_t})]
        vals = self._conv(bm(grid + self.s0), kern, b0, b1, 'backward')
        if mode == 'direct':
            return interp1d(grid + self.s0, vals, fill_value='extrapolate')
        return interp1d(vals, grid + self.s0, fill_value='extrapolate')

    @staticmethod
    def _distribute(n_steps, T):
        """Distribute n_steps across intervals proportionally to their length."""
        r = n_steps * np.diff(T, prepend=0)
        s = r.astype(int)
        rem = int(n_steps * T[-1]) - s.sum()
        if rem > 0:
            s[np.argpartition(-(r - s), rem)[:rem]] += 1
        return s.tolist()
