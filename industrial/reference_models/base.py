"""
Abstract base class for Bass SV reference models.

Every concrete model must implement the interface declared here.
Shared finite-difference helpers used by all PDE solvers live at the
bottom of this module so they can be imported by each model file.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import lil_matrix


class ReferenceModel(ABC):
    """
    Interface for a Bass SV reference process.

    A reference model provides:
    - a transition kernel   k(dt, x)   for the Bass A-operator convolutions
    - a marginal CDF/QF    cdf / qf    for Brenier-map construction
    - path simulation      simulate    for direct and Ito Bass simulation
    - a PDE solver         solve_pde   for Kolmogorov backward equation (Ito mode)

    All concrete models inherit from this class and must implement every
    abstractmethod. simulate_with_increments is optional and raises
    NotImplementedError by default; implement it to enable Ito transport.
    """

    @property
    @abstractmethod
    def s0(self) -> float:
        """Initial spot / forward level."""

    @abstractmethod
    def variance(self, T: float) -> float:
        """
        Total variance Var[S_T - S_0] of the reference process at horizon T.
        Used to size the Bass calibration grid.
        """

    @abstractmethod
    def kernel(self, dt: float, x: np.ndarray) -> np.ndarray:
        """
        Transition kernel density of the increment S_{t+dt} - S_t.

        Parameters
        ----------
        dt : float        time step
        x  : np.ndarray   evaluation points (increment grid, centered at 0)

        Returns
        -------
        pdf : np.ndarray  same shape as x, integrates to 1 over x
        """

    @abstractmethod
    def cdf(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        CDF of the spot process: P(S_t <= x).

        Parameters
        ----------
        t : float        time horizon
        x : np.ndarray   spot values
        """

    @abstractmethod
    def qf(self, t: float, u) -> np.ndarray:
        """
        Quantile function (inverse CDF): F_t^{-1}(u).

        Parameters
        ----------
        t : float           time horizon
        u : array-like      probability levels in (0, 1)
        """

    @abstractmethod
    def simulate(self, n_paths: int, T: float, n_steps: int,
                 seed=None) -> np.ndarray:
        """
        Simulate reference paths by Euler discretisation.

        Returns
        -------
        paths : (n_paths, n_steps + 1)  with paths[:, 0] == s0
        """

    def simulate_with_increments(self, n_paths: int, T: float, n_steps: int,
                                 seed=None):
        """
        Simulate paths and return driving noise increments needed for Ito transport.

        Concrete models override this to return a model-specific tuple
        (paths, state_paths, dW, [dZ, ...]). The base implementation raises
        NotImplementedError so that unsupported models fail early.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement simulate_with_increments. "
            "Ito simulation requires this method."
        )

    @abstractmethod
    def solve_pde(self, g0, state0, T: float, **kw) -> dict:
        """
        Solve the backward Kolmogorov PDE:
            f_t + L*f = 0,   f(T, .) = g0(.)

        Used to obtain the gradient fields grad(f) needed for Ito transport
        dX = grad(f)(t, Y_t) * dY_t.

        Parameters
        ----------
        g0     : callable   terminal condition (Brenier map phi_i)
        state0 : float or tuple  initial state of the reference process
        T      : float      time horizon for this interval
        **kw   : grid / solver keyword arguments (Nx, Nt, fast, ...)

        Returns
        -------
        dict with keys:
            t_grid   - (Nt+1,) time grid (forward)
            *_grid   - spatial grids (model-specific)
            f_time   - (Nt+1, ...) solution stored backward-in-time
            *_time   - gradient arrays needed for dX = grad(f) * dY
        """


# Shared finite-difference helpers for all 2-D PDE solvers.
# Module-level so any model file can import them without inheriting from a concrete base.

def _fd1(n: int, h: float, bc: str = 'neumann'):
    """Central finite-difference first-derivative matrix (n x n)."""
    D = lil_matrix((n, n))
    for i in range(1, n - 1):
        D[i, i - 1] = -0.5 / h
        D[i, i + 1] =  0.5 / h
    if bc == 'one-sided':
        D[0,   0] = -1 / h;  D[0,   1] =  1 / h
        D[n-1, n-2] = -1 / h;  D[n-1, n-1] = 1 / h
    return D.tocsr()


def _fd2(n: int, h: float, bc: str = 'neumann'):
    """Central finite-difference second-derivative matrix (n x n)."""
    D = lil_matrix((n, n))
    for i in range(1, n - 1):
        D[i, i-1] =  1 / h**2
        D[i, i]   = -2 / h**2
        D[i, i+1] =  1 / h**2
    if bc == 'neumann':
        D[0,   0] = -2/h**2;  D[0,   1]   = 2/h**2
        D[n-1, n-2] = 2/h**2;  D[n-1, n-1] = -2/h**2
    else:  # 'plain' - Dirichlet handled externally
        D[0,   0] = -2/h**2;  D[0,   1]   = 1/h**2
        D[n-1, n-2] = 1/h**2;  D[n-1, n-1] = -2/h**2
    return D.tocsr()


def _dirichlet_boundary_2d(Ns: int, N2: int, s_grid: np.ndarray, g0):
    """
    Return (idx, bvec) for Dirichlet BCs on the first (s/y) dimension
    of a flattened Ns x N2 grid stored in C order.
    """
    idx  = np.array([i * N2 + j for i in [0, Ns - 1] for j in range(N2)])
    bvec = np.zeros(Ns * N2)
    for j in range(N2):
        bvec[0 * N2 + j]        = g0(s_grid[0])
        bvec[(Ns - 1) * N2 + j] = g0(s_grid[-1])
    return idx, bvec


def _impose_dirichlet(M, rhs: np.ndarray, idx: np.ndarray, bvec: np.ndarray):
    """Overwrite rows idx of M with identity and set rhs[idx] = bvec[idx]."""
    M = M.tolil()
    for k in idx:
        M.rows[k] = [k];  M.data[k] = [1.0]
    rhs = rhs.copy();  rhs[idx] = bvec[idx]
    return M.tocsr(), rhs
