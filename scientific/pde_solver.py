"""
PDE solvers for Bass LV transport.

Two reference models are supported:

  StableNormalSABRPDE  - arithmetic SABR (beta=0) Kolmogorov backward PDE on the
                         2-D state space (y, log alpha). Used for Bass LV calibration.

  solve_brownian_transport_pde - 1-D heat equation for Brownian motion reference.
                                  Used for smile-dynamics experiments.

Both solvers use an implicit Crank-Nicolson scheme with Dirichlet boundary
conditions and return time-reversed coefficient arrays so that index 0
corresponds to the initial time of the interval.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, eye, kron, lil_matrix
from scipy.sparse.linalg import factorized


# Finite-difference building blocks

def _fd1(n, h, bc='neumann'):
    D = lil_matrix((n, n))
    for i in range(1, n - 1):
        D[i, i - 1] = -0.5 / h
        D[i, i + 1] =  0.5 / h
    if bc == 'one-sided':
        D[0, 0] = -1.0 / h;  D[0, 1] =  1.0 / h
        D[n-1, n-2] = -1.0 / h;  D[n-1, n-1] = 1.0 / h
    elif bc != 'neumann':
        raise ValueError(f'Unknown bc={bc!r}')
    return D.tocsr()


def _fd2(n, h, bc='neumann'):
    D = lil_matrix((n, n))
    for i in range(1, n - 1):
        D[i, i-1] =  1.0 / h**2
        D[i, i]   = -2.0 / h**2
        D[i, i+1] =  1.0 / h**2
    if bc == 'neumann':
        D[0, 0] = -2.0/h**2;  D[0, 1]     = 2.0/h**2
        D[n-1, n-2] = 2.0/h**2;  D[n-1, n-1] = -2.0/h**2
    elif bc == 'plain':
        D[0, 0] = -2.0/h**2;  D[0, 1]     = 1.0/h**2
        D[n-1, n-2] = 1.0/h**2;  D[n-1, n-1] = -2.0/h**2
    else:
        raise ValueError(f'Unknown bc={bc!r}')
    return D.tocsr()


def _flat(i, j, Nu):
    return i * Nu + j


def _y_boundary_indices(Ny, Nu):
    return np.array(
        [_flat(0, j, Nu) for j in range(Nu)] +
        [_flat(Ny-1, j, Nu) for j in range(Nu)],
        dtype=int,
    )


def _terminal_boundary_vec(g0, y_grid, Nu):
    Ny = len(y_grid)
    b  = np.zeros(Ny * Nu)
    for j in range(Nu):
        b[_flat(0,    j, Nu)] = g0(y_grid[0])
        b[_flat(Ny-1, j, Nu)] = g0(y_grid[-1])
    return b


def _impose_dirichlet_rows(M, idx):
    M = M.tolil()
    for k in idx:
        M.rows[k] = [k];  M.data[k] = [1.0]
    return M.tocsr()


def _impose_dirichlet_rhs(rhs, bvec, idx):
    rhs = rhs.copy()
    rhs[idx] = bvec[idx]
    return rhs


# SABR Kolmogorov operator

def _sabr_log_operator(y_grid, u_grid, nu, rho):
    """
    Backward Kolmogorov operator for arithmetic SABR in (y, u=log alpha) coordinates.

    L*f = 0.5*alpha^2*f_yy + rho*nu*alpha*f_{yu} + 0.5*nu^2*f_{uu} - 0.5*nu^2*f_u
    """
    Ny, Nu = len(y_grid), len(u_grid)
    hy, hu = y_grid[1] - y_grid[0], u_grid[1] - u_grid[0]

    a = np.exp(u_grid)
    A_coef = 0.5 * a**2
    B_coef = rho * nu * a
    C_coef = 0.5 * nu**2
    D_coef = -0.5 * nu**2

    D1y = _fd1(Ny, hy, bc='one-sided')
    D2y = _fd2(Ny, hy, bc='plain')
    D1u = _fd1(Nu, hu, bc='neumann')
    D2u = _fd2(Nu, hu, bc='neumann')
    Iy  = eye(Ny, format='csr')
    Iu  = eye(Nu, format='csr')

    A_vec = np.tile(A_coef, Ny)
    B_vec = np.tile(B_coef, Ny)

    return (
        diags(A_vec) @ kron(D2y, Iu, format='csr')
        + diags(B_vec) @ kron(D1y, D1u, format='csr')
        + C_coef * kron(Iy, D2u, format='csr')
        + D_coef * kron(Iy, D1u, format='csr')
    ).tocsr()


# SABR PDE solver

class StableNormalSABRPDE:
    """
    Configuration for the arithmetic SABR (beta=0) backward PDE.

    Solves f_t + L*f = 0 backward from terminal condition g0(y).

    Parameters
    ----------
    nu, rho  : SABR vol-of-vol and dW-dZ correlation
    T        : interval length
    Ny, Nu   : grid points in y and log(alpha) directions
    Nt       : number of time steps
    y_width  : grid half-width in units of alpha*sqrt(T)  (y-direction)
    u_width  : grid half-width in units of nu*sqrt(T)  (log(alpha) direction)
    """
    def __init__(self, nu, rho, T, Ny, Nu, Nt, y_width=10.0, u_width=5.5):
        self.nu      = float(nu)
        self.rho     = float(rho)
        self.T       = float(T)
        self.Ny      = int(Ny)
        self.Nu      = int(Nu)
        self.Nt      = int(Nt)
        self.y_width = float(y_width)
        self.u_width = float(u_width)

    def build_grid(self, y0, alpha):
        y_std  = alpha * np.sqrt(self.T)
        y_grid = np.linspace(y0 - self.y_width * y_std,
                             y0 + self.y_width * y_std, self.Ny)
        u0     = np.log(alpha)
        u_hw   = max(self.u_width * self.nu * np.sqrt(self.T), 1e-3)
        u_grid = np.linspace(u0 - u_hw, u0 + u_hw, self.Nu)
        return y_grid, u_grid

    @staticmethod
    def compute_derivatives(F, y_grid, u_grid):
        p_y = np.gradient(F, y_grid, axis=0, edge_order=2)
        p_u = np.gradient(F, u_grid, axis=1, edge_order=2)
        p_a = p_u / np.exp(u_grid)[np.newaxis, :]
        return p_y, p_a


def solve_pde_time_grid(solver, g0, y0, alpha, verbose=True):
    """
    Solve the SABR backward Kolmogorov PDE and store the solution and
    its spatial derivatives at every time step.

    Parameters
    ----------
    solver  : StableNormalSABRPDE instance
    g0      : callable, terminal condition f(T, y) = g0(y)
    y0      : reference starting price (determines grid centre)
    alpha   : initial SABR vol (determines grid width)
    verbose : print progress

    Returns
    -------
    dict with keys:
      't_grid'  : (Nt+1,)  forward time grid [0, T]
      'y_grid'  : (Ny,)
      'u_grid'  : (Nu,)   in log(alpha) space
      'a_grid'  : (Nu,)   alpha = exp(u)
      'f_time'  : (Nt+1, Ny, Nu)  f at each forward time step
      'py_time' : (Nt+1, Ny, Nu)  df/dy
      'pa_time' : (Nt+1, Ny, Nu)  df/d(alpha)  (= p_u / alpha)
    """
    y_grid, u_grid = solver.build_grid(y0, alpha)
    Ny, Nu = len(y_grid), len(u_grid)
    N  = Ny * Nu
    dt = solver.T / solver.Nt

    Y_mesh, _ = np.meshgrid(y_grid, u_grid, indexing='ij')
    U = g0(Y_mesh).ravel(order='C')

    L = _sabr_log_operator(y_grid, u_grid, solver.nu, solver.rho)
    I = eye(N, format='csr')

    b_idx  = _y_boundary_indices(Ny, Nu)
    b_vec  = _terminal_boundary_vec(g0, y_grid, Nu)
    U[b_idx] = b_vec[b_idx]

    f_tau  = np.zeros((solver.Nt + 1, Ny, Nu))
    py_tau = np.zeros_like(f_tau)
    pa_tau = np.zeros_like(f_tau)

    F0 = U.reshape((Ny, Nu), order='C')
    py0, pa0 = solver.compute_derivatives(F0, y_grid, u_grid)
    f_tau[0], py_tau[0], pa_tau[0] = F0, py0, pa0

    if verbose:
        print(f'Solving PDE: Ny={Ny}, Nu={Nu}, Nt={solver.Nt}')
        print(f'  y in [{y_grid[0]:.3f}, {y_grid[-1]:.3f}]')
        print(f'  alpha in [{np.exp(u_grid[0]):.4f}, {np.exp(u_grid[-1]):.4f}]')

    M_left  = _impose_dirichlet_rows(I - 0.5*dt*L, b_idx)
    M_right = I + 0.5*dt*L
    solve   = factorized(M_left.tocsc())

    for n in range(solver.Nt):
        rhs = _impose_dirichlet_rhs(M_right @ U, b_vec, b_idx)
        U   = solve(rhs)
        F   = U.reshape((Ny, Nu), order='C')
        py, pa = solver.compute_derivatives(F, y_grid, u_grid)
        f_tau[n+1]  = F
        py_tau[n+1] = py
        pa_tau[n+1] = pa
        if verbose and (n+1) % max(1, solver.Nt // 5) == 0:
            print(f'  step {n+1}/{solver.Nt}')

    return {
        't_grid':  np.linspace(0.0, solver.T, solver.Nt + 1),
        'y_grid':  y_grid,
        'u_grid':  u_grid,
        'a_grid':  np.exp(u_grid),
        'f_time':  f_tau[::-1].copy(),
        'py_time': py_tau[::-1].copy(),
        'pa_time': pa_tau[::-1].copy(),
    }


# SABR path simulation

def simulate_sabr(y0, alpha, nu, rho, T, n_paths, n_steps, seed=None):
    """
    Simulate arithmetic SABR (beta=0) paths.

        dY = A*dW,    d(log A) = nu*dZ - 0.5*nu^2*dt,    corr(dW, dZ) = rho

    Returns
    -------
    Y     : (n_paths, n_steps+1)  price paths
    A     : (n_paths, n_steps+1)  vol paths
    dW    : (n_paths, n_steps)    unit Brownian increments for Y
    dZ    : (n_paths, n_steps)    unit Brownian increments for log A
    """
    rng   = np.random.default_rng(seed)
    dt    = T / n_steps
    sq_dt = np.sqrt(dt)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    dW = Z1 * sq_dt
    dZ = (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2) * sq_dt

    log_A = np.zeros((n_paths, n_steps + 1))
    log_A[:, 0] = np.log(alpha)
    for i in range(n_steps):
        log_A[:, i+1] = log_A[:, i] + nu * dZ[:, i] - 0.5*nu**2*dt
    A = np.exp(log_A)

    Y = np.zeros((n_paths, n_steps + 1))
    Y[:, 0] = y0
    for i in range(n_steps):
        Y[:, i+1] = Y[:, i] + A[:, i] * dW[:, i]

    return Y, A, dW, dZ


def apply_ito_transport(Y_paths, A_paths, dW, dZ, pde_data, nu):
    """
    Apply the Ito transport formula to SABR paths.

        dX = A*f_y*dW + nu*A*f_a*dZ

    where f solves the Kolmogorov PDE with terminal condition = Brenier map.

    Parameters
    ----------
    Y_paths, A_paths : SABR reference paths from simulate_sabr
    dW, dZ           : raw unit increments from simulate_sabr
    pde_data         : dict from solve_pde_time_grid
    nu               : SABR vol-of-vol

    Returns
    -------
    X : (n_paths, n_steps+1)  Bass LV price paths
    """
    y_grid, u_grid   = pde_data['y_grid'], pde_data['u_grid']
    f_time, py_time, pa_time = (
        pde_data['f_time'], pde_data['py_time'], pde_data['pa_time'])
    n_paths, n_times = Y_paths.shape
    X = np.zeros_like(Y_paths)

    X[:, 0] = RegularGridInterpolator(
        (y_grid, u_grid), f_time[0], bounds_error=False, fill_value=None
    )(np.column_stack([Y_paths[:, 0], np.log(A_paths[:, 0])]))

    for k in range(n_times - 1):
        pts = np.column_stack([Y_paths[:, k], np.log(A_paths[:, k])])
        interp = lambda arr: RegularGridInterpolator(
            (y_grid, u_grid), arr, bounds_error=False, fill_value=None)(pts)
        py  = interp(py_time[k])
        pa  = interp(pa_time[k])
        A_k = A_paths[:, k]
        X[:, k+1] = X[:, k] + A_k * py * dW[:, k] + nu * A_k * pa * dZ[:, k]

    return X


# Brownian transport PDE (for smile-dynamics experiments)

def solve_brownian_transport_pde(g0, x0, sigma0, T=1.0, Nx=501, Nt=250,
                                 x_width=7.0, verbose=True):
    """
    Solve the 1-D heat equation for a Brownian motion reference process.

        f_t + 0.5*sigma0^2 * f_xx = 0,    f(T, x) = g0(x)

    Used for smile-dynamics experiments with a Brownian (sigma-SABR, nu=0) reference.

    Returns
    -------
    dict with keys: t_grid, x_grid, f_time, fx_time
    """
    x_std  = sigma0 * np.sqrt(T)
    x_grid = np.linspace(x0 - x_width*x_std, x0 + x_width*x_std, Nx)
    hx = x_grid[1] - x_grid[0]
    dt = T / Nt

    u = g0(x_grid).copy()

    D2 = lil_matrix((Nx, Nx))
    for i in range(1, Nx - 1):
        D2[i, i-1] =  1.0/hx**2
        D2[i, i]   = -2.0/hx**2
        D2[i, i+1] =  1.0/hx**2
    D2 = D2.tocsr()

    L  = 0.5 * sigma0**2 * D2
    I  = eye(Nx, format='csr')
    b_idx = np.array([0, Nx-1], dtype=int)
    b_val = g0(x_grid)

    f_tau  = np.zeros((Nt + 1, Nx))
    fx_tau = np.zeros_like(f_tau)
    f_tau[0]  = u
    fx_tau[0] = np.gradient(u, x_grid, edge_order=2)

    M_left = I - 0.5*dt*L
    M_left = M_left.tolil()
    for idx in b_idx:
        M_left.rows[idx] = [idx];  M_left.data[idx] = [1.0]
    M_left  = M_left.tocsr()
    M_right = I + 0.5*dt*L
    solve   = factorized(M_left.tocsc())

    if verbose:
        print(f'Solving Brownian transport PDE: Nx={Nx}, Nt={Nt}')
        print(f'  x in [{x_grid[0]:.3f}, {x_grid[-1]:.3f}]')

    for n in range(Nt):
        rhs = M_right @ u
        rhs[b_idx] = b_val[b_idx]
        u  = solve(rhs)
        f_tau[n+1]  = u
        fx_tau[n+1] = np.gradient(u, x_grid, edge_order=2)

    return {
        't_grid':  np.linspace(0.0, T, Nt + 1),
        'x_grid':  x_grid,
        'f_time':  f_tau[::-1].copy(),
        'fx_time': fx_tau[::-1].copy(),
    }
