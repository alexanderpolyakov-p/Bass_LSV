from dataclasses import dataclass
import numpy as np
import cmath
from math import log, pi, inf
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.interpolate import interp1d
from utils import *


# Heston stochastic-volatility model
@dataclass
class Heston:
    s0: float       # initial spot
    v0: float       # initial variance
    kappa: float    # mean-reversion speed
    theta: float    # long-run variance
    xi: float       # vol-of-vol
    rho: float      # dW-dZ correlation

    def __post_init__(self):
        # vectorize call price so it can be applied to strike arrays
        self.call_price = np.vectorize(self._call_price, excluded='self')

    # characteristic function of log(S_t)
    def _cf(self, u: float | complex, t: float) -> complex:
        d = cmath.sqrt((1j*self.rho*self.xi*u - self.kappa)**2 + self.xi**2*(1j*u + u**2))
        g = ((1j*self.rho*self.xi*u - self.kappa + d) / (1j*self.rho*self.xi*u - self.kappa - d))
        C = self.kappa*self.theta/self.xi**2 * ((self.kappa - 1j*self.rho*self.xi*u - d)*t -
                                                2*cmath.log((1 - g*cmath.exp(-d*t))/(1-g)))
        D = (self.kappa - 1j*self.rho*self.xi*u - d)/self.xi**2 * ((1-cmath.exp(-d*t)) /
                                                                   (1-g*cmath.exp(-d*t)))
        return cmath.exp(C + D*self.v0)

    # call price with maturity t and strike k via Carr-Madan integral
    def _call_price(self, t: float, k: float) -> float:
        def integrand(u):
            return (cmath.exp(1j*u*log(self.s0/k)) / (1j*u) *
                    (self._cf(u-1j, t) - k/self.s0 * self._cf(u, t))).real
        return self.s0 * ((1 - k/self.s0)/2 +
                          1/pi * quad(integrand, 0, inf, epsrel=1e-12, epsabs=1e-20)[0])

    # Black-Scholes implied vol
    def implied_vol(self, t: float | np.ndarray, k: float | np.ndarray) -> float | np.ndarray:
        return implied_vol(self.s0, t, k, self.call_price(t, k))

    # CDF of S_t at level s
    def cdf(self, t: float, s: float) -> float:
        def integrand(u):
            return (cmath.exp(1j*u*log(self.s0/s)) / (1j*u) * self._cf(u, t)).real
        return 0.5 - 1/pi * quad(integrand, 0, inf)[0]

    # quantile function at level p
    def quantile(self, t: float, p: float) -> float:
        return newton(lambda s: self.cdf(t, s) - p, self.s0)

    # interpolated CDF at time t over n quantile points
    # CDF is set to 0 left of the 1/(4n) quantile and 1 right of 1 - 1/(4n)
    def cdf_interpolate(self, t: float, n: int = 1000, kind: str = 'linear') -> callable:
        P = np.linspace(0, 1, n+1)
        s_p0 = self.quantile(t, 1/(4*n))
        s_p1 = self.quantile(t, 1 - 1/(4*n))
        S = [s_p0] + [self.quantile(t, p) for p in P[1:-1]] + [s_p1]
        return interp1d(S, P, fill_value=(0, 1), bounds_error=False, kind=kind)

    # interpolated quantile function at time t over n points
    def quantile_interpolate(self, t: float, n: int = 1000, kind: str = 'linear') -> callable:
        P = np.linspace(0, 1, n+1)
        s_p0 = self.quantile(t, 1/(4*n))
        s_p1 = self.quantile(t, 1 - 1/(4*n))
        S = [s_p0] + [self.quantile(t, p) for p in P[1:-1]] + [s_p1]
        return interp1d(P, S, bounds_error=True, kind=kind)
