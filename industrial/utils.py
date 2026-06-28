from dataclasses import dataclass
from typing import List, Callable
import numpy as np
import scipy.stats as st
from scipy.optimize import newton


def bs_call_price(s, sigma, t, k):
    """Black-Scholes call price (zero rates)."""
    k = np.asarray(k, float)
    d1 = (np.log(s / k) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * st.norm.cdf(d1) - k * st.norm.cdf(d2)


@np.vectorize
def implied_vol(s, t, k, v, theta=1):
    """BS implied volatility via Newton."""
    x = np.log(float(s) / float(k))
    p = float(v) / np.sqrt(float(s) * float(k))
    if np.isclose(x, 0):
        return -2.0 * st.norm.ppf((1 - p) / 2) / np.sqrt(float(t))
    def eq(sig):
        return (theta * (np.exp(x / 2) * st.norm.cdf(theta * (x / sig + sig / 2))
                         - np.exp(-x / 2) * st.norm.cdf(theta * (x / sig - sig / 2))) - p)
    try:
        return newton(eq, x0=np.sqrt(2 * abs(x))) / np.sqrt(float(t))
    except Exception:
        return np.nan


def mc_iv(s0, t, strikes, samples):
    """BS implied vols from MC terminal samples."""
    prices = np.mean(np.maximum(np.subtract.outer(samples, np.asarray(strikes)), 0), axis=0)
    return implied_vol(s0, t, strikes, prices)


@dataclass
class Density:
    cdf: Callable
    qf: Callable


@dataclass
class CalibrationResult:
    maturity: float
    error_evolution: List[float]
    iterations: int
    final_error: float
