"""
Shared utilities: Black-Scholes pricing and implied-vol inversion.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton


@np.vectorize
def bs_implied_vol(s, t, k, v, theta=1):
    """
    Black-Scholes implied volatility via Newton's method.

    Solves for σ such that C_BS(s, k, t, σ) = v, where C_BS is the
    standard Black-Scholes call price (zero rates, no dividends).

    Parameters
    ----------
    s     : spot / forward
    t     : time to maturity
    k     : strike
    v     : option price
    theta : 1 = call, -1 = put

    Returns
    -------
    float : implied volatility (or nan on failure)
    """
    x = np.log(s / k)
    p = v / np.sqrt(s * k)
    if np.isclose(x, 0):
        return -2.0 * norm.ppf((1.0 - p) / 2.0) / np.sqrt(t)

    def eq(sigma):
        return (theta * (np.exp(x/2) * norm.cdf(theta*(x/sigma + sigma/2))
                       - np.exp(-x/2) * norm.cdf(theta*(x/sigma - sigma/2))) - p)

    try:
        return newton(eq, x0=np.sqrt(2.0 * np.abs(x))) / np.sqrt(t)
    except Exception:
        return np.nan


# backward-compatible alias
implied_vol = bs_implied_vol


def bs_call_price(s, sigma, t, k):
    """Black-Scholes call price (zero rates)."""
    d1 = (np.log(s / k) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * norm.cdf(d1) - k * norm.cdf(d2)


def mc_iv(s0, t, strikes, terminal_samples):
    """
    Compute BS implied volatilities from Monte Carlo terminal samples.

    Parameters
    ----------
    s0               : initial spot (used as proxy for forward)
    t                : time to maturity
    strikes          : array of strikes
    terminal_samples : (n_paths,) simulated terminal prices

    Returns
    -------
    array of BS implied vols, one per strike
    """
    prices = np.mean(
        np.maximum(np.subtract.outer(terminal_samples, strikes), 0), axis=0)
    return bs_implied_vol(s0, t, strikes, prices)