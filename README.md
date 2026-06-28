# Bass SV

Implementation of the Bass stochastic-volatility model — a continuous martingale
calibrated exactly to a discrete set of market marginals via optimal transport
(Brenier maps) applied to a reference process.

## Structure

```
data/
    spx-2024-02-20.xlsx      S&P 500 option surface
    aapl_combined_all_data.csv
    surface_storage.py

scientific/                  Step-by-step notebooks
    bass_calibration.ipynb   Main walkthrough: SVI → SABR → Bass transport
    01_sabr_accuracy.ipynb
    02_binned_model.ipynb
    03_smile_dynamics.ipynb
    heston.py / pde_solver.py / utils.py / plot_style.py

industrial/                  Clean class-based implementation
    bass_sv.py               BassSV — calibration + simulation
    svi.py                   SVI surface fit
    utils.py                 Density, CalibrationResult, BS helpers
    reference_models/
        base.py              ReferenceModel (abstract)
        brownian_motion.py   BrownianMotion
        arithmetic_sabr.py   ArithmeticSABR (β=0, Hagan + Breeden-Litsenberger)
        arithmetic_heston.py ArithmeticHeston
    calibration/
        market_calibration.ipynb   SPX calibration demo (BM + SABR, 8 maturities)
```

## Quick start

```python
from bass_sv import BassSV
from reference_models import BrownianMotion, ArithmeticSABR
from utils import Density

# build market marginals (cdf/qf callables per maturity)
market_marginals = {T1: Density(cdf=..., qf=...), T2: Density(...)}

# calibrate
ref   = ArithmeticSABR(s0=100, alpha=15, rho=-0.3, nu=0.6)
model = BassSV(s0=100, reference_model=ref, market_marginals=market_marginals)
model.calibrate(N=20_000, nsigma=7, tol=1e-3, max_iter=100, verbose=True)

# sample terminal distribution at maturity i
samples = model.simulate_smile(interval_idx=1, n_paths=500_000)
```

## How it works

1. **SVI** fits a no-arbitrage vol surface to market quotes.
2. **Reference model** (BrownianMotion or ArithmeticSABR) provides a tractable
   transition kernel.
3. **Banach iteration** solves for the alpha-measure CDF F_α on each interval via
   the A-operator fixed point.
4. **Brenier map** φ_i = Q_{μ_i} ∘ (k * F_α) transports reference paths to the
   market marginal exactly.

Convolution uses PyTorch FFT (O(N log N)).

## Dependencies

```
numpy scipy torch pandas matplotlib openpyxl
```
