"""
calibration.py -- the Vasicek calibration of the monthly 10Y OLO under P (OLS) and the
historical term spread s_term = mean(10Y - 1Y), from the cached NBB data.

Vasicek SDE:       dy = kappa (theta - y) dt + sigma dW
Discretised OLS:   dy_m = a + b y_m + e_m   ->   kappa = -b/dt, theta = -a/b, sigma = std(e)/sqrt(dt)
"""
import functools

import numpy as np

from pension.rates.data import load_olo, load_olo_short


def calibrate_vasicek(y, dt=1 / 12):
    """OLS calibration of a monthly series y (decimal); y0 is its last value."""
    y_t, dy = y[:-1], y[1:] - y[:-1]
    X = np.column_stack([np.ones(len(y_t)), y_t])
    a, b = np.linalg.lstsq(X, dy, rcond=None)[0]
    eps = dy - (a + b * y_t)
    return dict(kappa=-b / dt, theta=-a / b, sigma=np.std(eps, ddof=2) / np.sqrt(dt), y0=y[-1])


@functools.lru_cache(maxsize=1)
def calibration():
    """Once per process: the Vasicek parameters of the 10Y, s_term, the monthly 10Y
    history (oldest first; the last value is model t0) and the date of t0."""
    df10, _, _ = load_olo()
    _, df_short = load_olo_short()
    assert np.array_equal(df10["DATE"].values, df_short["DATE"].values), "10Y and 1Y must share dates"
    y10, r = df10["YIELD"].values / 100.0, df_short["YIELD"].values / 100.0
    return dict(vasicek=calibrate_vasicek(y10), spread=float((y10 - r).mean()),
                hist10Y=y10, t0=df10["DATE"].iloc[-1])
