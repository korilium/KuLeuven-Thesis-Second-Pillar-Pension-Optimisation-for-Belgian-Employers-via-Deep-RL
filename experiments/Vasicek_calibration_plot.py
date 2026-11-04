"""Diagnostic plots for the OLS Vasicek calibration in `calibrate_vasicek`.

Reproduces the intermediate quantities of the calibration (y_t, dy, a, b, eps)
and shows, in four panels:
  1. the input series with theta_hat and y0,
  2. the regression dy = a + b * y_t with the fitted line and its zero crossing,
  3. the residuals over time,
  4. the residual distribution against the normal density implied by sigma_hat.

Run from the repository root:
    python -m experiments.vasicek_calibration_plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def ols_components(y, dt=1 / 12):
    """Same computation as calibrate_vasicek, but returning the intermediates."""
    y = np.asarray(y, dtype=float)
    y_t, dy = y[:-1], y[1:] - y[:-1]
    X = np.column_stack([np.ones(len(y_t)), y_t])
    a, b = np.linalg.lstsq(X, dy, rcond=None)[0]
    eps = dy - (a + b * y_t)
    params = dict(
        kappa=-b / dt,
        theta=-a / b,
        sigma=np.std(eps, ddof=2) / np.sqrt(dt),
        y0=y[-1],
    )
    r2 = 1 - np.sum(eps**2) / np.sum((dy - dy.mean()) ** 2)
    return dict(y=y, y_t=y_t, dy=dy, a=a, b=b, eps=eps, params=params, r2=r2, dt=dt)


def plot_vasicek_calibration(y, dates=None, dt=1 / 12, pct=True, title=None):
    """Four-panel diagnostic of the OLS Vasicek calibration.

    y      : monthly series in decimals (as calibrate_vasicek expects)
    dates  : optional array of dates of the same length as y
    pct    : display in percent (True) or decimals (False); estimation is unaffected
    """
    c = ols_components(y, dt)
    p, a, b = c["params"], c["a"], c["b"]
    s = 100.0 if pct else 1.0
    unit = "%" if pct else "decimal"
    t = np.arange(len(c["y"])) if dates is None else np.asarray(dates)
    mean_reverting = b < 0

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(title or "OLS calibration of the Vasicek model", fontsize=13)

    # 1. Input series
    a1 = ax[0, 0]
    a1.plot(t, c["y"] * s, lw=1.2, label="y")
    if mean_reverting:
        a1.axhline(p["theta"] * s, ls="--", color="C1", label=rf"$\hat\theta$ = {p['theta'] * s:.2f}")
    a1.scatter([t[-1]], [p["y0"] * s], color="C3", zorder=3, label=rf"$y_0$ = {p['y0'] * s:.2f}")
    a1.set_title("1. Input series")
    a1.set_ylabel(f"level ({unit})")
    a1.legend(fontsize=9)

    # 2. Regression dy on y_t
    a2 = ax[0, 1]
    a2.scatter(c["y_t"] * s, c["dy"] * s, s=12, alpha=0.5, label=r"$(y_t, \Delta y_t)$")
    grid = np.linspace(c["y_t"].min(), c["y_t"].max(), 200)
    a2.plot(grid * s, (a + b * grid) * s, color="C1", lw=2, label=r"fit $a + b\,y_t$")
    a2.axhline(0, color="grey", lw=0.8)
    if mean_reverting:
        a2.axvline(p["theta"] * s, ls="--", color="C1", lw=1, label=r"$\hat\theta = -a/b$")
    a2.set_title(r"2. Regression $\Delta y_t = a + b\,y_t + e_t$")
    a2.set_xlabel(f"$y_t$ ({unit})")
    a2.set_ylabel(rf"$\Delta y_t$ ({unit})")
    a2.legend(fontsize=9, loc="upper right")
    txt = (
        f"a = {a:.3e}\nb = {b:.3e}\n"
        rf"$\hat\kappa$ = $-b/\Delta t$ = {p['kappa']:.3f}" "\n"
        rf"$\hat\theta$ = $-a/b$ = {p['theta'] * s:.2f} {unit}" "\n"
        rf"$\hat\sigma$ = {p['sigma'] * s:.2f} {unit}" "\n"
        rf"$R^2$ = {c['r2']:.3f}"
    )
    if not mean_reverting:
        txt += "\nb >= 0: no mean reversion"
    a2.text(0.02, 0.03, txt, transform=a2.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # 3. Residuals over time
    a3 = ax[1, 0]
    sd_e = p["sigma"] * np.sqrt(dt)
    a3.plot(t[1:], c["eps"] * s, lw=0.9)
    a3.axhline(0, color="grey", lw=0.8)
    for k in (-2, 2):
        a3.axhline(k * sd_e * s, ls=":", color="C3", lw=1)
    rho1 = np.corrcoef(c["eps"][:-1], c["eps"][1:])[0, 1]
    a3.set_title(rf"3. Residuals over time ($\pm 2\hat\sigma\sqrt{{\Delta t}}$; lag-1 autocorr = {rho1:.2f})")
    a3.set_ylabel(f"$e_t$ ({unit})")

    # 4. Residual distribution vs implied normal
    a4 = ax[1, 1]
    a4.hist(c["eps"] * s, bins=40, density=True, alpha=0.6, label="residuals")
    xs = np.linspace(*a4.get_xlim(), 300)
    a4.plot(xs, stats.norm.pdf(xs, 0, sd_e * s), color="C1", lw=2,
            label=rf"$N(0, (\hat\sigma\sqrt{{\Delta t}})^2)$")
    jb = stats.jarque_bera(c["eps"])
    a4.set_title(f"4. Residual distribution (excess kurtosis = {stats.kurtosis(c['eps']):.2f}, "
                 f"JB p = {jb.pvalue:.3f})")
    a4.set_xlabel(f"$e_t$ ({unit})")
    a4.legend(fontsize=9)

    fig.tight_layout()
    return fig, c


if __name__ == "__main__":
    # Replace this block with your own monthly series (in decimals), e.g. the
    # 10-year OLO yield loaded through pension.rates.data, plus its dates.
    rng = np.random.default_rng(0)
    dt, kappa, theta, sigma, n = 1 / 12, 0.3, 0.025, 0.008, 240
    e = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - e**2) / (2 * kappa))
    y = np.empty(n)
    y[0] = 0.01
    for i in range(1, n):
        y[i] = theta + (y[i - 1] - theta) * e + sd * rng.standard_normal()
    dates = np.arange("2006-01", "2026-01", dtype="datetime64[M]")

    fig, comps = plot_vasicek_calibration(y, dates=dates, title="Synthetic Vasicek series (demo)")

    # Cross-check against the project's own function, if it is importable.
    try:
        from pension.rates.calibration import calibrate_vasicek
        ref = calibrate_vasicek(y, dt=dt)
        same = all(np.isclose(ref[k], comps["params"][k]) for k in ref)
        print("Matches pension.rates.calibration.calibrate_vasicek:", same)
    except ImportError:
        print("calibrate_vasicek not importable here; skipped cross-check.")

    print({k: round(float(v), 5) for k, v in comps["params"].items()})
    plt.show()