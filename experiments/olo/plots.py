"""Plots of the rate engine: Vasicek fit, NSS forward curve, simulated paths.

Moved out of pension/rates so the engine itself has no plotting or file output.
Run from the repo root to redo the calibration figures on the cached NBB data:
    python experiments/olo/plots.py
Figures go to results/figs/olo/.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from pension.rates.calibration import (calibrateVasicek, bootstrapForwardCurve,
                                       nss_yield, nss_forward)
from pension.rates.data import load_olo
from pension.rates.simulation import simulateVasicek, simulateHullWhite

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "figs", "olo")


def plotVasicekCalibration(df10Y, results):
    """Four-panel visual validation of the Vasicek OLS fit (was a commented-out
    block in calibration.py)."""
    os.makedirs(OUT, exist_ok=True)

    # unpack results for plotting
    kappa     = results["kappa"]
    sigma    = results["sigma"]
    theta     = results["theta"]
    r         = results["arrays"]["r"]
    r_t       = results["arrays"]["r_t"]
    dr        = results["arrays"]["dr"]
    eps       = results["arrays"]["eps"]
    dt        = results["arrays"]["dt"]
    r0       = results["r0"]


    # ── 4. Validation plots ───────────────────────────────────────────────────
    dr_fit     = -kappa * (r_t - theta) * dt
    t_grid     = np.arange(len(r)) * dt
    mean_path  = theta + (r0 - theta) * np.exp(-kappa * t_grid)
    std_band   = sigma * np.sqrt((1 - np.exp(-2 * kappa * t_grid)) / (2 * kappa))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Vasicek Calibration — Visual Validation (10Y Belgian OLO)",
                 fontsize=14, fontweight="bold")

    # Panel 1 — Actual vs mean path
    ax1 = axes[0, 0]
    ax1.plot(df10Y["DATE"], r * 100,
             color="steelblue", linewidth=1.5, label="Actual 10Y OLO")
    ax1.plot(df10Y["DATE"], mean_path * 100,
             color="red", linewidth=1.5, linestyle="--", label="Vasicek mean path")
    ax1.fill_between(df10Y["DATE"],
                     (mean_path - 2 * std_band) * 100,
                     (mean_path + 2 * std_band) * 100,
                     alpha=0.15, color="red", label="±2σ band")
    ax1.axhline(theta * 100, color="darkred", linestyle=":",
                linewidth=1, label=f"θ = {theta*100:.2f}%")
    ax1.set_title("Actual vs Vasicek Mean Path")
    ax1.set_ylabel("Yield (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — OLS fit
    ax2 = axes[0, 1]
    ax2.scatter(r_t * 100, dr * 100,
                alpha=0.4, s=15, color="steelblue", label="Actual Δr")
    ax2.plot(r_t * 100, dr_fit * 100,
             color="red", linewidth=2, label="OLS fit")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("OLS Fit: Δr vs r(t)")
    ax2.set_xlabel("r(t) (%)")
    ax2.set_ylabel("Δr (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — Residuals over time
    ax3 = axes[1, 0]
    ax3.plot(df10Y["DATE"][1:], eps * 100,
             color="steelblue", linewidth=0.8)
    ax3.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax3.fill_between(df10Y["DATE"][1:], eps * 100, 0,
                     alpha=0.2, color="steelblue")
    ax3.set_title("Residuals over Time")
    ax3.set_ylabel("Residual (%)")
    ax3.grid(True, alpha=0.3)

    # Panel 4 — Residual distribution
    ax4 = axes[1, 1]
    ax4.hist(eps * 100, bins=40, density=True,
             color="steelblue", alpha=0.6, label="Residuals")
    x_norm  = np.linspace(eps.min(), eps.max(), 200) * 100
    std_res = np.std(eps * 100)
    mu_res  = np.mean(eps * 100)
    norm_pdf = (1 / (std_res * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * ((x_norm - mu_res) / std_res) ** 2)
    ax4.plot(x_norm, norm_pdf, color="red", linewidth=2, label="Normal fit")
    ax4.set_title("Residual Distribution")
    ax4.set_xlabel("Residual (%)")
    ax4.set_ylabel("Density")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/vasicek_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {OUT}/vasicek_validation.png")






def plotForwardCurve(
    curve:      dict,
    maturities: np.ndarray,
    yields:     np.ndarray,
):
    """
    3-panel validation plot:
      Panel 1 — observed yields vs NSS fit
      Panel 2 — instantaneous forward curve
      Panel 3 — df/dT (smoothness check)
    """
 
    os.makedirs(OUT, exist_ok=True)
 
    t     = curve["t_grid"]
    f     = curve["f"]
    Y     = curve["Y"]
    df_dT = curve["df_dT"]
    P     = curve["P"]
 
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("NSS Forward Curve Bootstrap — Belgian OLO (Apr 2026)",
                 fontsize=13, fontweight="bold")
 
    # ── Panel 1: yield curve fit ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(maturities, yields,
                color="steelblue", zorder=5, s=50, label="Observed OLO yields")
    ax1.plot(t, Y * 100, color="crimson", linewidth=2, label="NSS fit Y(0,T)")
    ax1.set_title("Yield Curve Fit")
    ax1.set_xlabel("Maturity T (years)")
    ax1.set_ylabel("Yield (%)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
 
    # Annotate fit quality
    y_hat = nss_yield(maturities, *curve["params"])
    rmse  = np.sqrt(np.mean(((y_hat - yields / 100) * 10_000) ** 2))
    ax1.annotate(f"RMSE = {rmse:.2f} bps", xy=(0.05, 0.95),
                 xycoords="axes fraction", fontsize=9,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
 
    # ── Panel 2: forward rate curve ───────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(t, f * 100, color="steelblue", linewidth=2,
             label="Forward rate f(0,T) [NSS analytic]")
    ax2.plot(t, Y * 100, color="crimson", linewidth=1.5, linestyle="--",
             label="Yield curve Y(0,T)")
    ax2.set_title("Instantaneous Forward Curve f(0,T)")
    ax2.set_xlabel("Maturity T (years)")
    ax2.set_ylabel("Rate (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
 
    # ── Panel 3: df/dT smoothness ─────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(t, df_dT * 100, color="darkorange", linewidth=1.8,
             label="df/dT  (analytic NSS)")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax3.set_title("Forward Rate Slope df/dT\n(smoothness check — fed into θ(t))")
    ax3.set_xlabel("Maturity T (years)")
    ax3.set_ylabel("df/dT  (% per year)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(f"{OUT}/nss_forward_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {OUT}/nss_forward_curve.png")


def plotSimulation(paths: np.ndarray, results: dict, dt: float = 1/12):
    """
    2-panel plot: sample paths + terminal rate distribution.
    """

    theta = results["theta"]
    r0    = results["r0"]

    n_steps, n_paths = paths.shape
    t_axis   = np.arange(n_steps) * dt
    terminal = paths[-1, :] * 100

    # ── Percentile bands ──────────────────────────────────────────────────
    p05 = np.percentile(paths,  5, axis=1) * 100
    p25 = np.percentile(paths, 25, axis=1) * 100
    p50 = np.percentile(paths, 50, axis=1) * 100
    p75 = np.percentile(paths, 75, axis=1) * 100
    p95 = np.percentile(paths, 95, axis=1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Vasicek Monte Carlo — {n_paths} paths | "
        f"{int(n_steps * dt)}Y horizon | "
        f"θ={theta*100:.2f}% | κ={results['kappa']:.4f}",
        fontsize=12, fontweight="bold"
    )

    # Panel 1 — Sample paths + percentile bands
    ax1 = axes[0]
    for i in range(min(100, n_paths)):
        ax1.plot(t_axis, paths[:, i] * 100,
                 color="steelblue", alpha=0.06, linewidth=0.5)
    ax1.fill_between(t_axis, p05, p95, alpha=0.12, color="red", label="5–95th pct")
    ax1.fill_between(t_axis, p25, p75, alpha=0.22, color="red", label="25–75th pct")
    ax1.plot(t_axis, p50,          color="red",     linewidth=2,   label="Median")
    ax1.axhline(theta * 100,       color="darkred", linewidth=1.2,
                linestyle=":",     label=f"θ = {theta*100:.2f}%")
    ax1.axhline(r0 * 100,          color="black",   linewidth=1,
                linestyle="--",    label=f"r₀ = {r0*100:.2f}%")
    ax1.set_title("Simulated Rate Paths")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Rate (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Terminal rate distribution
    ax2 = axes[1]
    ax2.hist(terminal, bins=60, density=True,
             color="steelblue", alpha=0.6, label="Terminal rates")
    ax2.axvline(np.mean(terminal),         color="red",     linewidth=2,
                label=f"Mean  = {np.mean(terminal):.2f}%")
    ax2.axvline(theta * 100,               color="darkred", linewidth=1.5,
                linestyle=":",             label=f"θ     = {theta*100:.2f}%")
    ax2.axvline(np.percentile(terminal,  5), color="orange", linewidth=1.2,
                linestyle="--",            label=f"p5    = {np.percentile(terminal, 5):.2f}%")
    ax2.axvline(np.percentile(terminal, 95), color="orange", linewidth=1.2,
                linestyle="--",            label=f"p95   = {np.percentile(terminal,95):.2f}%")
    ax2.set_title(f"Terminal Rate Distribution (Year {int(n_steps * dt)})")
    ax2.set_xlabel("Rate (%)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/vasicek_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {OUT}/vasicek_simulation.png")


def plotSimulationHW(
    paths: np.ndarray,
    curve: dict,
    kappa: float,
    sigma: float,
    dt:    float = 1/12,
):
    """
    2-panel plot mirroring plotSimulation, with a no-arbitrage validation:
    the empirical mean path must lie on the theoretical conditional mean
    alpha(t), which sits a small (Q-convexity) gap above the forward curve f(0,t).
    """

    os.makedirs(OUT, exist_ok=True)

    n_steps, n_paths = paths.shape
    t_axis = np.arange(n_steps) * dt

    params = curve["params"]
    f0t    = nss_forward(t_axis, *params)
    alpha  = f0t + (sigma**2 / (2 * kappa**2)) * (1 - np.exp(-kappa * t_axis))**2

    emp_mean = paths.mean(axis=1) * 100
    terminal = paths[-1, :] * 100

    p05 = np.percentile(paths,  5, axis=1) * 100
    p25 = np.percentile(paths, 25, axis=1) * 100
    p75 = np.percentile(paths, 75, axis=1) * 100
    p95 = np.percentile(paths, 95, axis=1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Hull-White Monte Carlo — {n_paths} paths | "
        f"{int(n_steps * dt)}Y horizon | κ={kappa:.4f} | σ={sigma*100:.2f}%",
        fontsize=12, fontweight="bold"
    )

    # Panel 1 — sample paths, bands, and the mean-vs-alpha no-arb check
    ax1 = axes[0]
    for i in range(min(100, n_paths)):
        ax1.plot(t_axis, paths[:, i] * 100,
                 color="steelblue", alpha=0.06, linewidth=0.5)
    ax1.fill_between(t_axis, p05, p95, alpha=0.12, color="red", label="5–95th pct")
    ax1.fill_between(t_axis, p25, p75, alpha=0.22, color="red", label="25–75th pct")
    ax1.plot(t_axis, f0t * 100,    color="black",  linewidth=1.5, linestyle="--",
             label="f(0,t) — forward curve")
    ax1.plot(t_axis, alpha * 100,  color="darkred", linewidth=1.6, linestyle=":",
             label="α(t) — theoretical mean")
    ax1.plot(t_axis, emp_mean,     color="gold",   linewidth=1.4,
             label="empirical mean (should match α)")
    ax1.set_title("Simulated Rate Paths  (mean must track α(t))")
    ax1.set_xlabel("Years"); ax1.set_ylabel("Rate (%)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Panel 2 — terminal distribution
    ax2 = axes[1]
    ax2.hist(terminal, bins=60, density=True,
             color="steelblue", alpha=0.6, label="Terminal rates")
    ax2.axvline(np.mean(terminal), color="red", linewidth=2,
                label=f"Mean = {np.mean(terminal):.2f}%")
    ax2.axvline(alpha[-1] * 100, color="darkred", linewidth=1.5, linestyle=":",
                label=f"α(T) = {alpha[-1]*100:.2f}%")
    ax2.axvline(np.percentile(terminal, 5),  color="orange", linewidth=1.2,
                linestyle="--", label=f"p5  = {np.percentile(terminal,5):.2f}%")
    ax2.axvline(np.percentile(terminal, 95), color="orange", linewidth=1.2,
                linestyle="--", label=f"p95 = {np.percentile(terminal,95):.2f}%")
    ax2.set_title(f"Terminal Rate Distribution (Year {int(n_steps * dt)})")
    ax2.set_xlabel("Rate (%)"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/hull_white_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {OUT}/hull_white_simulation.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df10Y, maturities, yields = load_olo()

    results = calibrateVasicek(df10Y)
    plotVasicekCalibration(df10Y, results)

    curve = bootstrapForwardCurve(maturities, yields)
    plotForwardCurve(curve, maturities, yields)

    paths = simulateVasicek(kappa=results["kappa"], theta=results["theta"], sigma=results["sigma"],
                            r0=results["r0"], T=40, n_paths=5000, dt=1/12)
    print(f"Vasicek  mean terminal rate {paths[-1].mean()*100:.4f} %   std {paths[-1].std()*100:.4f} %")
    plotSimulation(paths, results)

    paths = simulateHullWhite(curve=curve, kappa=results["kappa"], sigma=results["sigma"],
                              T=40, n_paths=5000, dt=1/12)
    print(f"Hull-White r(0) {paths[0, 0]*100:.4f} %  (= f(0,0) = β0+β1)   "
          f"mean terminal rate {paths[-1].mean()*100:.4f} %")
    plotSimulationHW(paths, curve, results["kappa"], results["sigma"])
