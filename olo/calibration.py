from olo.data.extract import extractDataYieldNBB
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt


dfYield = extractDataYieldNBB(startPeriod="2000-01")


df10Y = dfYield[dfYield["IROLOBE2_MATUR"] == "10Y"].copy().reset_index(drop=True)


def calibrateVasicek(df10Y: pd.DataFrame):

    """
    Calibrate Vasicek parameters from a 10Y OLO yield time series using OLS.

    Vasicek SDE:       dr = κ(θ - r)dt + σ dW
    Discretised OLS:   Δr = a + b·r(t) + ε

    Mapping:
        κ = -b / Δt
        θ = -a / b
        σ = std(ε) / √Δt
    Returns
    -------
    dict with parameters, diagnostics, and arrays needed for plotting
    """

    #preperation for calibration of Vasicek model 
    r = df10Y["YIELD"].values / 100 # type: ignore # % -> decimal (0.01 = 1%)

    dt = 1/12 # Monthly timestep (1 year = 12 months)

    r_t = r[:-1] # t = 0, 1, ..., T-1

    r_t1 = r[1:]  # t = 1, 2, ..., T

    dr = r_t1 - r_t # dr = r(t+1) - r(t) = κ(θ - r(t))dt + σ√(dt)ε

    # OLS regression to estimate κ and θ 

    X      = np.column_stack([np.ones(len(r_t)), r_t])
    coeffs = np.linalg.lstsq(X, dr, rcond=None)[0]
    a, b   = coeffs

    eps = dr - (a + b * r_t) # Residuals from the regression
    sigma_eps = np.std(eps, ddof=2)


    # recover Vasicek parameters from OLS coefficients
    kappa = -b / dt   # mean reversion speed κ = -b / dt
    theta = -a / b     # long-term mean θ = -a / b

    sigma = sigma_eps / np.sqrt(dt) # volatility σ = std(ε) / sqrt(dt)
    r0 = r[-1] # initial short rate (first observed yield)

    #diagnostics 
    n = len(r_t)
    var_b = sigma_eps**2 * np.linalg.inv(X.T @ X)[1, 1]
    t_b   = b / np.sqrt(var_b)
    p_b   = 2 * t_dist.sf(np.abs(t_b), df=n - 2)
    r2    = 1 - np.var(eps) / np.var(dr)


    #print summary of calibration results
    print(f"Estimated Vasicek parameters:")
    print(f"  κ (mean reversion speed): {kappa:.4f}")
    print(f"  θ (long-term mean): {theta:.4f}")
    print(f"  σ (volatility): {sigma:.4f}")
    print(f"  r0 (initial short rate): {r0:.4f}")
    print("\nDiagnostics:")
    print(f"  t-statistic for b: {t_b:.4f}")
    print(f"  p-value for b: {p_b:.4f}")
    print(f"  R-squared: {r2:.4f}") 

    return {
        # ── Parameters ────────────────────────────────────────────────────
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "r0":    r0,
        # ── Diagnostics ───────────────────────────────────────────────────
        "diagnostics": {
            "n_obs":           n,
            "R2":              r2,
            "t_stat_b":        t_b,
            "p_val_b":         p_b,
            "half_life_years": np.log(2) / kappa,
        },
        # ── Arrays for plotting ───────────────────────────────────────────
        "arrays": {
            "r":         r,
            "r_t":       r_t,
            "dr":        dr,
            "dt":        dt,
            "eps":       eps,
        },
    }



results = calibrateVasicek(df10Y)



#unpack results for plotting
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
plt.savefig("tests/vasicek_validation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → tests/vasicek_validation.png")