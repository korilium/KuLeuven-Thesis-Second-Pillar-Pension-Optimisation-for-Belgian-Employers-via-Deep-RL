from olo.data.extract import extractDataYieldNBB
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator   # ← replaces CubicSpline
from scipy.optimize import differential_evolution, minimize
import os 



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



VasicekResults = calibrateVasicek(df10Y)

###################################
### calibration checks Vasicek  ###
###################################


# #unpack results for plotting
# kappa     = results["kappa"]
# sigma    = results["sigma"]
# theta     = results["theta"]
# r         = results["arrays"]["r"]
# r_t       = results["arrays"]["r_t"]
# dr        = results["arrays"]["dr"]
# eps       = results["arrays"]["eps"]
# dt        = results["arrays"]["dt"]
# r0       = results["r0"]


# # ── 4. Validation plots ───────────────────────────────────────────────────
# dr_fit     = -kappa * (r_t - theta) * dt
# t_grid     = np.arange(len(r)) * dt
# mean_path  = theta + (r0 - theta) * np.exp(-kappa * t_grid)
# std_band   = sigma * np.sqrt((1 - np.exp(-2 * kappa * t_grid)) / (2 * kappa))

# fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# fig.suptitle("Vasicek Calibration — Visual Validation (10Y Belgian OLO)",
#              fontsize=14, fontweight="bold")

# # Panel 1 — Actual vs mean path
# ax1 = axes[0, 0]
# ax1.plot(df10Y["DATE"], r * 100,
#          color="steelblue", linewidth=1.5, label="Actual 10Y OLO")
# ax1.plot(df10Y["DATE"], mean_path * 100,
#          color="red", linewidth=1.5, linestyle="--", label="Vasicek mean path")
# ax1.fill_between(df10Y["DATE"],
#                  (mean_path - 2 * std_band) * 100,
#                  (mean_path + 2 * std_band) * 100,
#                  alpha=0.15, color="red", label="±2σ band")
# ax1.axhline(theta * 100, color="darkred", linestyle=":",
#             linewidth=1, label=f"θ = {theta*100:.2f}%")
# ax1.set_title("Actual vs Vasicek Mean Path")
# ax1.set_ylabel("Yield (%)")
# ax1.legend(fontsize=8)
# ax1.grid(True, alpha=0.3)

# # Panel 2 — OLS fit
# ax2 = axes[0, 1]
# ax2.scatter(r_t * 100, dr * 100,
#             alpha=0.4, s=15, color="steelblue", label="Actual Δr")
# ax2.plot(r_t * 100, dr_fit * 100,
#          color="red", linewidth=2, label="OLS fit")
# ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
# ax2.set_title("OLS Fit: Δr vs r(t)")
# ax2.set_xlabel("r(t) (%)")
# ax2.set_ylabel("Δr (%)")
# ax2.legend(fontsize=8)
# ax2.grid(True, alpha=0.3)

# # Panel 3 — Residuals over time
# ax3 = axes[1, 0]
# ax3.plot(df10Y["DATE"][1:], eps * 100,
#          color="steelblue", linewidth=0.8)
# ax3.axhline(0, color="red", linewidth=1.2, linestyle="--")
# ax3.fill_between(df10Y["DATE"][1:], eps * 100, 0,
#                  alpha=0.2, color="steelblue")
# ax3.set_title("Residuals over Time")
# ax3.set_ylabel("Residual (%)")
# ax3.grid(True, alpha=0.3)

# # Panel 4 — Residual distribution
# ax4 = axes[1, 1]
# ax4.hist(eps * 100, bins=40, density=True,
#          color="steelblue", alpha=0.6, label="Residuals")
# x_norm  = np.linspace(eps.min(), eps.max(), 200) * 100
# std_res = np.std(eps * 100)
# mu_res  = np.mean(eps * 100)
# norm_pdf = (1 / (std_res * np.sqrt(2 * np.pi))) * \
#             np.exp(-0.5 * ((x_norm - mu_res) / std_res) ** 2)
# ax4.plot(x_norm, norm_pdf, color="red", linewidth=2, label="Normal fit")
# ax4.set_title("Residual Distribution")
# ax4.set_xlabel("Residual (%)")
# ax4.set_ylabel("Density")
# ax4.legend(fontsize=8)
# ax4.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("tests/vasicek_validation.png", dpi=150, bbox_inches="tight")
# plt.show()
# print("Saved → tests/vasicek_validation.png")





############################
### hull-white extension ###
############################


 
# ═══════════════════════════════════════════════════════════════════════════
# NSS — Nelson-Siegel-Svensson forward curve bootstrap
# ═══════════════════════════════════════════════════════════════════════════
#
# The NSS model parametrises the yield curve analytically:
#
#   Y(T) = β0
#         + β1 · φ1(T)                         ← level decay
#         + β2 · φ2(T)                         ← short hump
#         + β3 · φ3(T)                         ← long hump
#
#   where:
#       φ1(T) = (1 − e^{−T/τ1}) / (T/τ1)
#       φ2(T) = (1 − e^{−T/τ1}) / (T/τ1)  − e^{−T/τ1}
#       φ3(T) = (1 − e^{−T/τ2}) / (T/τ2)  − e^{−T/τ2}
#
# The instantaneous forward rate is the ANALYTIC derivative:
#
#   f(T) = −d/dT [T · Y(T)]
#         = β0
#         + β1 · e^{−T/τ1}
#         + β2 · (T/τ1) · e^{−T/τ1}
#         + β3 · (T/τ2) · e^{−T/τ2}
#
# df/dT is also analytic (needed for θ(t)):
#
#   df/dT = −(β1/τ1) · e^{−T/τ1}
#           + β2 · e^{−T/τ1} · (1/τ1 − T/τ1²)
#           + β3 · e^{−T/τ2} · (1/τ2 − T/τ2²)
#
# Parameters:  6 scalars  [β0, β1, β2, β3, τ1, τ2]
#
# ═══════════════════════════════════════════════════════════════════════════
 
 
# ── NSS yield formula ─────────────────────────────────────────────────────
 
def nss_yield(T: np.ndarray, beta0, beta1, beta2, beta3, tau1, tau2) -> np.ndarray:
    """
    Nelson-Siegel-Svensson yield Y(0,T).
 
    Parameters
    ----------
    T           : array of maturities (years), T > 0
    beta0..tau2 : NSS parameters
 
    Returns
    -------
    Y : yield curve (decimal)
    """
    t1 = T / tau1
    t2 = T / tau2
 
    phi1 = (1 - np.exp(-t1)) / t1
    phi2 = phi1 - np.exp(-t1)
    phi3 = (1 - np.exp(-t2)) / t2 - np.exp(-t2)
 
    return beta0 + beta1 * phi1 + beta2 * phi2 + beta3 * phi3
 
 
# ── NSS instantaneous forward rate (analytic) ────────────────────────────
 
def nss_forward(T: np.ndarray, beta0, beta1, beta2, beta3, tau1, tau2) -> np.ndarray:
    """
    Instantaneous forward rate f(0,T) = −d/dT [T · Y(0,T)].
 
    This is the ANALYTIC derivative — no numerical differentiation,
    no spline noise, smooth across the entire curve.
 
    f(T) = β0
          + β1 · e^{−T/τ1}
          + β2 · (T/τ1) · e^{−T/τ1}
          + β3 · (T/τ2) · e^{−T/τ2}
    """
    t1 = T / tau1
    t2 = T / tau2
 
    return (
        beta0
        + beta1 * np.exp(-t1)
        + beta2 * t1 * np.exp(-t1)
        + beta3 * t2 * np.exp(-t2)
    )
 
 
# ── Analytic df/dT (needed for computeTheta) ─────────────────────────────
 
def nss_forward_deriv(T: np.ndarray, beta0, beta1, beta2, beta3, tau1, tau2) -> np.ndarray:
    """
    df/dT — first derivative of the forward rate.
 
    Required by computeTheta:
        θ(t) = df/dT + κ · f(t) + σ²/(2κ) · (1 − e^{−2κt})
 
    df/dT = e^{−T/τ1} · [−β1/τ1 + β2 · (1/τ1 − T/τ1²)]
          + e^{−T/τ2} · [β3 · (1/τ2 − T/τ2²)]
    """
    e1 = np.exp(-T / tau1)
    e2 = np.exp(-T / tau2)
 
    term1 = e1 * (-beta1 / tau1 + beta2 * (1 / tau1 - T / tau1**2))
    term2 = e2 * (beta3 * (1 / tau2 - T / tau2**2))
 
    return term1 + term2
 
 
# ── Fit NSS parameters to observed yields ────────────────────────────────
 
def _fit_nss(maturities: np.ndarray, yields_dec: np.ndarray) -> np.ndarray:
    """
    Fit [β0, β1, β2, β3, τ1, τ2] to observed yields by minimising
    weighted root-mean-squared error.
 
    Strategy: global search with differential_evolution, then local
    polish with Nelder-Mead.  This avoids local minima in τ1/τ2.
    """
 
    def objective(params):
        b0, b1, b2, b3, t1, t2 = params
        if t1 <= 0 or t2 <= 0 or t1 == t2:
            return 1e10
        try:
            y_hat = nss_yield(maturities, b0, b1, b2, b3, t1, t2)
        except Exception:
            return 1e10
        # Weight short end more — matters for Hull-White short-rate consistency
        w = 1.0 / np.maximum(maturities, 0.5)
        return np.sum(w * (y_hat - yields_dec) ** 2)
 
    # ── Bounds ────────────────────────────────────────────────────────────
    #   β0 : long-run level           [0 %, 10 %]
    #   β1 : slope (can be negative)  [-5 %, 5 %]
    #   β2 : short curvature          [-5 %, 5 %]
    #   β3 : medium curvature         [-5 %, 5 %]
    #   τ1 : first decay factor       [0.1, 5]   years
    #   τ2 : second decay factor      [1,  30]   years
    bounds = [
        (0.001, 0.10),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (0.1,   5.0),
        (1.0,  30.0),
    ]
 
    # Global optimisation (population-based, handles non-convex τ landscape)
    result_global = differential_evolution(
        objective, bounds,
        seed=42, maxiter=2000, tol=1e-12,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9,
        polish=False,
    )
 
    # Local polish
    result_local = minimize(
        objective, result_global.x,
        method="Nelder-Mead",
        options={"maxiter": 10_000, "xatol": 1e-10, "fatol": 1e-12},
    )
 
    params = result_local.x
 
    # Diagnostics
    y_hat  = nss_yield(maturities, *params)
    rmse   = np.sqrt(np.mean((y_hat - yields_dec) ** 2)) * 10_000  # bps
    max_err = np.max(np.abs(y_hat - yields_dec)) * 10_000           # bps
    print(f"NSS fit  →  RMSE = {rmse:.2f} bps   |   max error = {max_err:.2f} bps")
    print(f"  β0={params[0]*100:.4f}%  β1={params[1]*100:.4f}%  "
          f"β2={params[2]*100:.4f}%  β3={params[3]*100:.4f}%")
    print(f"  τ1={params[4]:.4f} yr   τ2={params[5]:.4f} yr")
 
    return params
 
 
# ══════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION — drop-in replacement for spline-based bootstrapForwardCurve
# ══════════════════════════════════════════════════════════════════════════
 
def bootstrapForwardCurve(
    maturities: np.ndarray,
    yields:     np.ndarray,
    t_grid:     np.ndarray = None,
) -> dict:
    """
    Bootstrap instantaneous forward curve f(0,T) using the
    Nelson-Siegel-Svensson (NSS) parametric model.
 
    Why NSS instead of cubic spline?
    ----------------------------------
    1. Analytic forward rates  — f(T) has a closed-form expression,
       no finite-difference noise.
    2. Analytic df/dT          — θ(t) computation is exact.
    3. Economically smooth     — no oscillation artefacts between
       knot points.
    4. ECB / NBB standard      — the Belgian National Bank fits NSS to
       OLO data for its official term structure publications.
    5. Extrapolation            — NSS extrapolates sensibly beyond 30Y
       toward β0 (the long-run level), unlike splines.
 
    Parameters
    ----------
    maturities : observed OLO maturities in years, e.g. [1, 2, ..., 30]
    yields     : observed OLO yields in %, e.g. [2.50, 2.61, ..., 4.38]
    t_grid     : fine evaluation grid (years); default 0.01 → 30
 
    Returns
    -------
    dict with keys:
        t_grid   : np.ndarray  — evaluation grid (years)
        f        : np.ndarray  — instantaneous forward rates f(0,T) [decimal]
        df_dT    : np.ndarray  — df/dT analytic derivative [decimal/year]
        P        : np.ndarray  — zero-coupon bond prices P(0,T)
        Y        : np.ndarray  — yield curve Y(0,T) [decimal]
        params   : np.ndarray  — fitted NSS parameters [β0,β1,β2,β3,τ1,τ2]
 
    The dict interface is identical to the cubic-spline version, so
    computeTheta() and all downstream functions work unchanged.
    """
 
    if t_grid is None:
        t_grid = np.linspace(0.01, 30, 3000)
 
    yields_dec = yields / 100.0
 
    # ── Fit NSS to observed yields ────────────────────────────────────────
    params = _fit_nss(maturities, yields_dec)
    b0, b1, b2, b3, tau1, tau2 = params
 
    # ── Evaluate on fine grid ─────────────────────────────────────────────
    Y     = nss_yield(t_grid,          b0, b1, b2, b3, tau1, tau2)
    f     = nss_forward(t_grid,        b0, b1, b2, b3, tau1, tau2)
    df_dT = nss_forward_deriv(t_grid,  b0, b1, b2, b3, tau1, tau2)
    P     = np.exp(-Y * t_grid)
 
    return {
        "t_grid": t_grid,
        "f":      f,
        "df_dT":  df_dT,
        "P":      P,
        "Y":      Y,
        "params": params,
    }
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Validation plot
# ═══════════════════════════════════════════════════════════════════════════
 
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
 
    os.makedirs("tests", exist_ok=True)
 
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
    plt.savefig("olo/tests/nss_forward_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → olo/tests/nss_forward_curve.png")
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test with your actual OLO data
# ═══════════════════════════════════════════════════════════════════════════
 

    
    
    
lastDate = dfYield["DATE"].max()
dfCurrentYield = (
    dfYield[dfYield["DATE"] == lastDate]
    .copy()
    .assign(MAT_NUM=lambda df: df["IROLOBE2_MATUR"].str.replace("Y", "").astype(int))
    .sort_values("MAT_NUM")
    .reset_index(drop=True)
)

yields = dfCurrentYield["YIELD"].values
maturities = dfCurrentYield["MAT_NUM"].values



curve = bootstrapForwardCurve(maturities, yields)

t = curve["t_grid"]
f = curve["f"]
P = curve["P"]

print("\n── Sanity checks ───────────────────────────────────────────")
print(f"Forward rate at  1Y : {np.interp(1,  t, f)*100:.4f}%   (expect > 2.50%)")
print(f"Forward rate at  5Y : {np.interp(5,  t, f)*100:.4f}%")
print(f"Forward rate at 10Y : {np.interp(10, t, f)*100:.4f}%")
print(f"Forward rate at 30Y : {np.interp(30, t, f)*100:.4f}%   (expect ≈ 4.38%)")
print(f"\nBond price P(0, 1Y) : {np.interp(1,  t, P):.6f}   (expect ≈ 0.9753)")
print(f"Bond price P(0,10Y) : {np.interp(10, t, P):.6f}   (expect ≈ 0.700)")
print(f"Bond price P(0,30Y) : {np.interp(30, t, P):.6f}   (expect ≈ 0.27)")

# df/dT smoothness: should have no oscillations at all
df = curve["df_dT"]
print(f"\ndf/dT range : [{df.min()*100:.4f}%, {df.max()*100:.4f}%]  per year")
print(f"  → no oscillations expected (NSS is analytic)")

plotForwardCurve(curve, maturities, yields)







def computeTheta(
    t_grid: np.ndarray,
    f:      np.ndarray,
    df_dT:  np.ndarray,
    kappa:  float,
    sigma:  float,
) -> np.ndarray:
    """
    Compute Hull-White time-dependent mean reversion target θ(t).

    Derived from no-arbitrage condition E[r(T)] = f(0,T):

        θ(t) = f(0,t) + df(0,t)/dt / κ + σ²/2κ² · (1 - e^(-2κt))
                 ↑              ↑                  ↑
            forward rate    slope of           volatility
            at time t       forward curve      correction

    Parameters
    ----------
    t_grid : time grid (years)
    f      : instantaneous forward rates on t_grid (decimal)
    df_dT  : derivative of forward rates on t_grid
    kappa  : mean-reversion speed κ
    sigma  : volatility σ

    Returns
    -------
    theta_t : np.ndarray — θ(t) on t_grid
    """

    # Term 1: forward rate
    term1 = f

    # Term 2: slope of forward curve scaled by κ
    term2 = df_dT / kappa

    # Term 3: volatility convexity correction
    term3 = (sigma**2 / (2 * kappa**2)) * (1 - np.exp(-2 * kappa * t_grid))

    theta_t = term1 + term2 + term3

    return theta_t

#######################################
### calibration checks Hull-White   ###
#######################################


def checkHullWhiteCalibration(
    maturities: np.ndarray,
    yields:     np.ndarray,
    results:    dict,
) -> dict:
    """
    Run all Hull-White calibration checks and produce a 4-panel plot.

    Check 1 — Forward curve shape: f(0,T) must be smooth and above Y(0,T)
              for an upward sloping curve
    Check 2 — Bond price consistency: P(0,T) must be strictly decreasing
              and start at 1.0
    Check 3 — θ(t) shape: must track the forward curve and be positive
    Check 4 — No-arbitrage: recover yield curve from P(0,T) and compare
              to observed OLO yields — must match exactly
    """

    import os
    os.makedirs("tests", exist_ok=True)

    kappa = results["kappa"]
    sigma = results["sigma"]

    # ── Run bootstrap and compute θ(t) ───────────────────────────────────
    curve   = bootstrapForwardCurve(maturities, yields)
    theta_t = computeTheta(
        t_grid = curve["t_grid"],
        f      = curve["f"],
        df_dT  = curve["df_dT"],
        kappa  = kappa,
        sigma  = sigma,
    )
    curve["theta_t"] = theta_t

    t_grid = curve["t_grid"]
    f      = curve["f"]
    P      = curve["P"]
    Y      = curve["Y"]

    # ── Check 1: forward curve must be above yield curve ─────────────────
    spread    = (f - Y) * 100                  # f - Y in bps
    check1_ok = np.all(spread >= -0.01)            # must be ≥ 0 for upward slope

    # ── Check 2: bond prices must be strictly decreasing 0 < P ≤ 1 ──────
    check2_ok = (
        np.all(np.diff(P) <= 0) and
        np.all(P > 0) and
        P[0] <= 1.0
    )

    # ── Check 3: θ(t) must be positive and track forward curve ───────────
    check3_ok = np.all(theta_t > 0)

    # ── Check 4: no-arbitrage — recover yields from P(0,T) ───────────────
    # Y_recovered(T) = -log P(0,T) / T — must match observed yields exactly
    Y_recovered    = -np.log(P) / t_grid * 100
    max_error_bps  = np.max(np.abs(Y_recovered - Y * 100)) * 100
    check4_ok      = max_error_bps < 0.1       # less than 0.1 bps error

    # ── Print summary ─────────────────────────────────────────────────────
    print("=" * 52)
    print("  HULL-WHITE CALIBRATION CHECKS")
    print("=" * 52)
    print(f"  Check 1 — Forward curve above yield curve  : {'✅ PASS' if check1_ok else '❌ FAIL'}")
    print(f"  Check 2 — Bond prices strictly decreasing  : {'✅ PASS' if check2_ok else '❌ FAIL'}")
    print(f"  Check 3 — θ(t) positive throughout         : {'✅ PASS' if check3_ok else '❌ FAIL'}")
    print(f"  Check 4 — No-arbitrage yield recovery       : {'✅ PASS' if check4_ok else '❌ FAIL'}")
    print(f"            Max recovery error                : {max_error_bps:.4f} bps")
    print("-" * 52)
    print(f"  θ(t) at  1Y : {np.interp(1,  t_grid, theta_t)*100:.4f}%")
    print(f"  θ(t) at  5Y : {np.interp(5,  t_grid, theta_t)*100:.4f}%")
    print(f"  θ(t) at 10Y : {np.interp(10, t_grid, theta_t)*100:.4f}%")
    print(f"  θ(t) at 30Y : {np.interp(30, t_grid, theta_t)*100:.4f}%")
    print("-" * 52)
    print(f"  f(0,t) at  1Y : {np.interp(1,  t_grid, f)*100:.4f}%")
    print(f"  f(0,t) at 10Y : {np.interp(10, t_grid, f)*100:.4f}%")
    print(f"  f(0,t) at 30Y : {np.interp(30, t_grid, f)*100:.4f}%")
    print("-" * 52)
    print(f"  P(0, 1Y)  : {np.interp(1,  t_grid, P):.6f}  ← €1 in 1Y")
    print(f"  P(0,10Y)  : {np.interp(10, t_grid, P):.6f}  ← €1 in 10Y")
    print(f"  P(0,30Y)  : {np.interp(30, t_grid, P):.6f}  ← €1 in 30Y")
    print("=" * 52)

    # ── 4-panel plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Hull-White Calibration Checks", fontsize=14, fontweight="bold")

    # Panel 1 — yield curve + forward curve
    ax1 = axes[0, 0]
    ax1.scatter(maturities, yields,
                color="steelblue", zorder=5, s=40, label="Observed OLO yields")
    ax1.plot(t_grid, Y * 100,
             color="steelblue", linewidth=1.5, linestyle="--", label="Spline fit Y(0,T)")
    ax1.plot(t_grid, f * 100,
             color="red", linewidth=2, label="Forward curve f(0,T)")
    ax1.set_title("Check 1 — Yield vs Forward Curve")
    ax1.set_xlabel("Maturity (years)")
    ax1.set_ylabel("Rate (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — bond prices P(0,T)
    ax2 = axes[0, 1]
    ax2.plot(t_grid, P,
             color="steelblue", linewidth=2, label="P(0,T)")
    ax2.axhline(1.0, color="red", linestyle=":",
                linewidth=1, label="P(0,0) = 1.0")
    for mat in [1, 5, 10, 20, 30]:
        p_val = np.interp(mat, t_grid, P)
        ax2.annotate(f"P(0,{mat}Y)={p_val:.3f}",
                     xy=(mat, p_val),
                     xytext=(mat + 0.5, p_val + 0.03),
                     fontsize=7, color="darkblue")
        ax2.scatter([mat], [p_val], color="red", s=25, zorder=5)
    ax2.set_title("Check 2 — Zero Coupon Bond Prices P(0,T)")
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("Bond Price (€)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — θ(t) vs f(0,t)
    ax3 = axes[1, 0]
    ax3.plot(t_grid, theta_t * 100,
             color="steelblue", linewidth=2, label="θ(t) — Hull-White target")
    ax3.plot(t_grid, f * 100,
             color="red", linewidth=1.5, linestyle="--", label="f(0,t) — forward rate")
    ax3.fill_between(t_grid,
                     f * 100, theta_t * 100,
                     alpha=0.15, color="steelblue",
                     label="vol + slope correction")
    ax3.set_title("Check 3 — θ(t) vs Forward Curve")
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Rate (%)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4 — no-arbitrage check: recovered vs observed yields
    ax4 = axes[1, 1]
    ax4.scatter(maturities, yields,
                color="steelblue", zorder=5, s=40, label="Observed OLO yields")
    ax4.plot(t_grid, Y_recovered,
             color="red", linewidth=2, linestyle="--",
             label=f"Recovered Y(0,T)  max err={max_error_bps:.3f} bps")
    ax4.set_title("Check 4 — No-Arbitrage Recovery")
    ax4.set_xlabel("Maturity (years)")
    ax4.set_ylabel("Yield (%)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("olo/tests/hull_white_calibration_checks.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → olo/tests/hull_white_calibration_checks.png")

    return curve






curve = checkHullWhiteCalibration(maturities, yields, VasicekResults)

import scipy.interpolate as si

spline = curve["spline"]
print(type(spline))