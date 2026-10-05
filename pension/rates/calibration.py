
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.optimize import differential_evolution, minimize




def calibrateVasicek(df10Y: pd.DataFrame, verbose: bool = True):

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
    r0 = r[-1] # initial rate: the most recent observed yield

    #diagnostics 
    n = len(r_t)
    var_b = sigma_eps**2 * np.linalg.inv(X.T @ X)[1, 1]
    t_b   = b / np.sqrt(var_b)
    p_b   = 2 * t_dist.sf(np.abs(t_b), df=n - 2)
    r2    = 1 - np.var(eps) / np.var(dr)


    #print summary of calibration results
    if verbose:
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


def calibrateVasicekShort(df_short: pd.DataFrame, df10Y: pd.DataFrame, tau: float = 10.0):
    """Real-world (P) Vasicek short rate for RATE_MODEL="vasicek_short".

    kappa, theta_P, sigma: calibrateVasicek on the monthly short-rate proxy.
    theta_Q: the risk-neutral long-run mean used ONLY to map r to the tau-year
    yield with the affine Vasicek price (pricing.vasicekBondPrice). It is set so
    that the model's mean (tau-year - short) spread over the history equals the
    observed one; the implied market price of risk is
        phi = (theta_P - theta_Q) * kappa / sigma   (constant).
    Because only the mean spread is matched, the in-sample bias of the
    reconstructed yield is zero by construction; the RMSE is the informative fit.

    The t0 offset. The model 10Y at the observed r0 misses the last observed 10Y
    by e_t0 (the last in-sample residual e_t = y10_obs - y10_model(r_obs)). The
    scenarios add a DETERMINISTIC decaying offset e_t0 * phi_e^m (m months ahead),
    so they start on the observed 10Y and converge to the mean-spread model.
    phi_e is the AR(1) coefficient of e_t (OLS, no intercept: e has mean zero by
    construction). It is used if its decay is better identified than the short
    rate's own mean reversion -- t-stat of (1 - phi_e) above |t-stat of b| --
    else the decay falls back to e^{-kappa dt}. Being deterministic, the offset
    simulates no slope risk, so the dispersion of G_t and mu_t is probably
    understated.

    Returns the parameters, r0 (last observed short rate), a "fit" dict
    (in-sample RMSE / bias / correlation of the reconstructed 10Y from the
    observed short rate, and the jump at t0 between the model 10Y at r0 and the
    last observed 10Y) and an "offset" dict (e_t0, phi_e, its standard error,
    half-life, the monthly decay used and whether it is the kappa fallback)."""
    from pension.rates.pricing import vasicekBondPrice
    assert np.array_equal(df_short["DATE"].values, df10Y["DATE"].values), \
        "short and 10Y series must share their dates"
    vas = calibrateVasicek(df_short, verbose=False)
    kappa, theta_P, sigma = vas["kappa"], vas["theta"], vas["sigma"]
    r = df_short["YIELD"].values / 100.0
    y10 = df10Y["YIELD"].values / 100.0

    # model yield y(r) = (B r - log A) / tau, log A linear in theta_Q:
    # mean y(r) = mean y10  <=>  log A = B mean(r) - tau mean(y10)
    B = (1 - np.exp(-kappa * tau)) / kappa
    logA = B * r.mean() - tau * y10.mean()
    theta_Q = (logA + sigma**2 * B**2 / (4 * kappa)) / (B - tau) + sigma**2 / (2 * kappa**2)
    phi = (theta_P - theta_Q) * kappa / sigma

    fit = -np.log(vasicekBondPrice(r, kappa, theta_Q, sigma, tau)) / tau
    err = fit - y10

    e = y10 - fit                                        # residual, mean 0 by construction
    e0, e1 = e[1:], e[:-1]
    phi_e = float((e0 @ e1) / (e1 @ e1))
    s2 = float(((e0 - phi_e * e1) ** 2).sum() / (len(e0) - 1))
    se = float(np.sqrt(s2 / (e1 @ e1)))
    t_decay = (1 - phi_e) / se
    use_ar1 = 0.0 < phi_e < 1.0 and t_decay > abs(vas["diagnostics"]["t_stat_b"])
    decay = phi_e if use_ar1 else float(np.exp(-kappa / 12))      # monthly steps
    offset = {"e_t0": float(e[-1]), "phi_e": phi_e, "se": se, "t_decay": float(t_decay),
              "half_life_years": float(np.log(0.5) / np.log(decay) / 12),
              "monthly_decay": decay, "fallback_kappa": not use_ar1}
    return {
        "kappa": kappa, "theta_P": theta_P, "theta_Q": theta_Q, "sigma": sigma,
        "phi": phi, "r0": r[-1], "tau": tau, "diagnostics": vas["diagnostics"],
        "fit": {"rmse": float(np.sqrt(np.mean(err**2))), "bias": float(err.mean()),
                "corr": float(np.corrcoef(fit, y10)[0, 1]),
                "rmse_last24": float(np.sqrt(np.mean(err[-24:]**2))),
                "model_10y_t0": float(fit[-1]), "observed_10y_t0": float(y10[-1]),
                "jump_t0": float(fit[-1] - y10[-1]),
                "spread_obs": float((y10 - r).mean())},
        "offset": offset,
        "arrays": {"r": r, "y10": y10, "y10_fit": fit},
    }


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
 
    Required by computeTheta (κ(θ(t) − r) drift convention):
        θ(t) = f(t) + (df/dT)/κ + σ²/(2κ²) · (1 − e^{−2κt})
 
    df/dT = e^{−T/τ1} · [−β1/τ1 + β2 · (1/τ1 − T/τ1²)]
          + e^{−T/τ2} · [β3 · (1/τ2 − T/τ2²)]
    """
    e1 = np.exp(-T / tau1)
    e2 = np.exp(-T / tau2)
 
    term1 = e1 * (-beta1 / tau1 + beta2 * (1 / tau1 - T / tau1**2))
    term2 = e2 * (beta3 * (1 / tau2 - T / tau2**2))
 
    return term1 + term2
 
 
# ── Fit NSS parameters to observed yields ────────────────────────────────
 
def _fit_nss(maturities: np.ndarray, yields_dec: np.ndarray, verbose: bool = True) -> np.ndarray:
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
    if verbose:
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
    verbose:    bool = True,
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
    params = _fit_nss(maturities, yields_dec, verbose)
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










 
