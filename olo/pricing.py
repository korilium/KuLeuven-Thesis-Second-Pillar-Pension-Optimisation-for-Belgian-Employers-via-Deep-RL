import os
import numpy as np
import matplotlib.pyplot as plt
from olo.calibration import nss_yield, nss_forward    # already in your module


# ══════════════════════════════════════════════════════════════════════════
# Vasicek ZCB price  (unchanged — constant-θ closed form)
# ══════════════════════════════════════════════════════════════════════════

def vasicekBondPrice(r, kappa, theta, sigma, tau):
    """
    Vasicek zero-coupon bond price P(t, t+τ) = A(τ)·exp(-B(τ)·r(t)).
    Constant long-run mean θ → a single model-implied curve (does NOT fit market).
    """
    B = (1 - np.exp(-kappa * tau)) / kappa
    log_A = (theta - sigma**2 / (2 * kappa**2)) * (B - tau) \
            - (sigma**2 * B**2) / (4 * kappa)
    return np.exp(log_A - B * r)


# ══════════════════════════════════════════════════════════════════════════
# Hull-White ZCB price  (analytic NSS — valid at ANY t, T, including T > 30Y)
# ══════════════════════════════════════════════════════════════════════════

def _Pmarket(T, params):
    """Analytic market discount factor P^M(0,T) = exp(-Y_NSS(0,T)·T), any T ≥ 0."""
    T = np.asarray(T, dtype=float)
    safe = np.maximum(T, 1e-12)
    return np.where(T < 1e-12, 1.0, np.exp(-nss_yield(safe, *params) * safe))


def hullWhiteBondPrice(r, t, kappa, sigma, tau, curve):
    """
    Hull-White zero-coupon bond price P(t, t+τ) given the short rate r(t).

        P(t,T) = [P^M(0,T)/P^M(0,t)]
                 · exp( B·f(0,t) − σ²/(4κ)·B²·(1−e^{−2κt}) − B·r(t) )
        B = (1 − e^{−κτ}) / κ,   τ = T − t

    The B·r term is IDENTICAL to Vasicek; only log A differs, anchoring the
    price to today's NSS curve so the model is arbitrage-free.

    IMPORTANT — this version evaluates the NSS curve ANALYTICALLY from
    curve["params"], not by interpolating curve["t_grid"]. That matters:

      • Exact at t = 0 (reproduces the market curve to machine precision when
        r = f(0,0); the grid-interp version was ~6 bps off).
      • Valid for T > 30Y. The WAP rate needs the 10Y OLO at future dates out
        to a 40Y horizon (so T up to 50Y); interpolating a 30Y grid clamps
        P beyond 30Y and produces 180–360 bps errors / negative yields.

    Parameters
    ----------
    r     : short rate(s) r(t) — scalar or array over paths (decimal)
    t     : current time (years since t=0)
    kappa : mean-reversion speed κ (from Vasicek calibration)
    sigma : volatility σ           (from Vasicek calibration)
    tau   : time to maturity τ = T − t (years)
    curve : NSS curve dict; must contain "params" = [b0,b1,b2,b3,τ1,τ2]

    Returns
    -------
    P : bond price, same shape as r — value at t of €1 paid at t+τ.
        At t = 0 with r = f(0,0) = β0+β1, returns P^M(0,τ) exactly.
    """
    params = curve["params"]
    T = t + tau
    P_t = _Pmarket(t, params)
    P_T = _Pmarket(T, params)
    f_t = nss_forward(np.maximum(t, 0.0), *params)            # exact f(0,t), incl. t=0
    B = (1 - np.exp(-kappa * tau)) / kappa
    log_A = (
        np.log(P_T / P_t)
        + B * f_t
        - (sigma**2 / (4 * kappa)) * B**2 * (1 - np.exp(-2 * kappa * t))
    )
    return np.exp(log_A - B * r)


# ══════════════════════════════════════════════════════════════════════════
# Future yield reconstruction  →  this is the bridge to the WAP rate
# ══════════════════════════════════════════════════════════════════════════

def reconstructFutureYield(
    paths: np.ndarray,
    kappa: float,
    sigma: float,
    curve: dict,
    tau:   float = 10.0,
    dt:    float = 1/12,
) -> np.ndarray:
    """
    Reconstruct the model-implied τ-year yield Y(t, t+τ) at every (time, path),
    from the simulated short rate via the Hull-White bond price:

        Y(t, t+τ) = − ln P(t, t+τ; r(t)) / τ

    With tau=10 this is the future 10Y OLO that drives the WAP fixing — the
    same quantity, per path, that you then 24-month-average and run through
    computeWAPRate. This is what makes the guarantee and the reserve correlated
    through the common OLO driver.

    Returns
    -------
    Y : np.ndarray, same shape as paths — the τ-year yield (decimal) on each path.
    """
    n_steps, n_paths = paths.shape
    Y = np.empty_like(paths)
    for i in range(n_steps):
        P = hullWhiteBondPrice(paths[i, :], t=i * dt, kappa=kappa,
                               sigma=sigma, tau=tau, curve=curve)
        Y[i, :] = -np.log(P) / tau
    return Y


# ══════════════════════════════════════════════════════════════════════════
# Cumulative discount factors  (for path-wise APV discounting)
# ══════════════════════════════════════════════════════════════════════════

def computeCumulativeDiscountFactors(paths, dt=1/12):
    """
    D(0,t) = exp(-∫₀ᵗ r ds) ≈ exp(-Σ r·dt)  per path  → feeds APV = Σ CF·D.

    Note: the Riemann sum carries a small discretisation bias (E[D] sits a few
    bps above the analytic P^M(0,t) at monthly dt). Fine for APV; if you ever
    need bias-free discounting use the analytic P^M(0,t) instead.
    """
    D = np.ones_like(paths)
    D[1:, :] = np.exp(-np.cumsum(paths[:-1, :] * dt, axis=0))
    return D


# ── Runner notes (fixes for the script you had) ───────────────────────────
# Three bugs in the previous runner:
#   1. `curve` was never built — call bootstrapForwardCurve first.
#   2. `actual_maturities` / `actual_yields` were undefined — use maturities/yields.
#   3. the t=0 Hull-White curve was drawn at the Vasicek r0 (the 10Y level); it
#      must be drawn at the instantaneous short rate f(0,0)=β0+β1, at which point
#      it overlays the OLO market curve EXACTLY (that's the no-arbitrage property).
#
# from olo.calibration import bootstrapForwardCurve, calibrateVasicek
# results = calibrateVasicek(df10Y)
# curve   = bootstrapForwardCurve(maturities, yields)     # provides "params"
# k, s    = results["kappa"], results["sigma"]
# f00     = curve["params"][0] + curve["params"][1]       # f(0,0) = β0+β1
#
# yields_hw = [-np.log(hullWhiteBondPrice(f00, t=0, kappa=k, sigma=s,
#                                         tau=tau, curve=curve)) / tau * 100
#              for tau in maturities]          # ← overlays the OLO market line