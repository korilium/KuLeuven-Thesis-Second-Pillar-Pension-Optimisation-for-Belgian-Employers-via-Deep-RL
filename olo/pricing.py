

import numpy as np
import matplotlib.pyplot as plt
import os


def vasicekBondPrice(
    r:     np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    tau:   float,
) -> np.ndarray:
    
    """
    Vasicek zero-coupon bond price P(t, t+τ).
    Derived from:
        P(t,T) = E[exp(-∫ₜᵀ r(s)ds)]     definition
               = exp(-M + V²/2)            moment generating function
               = A(τ) · exp(-B(τ) · r(t)) closed form
    Parameters
    ----------
    r     : current short rate(s) — scalar or np.ndarray (decimal)
    kappa : mean-reversion speed  κ
    theta : long-run mean         θ (decimal)
    sigma : volatility            σ
    tau   : time to maturity in years (τ = T - t)
    Returns
    -------
    P : bond price — same shape as r
        Interpretation: value today of €1 received in τ years
    """
    # ── B(τ) = (1 - e^(-κτ)) / κ ─────────────────────────────────────────
    # Comes from integrating E[r(s)] = θ + (r(t)-θ)e^(-κ(s-t)) over [t,T]
    B = (1 - np.exp(-kappa * tau)) / kappa
    # ── log A(τ) = (θ - σ²/2κ²)(B(τ) - τ) - σ²B(τ)²/4κ ─────────────────
    # Comes from V²/2 - θτ + θB(τ) after substituting M and V²
    log_A = (theta - sigma**2 / (2 * kappa**2)) * (B - tau) \
            - (sigma**2 * B**2) / (4 * kappa)
    # ── P(t,T) = A(τ) · exp(-B(τ) · r(t)) ───────────────────────────────
    return np.exp(log_A - B * r)

def hullWhiteBondPrice(
    r:     np.ndarray,
    t:     float,          # ← NEW: current time (years since t=0)
    kappa: float,
                           # ← REMOVED: theta (no longer needed)
    sigma: float,
    tau:   float,
    curve: dict,           # ← NEW: NSS curve dict from bootstrapForwardCurve()
) -> np.ndarray:
    T = t + tau
    # ── Interpolate market values from NSS curve ──────────────────────────
    P_market_t = 1.0 if t < 1e-10 else np.interp(t, curve["t_grid"], curve["P"])
    P_market_T = np.interp(T, curve["t_grid"], curve["P"])
    f_t        = np.interp(max(t, curve["t_grid"][0]), curve["t_grid"], curve["f"])
    # ── B(τ) = (1 - e^(-κτ)) / κ  ← IDENTICAL to Vasicek ────────────────
    B = (1 - np.exp(-kappa * tau)) / kappa
    # ── log A(t,T) — this is the only part that changes ───────────────────
    # Vasicek:    log A = (θ - σ²/2κ²)(B - τ) - σ²B²/4κ
    # Hull-White: log A = log[P^M(0,T)/P^M(0,t)] + B·f(0,t) - σ²/(4κ)·B²·(1-e^{-2κt})
    log_A = (
        np.log(P_market_T / P_market_t)                          # no-arb anchor
        + B * f_t                                                 # forward slope correction
        - (sigma**2 / (4 * kappa)) * B**2 * (1 - np.exp(-2 * kappa * t))  # variance adj.
    )
    # ── P(t,T) = exp(log A - B·r(t)) ← IDENTICAL to Vasicek ─────────────
    return np.exp(log_A - B * r)

def computeCumulativeDiscountFactors(
    paths: np.ndarray,
    dt:    float = 1/12,
) -> np.ndarray:
    """
    Cumulative discount factor D(0,t) for each path.
    Definition:
        D(0,t) = exp(-∫₀ᵗ r(s) ds)
               ≈ exp(-Σₛ r(s)·dt)     Riemann sum approximation
    This is what goes directly into the APV:
        APV = Σₜ cash_flow(t) · D(0,t)
    Parameters
    ----------
    paths : simulated rate paths shape (n_steps+1, n_paths)
    dt    : timestep (1/12 for monthly)
    Returns
    -------
    D : shape (n_steps+1, n_paths)
        D[0, :] = 1.0    no discounting at t=0
        D[t, :] = present value of €1 at time t
    """
    D        = np.ones_like(paths)
    D[1:, :] = np.exp(-np.cumsum(paths[:-1, :] * dt, axis=0))
    return D

def plotBondPricing(
    paths:   np.ndarray,
    D:       np.ndarray,
    results: dict,
    curve:   dict,          # ← ADD this parameter
    dt:      float = 1/12,
):
    os.makedirs("tests", exist_ok=True)
    kappa = results["kappa"]
    theta = results["theta"]
    sigma = results["sigma"]
    r0    = results["r0"]
    n_steps = paths.shape[0]
    t_axis  = np.arange(n_steps) * dt
    maturities = np.linspace(0.1, 30, 200)
    # ── Panel 1: BOTH yield curves ────────────────────────────────────────
    yield_curve_vasicek = np.array([
        -np.log(vasicekBondPrice(theta, kappa, theta, sigma, tau)) / tau
        for tau in maturities
    ]) * 100
    yield_curve_hw = np.array([                                  # ← ADD
        -np.log(hullWhiteBondPrice(r0, t=0, kappa=kappa,
                                   sigma=sigma, tau=tau,
                                   curve=curve)) / tau
        for tau in maturities
    ]) * 100
    # ── Panel 2: BOTH 1Y bond prices ─────────────────────────────────────
    P_1y_vasicek = vasicekBondPrice(paths, kappa, theta, sigma, tau=1.0)
    P_1y_hw = np.array([                                         # ← ADD
        hullWhiteBondPrice(paths[i, :], t=i*dt, kappa=kappa,
                           sigma=sigma, tau=1.0, curve=curve)
        for i in range(n_steps)
    ])
    # percentiles for both
    p05_V, p50_V, p95_V = [np.percentile(P_1y_vasicek, q, axis=1) for q in [5,50,95]]
    p05_H, p50_H, p95_H = [np.percentile(P_1y_hw,      q, axis=1) for q in [5,50,95]]
    # ── Panel 3: discount factor (unchanged) ──────────────────────────────
    p05_D, p50_D, p95_D = [np.percentile(D, q, axis=1) for q in [5,50,95]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Vasicek vs Hull-White — Bond Pricing & Discount Factors",
                 fontsize=13, fontweight="bold")
    # Panel 1 — yield curves
    ax1 = axes[0]
    ax1.plot(curve["t_grid"], curve["Y"] * 100,                  # ← ADD market
             color="steelblue", linewidth=2.5, label="OLO market")
    ax1.plot(maturities, yield_curve_vasicek,
             color="red", linewidth=1.8, linestyle="--",
             label=f"Vasicek  θ={theta*100:.2f}%")
    ax1.plot(maturities, yield_curve_hw,                         # ← ADD HW
             color="darkorange", linewidth=1.8, linestyle=":",
             label="Hull-White (r₀, t=0)")
    ax1.set_title("Yield Curve: Market vs Vasicek vs Hull-White")
    ax1.set_xlabel("Maturity τ (years)")
    ax1.set_ylabel("Yield (%)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    # Panel 2 — 1Y bond price
    ax2 = axes[1]
    ax2.fill_between(t_axis, p05_V, p95_V, alpha=0.15, color="red")
    ax2.plot(t_axis, p50_V, color="red", linewidth=1.8,
             linestyle="--", label="Vasicek median")
    ax2.fill_between(t_axis, p05_H, p95_H, alpha=0.15, color="darkorange")  # ← ADD
    ax2.plot(t_axis, p50_H, color="darkorange", linewidth=1.8,              # ← ADD
             linestyle=":", label="Hull-White median")
    ax2.set_title("1Y Bond Price P(r(t), 1Y) over Time")
    ax2.set_xlabel("Years")
    ax2.set_ylabel("Bond Price (€)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    # Panel 3 — discount factors (unchanged)
    ax3 = axes[2]
    ax3.fill_between(t_axis, p05_D, p95_D, alpha=0.2, color="steelblue", label="5–95th pct")
    ax3.plot(t_axis, p50_D, color="steelblue", linewidth=2, label="Median D(0,t)")
    ax3.set_title("Cumulative Discount Factor D(0,t)")
    ax3.set_xlabel("Years")
    ax3.set_ylabel("D(0,t)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("tests/vasicek_vs_hw_bond_pricing.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    
# ── Standalone yield curve comparison — add HW line ───────────────────────

def vasicekYield(r, kappa, theta, sigma, tau):
    B     = (1 - np.exp(-kappa * tau)) / kappa
    log_A = (theta - sigma**2 / (2*kappa**2)) * (B - tau) \
            - (sigma**2 * B**2) / (4 * kappa)
    P     = np.exp(log_A - B * r)
    return -np.log(P) / tau * 100



from olo.calibration import calibrateVasicek
from olo.data.extract import extractDataYieldNBB

##############
#read in data#
##############

dfYield = extractDataYieldNBB(startPeriod="2000-01")



#############
# dataManip #
#############

df10Y = dfYield[dfYield["IROLOBE2_MATUR"] == "10Y"].copy().reset_index(drop=True)


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

#calibrate 

results = calibrateVasicek(df10Y)


kappa = results["kappa"]
theta = results["theta"]
sigma = results["sigma"]
r0    = results["r0"]

yields_at_r0    = [vasicekYield(r0,    kappa, theta, sigma, tau) for tau in maturities]
yields_at_theta = [vasicekYield(theta, kappa, theta, sigma, tau) for tau in maturities]
yields_hw = [                                                    # ← ADD
    -np.log(hullWhiteBondPrice(r0, t=0, kappa=kappa,
                               sigma=sigma, tau=tau,
                               curve=curve)) / tau * 100
    for tau in maturities
    
]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(actual_maturities, actual_yields,
        color="steelblue", linewidth=2.5, marker="o", markersize=4,
        label="Actual Belgian OLO (Apr 24 2026)")
ax.plot(maturities, yields_at_r0,
        color="red", linewidth=2, linestyle="--",
        label=f"Vasicek at r₀ = {r0*100:.2f}%")
ax.plot(maturities, yields_at_theta,
        color="darkred", linewidth=1.5, linestyle=":",
        label=f"Vasicek at θ = {theta*100:.2f}%")
ax.plot(maturities, yields_hw,                                   # ← ADD
        color="darkorange", linewidth=2, linestyle="-.",
        label="Hull-White at r₀ (t=0)")
ax.set_title("Belgian OLO Yield Curve — Actual vs Vasicek vs Hull-White",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Yield (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tests/yield_curve_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
