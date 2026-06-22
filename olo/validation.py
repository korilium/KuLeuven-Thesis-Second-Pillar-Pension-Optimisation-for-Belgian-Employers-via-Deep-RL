from olo.data.extract import extractDataYieldNBB
from olo.calibration import bootstrapForwardCurve, plotForwardCurve, computeTheta, calibrateVasicek, checkHullWhiteNoArbitrage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

import os


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

VasicekResults = calibrateVasicek(df10Y)

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




def checkHullWhiteNoArbitrage(
    curve:   dict,
    results: dict,
    maturities=None,
    yields=None,
) -> dict:
    """
    Genuine no-arbitrage check for the calibrated Hull-White model.
 
    Unlike the yield-recovery check (which inverts P = exp(-Y*T) and is therefore
    true by construction), this prices zero-coupon bonds *through the Hull-White
    analytic formula* from the calibrated theta(t), kappa, sigma, and confirms they
    reproduce the NSS market curve. A near-zero residual is real evidence that
    computeTheta() inverts the term structure correctly.
 
    Affine ZCB price for dr = kappa(theta(t) - r)dt + sigma dW :
 
        P_HW(0,T) = exp( -M(T) + 1/2 V(T) )
 
        M(T) = r(0) B(0,T) + INT_0^T theta(u)(1 - e^{-kappa(T-u)}) du
             = r(0) B(0,T) + I1(T) - e^{-kappa T} I2(T)
        V(T) = (sigma^2/kappa^2) [ T - 2 B(0,T) + (1 - e^{-2 kappa T})/(2 kappa) ]
        B(0,T) = (1 - e^{-kappa T}) / kappa
        I1(T) = INT_0^T theta(u) du,   I2(T) = INT_0^T theta(u) e^{kappa u} du
 
    Critical detail
    ---------------
    r(0) is the INSTANTANEOUS short rate implied by today's curve, f(0,0) = beta0+beta1,
    NOT the historical 10Y proxy used to calibrate kappa/sigma. Feeding the 10Y in here
    breaks the fit by ~200 bps - which is exactly the kind of error this check exists
    to catch.
    """
    kappa = results["kappa"]
    sigma = results["sigma"]
 
    t        = curve["t_grid"]
    Y        = curve["Y"]
    P_market = curve["P"]                      # NSS bond prices exp(-Y*t)
    theta    = curve.get("theta_t")
    if theta is None:
        from olo.calibration import computeTheta
        theta = computeTheta(t, curve["f"], curve["df_dT"], kappa, sigma)
 
    # --- exact t=0 anchors from the fitted NSS params (no integration stub) ---
    b0, b1, b2, b3, tau1, tau2 = curve["params"]
    f0   = b0 + b1                             # f(0,0)
    fd0  = (b2 - b1) / tau1 + b3 / tau2        # f'(0,0)
    th0  = f0 + fd0 / kappa                    # theta(0)  (vol term vanishes at 0)
    r0   = f0                                  # short rate fed into the bond formula
 
    # --- cumulative integrals of theta, anchored at a true t=0 ---
    t_aug  = np.insert(t, 0, 0.0)
    th_aug = np.insert(theta, 0, th0)
    I1 = cumulative_trapezoid(th_aug,                      t_aug, initial=0.0)[1:]
    I2 = cumulative_trapezoid(th_aug * np.exp(kappa*t_aug), t_aug, initial=0.0)[1:]
 
    B   = (1 - np.exp(-kappa * t)) / kappa
    B2  = (1 - np.exp(-2 * kappa * t)) / (2 * kappa)
    M   = r0 * B + (I1 - np.exp(-kappa * t) * I2)
    V   = (sigma**2 / kappa**2) * (t - 2 * B + B2)
 
    P_hw = np.exp(-M + 0.5 * V)
 
    # --- compare on a yield basis (bps) ---
    Y_hw    = -np.log(P_hw) / t
    err_bps = (Y_hw - Y) * 1e4
    max_err = np.max(np.abs(err_bps))
    passed  = max_err < 1.0                    # < 1 bp = calibration inverts the curve
 
    print("=" * 56)
    print("  HULL-WHITE ANALYTIC NO-ARBITRAGE CHECK")
    print("=" * 56)
    print(f"  r(0) = f(0,0) instantaneous short rate : {r0*100:.4f}%")
    print(f"  Max |Y_HW - Y_NSS| across curve        : {max_err:.4f} bps")
    print(f"  Verdict (< 1 bp)                        : {'PASS' if passed else 'FAIL'}")
    print("-" * 56)
    for mat in [1, 5, 10, 30]:
        print(f"  T={mat:>2}Y   P_HW={np.interp(mat,t,P_hw):.6f}   "
              f"P_NSS={np.interp(mat,t,P_market):.6f}   "
              f"err={np.interp(mat,t,err_bps):+.4f} bps")
    print("=" * 56)
 
    # --- plot ---
    os.makedirs("olo/tests", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hull-White Analytic No-Arbitrage Check", fontsize=13, fontweight="bold")
 
    ax1 = axes[0]
    ax1.plot(t, P_market, color="steelblue", linewidth=3, alpha=0.5,
             label="P_NSS(0,T) — market")
    ax1.plot(t, P_hw, color="crimson", linewidth=1.4, linestyle="--",
             label="P_HW(0,T) — analytic HW")
    if maturities is not None and yields is not None:
        ax1.scatter(maturities, np.exp(-np.asarray(yields)/100*np.asarray(maturities)),
                    color="black", s=30, zorder=5, label="Observed nodes")
    ax1.set_title("Bond Prices: HW analytic vs NSS market")
    ax1.set_xlabel("Maturity T (years)"); ax1.set_ylabel("P(0,T)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
 
    ax2 = axes[1]
    ax2.plot(t, err_bps, color="darkorange", linewidth=1.8)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.fill_between(t, err_bps, 0, alpha=0.2, color="darkorange")
    ax2.set_title(f"Repricing error  (max = {max_err:.4f} bps)")
    ax2.set_xlabel("Maturity T (years)"); ax2.set_ylabel("Y_HW - Y_NSS (bps)")
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("olo/tests/hull_white_no_arbitrage.png", dpi=150, bbox_inches="tight")
    print("Saved -> olo/tests/hull_white_no_arbitrage.png")
 
    return {
        "parameters":  {"kappa": kappa, "sigma": sigma, "r0_inst": r0},
        "diagnostics": {"max_err_bps": max_err, "passed": passed,
                        "err_1Y":  float(np.interp(1, t, err_bps)),
                        "err_10Y": float(np.interp(10, t, err_bps)),
                        "err_30Y": float(np.interp(30, t, err_bps))},
        "arrays":      {"t_grid": t, "P_hw": P_hw, "P_market": P_market,
                        "err_bps": err_bps, "theta_t": theta},
    }


curve = checkHullWhiteCalibration(maturities, yields, VasicekResults)


noarb = checkHullWhiteNoArbitrage(curve, VasicekResults, maturities, yields)


plt.show()