import numpy as np
import matplotlib.pyplot as plt
from olo.calibration import calibrateVasicek, nss_forward
from olo.data.extract import extractDataYieldNBB
import os 


def simulateVasicek(
    kappa:   float,
    theta:   float,
    sigma:   float,
    r0:      float,
    T:       int   = 40,
    n_paths: int   = 1000,
    dt:      float = 1/12,
    seed:    int   = 42,
) -> np.ndarray:
    """
    Simulate Vasicek short-rate paths using exact discretisation.

    Exact solution:
        r(t+dt) = r(t)·e^(-κdt) + θ(1 - e^(-κdt)) + σ·√((1-e^(-2κdt))/2κ)·ε

    Parameters
    ----------
    kappa   : mean-reversion speed
    theta   : long-run mean (decimal)
    sigma   : volatility
    r0      : initial rate (decimal)
    T       : horizon in years
    n_paths : number of Monte Carlo paths
    dt      : timestep — must match calibration (1/12 for monthly)
    seed    : random seed for reproducibility

    Returns
    -------
    paths : np.ndarray shape (n_steps+1, n_paths)
            rows = timesteps, columns = simulation paths
    """

    np.random.seed(seed)
    n_steps = int(T / dt)

    # ── Exact discretisation coefficients (computed once) ─────────────────
    e_kdt  = np.exp(-kappa * dt)
    drift  = theta * (1 - e_kdt)
    diff   = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

    # ── Initialise path matrix ────────────────────────────────────────────
    paths       = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = r0

    # ── Simulate (vectorised across all paths) ────────────────────────────
    eps = np.random.normal(0, 1, size=(n_steps, n_paths))
    for t in range(n_steps):
        paths[t+1, :] = paths[t, :] * e_kdt + drift + diff * eps[t, :]

    return paths


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
    plt.savefig("olo/tests/vasicek_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → tests/vasicek_simulation.png")



########################
### hull-white model ###
########################


def simulateHullWhite(
    curve:   dict,
    kappa:   float,
    sigma:   float,
    T:       int   = 40,
    n_paths: int   = 1000,
    dt:      float = 1/12,
    seed:    int   = 42,
) -> np.ndarray:
    """
    Simulate Hull-White short-rate paths using EXACT step-wise discretisation
    via the shifted decomposition r(t) = x(t) + alpha(t).

    Hull-White is Vasicek with a time-varying target:
        dr = kappa(theta(t) - r)dt + sigma dW

    Writing r(t) = x(t) + alpha(t) with x a zero-mean OU process
    (dx = -kappa x dt + sigma dW, x(0)=0), the simulation is exact:

        r(t+dt) = alpha(t+dt) + (r(t) - alpha(t)) e^{-kappa dt}
                  + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) * eps

        alpha(t) = f(0,t) + sigma^2/(2 kappa^2) (1 - e^{-kappa t})^2

    Only the drift differs from Vasicek; the diffusion term is identical.
    The deterministic shift alpha(t) uses f(0,t) directly from the NSS fit
    (analytic, so it extrapolates sensibly past 30Y toward beta0). This is the
    SAME no-arbitrage fit as computeTheta: theta(t) = alpha(t) + alpha'(t)/kappa.

    Parameters
    ----------
    curve   : dict from bootstrapForwardCurve / checkHullWhiteCalibration;
              must contain "params" = [b0,b1,b2,b3,tau1,tau2]
    kappa   : mean-reversion speed (from Vasicek calibration)
    sigma   : volatility            (from Vasicek calibration)
    T       : horizon in years (40 for the pension horizon)
    n_paths : number of Monte Carlo paths
    dt      : timestep — must match calibration (1/12 for monthly)
    seed    : random seed for reproducibility

    Returns
    -------
    paths : np.ndarray shape (n_steps+1, n_paths)
            rows = timesteps, columns = simulation paths.
            r(0) = alpha(0) = f(0,0) = beta0 + beta1 by construction.

    Note
    ----
    The 30Y-40Y segment of f(0,t) is an NSS *extrapolation* toward beta0, not
    data — the OLO curve only supplies maturities out to 30Y. Being Gaussian,
    the model admits negative rates (realistic for Belgium 2015-2022); if a hard
    floor is ever required downstream, the shifted-Hull-White variant is the
    drop-in mitigation.
    """

    np.random.seed(seed)
    n_steps = int(round(T / dt))
    t_axis  = np.arange(n_steps + 1) * dt

    # ── Deterministic shift alpha(t) on the simulation grid ───────────────
    params = curve["params"]
    f0t    = nss_forward(t_axis, *params)                       # f(0,t), analytic
    alpha  = f0t + (sigma**2 / (2 * kappa**2)) * (1 - np.exp(-kappa * t_axis))**2

    # ── Exact OU coefficients (computed once — identical to Vasicek) ──────
    e_kdt = np.exp(-kappa * dt)
    diff  = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

    # ── Initialise: r(0) = alpha(0) = f(0,0) ──────────────────────────────
    paths       = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = alpha[0]

    # ── Simulate (vectorised across paths; only drift differs from Vasicek)
    eps = np.random.normal(0, 1, size=(n_steps, n_paths))
    for t in range(n_steps):
        paths[t+1, :] = alpha[t+1] + (paths[t, :] - alpha[t]) * e_kdt + diff * eps[t, :]

    return paths


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

    os.makedirs("olo/tests", exist_ok=True)

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
    plt.savefig("olo/tests/hull_white_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → olo/tests/hull_white_simulation.png")


# ── Run (real data) ───────────────────────────────────────────────────────


# ── Run ───────────────────────────────────────────────────────────────────

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

#######################
# vasicek simulations #
#######################

results = calibrateVasicek(df10Y)

paths = simulateVasicek(
    kappa   = results["kappa"],
    theta   = results["theta"],
    sigma   = results["sigma"],
    r0      = results["r0"],
    T       = 40,        # 40-year pension horizon
    n_paths = 5000,
    dt      = 1/12,      # monthly — matches calibration
)

print(f"Paths shape        : {paths.shape}")
print(f"Mean terminal rate : {paths[-1,:].mean()*100:.4f} %")
print(f"Std  terminal rate : {paths[-1,:].std()*100:.4f} %")
print(f"Min  terminal rate : {paths[-1,:].min()*100:.4f} %")
print(f"Max  terminal rate : {paths[-1,:].max()*100:.4f} %")

plotSimulation(paths, results)


os.getcwd()


from olo.calibration import calibrateVasicek, bootstrapForwardCurve, computeTheta
VasicekResults = calibrateVasicek(df10Y)
curve = bootstrapForwardCurve(maturities, yields)         # provides "params"
#
paths = simulateHullWhite(
    curve   = curve,
    kappa   = VasicekResults["kappa"],
    sigma   = VasicekResults["sigma"],
   T       = 40,
    n_paths = 5000,
    dt      = 1/12,
)
print(f"Paths shape        : {paths.shape}")
print(f"r(0)               : {paths[0,0]*100:.4f} %  (= f(0,0) = β0+β1)")
print(f"Mean terminal rate : {paths[-1,:].mean()*100:.4f} %")
print(f"Std  terminal rate : {paths[-1,:].std()*100:.4f} %")
plotSimulationHW(paths, curve, VasicekResults["kappa"], VasicekResults["sigma"])