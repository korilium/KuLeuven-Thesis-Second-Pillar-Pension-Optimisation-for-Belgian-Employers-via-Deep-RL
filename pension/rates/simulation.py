import numpy as np
from pension.rates.calibration import nss_forward


def simulateVasicek(
    kappa:   float,
    theta:   float,
    sigma:   float,
    r0:      float,
    T:       int   = 40,
    n_paths: int   = 1000,
    dt:      float = 1/12,
    seed:    int   = 42,
    rng:     np.random.Generator = None,
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
    seed    : random seed for reproducibility (legacy global-seed behaviour)
    rng     : optional np.random.Generator; when given it supplies the shocks
              and `seed` is ignored, so a caller can own the random stream

    Returns
    -------
    paths : np.ndarray shape (n_steps+1, n_paths)
            rows = timesteps, columns = simulation paths
    """

    if rng is None:
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
    eps = (np.random.normal(0, 1, size=(n_steps, n_paths)) if rng is None
           else rng.standard_normal((n_steps, n_paths)))
    for t in range(n_steps):
        paths[t+1, :] = paths[t, :] * e_kdt + drift + diff * eps[t, :]

    return paths


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
    rng:     np.random.Generator = None,
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
    seed    : random seed for reproducibility (legacy global-seed behaviour)
    rng     : optional np.random.Generator; when given it supplies the shocks
              and `seed` is ignored, so a caller can own the random stream

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

    if rng is None:
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
    eps = (np.random.normal(0, 1, size=(n_steps, n_paths)) if rng is None
           else rng.standard_normal((n_steps, n_paths)))
    for t in range(n_steps):
        paths[t+1, :] = alpha[t+1] + (paths[t, :] - alpha[t]) * e_kdt + diff * eps[t, :]

    return paths
