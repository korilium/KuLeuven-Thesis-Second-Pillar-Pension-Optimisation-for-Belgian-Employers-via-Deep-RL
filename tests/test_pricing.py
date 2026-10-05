"""
Tests for pension/rates/pricing.py (the Vasicek / Hull-White bond pricer).

Run:  pytest tests/test_pricing.py      (the integration tests are marked slow)

Design
------
• DETERMINISTIC tests pin the pricer against analytic targets at machine
  precision (no Monte-Carlo noise). These are the real proof of correctness.
• INTEGRATION tests couple an independent reference simulator with the pricer
  via the no-arbitrage tower property. Their residual is Monte-Carlo +
  monthly-discretisation (a few to ~20 bps), NOT formula error — so they use a
  fixed seed and a generous tolerance and serve as consistency checks.

All tests use a synthetic NSS curve, so they are fully reproducible and need no
NBB data. The starred test (test_hw_recovers_market_at_t0) also runs on your
REAL curve — see the note at the bottom.
"""

import numpy as np
import pytest

from pension.rates.calibration import nss_yield, nss_forward
from pension.rates.pricing import (
    vasicekBondPrice,
    hullWhiteBondPrice,
    reconstructFutureYield,
    computeCumulativeDiscountFactors,
)

# ── Synthetic, Belgian-style fixtures ─────────────────────────────────────
SYNTH = np.array([0.035, -0.020, 0.020, 0.010, 2.0, 10.0])   # β0,β1,β2,β3,τ1,τ2
KAPPA, SIGMA = 0.15, 0.010
F00 = SYNTH[0] + SYNTH[1]                                     # f(0,0) = β0+β1

def synth_curve():
    tg = np.linspace(0.01, 50, 5000)
    return {"t_grid": tg, "f": nss_forward(tg, *SYNTH),
            "Y": nss_yield(tg, *SYNTH), "P": np.exp(-nss_yield(tg, *SYNTH) * tg),
            "params": SYNTH}

def Pmkt(T):
    return np.exp(-nss_yield(T, *SYNTH) * T)


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC PRICER TESTS  (machine precision)
# ══════════════════════════════════════════════════════════════════════════

def test_hw_zero_maturity_is_one():
    # A bond maturing now (τ=0) is worth exactly €1, any r, any t.
    c = synth_curve()
    for t in [0.0, 3.0, 10.0]:
        for r in [-0.01, 0.0, 0.03, 0.06]:
            P = hullWhiteBondPrice(r, t=t, kappa=KAPPA, sigma=SIGMA, tau=0.0, curve=c)
            assert abs(P - 1.0) < 1e-12

def test_vasicek_zero_maturity_is_one():
    for r in [-0.01, 0.0, 0.03, 0.06]:
        P = vasicekBondPrice(r, KAPPA, 0.03, SIGMA, tau=0.0)
        assert abs(P - 1.0) < 1e-12

def test_hw_recovers_market_at_t0():
    # *** THE no-arbitrage property: at t=0 with r=f(0,0), P_HW = P_market exactly,
    # including beyond 30Y. This is the test that must hold on the REAL curve too.
    c = synth_curve()
    for T in [1, 5, 10, 30, 45]:
        P = hullWhiteBondPrice(F00, t=0.0, kappa=KAPPA, sigma=SIGMA, tau=T, curve=c)
        assert abs(P - Pmkt(T)) < 1e-9, f"T={T}: {P} vs {Pmkt(T)}"

def test_hw_yield_reconstruction_at_t0():
    # -ln P_HW(0,T)/T must equal the NSS market yield.
    c = synth_curve()
    for T in [2, 7, 20, 40]:
        P = hullWhiteBondPrice(F00, t=0.0, kappa=KAPPA, sigma=SIGMA, tau=T, curve=c)
        y = -np.log(P) / T
        assert abs(y - nss_yield(T, *SYNTH)) < 1e-9

def test_hw_monotone_decreasing_in_r():
    # Higher short rate → lower bond price (∂P/∂r = -B·P < 0).
    c = synth_curve()
    r = np.linspace(-0.02, 0.08, 60)
    P = hullWhiteBondPrice(r, t=5.0, kappa=KAPPA, sigma=SIGMA, tau=10.0, curve=c)
    assert np.all(np.diff(P) < 0)

def test_hw_monotone_decreasing_in_tau():
    # For this upward curve, longer maturity → lower price.
    c = synth_curve()
    taus = np.linspace(0.1, 40, 100)
    P = np.array([hullWhiteBondPrice(0.03, t=3.0, kappa=KAPPA, sigma=SIGMA,
                                     tau=tt, curve=c) for tt in taus])
    assert np.all(np.diff(P) < 0)

def test_hw_price_in_unit_interval():
    # Prices are always positive; they are <= 1 only for a non-negative short rate.
    # The model is Gaussian and admits negative rates on purpose, and there a
    # bond is rightly worth more than par (r = -1% gives P > 1).
    c = synth_curve()
    r = np.array([-0.01, 0.02, 0.05])
    for t in [0.0, 5.0, 20.0]:
        for tau in [0.5, 5.0, 15.0]:
            P = hullWhiteBondPrice(r, t=t, kappa=KAPPA, sigma=SIGMA, tau=tau, curve=c)
            assert np.all(P > 0) and np.all(P[r >= 0] <= 1.0 + 1e-12)

def test_hw_valid_beyond_30y():
    # The fix that matters for WAP: pricing past the 30Y data must stay sane.
    c = synth_curve()
    for t in [10.0, 25.0, 35.0]:
        P = hullWhiteBondPrice(F00, t=t, kappa=KAPPA, sigma=SIGMA, tau=10.0, curve=c)
        y = -np.log(P) / 10.0
        assert 0.0 < P <= 1.0
        assert 0.0 < y < 0.10, f"reconstructed 10Y yield at t={t} = {y:.4f} is implausible"

def test_hw_flat_curve_analytic_target():
    # With a perfectly flat curve f(0,t)=c, P_HW has a clean closed form:
    #   P(t,T;r) = exp(-cτ + B(c-r) - σ²/(4κ)·B²·(1-e^{-2κt}))
    c_rate = 0.03
    flat = np.array([c_rate, 0.0, 0.0, 0.0, 2.0, 10.0])
    curve = {"params": flat}
    for t, tau, r in [(5.0, 10.0, 0.02), (12.0, 3.0, 0.05), (0.0, 8.0, c_rate)]:
        got = hullWhiteBondPrice(r, t=t, kappa=KAPPA, sigma=SIGMA, tau=tau, curve=curve)
        B = (1 - np.exp(-KAPPA * tau)) / KAPPA
        target = np.exp(-c_rate * tau + B * (c_rate - r)
                        - (SIGMA**2 / (4 * KAPPA)) * B**2 * (1 - np.exp(-2 * KAPPA * t)))
        assert abs(got - target) < 1e-12

def test_vasicek_asymptotic_yield():
    # Long-maturity Vasicek yield → R∞ = θ - σ²/(2κ²).
    theta = 0.03
    Rinf = theta - SIGMA**2 / (2 * KAPPA**2)
    y = -np.log(vasicekBondPrice(0.02, KAPPA, theta, SIGMA, tau=1000.0)) / 1000.0
    assert abs(y - Rinf) < 2e-4      # within 2 bps at τ=1000


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS  (reference simulator + pricer; fixed seed; consistency)
# ══════════════════════════════════════════════════════════════════════════

def _reference_hw_paths(T=15, n_paths=100_000, dt=1/12, seed=42):
    """Independent exact Hull-White simulator (shifted decomposition)."""
    np.random.seed(seed)
    n = int(round(T / dt)); t_axis = np.arange(n + 1) * dt
    f = nss_forward(t_axis, *SYNTH)
    alpha = f + (SIGMA**2 / (2 * KAPPA**2)) * (1 - np.exp(-KAPPA * t_axis))**2
    e = np.exp(-KAPPA * dt); diff = SIGMA * np.sqrt((1 - np.exp(-2 * KAPPA * dt)) / (2 * KAPPA))
    paths = np.zeros((n + 1, n_paths)); paths[0, :] = alpha[0]
    z = np.random.normal(size=(n, n_paths))
    for i in range(n):
        paths[i+1, :] = alpha[i+1] + (paths[i, :] - alpha[i]) * e + diff * z[i, :]
    return paths

@pytest.mark.slow
def test_tower_property():
    # E^Q[ D(0,t)·P(t,t+τ; r(t)) ] == P_market(0,t+τ).  Couples sim + pricer.
    # Residual is MC + monthly discretisation (~20 bps), not formula error.
    c = synth_curve(); dt = 1/12
    paths = _reference_hw_paths(T=15, n_paths=100_000, dt=dt, seed=42)
    D = computeCumulativeDiscountFactors(paths, dt=dt)
    i = int(round(5 / dt))
    P = hullWhiteBondPrice(paths[i, :], t=5.0, kappa=KAPPA, sigma=SIGMA, tau=10.0, curve=c)
    est = (D[i, :] * P).mean()
    err_bps = abs(est - Pmkt(15)) / Pmkt(15) * 1e4
    assert err_bps < 35, f"tower property off by {err_bps:.1f} bps"

@pytest.mark.slow
def test_expected_discount_factor():
    # E^Q[D(0,T)] == P_market(0,T) (simulator alone).
    dt = 1/12
    paths = _reference_hw_paths(T=15, n_paths=100_000, dt=dt, seed=42)
    D = computeCumulativeDiscountFactors(paths, dt=dt)
    err_bps = abs(D[-1, :].mean() - Pmkt(15)) / Pmkt(15) * 1e4
    assert err_bps < 25, f"E[D] off by {err_bps:.1f} bps"


# ══════════════════════════════════════════════════════════════════════════
# HELPER TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_reconstruct_future_yield_t0_row():
    # At t=0 every path sits at f(0,0); the reconstructed τ-yield must equal the
    # market yield nss_yield(τ) on the whole first row.
    c = synth_curve()
    paths = _reference_hw_paths(T=12, n_paths=2000, dt=1/12, seed=7)
    Y = reconstructFutureYield(paths, kappa=KAPPA, sigma=SIGMA, curve=c, tau=10.0, dt=1/12)
    assert np.allclose(Y[0, :], nss_yield(10.0, *SYNTH), atol=1e-9)

def test_cumulative_discount_factors_basic():
    paths = _reference_hw_paths(T=10, n_paths=1000, dt=1/12, seed=3)
    D = computeCumulativeDiscountFactors(paths, dt=1/12)
    assert D.shape == paths.shape
    assert np.all(D[0, :] == 1.0)        # no discounting at t=0
    assert np.all(D > 0)                  # discount factors stay positive
    assert D[-1, :].mean() < 1.0         # net positive discounting on average
