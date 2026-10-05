"""The rate engine as the pension contract sees it: WAP rates and degenerate scenarios."""
import pytest

from pension import checks
from pension.rates.wap import wap_formula


def test_wap_formula_reproduces_fsma_rates():
    for year, (ours, published) in checks.wap_vs_fsma().items():
        assert abs(ours - published) < 1e-9, (year, ours, published)


def test_wap_formula_bounds_and_rounding():
    assert wap_formula(-0.01) == pytest.approx(0.0175)      # floor
    assert wap_formula(0.10) == pytest.approx(0.0375)       # cap
    assert wap_formula(0.0300) == pytest.approx(0.0250)     # 0.85*3.00% = 2.55% -> 2.50%
    assert wap_formula(0.0310) == pytest.approx(0.0275)     # 0.85*3.10% = 2.635% -> 2.75%


@pytest.mark.parametrize("model", ["hull_white", "vasicek", "vasicek_short"])
def test_wap_rates_of_scenarios(p, model):
    st = checks.wap_scenario_stats(p, model, n=500)
    assert st["min"] >= 0.0175 - 1e-12 and st["max"] <= 0.0375 + 1e-12
    assert st["grid_err"] < 1e-9
    assert st["G0"] == pytest.approx(0.025) and st["G0_spread"] == 0.0   # today's fixing, observed


def test_degenerate_scenario_is_the_constant_model(p, grid, entry):
    assert checks.degenerate_rates_gap(p, grid["Fg"], grid["rg"], entry, 2000, 7) < 1e-10


def test_vasicek_short_calibration():
    """phi matches the historical mean (10Y - short) spread, so the in-sample bias
    of the reconstructed 10Y is zero; the short rate starts at the observed proxy."""
    f = checks.vasicek_short_fit()
    assert f["maturity"] == "1Y"
    assert abs(f["bias"]) < 1e-12
    assert f["kappa"] > 0 and f["sigma"] > 0


def test_vasicek_short_scenarios(p):
    """Year 0 starts at the observed short rate and, thanks to the decaying
    offset, at the observed 10Y; the offset decays at the calibrated rate."""
    import numpy as np
    from pension.economy import draw_rate_scenarios, rate_calibration, vasicek_short_params
    from pension.rates.pricing import vasicekBondPrice
    q = p.replace(RATE_MODEL="vasicek_short")
    sc = draw_rate_scenarios(300, p=q)
    vp = vasicek_short_params(q)
    assert np.allclose(sc["r"][0], vp["r0"])
    assert np.allclose(sc["y10"][0], rate_calibration()["hist10Y"][-1], atol=1e-12)
    assert sc["acc"].shape == (q.T, 300) and np.all(sc["acc"] > 0)
    affine = -np.log(vasicekBondPrice(sc["r"], vp["kappa"], vp["theta_Q"], vp["sigma"], 10.0)) / 10.0
    off = vp["e_t0"] * vp["monthly_decay"] ** (12 * np.arange(q.T + 1))
    np.testing.assert_allclose(sc["y10"] - affine, np.repeat(off[:, None], 300, 1), atol=1e-12)


def test_vasicek_short_long_rate_anchor(p):
    """LONG_RATE_P replaces theta_P; theta_Q (the mean-spread fit) is unchanged."""
    q0, q1 = p.replace(RATE_MODEL="vasicek_short"), p.replace(RATE_MODEL="vasicek_short", LONG_RATE_P=0.0225)
    f0, f1 = checks.vasicek_short_fit(q0), checks.vasicek_short_fit(q1)
    assert f1["theta_P"] == 0.0225 and f0["theta_P"] == f0["theta_P_ols"]
    assert f0["theta_Q"] == f1["theta_Q"] and f0["phi"] != f1["phi"]


# --- hull_white_p (stage 1b checks) -------------------------------------------------
def test_hw_p_reproduces_hull_white_at_phi_zero_and_legacy_sigma():
    assert checks.hw_p_link(n=50) == dict(r=0.0, y10=0.0, scenario_r=0.0)


@pytest.mark.parametrize("long_rate", [None, 0.0225])
def test_hw_p_fits_todays_curve(p, long_rate):
    f = checks.hw_p_t0_fit(p.replace(RATE_MODEL="hull_white_p", LONG_RATE_P=long_rate))
    assert abs(f["model_10y_t0"] - f["nss_10y"]) < 1e-12 and abs(f["r0"] - f["f00"]) < 1e-12
    assert abs(f["model_10y_t0"] - f["observed_10y"]) < 3 * f["nss_rmse"]
    assert (f["phi"] == 0.0) == (long_rate is None)


@pytest.mark.parametrize("long_rate", [None, 0.0225])
def test_hw_p_real_world_drift(p, long_rate):
    for t, d in checks.hw_p_drift(p.replace(RATE_MODEL="hull_white_p", LONG_RATE_P=long_rate),
                                  n=2000).items():
        assert abs(d["mc"] - d["analytic"]) < 4 * d["se"], (t, d)


def test_hw_p_hits_the_10y_volatility(p):
    v = checks.vol_target(p, n=300)
    assert abs(v["hull_white_p"] / v["historical"] - 1) < 0.05
    assert abs(v["hull_white"] / v["hull_white_p"] - v["B10_over_10"]) < 0.05


@pytest.mark.parametrize("model,long_rate", [("hull_white_p", None), ("hull_white_p", 0.0225),
                                             ("vasicek_short", None), ("vasicek_short", 0.0225)])
def test_closed_form_accrual_matches_monte_carlo(p, model, long_rate):
    acc = checks.accrual_accuracy(p.replace(RATE_MODEL=model, LONG_RATE_P=long_rate), n=2000)
    for t, d in acc.items():
        assert abs(d["rel_err"]) < 4 * d["rel_se"] + 1e-3, (t, d)


def test_no_real_world_accrual_for_q_models(p):
    from pension.rates.accrual import closed_form_accrual
    for model in ("hull_white", "constant"):
        with pytest.raises(ValueError):
            closed_form_accrual(0, 0.03, p.replace(RATE_MODEL=model))
