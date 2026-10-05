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


@pytest.mark.parametrize("model", ["hull_white", "vasicek"])
def test_wap_rates_of_scenarios(p, model):
    st = checks.wap_scenario_stats(p, model, n=500)
    assert st["min"] >= 0.0175 - 1e-12 and st["max"] <= 0.0375 + 1e-12
    assert st["grid_err"] < 1e-9
    assert st["G0"] == pytest.approx(0.025) and st["G0_spread"] == 0.0   # today's fixing, observed


def test_degenerate_scenario_is_the_constant_model(p, grid, entry):
    assert checks.degenerate_rates_gap(p, grid["Fg"], grid["rg"], entry, 2000, 7) < 1e-10
