"""Structural properties the DP oracle must have (pension.checks), small grid."""
import numpy as np
import pytest

from pension import checks
from pension.objective import OBJECTIVES

N, SEED = 2000, 3


def test_scale_invariance(p, grid):
    assert checks.scale_invariance(p, grid["Fg"], grid["rg"], grid["ag"], grid["nq"], N, SEED) < 1e-12


def test_delta_e_is_a_lambda_change(p, grid):
    gap, _ = checks.lambda_reparam(p, grid["Fg"], grid["rg"], grid["ag"], grid["nq"])
    assert gap < 1e-6


def test_eta_one_is_log(p):
    to_log, jump = checks.eta_log_limit(p)
    assert to_log < 1e-12 and jump < 1e-3


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_full_service_leaver_is_a_stayer(p, grid, objective):
    assert checks.leaver_terminal_gap(p, grid["Fg"], grid["rg"], objective) < 1e-12


def test_cost_and_adequacy_increase_with_lambda(p, grid, entry):
    cost, sty = checks.lambda_monotonicity(p, grid["Fg"], grid["rg"], grid["ag"], grid["nq"],
                                           entry, N, SEED)
    assert np.all(np.diff(cost) > 0) and np.all(np.diff(sty) > 0), (cost, sty)


def test_no_capacity_adds_nothing(p, grid, entry):
    assert abs(checks.no_capacity_gap(p, grid["Fg"], grid["rg"], grid["nq"], entry, N, SEED)) < 0.01


@pytest.mark.slow
def test_timing_neutral_when_discounts_equal_mu(p, entry):
    import pension.dp as dp
    Fg, rg = dp.make_F_grid(n=73), dp.make_rho_grid(n=71)
    early, late = checks.timing_neutrality(p, Fg, rg, 5, entry, N, SEED)
    assert abs(early - late) / max(early, late) < 0.08, (early, late)   # measured 1.7% (REVIEW M6)
