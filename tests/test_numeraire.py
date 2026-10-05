"""The employer's numeraire (pension/numeraire.py): stage-3 checks a-e."""
import numpy as np
import pytest

from pension import checks


def test_env_contract_both_numeraires(p, grid, entry):
    small = {k: v[:200] for k, v in entry.items()}
    for num in ("retirement", "discounted"):
        assert checks.env_contract(p.replace(EMPLOYER_NUMERAIRE=num), grid["Fg"], grid["rg"],
                                   grid["ag"], grid["nq"], small, 200, 7) < 1e-12


def test_reduced_form_matches_step(p):
    assert checks.reduced_form_gap(p) < 1e-12


@pytest.mark.parametrize("lam,r,de", [(0.5, 0.03, 0.03), (0.5, 0.05, 0.03), (0.3, 0.02, 0.04),
                                      (0.7, 0.04, 0.01), (0.5, 0.0, 0.03)])
def test_retirement_equals_discounted_at_lambda_prime(p, grid, lam, r, de):
    """(b) The two objectives differ by a positive factor: same policy, V in ratio."""
    dpol, dV, _ = checks.numeraire_equivalence(p, grid["Fg"], grid["rg"], grid["ag"], grid["nq"],
                                               lam, r, de)
    assert dpol < 1e-6 and dV < 1e-10, (dpol, dV)


def test_face_value_premiums(p, grid, entry):
    """(c) SHORT_RATE = s = 0: premiums at face value."""
    f, d = checks.face_value_premiums(p, grid["Fg"], grid["rg"], entry, 2000, 7)
    assert f == 0.0 and d < 1e-12


@pytest.mark.slow
def test_timing_neutral_at_short_rate_equal_mu(p, entry):
    """(d) SHORT_RATE = MU = G: a roughly level schedule."""
    import pension.dp as dp
    Fg, rg = dp.make_F_grid(n=73), dp.make_rho_grid(n=71)
    early, late = checks.timing_neutrality(p.replace(EMPLOYER_NUMERAIRE="retirement"),
                                           Fg, rg, 5, entry, 2000, 3)
    assert abs(early - late) / max(early, late) < 0.08, (early, late)   # measured 1.7% (REVIEW M6)


def test_legacy_drift_tilts_towards_early_funding(p, entry):
    """REVIEW M6: without the -sigma^2/2 correction the reserve's mean return is
    MU + SIGMA_R^2/2 > SHORT_RATE, so funding early pays; measured early 7.9% vs
    late 6.7% of salary."""
    import pension.dp as dp
    Fg, rg = dp.make_F_grid(n=73), dp.make_rho_grid(n=71)
    early, late = checks.timing_neutrality(
        p.replace(EMPLOYER_NUMERAIRE="retirement", DRIFT_CORRECTION=False), Fg, rg, 5, entry, 2000, 3)
    assert (early - late) / early > 0.10, (early, late)


def test_accrual_accuracy_vasicek(p):
    """(e) for the canonical model; the robustness models are in test_rates."""
    for t, d in checks.accrual_accuracy(p.replace(RATE_MODEL="vasicek"), n=2000).items():
        assert abs(d["rel_err"]) < 4 * d["rel_se"] + 1e-5, (t, d)   # REVIEW M12: no 10 bp floor


def test_headline_runs(p, grid, entry):
    h = checks.headline(p, grid["Fg"], grid["rg"], grid["nq"], entry, 2000, 7, na=7)
    assert len(h["by_decade"]) == 5 and h["cost"] > 0
