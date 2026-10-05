"""Printed simulation report for the committed model (pension.dp).

Runs simulations and prints the numbers -- no pass/fail verdicts; read the
quantities and judge them. Complements experiments/dynpro/*_suite.py, which produce
figures; this file produces numbers and stays fast.

All Monte-Carlo goes through dp.simulate(, p=P) (the shared committed engine: Belgian
churn, split discount, service-pro-rated target) rather than a local rollout, so
these numbers are directly comparable with the suites'.

Run:
    python experiments/dynpro/report.py            # everything
    python experiments/dynpro/report.py cohort     # one section
    python experiments/dynpro/report.py --rates=vasicek   # under a rate model (also: hull_white_p)

Under a rate model dp.solve is certainty-equivalent and dp.simulate path-wise
(see Params.RATE_MODEL); the mu sweep in diagnostics is then skipped, since mu
is a path-wise book yield rather than a parameter.
"""
import numpy as np

import pension.dp as dp

P = dp.DEFAULT        # the parameters of the report; --rates=<model> sets RATE_MODEL



# ============ 1. cohort outcome distribution ============
def cohort(nF=121, nR=91, na=26, nq=5, n_paths=30000, seed=7,
           band_pct=(0.02, 0.15), years=(0, 5, 10, 15, 22, 30, 44)):
    """Solve the committed policy, simulate a fresh-plan cohort, report the
    spread of outcomes it produces."""
    Fg, rg, _ = dp.grids(nF, nR, na)
    lo, hi = band_pct[0] / P.GAMMA, band_pct[1] / P.GAMMA
    ag = np.linspace(lo, hi, na)
    pol = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, p=P)["policy"]

    rng = np.random.default_rng(seed)
    R0, L0, S0 = dp.new_plan_init(n_paths, rng)
    r = dp.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, band=(lo, hi),
                    n_paths=n_paths, seed=seed, track=True, p=P)

    RR = r["RR_tot"]; stay = r["stay"]
    print(f"cohort: {n_paths} paths, banded {band_pct[0]:.0%}-{band_pct[1]:.0%} of salary, "
          f"grid ({nF},{nR},{na},q{nq})")
    print("\n  total replacement rate (legal + 2nd pillar)")
    print(f"    median      {np.median(RR):.3f}      target {P.RR_TARGET}   legal floor {P.RR_LEGAL}")
    print(f"    P10 / P90   {np.quantile(RR, 0.10):.3f} / {np.quantile(RR, 0.90):.3f}")
    print(f"    P(RR < target)   {np.mean(RR < P.RR_TARGET):.2f}")
    print(f"    P(RR < legal+5pp) {np.mean(RR < P.RR_LEGAL + 0.05):.2f}")

    print("\n  stayers vs leavers")
    print(f"    stayers   share {stay.mean():.2f}   median RR {r['sty']:.3f}")
    print(f"    leavers   share {1 - stay.mean():.2f}   median RR {r['lea']:.3f}")
    print(f"    expected full-career share (survival to T) {dp.survival(dp.tenure_hazard, p=P)[P.T]:.2f}")

    print("\n  employer side")
    print(f"    cost (PV per unit final salary)  {r['cost']:.4f}")
    print(f"    career-average contribution      {r['avg']:.1f}% of salary")
    print(f"    joint value                      {r['joint']:+.4f}")

    print("\n  contribution schedule (% of salary)")
    print("    year " + "".join(f"{y:>7}" for y in years))
    print("    pct  " + "".join(f"{r['c_by'][y]:>7.1f}" for y in years))
    return r


# ============ 2. model-health diagnostics ============
def diagnostics(nF=73, nR=91, na=15, nq=5, n_paths=8000, seed=3):
    """The quantities the old assertion suite used to check, printed as numbers."""
    Fg, rg, ag = dp.grids(nF, nR, na)

    print("quadrature")
    zR, zL, wq = dp.gauss_hermite_2d(7)
    print(f"    2D GH weights sum   1{wq.sum() - 1:+.2e}")

    print("\nleaver value (deterministic paid-up, pro-rated target)")
    Phi = dp.paidup_service(Fg, rg, p=P)
    print(f"    max|Phi[T] - dp.terminal|   {np.abs(Phi[P.T] - dp.terminal(Fg, rg, p=P)).max():.2e}")

    print("\nsolve health")
    o1 = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, p=P)
    pol = o1["policy"]
    o2 = dp.solve(Fg=Fg, rg=rg, ag=ag, n_quad=nq, p=P)
    bang = float(np.mean((pol == ag.min()) | (pol == ag.max())))
    print(f"    grid-mean a*        {pol.mean():.3f}")
    print(f"    at a corner         {bang:.2f} of states")
    print(f"    policy min / max    {pol.min():.3f} / {pol.max():.3f}")
    print(f"    non-finite in V     {int((~np.isfinite(o1['V'])).sum())}")
    print(f"    repeat solve identical   {np.array_equal(pol, o2['policy'])}")
    print(f"    F=1 is a grid node       {np.isclose(Fg[np.argmin(np.abs(Fg - 1))], 1.0)}")

    print("\ngrid convergence")
    c = dp.solve(Fg=dp.make_F_grid(n=73), rg=dp.make_rho_grid(n=31),
                           ag=dp.make_a_grid(n=15), n_quad=5, p=P)["policy"].mean()
    f = dp.solve(Fg=dp.make_F_grid(n=145), rg=dp.make_rho_grid(n=61),
                           ag=dp.make_a_grid(n=31), n_quad=7, p=P)["policy"].mean()
    print(f"    grid-mean a*  coarse {c:.3f}   fine {f:.3f}   diff {abs(f - c):.3f}")

    print("\nscale-freeness (homogeneity degree 0)")
    kappa = 5.0
    r1 = dp.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=20.0, n_paths=n_paths, seed=seed, p=P)
    rk = dp.simulate(pol, Fg, rg, R0=kappa, L0=kappa, S0=kappa * 20.0, n_paths=n_paths, seed=seed, p=P)
    rel = np.max(np.abs(rk["RR"] - r1["RR"]) / np.abs(r1["RR"]))
    print(f"    max rel. diff in RR when (R,L,S) scaled by {kappa:g}   {rel:.2e}")

    print("\neconomic monotonicity")
    sty = []
    for a_const in (0.2, 0.4, 0.6, 0.8):
        pol_a = dp.const_policy(a_const, P.T, len(Fg), len(rg))
        sty.append(dp.simulate(pol_a, Fg, rg, S0=20.0, n_paths=n_paths, seed=seed, p=P)["sty"])
    print(f"    stayer RR vs constant a=0.2/0.4/0.6/0.8   {' '.join(f'{x:.3f}' for x in sty)}")
    print(f"    increasing in a                            {bool(np.all(np.diff(sty) > 0))}")

    if P.RATE_MODEL != "constant":
        print(f"    mu sweep skipped under RATE_MODEL={P.RATE_MODEL}: mu is path-wise")
    else:
        pol6 = dp.const_policy(0.6, P.T, len(Fg), len(rg))
        lo_rr = dp.simulate(pol6, Fg, rg, S0=20.0, n_paths=n_paths, seed=seed, p=P.replace(MU=0.01))["tot"]
        hi_rr = dp.simulate(pol6, Fg, rg, S0=20.0, n_paths=n_paths, seed=seed, p=P.replace(MU=0.05))["tot"]
        print(f"    median RR at mu=1% / mu=5%                 {lo_rr:.3f} / {hi_rr:.3f}")

    print("\nleaver vs stayer (banded policy)")
    LO, HI = 0.02 / P.GAMMA, 0.15 / P.GAMMA
    polb = dp.solve(Fg=Fg, rg=rg, ag=np.linspace(LO, HI, 26), n_quad=nq, p=P)["policy"]
    rb = dp.simulate(polb, Fg, rg, S0=20.0, band=(LO, HI), n_paths=n_paths, seed=seed, p=P)
    print(f"    stayer median RR {rb['sty']:.3f}   leaver median RR {rb['lea']:.3f}")
    print(f"    career-average contribution {rb['avg']:.1f}%")


_ALL = {"cohort": cohort, "diagnostics": diagnostics}


def main(which=None):
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
            fn()


if __name__ == "__main__":
    import sys
    for arg in list(sys.argv[1:]):
        if arg.startswith("--rates="):
            P = P.replace(RATE_MODEL=arg.split("=", 1)[1]); sys.argv.remove(arg)
    print(f"RATE_MODEL = {P.RATE_MODEL}")
    main(sys.argv[1] if len(sys.argv) > 1 else None)
