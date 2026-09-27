"""Scenario suite: named parameter regimes with PREDICTED outcomes.

The other three suites sweep one parameter at a time around the committed
calibration. This one does something different: it runs a fixed battery of
*named regimes*, several of which have known, parameter-free answers, so it
functions as a correctness test of the economics rather than a sensitivity
picture. Regimes are defined by the ORDERING of rates -- MU vs G, DISC_ER vs MU,
DISC_EMP vs MU -- which one-at-a-time sweeps cannot reach.

Groups:
  A  analytic anchors     exact predictions (lambda=0 -> band floor, lambda=1 ->
                          ceiling, GAMMA->0 -> RR_LEGAL). These PASS/FAIL.
  B  drift gap MU - G     does the reserve out-grow the guarantee?
  C  discount structure   sign of (DISC_ER - MU) sets front-load vs back-load
  D  specification        SATIATE, ETA (incl. eta=1 log), BETA

Figures (-> figs/scenarios/):
  shapes   scenario_shapes.png   contribution schedules by regime (B and C)
  rates    scenario_rates.png    (G x delta_f) grid: delta_f sets the shape,
                                 G sets the level -- their ordering is NOT a regime
  figures  <slug>/*.png          the full six-figure set rendered under EACH
                                 scenario (policy_map, baseline_schedule,
                                 new_plan_profile, dca_schedules,
                                 sens_visitation_years, lambda_threshold)

Run:  python scenario_suite.py [table|shapes|rates|figures]
"""
import contextlib
import re
import time

import numpy as np
import common as c
import sensitivity_suite as _sens
import contribution_schedule_suite as _cs
import lambda_dial_suite as _ld

OUT = f"{c.OUT}/scenarios"   # this suite writes only here

# grid: nR >= 71 required at rho lo=0.01 -- the schedule-shape columns are not
# converged below that (widening the range without raising n coarsens the step).
GRID = dict(nF=73, nR=71, na=20, nq=5)
N_PATHS = 15000
SEED = 7
BAND_PCT = (0.02, 0.15)

# (label, group, overrides, expectation)  -- expectation is checked only for group A
SCENARIOS = [
    ("BASE (committed)",             "base", {},                                    None),

    ("lambda=0  pure employer",      "A", dict(LAMBDA=0.0),                          "floor"),
    ("lambda=1  pure employee",      "A", dict(LAMBDA=1.0),                          "ceiling"),
    ("GAMMA->0  no capacity",        "A", dict(GAMMA=0.01),                          "legal"),
    ("deterministic sigma=0",        "A", dict(SIGMA_R=0.0, SIGMA_L=0.0),            None),

    ("MU<G  underwater (1%/3%)",     "B", dict(MU=0.01, G=0.03),                     None),
    ("MU=G  at the money",           "B", dict(MU=0.03, G=0.03),                     None),
    ("MU>G  floating (5%/1.75%)",    "B", dict(MU=0.05, G=0.0175),                   None),

    ("DISC_ER<MU  (0.02)",           "C", dict(DISC_ER=0.02),                        None),
    ("DISC_ER=MU  (0.03)",           "C", dict(DISC_ER=0.03),                        None),
    ("DISC_ER>MU  (0.05 committed)", "C", dict(DISC_ER=0.05),                        None),
    ("DISC_EMP=0.02 (== a LAMBDA change)", "C", dict(DISC_EMP=0.02),                 None),

    ("SATIATE=True",                 "D", dict(SATIATE=True),                        None),
    ("ETA=1  log (ergodicity)",      "D", dict(ETA=1.0),                             None),
    ("ETA=5  high aversion",         "D", dict(ETA=5.0),                             None),
    ("BETA=0  hard argmax",          "D", dict(BETA=0.0),                            None),
]

_GROUP_TITLE = {"base": "committed baseline",
                "A": "A. analytic anchors (exact predictions)",
                "B": "B. drift gap  MU - G",
                "C": "C. discount structure",
                "D": "D. specification toggles"}


def _evaluate(overrides, **grid):
    """Solve + simulate under one set of dp.* overrides, always restoring them."""
    g = {**GRID, **grid}
    Fg, rg, _ = c.grids(g["nF"], g["nR"])
    try:
        for k, v in overrides.items():
            setattr(c.dp, k, v)
        lo, hi = BAND_PCT[0] / c.dp.GAMMA, min(BAND_PCT[1] / c.dp.GAMMA, 1.0)
        lo = min(lo, hi)                       # GAMMA->0 can invert the band
        pol = c.solve(Fg, rg, np.linspace(lo, hi, g["na"]), g["nq"])["policy"]
        rng = np.random.default_rng(SEED)
        R0, L0, S0 = c.new_plan_init(N_PATHS, rng)
        r = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, band=(lo, hi),
                       n_paths=N_PATHS, seed=SEED, track=True)
        r["band"] = (lo * c.dp.GAMMA * 100, hi * c.dp.GAMMA * 100)
        r["early"] = float(np.mean(r["c_by"][:10]))
        r["late"] = float(np.mean(r["c_by"][35:]))
        return r
    finally:
        c.restore()                            # never leak a mutated global


def _verdict(kind, r):
    """Group-A anchors have exact, parameter-free predictions."""
    floor, ceiling = r["band"]
    if kind == "floor":    ok = abs(r["avg"] - floor) < 0.2          # funds the minimum
    elif kind == "ceiling":ok = abs(r["avg"] - ceiling) < 0.2        # funds to capacity
    elif kind == "legal":  ok = abs(r["sty"] - c.dp.RR_LEGAL) < 0.05  # RR collapses to legal
    else:                  return "", True
    return ("PASS" if ok else "FAIL"), ok


# ============ 1. the table ============
def table():
    c.ensure_out(OUT)
    print(f"grid {GRID}   band {BAND_PCT[0]:.0%}-{BAND_PCT[1]:.0%}   {N_PATHS} paths")
    print("early/late = mean contribution %% over years 0-9 and 35-44\n")
    hdr = "  %-30s %7s %7s %8s %7s %7s %7s  %s" % (
        "scenario", "styRR", "leaRR", "avg", "cost", "early", "late", "check")
    results, failures, group = {}, [], None
    for label, grp, ov, exp in SCENARIOS:
        if grp != group:
            group = grp
            print(("\n" if results else "") + f"  --- {_GROUP_TITLE[grp]} ---")
            print(hdr)
        r = _evaluate(ov)
        results[label] = r
        mark, ok = _verdict(exp, r) if exp else ("", True)
        if not ok: failures.append(label)
        print("  %-30s %7.3f %7.3f %7.1f%% %7.2f %7.1f %7.1f  %s"
              % (label, r["sty"], r["lea"], r["avg"], r["cost"], r["early"], r["late"], mark))

    print("\n  group-A anchors: %s" % ("all PASS" if not failures else "FAILED -> " + ", ".join(failures)))
    print("  globals restored: MU=%.3f DISC_ER=%.3f LAMBDA=%.2f SATIATE=%s ETA=%.1f BETA=%.3f"
          % (c.dp.MU, c.dp.DISC_ER, c.dp.LAMBDA, c.dp.SATIATE, c.dp.ETA, c.dp.BETA))
    return results


# ============ 2. schedule shapes by regime ============
# NOTE on the panels below. delta_e is NOT shown as a regime: it enters the objective
# only as exp(-delta_e*T), a constant on the employee leg, so it is exactly redundant
# with LAMBDA up to a positive rescale of the objective (which leaves the argmax
# alone). Any (LAMBDA, d_e') has an equivalent (LAMBDA'', d_e):
#     LAMBDA'' = A / (A + B*exp(-d_e*T)),  A = LAMBDA*exp(-d_e'*T),  B = 1-LAMBDA
# verified to max|dPolicy| ~ 6e-11. So delta_e has no relationship to MU, G or
# delta_f -- it never interacts with them -- and a delta_e panel would be a LAMBDA
# panel mislabelled.
def shapes():
    c.ensure_out(OUT)
    # the committed case is EXCLUDED from each panel's lines: MU=G and DISC_ER=0.05
    # ARE the committed values, so plotting them would sit exactly on the baseline
    # and hide it. The baseline is drawn last, on top, as the explicit reference.
    panels = [("B", "drift gap  $\\mu - G$", ("MU=G  at the money",)),
              ("C", "employer discount  $\\delta_f$", ("DISC_ER>MU  (0.05 committed)",))]
    fig, axes = c.plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.8),
                               sharey=True, constrained_layout=True)
    yrs = np.arange(c.dp.T)
    base = _evaluate({})
    for ax, (grp, title, skip) in zip(np.atleast_1d(axes), panels):
        rows = [(lb, ov) for lb, g, ov, _ in SCENARIOS if g == grp and lb not in skip]
        cols = c.plt.cm.viridis(np.linspace(0.12, 0.80, len(rows)))
        for (lb, ov), col in zip(rows, cols):
            r = _evaluate(ov)
            ax.plot(yrs, r["c_by"], lw=2, color=col,
                    label=f"{lb}  ({r['early']:.0f}$\\to${r['late']:.0f}%)")
            print("  %-34s early %5.1f%%  late %5.1f%%  styRR %.3f" % (lb, r["early"], r["late"], r["sty"]))
        # baseline LAST and on top, so it is never hidden by a coincident line
        ax.plot(yrs, base["c_by"], lw=3.4, color="white", zorder=4)
        ax.plot(yrs, base["c_by"], lw=2.2, color="#C1121F", ls=(0, (5, 2)), zorder=5,
                label=f"COMMITTED baseline  ({base['early']:.0f}$\\to${base['late']:.0f}%)")
        ax.set_xlabel("career year $t$"); ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="best"); ax.grid(True, alpha=0.25, lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("contribution (% of salary)")
    fig.suptitle("Schedule shape responds to the rate structure "
                 "(levels matter too -- see rates())", fontsize=12)
    fig.savefig(f"{OUT}/scenario_shapes.png", dpi=c.DPI); c.plt.close(fig)
    print(f"\nwrote {OUT}/scenario_shapes.png")


# ============ 2b. (G x delta_f): which rate sets the LEVEL, which sets the SHAPE ============
def rates(Gs=(0.0175, 0.03, 0.045), DERs=(0.02, 0.05)):
    """G and delta_f do not form a regime boundary by their ORDERING -- the schedule
    tracks delta_f alone (front-loaded at 0.02 for every G, back-loaded at 0.05 for
    every G) while G shifts the LEVEL. This grid shows that separation directly."""
    c.ensure_out(OUT)
    yrs = np.arange(c.dp.T)
    fig, axes = c.plt.subplots(1, len(DERs), figsize=(6.4 * len(DERs), 4.8),
                               sharey=True, constrained_layout=True)
    print("  %-18s %8s %8s %8s %8s" % ("(G, delta_f)", "styRR", "avg", "early", "late"))
    for ax, der in zip(np.atleast_1d(axes), DERs):
        cols = c.plt.cm.plasma(np.linspace(0.15, 0.75, len(Gs)))
        for G, col in zip(Gs, cols):
            r = _evaluate(dict(G=G, DISC_ER=der))
            ax.plot(yrs, r["c_by"], lw=2, color=col,
                    label=f"G={G:.4g}   styRR {r['sty']:.2f}")
            print("  (%.4f, %.3f)   %8.3f %7.1f%% %8.1f %8.1f"
                  % (G, der, r["sty"], r["avg"], r["early"], r["late"]))
        ax.set_xlabel("career year $t$")
        ax.set_title(f"$\\delta_f$ = {der:.0%}  "
                     + ("(front-loads)" if der < c.dp.MU else "(back-loads)"))
        ax.legend(frameon=False, fontsize=8.5, loc="best"); ax.grid(True, alpha=0.25, lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("contribution (% of salary)")
    fig.suptitle(r"$\delta_f$ selects the SHAPE; $G$ shifts the LEVEL  "
                 r"($\mu$ fixed at %.0f%%)" % (100 * c.dp.MU), fontsize=12)
    fig.savefig(f"{OUT}/scenario_rates.png", dpi=c.DPI); c.plt.close(fig)
    print(f"\nwrote {OUT}/scenario_rates.png")

# ============ 3. the full figure set, per scenario ============
# Reuses the existing suite functions wholesale -- their OUT is temporarily
# redirected into the scenario's own folder, so none of the plotting code is
# duplicated here. Grids are reduced: this is ~14 solves per scenario.
FIG_GRID = dict(nF=73, nR=71, na=20, nq=5)
FIG_PATHS = 12000

FIGURES = [
    ("policy_map",        _cs,   lambda g: _cs.policy_map(nF=g["nF"], nR=g["nR"], na=g["na"],
                                                          nq=g["nq"], years=(0, 10, 22, 44))),
    ("baseline_schedule", _cs,   lambda g: _cs.baseline_schedule(nF=g["nF"], nR=g["nR"], nq=g["nq"],
                                                                 n_paths=FIG_PATHS)),
    ("new_plan_profile",  _cs,   lambda g: _cs.new_plan_profile(nF=g["nF"], nR=g["nR"], na=g["na"],
                                                                nq=g["nq"], n_paths=FIG_PATHS)),
    ("dca_schedules",     _cs,   lambda g: _cs.dca_predictability(nF=g["nF"], nR=g["nR"], na=g["na"],
                                                                  nq=g["nq"], eps_list=(0.0, 0.25, 1.0),
                                                                  n_paths=FIG_PATHS)),
    ("visitation_years",  _sens, lambda g: _sens.state_visitation(nF=g["nF"], nR=g["nR"], na=g["na"],
                                                                  nq=g["nq"], n_paths=FIG_PATHS,
                                                                  years=(0, 10, 22, 44))),
    ("lambda_threshold",  _ld,   lambda g: _ld.lambda_threshold(lambdas=np.linspace(0.2, 0.8, 6),
                                                                nF=g["nF"], nR=g["nR"], na=g["na"],
                                                                nq=g["nq"], n_paths=FIG_PATHS)),
]


def _slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


@contextlib.contextmanager
def _redirect(path):
    """Point every suite module's OUT at `path` for the duration."""
    mods = {_sens, _cs, _ld}
    saved = {m: m.OUT for m in mods}
    try:
        for m in mods: m.OUT = path
        yield
    finally:
        for m, o in saved.items(): m.OUT = o


def figures(only=("base", "B", "C"), grid=None, figs=None):
    """Render the full figure set under each scenario, into figs/scenarios/<slug>/.

    `only`   groups or labels to run (default: baseline + the two shape-flipping
             regimes; pass "all" for every scenario -- that is ~14 solves each).
    `figs`   subset of FIGURES names to render.
    """
    g = {**FIG_GRID, **(grid or {})}
    picked = [x for x in SCENARIOS if only == "all" or x[1] in only or x[0] in only]
    chosen = [f for f in FIGURES if figs is None or f[0] in figs]
    print("%d scenario(s) x %d figure(s), grid %s, %d paths"
          % (len(picked), len(chosen), g, FIG_PATHS))
    t_all = time.time()
    for label, grp, ov, _ in picked:
        d = f"{OUT}/{_slug(label)}"
        c.ensure_out(d)
        print(f"\n--- [{grp}] {label}  ->  {d}")
        t0 = time.time()
        try:
            for k, v in ov.items():
                setattr(c.dp, k, v)
            with _redirect(d):
                for name, _mod, fn in chosen:
                    try:
                        fn(g)
                    except Exception as e:
                        print("    %-18s FAILED: %s" % (name, e))
        finally:
            c.restore()                 # never leak a mutated global
        print("    (%.0fs)" % (time.time() - t0))
    print("\ntotal %.0fs   globals: MU=%.3f DISC_ER=%.3f LAMBDA=%.2f SATIATE=%s BETA=%.3f"
          % (time.time() - t_all, c.dp.MU, c.dp.DISC_ER, c.dp.LAMBDA, c.dp.SATIATE, c.dp.BETA))


_ALL = {"table": table, "shapes": shapes, "rates": rates, "figures": figures}


def main(which=None):
    c.ensure_out(OUT)
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
