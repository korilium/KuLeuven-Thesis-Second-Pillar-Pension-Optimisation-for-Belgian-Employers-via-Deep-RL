"""Objective suite: how the value function shapes the optimal funding policy.

Every objective in pension/objective.py (OBJECTIVES) is solved on the common
protocol (common.GRID / N_PATHS / SEED / BAND_PCT, new-plan entry cohort, the
--rates regime) and compared on what it DOES, not on its value: values of
different objectives are in different units, policies and outcomes are not.

To try your own value function: add an Objective to OBJECTIVES in objective.py
(or pass one in `names` below as an Objective instance) and rerun.

Checks (asserted; the suite stops on the first failure):
  baseline   the "baseline" objective reproduces the committed formulas
  leaver     Phi[T] == terminal for EVERY objective (leaver at full service = stayer)
  lambda     delta_e is still exactly a LAMBDA change under "baseline"
  diagonal   each objective's own policy scores best under that objective
             (up to Monte-Carlo / grid noise), i.e. the oracle optimises what
             simulate() scores

Figures (-> figs/objectives/, or figs/rates_<model>/objectives/):
  maps       objective_maps.png        a*(F, rho) per objective at two years, with
                                       where the cohort actually is
  schedules  objective_schedules.png   contribution path per objective + a table of
                                       outcomes in money terms (baseline cost units)
  frontier   objective_frontier.png    each objective on the baseline lambda-frontier:
                                       on it = "just a different lambda", off it = new
  cross      objective_cross.png       policy optimised under i, scored under j, as
                                       normalised regret (0 = as good as j's own optimum,
                                       1 = no better than funding the band floor)

Run:  python objective_suite.py [checks|maps|schedules|frontier|cross] [--rates=...]
"""
import numpy as np
import common as c
from pension import checks as known   # (the suite function below is called checks)
import benchmark_suite as _bm
from pension.objective import OBJECTIVES

OUT = f"{c.OUT}/objectives"   # this suite writes only here

GRID, N_PATHS, SEED, BAND_PCT = c.GRID, c.N_PATHS, c.SEED, c.BAND_PCT
NAMES = list(OBJECTIVES)
COLORS = dict(zip(NAMES, ["#222222", "#1D9E75", "#C1121F", "#274690", "#E08D1C",
                          "#8E44AD", "#7F8C8D"]))


def _band():
    lo, hi = BAND_PCT[0] / c.P.GAMMA, min(BAND_PCT[1] / c.P.GAMMA, 1.0)
    return lo, hi


def _grids(g=None):
    g = {**GRID, **(g or {})}
    Fg, rg, _ = c.grids(g["nF"], g["nR"])
    lo, hi = _band()
    return Fg, rg, np.linspace(lo, hi, g["na"]), g["nq"]


_POL = {}

def _policy(name, g=None):
    """The banded optimum under objective `name` (memoised per grid and parameters)."""
    key = (name, tuple(sorted((g or {}).items())), c.P)     # Params is hashable (REVIEW m8)
    if key not in _POL:
        Fg, rg, ag, nq = _grids(g)
        _POL[key] = c.solve(Fg, rg, ag, nq, objective=name)["policy"]
    return _POL[key]


def _sim(pol, objective, g=None, **kw):
    Fg, rg, _, _ = _grids(g)
    return c.simulate(pol, Fg, rg, **c.entry(N_PATHS, SEED), band=_band(),
                      n_paths=N_PATHS, seed=SEED, objective=objective, **kw)


def _floor_policy(g=None):
    Fg, rg, _, _ = _grids(g)
    return c.const_policy(_band()[0], c.P.T, len(Fg), len(rg))


# --- checks -------------------------------------------------------------------
def _cross_matrix(names=NAMES):
    """J[i, j] = score under objective j of the policy optimised under objective i,
    plus each column's band-floor score for normalisation (pension.checks.cross_scores)."""
    Fg, rg, _, _ = _grids()
    return known.cross_scores(c.P, {n: _policy(n) for n in names}, Fg, rg,
                               c.entry(N_PATHS, SEED), N_PATHS, SEED, _band(), _band()[0])


@c.restores
def checks(tol=0.01):
    Fg, rg, ag, nq = _grids()
    dp, P = c.dp, c.P.replace(EMPLOYER_NUMERAIRE="discounted")   # the hand formula is the discounted one

    # baseline == the committed formulas, written out by hand
    Fc, rc = Fg[:, None], rg[None, :]
    rr = P.RR_LEGAL + np.maximum(Fc, 1.0) / (rc * P.ANNUITY)
    ref = (P.LAMBDA * P.RR_TARGET * P.ANNUITY * dp.u(rr / P.RR_TARGET, p=P) * np.exp(-P.DISC_EMP * P.T)
           - (1 - P.LAMBDA) * np.maximum(1.0 - Fc, 0.0) / rc * np.exp(-P.DISC_ER * P.T))
    d = float(np.abs(dp.terminal(Fg, rg, "baseline", p=P) - ref).max() / np.abs(ref).max())
    assert d < 1e-12, d
    print(f"  baseline   terminal == committed formula (rel {d:.1e})                  OK")

    for n in NAMES:
        d = known.leaver_terminal_gap(P, Fg, rg, n)
        assert d < 1e-12, (n, d)
    print(f"  leaver     Phi[T] == terminal for all {len(NAMES)} objectives                      OK")

    d, _ = known.lambda_reparam(P, Fg, rg, ag, nq, objective="baseline")
    assert d < 1e-6, d
    print(f"  lambda     delta_e == LAMBDA reparametrisation (max|dpolicy| {d:.1e})        OK")

    J, floor = _cross_matrix()
    gain, _ = known.diagonal_margin(J, floor)
    worst, arg = known.diagonal_margin(J, floor, relative_to="value")
    for j, n in enumerate(NAMES):
        print(f"  diagonal   {n:17s} best other ({NAMES[arg[j]]}) {worst[j]:+.4f} of |J|, "
              f"{gain[j]:+.3f} of the gain over the floor  {'OK' if worst[j] <= tol else 'FAIL'}")
    assert np.all(worst <= tol), "an objective is beaten by another objective's policy"


# --- figures ------------------------------------------------------------------
MAP_G = dict(nF=121, nR=91)


def maps(years=(5, 22), names=NAMES):
    c.ensure_out(OUT)
    Fg, rg, _, _ = _grids(MAP_G)
    RR = c.iso_rr(Fg, rg)
    fig, axes = c.plt.subplots(len(years), len(names), figsize=(2.9 * len(names), 2.9 * len(years) + 0.6),
                               sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    lo, hi = _band()
    mesh = None
    for j, n in enumerate(names):
        pol = _policy(n, MAP_G)
        r = _sim(pol, n, MAP_G, visits=True)
        for i, t in enumerate(years):
            ax = axes[i, j]
            mesh = ax.pcolormesh(Fg, rg, pol[t].T * c.P.GAMMA * 100, cmap="viridis",
                                 vmin=lo * c.P.GAMMA * 100, vmax=hi * c.P.GAMMA * 100, shading="auto")
            cs = ax.contour(Fg, rg, RR.T, levels=[c.P.RR_TARGET], colors="white", linewidths=1.0)
            _bm._contours(ax, r["visits"][t], Fg, rg, color="#ff4fd8")
            ax.set_yscale("log"); ax.axvline(1.0, color="white", lw=0.7, ls=":")
            if i == 0: ax.set_title(n, fontsize=10)
            if j == 0: ax.set_ylabel(f"year {t}\n$\\rho = S/L$")
            if i == len(years) - 1: ax.set_xlabel("$F = R/L$")
    fig.colorbar(mesh, ax=axes, shrink=0.8, label="optimal contribution (% of salary)")
    fig.suptitle("Optimal funding a*(F, rho) per value function  "
                 "(white: RR = target; magenta: where the cohort is, 50/90/99%)", fontsize=11)
    fig.savefig(f"{OUT}/objective_maps.png", dpi=c.DPI); c.plt.close(fig)
    print(f"  wrote {OUT}/objective_maps.png")


def _outcomes(names=NAMES):
    rows = {}
    for n in names:
        own = _sim(_policy(n), n)
        money = _sim(_policy(n), "baseline")      # same paths, scored in money terms
        rows[n] = dict(c_by=own["c_by"], sty=own["sty"], lea=own["lea"], avg=own["avg"],
                       early=float(np.mean(own["c_by"][:10])), late=float(np.mean(own["c_by"][35:])),
                       cost=money["cost"])
    return rows


def schedules(names=NAMES):
    c.ensure_out(OUT)
    rows = _outcomes(names)
    print(f"  rates {c.RATES}   band {BAND_PCT[0]:.0%}-{BAND_PCT[1]:.0%}   cost = PV per final "
          f"salary, baseline units\n")
    print("  %-17s %7s %7s %7s %7s %7s %8s   %s" % ("objective", "styRR", "leaRR", "avg%", "early",
                                                    "late", "cost", "note"))
    for n, m in rows.items():
        print("  %-17s %7.3f %7.3f %7.1f %7.1f %7.1f %8.4f   %s"
              % (n, m["sty"], m["lea"], m["avg"], m["early"], m["late"], m["cost"], OBJECTIVES[n].note))
    fig, ax = c.plt.subplots(figsize=(9, 5), constrained_layout=True)
    yrs = np.arange(c.P.T)
    for n, m in rows.items():
        ax.plot(yrs, m["c_by"], lw=2.6 if n == "baseline" else 1.6, color=COLORS.get(n),
                ls="--" if n == "baseline" else "-",
                label=f"{n}  (avg {m['avg']:.1f}%, stayer RR {m['sty']:.2f})")
    ax.set_xlabel("career year $t$"); ax.set_ylabel("contribution (% of salary)")
    ax.legend(frameon=False, fontsize=8); ax.grid(True, alpha=0.25)
    ax.set_title(f"Contribution schedule per value function  (rates: {c.RATES})")
    fig.savefig(f"{OUT}/objective_schedules.png", dpi=c.DPI); c.plt.close(fig)
    print(f"\n  wrote {OUT}/objective_schedules.png")


def frontier(names=NAMES, lambdas=_bm.FRONTIER_LAMBDAS[::2]):
    c.ensure_out(OUT)
    Fg, rg, _, _ = _grids()
    fr = _bm._frontier(Fg, rg, lambdas)            # baseline objective, lambda swept
    rows = _outcomes(names)
    fig, ax = c.plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.plot(fr["cost"], fr["sty"], "-o", color="#9aa0a6", ms=3, lw=1.5, label="baseline, $\\lambda$ swept")
    for lam, x, y in zip(fr["lam"][::2], fr["cost"][::2], fr["sty"][::2]):
        ax.annotate(f"{lam:.2f}", (x, y), fontsize=7, color="#9aa0a6", xytext=(4, -8),
                    textcoords="offset points")
    for n, m in rows.items():
        ax.scatter(m["cost"], m["sty"], s=70, color=COLORS.get(n), zorder=5, label=n,
                   marker="*" if n == "baseline" else "o")
    ax.set_xlabel("employer cost (PV per final salary, baseline units)")
    ax.set_ylabel("median stayer total RR")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(True, alpha=0.25)
    ax.set_title("Is a value function just a different $\\lambda$?\n"
                 "on the grey curve: yes;  above/below: it buys a different trade-off")
    fig.savefig(f"{OUT}/objective_frontier.png", dpi=c.DPI); c.plt.close(fig)
    print(f"  wrote {OUT}/objective_frontier.png")


def cross(names=NAMES):
    c.ensure_out(OUT)
    J, floor = _cross_matrix(names)
    gain = np.diag(J) - floor
    R = (np.diag(J)[None, :] - J) / np.where(np.abs(gain) > 0, gain, 1.0)[None, :]
    print("  normalised regret: row = policy optimised under, column = scored under")
    print("  %-17s" % "" + "".join(f"{n[:9]:>10s}" for n in names))
    for i, n in enumerate(names):
        print("  %-17s" % n + "".join(f"{v:10.3f}" for v in R[i]))
    fig, ax = c.plt.subplots(figsize=(1.1 * len(names) + 3, 1.0 * len(names) + 2), constrained_layout=True)
    im = ax.imshow(R, cmap="magma_r", vmin=0, vmax=max(1.0, float(np.nanmax(R))))
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if R[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("scored under"); ax.set_ylabel("policy optimised under")
    fig.colorbar(im, ax=ax, label="regret (0 = own optimum, 1 = band floor)")
    ax.set_title(f"What each value function gives up under the others  (rates: {c.RATES})", fontsize=10)
    fig.savefig(f"{OUT}/objective_cross.png", dpi=c.DPI); c.plt.close(fig)
    print(f"  wrote {OUT}/objective_cross.png")


_ALL = {"checks": checks, "maps": maps, "schedules": schedules, "frontier": frontier, "cross": cross}


def main(which=None):
    c.ensure_out(OUT)
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
