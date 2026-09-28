"""Market-plan benchmark: what does optimising the schedule actually buy?

The other suites compare the DP against variants of itself (flat, banded DCA,
pure DCA). This one compares it against plan designs employers actually use, on
the SAME protocol (common.GRID / N_PATHS / SEED / BAND_PCT), so the headline
claim -- "optimisation is worth X" -- has a reference point outside the model's
own family.

Designs covered, both scale-free and therefore expressible as a state-independent
schedule a(t) (see dp.schedule_policy):
  * FLAT      a fixed percentage of salary every year;
  * AGE SCALE a stepped percentage rising with age/tenure, the shape of
              economy.plan_age.

NOT covered: the two-tier formula that splits contributions at the social-security
wage ceiling. It compares a salary LEVEL against a fixed ceiling, while the model
is homogeneous of degree 0 in (R, L, S) -- the ceiling would have to enter the
state as a further ratio. Little is lost: if salary and ceiling are indexed at the
same rate then S/ceiling never moves and the two-tier plan degenerates to a flat
rate anyway; the design only bites when career progression outruns indexation.

SOURCING: the contribution LEVELS below (3/5/8%, and the age steps) are
placeholders chosen to bracket plausible Belgian practice. They need a DB2P /
Assuralia figure before being described as representative, and any age step must
stay inside the WAP non-discrimination bound. Confirm before citing.

Figures (-> figs/benchmark/):
  table      (printed)                 every design vs the DP on one protocol, plus
                                       cost at matched adequacy / adequacy at matched cost
  frontier   benchmark_frontier.png    the lambda-frontier with the designs on it
  schedules  benchmark_schedules.png   each design's shape vs the DP's

Run:  python benchmark_suite.py [table|frontier|schedules]
"""
import numpy as np
import common as c

OUT = f"{c.OUT}/benchmark"   # this suite writes only here

GRID, N_PATHS, SEED, BAND_PCT = c.GRID, c.N_PATHS, c.SEED, c.BAND_PCT


# --- the designs, as rate-of-salary schedules ------------------------------
def _flat(rate):
    return lambda t: rate

def _age(rate0, step, band):
    """The economy.plan_age shape: rate0 + step * (t // band)."""
    return lambda t: rate0 + step * (t // band)

DESIGNS = [
    ("flat 3% of salary",      _flat(0.03)),
    ("flat 5% of salary",      _flat(0.05)),
    ("flat 8% of salary",      _flat(0.08)),
    ("age scale 3% +1%/10y",   _age(0.03, 0.01, 10)),
    ("age scale 2% +1%/5y",    _age(0.02, 0.01, 5)),
]


def _metrics(r):
    return dict(cost=r["cost"], benefit=r["benefit"], joint=r["joint"],
                sty=r["sty"], lea=r["lea"], avg=r["avg"],
                early=float(np.mean(r["c_by"][:10])), late=float(np.mean(r["c_by"][35:])),
                c_by=r["c_by"])


def _run_design(rate_of_t, Fg, rg):
    """A market design: contribution set by the calendar, not by the state."""
    a_of_t = lambda t: min(rate_of_t(t) / c.dp.GAMMA, 1.0)
    pol = c.schedule_policy(a_of_t, len(Fg), len(rg))
    r = c.simulate(pol, Fg, rg, **c.entry(N_PATHS, SEED), n_paths=N_PATHS, seed=SEED)
    return _metrics(r)


def _run_dp(Fg, rg, banded=True, lam=None, **g):
    """The optimised schedule, banded to the protocol band or unconstrained."""
    gg = {**GRID, **g}
    with c.overrides(**({} if lam is None else dict(LAMBDA=float(lam)))):
        if banded:
            lo, hi = BAND_PCT[0] / c.dp.GAMMA, min(BAND_PCT[1] / c.dp.GAMMA, 1.0)
            ag, band = np.linspace(lo, hi, gg["na"]), (lo, hi)
        else:
            ag, band = c.dp.make_a_grid(n=gg["na"]), None
        pol = c.solve(Fg, rg, ag, gg["nq"])["policy"]
        r = c.simulate(pol, Fg, rg, **c.entry(N_PATHS, SEED), band=band,
                       n_paths=N_PATHS, seed=SEED)
    return _metrics(r)


# --- the lambda frontier, used for the matched comparisons -----------------
def _frontier(Fg, rg, lambdas, **g):
    rows = [(float(lam), _run_dp(Fg, rg, banded=True, lam=lam, **g)) for lam in lambdas]
    return dict(lam=np.array([x for x, _ in rows]),
                cost=np.array([m["cost"] for _, m in rows]),
                sty=np.array([m["sty"] for _, m in rows]),
                benefit=np.array([m["benefit"] for _, m in rows]))


def _matched(fr, m):
    """Two readings of the same gap, both along the frontier:

      cost at matched adequacy   -- what the DP spends to reach THIS design's styRR
      adequacy at matched cost   -- what styRR the DP reaches on THIS design's budget

    Both are interpolations, so a design outside the frontier's swept range gets
    None rather than a silently extrapolated number.
    """
    o = np.argsort(fr["cost"])
    cost, sty = fr["cost"][o], fr["sty"][o]
    dp_cost = (float(np.interp(m["sty"], sty, cost))
               if sty.min() <= m["sty"] <= sty.max() else None)
    dp_sty = (float(np.interp(m["cost"], cost, sty))
              if cost.min() <= m["cost"] <= cost.max() else None)
    return dp_cost, dp_sty


# ============ 1. the table ============
# The frontier is DOUBLY CENSORED by the contribution band, so a uniform lambda grid
# wastes most of its points. Measured at the committed calibration:
#     lambda <= 0.15  pinned at the band FLOOR    (cost 0.0750, styRR 0.507)
#     lambda >= 0.60  pinned at the band CEILING  (cost 0.5606, styRR 0.981)
# Everything that varies happens in between, and every market design lands at a cost
# of 0.11-0.30, i.e. around lambda 0.28-0.48. A uniform 0.1-0.95 grid put 7 of 10
# points on the two flat stretches and left 4 to cover the whole informative range.
# The grid below is concentrated there instead, with 0.15 and 0.60 kept as brackets.
# Interpolation between the points CHORDS a concave frontier, so every matched figure
# stays conservative: the optimum is at least this much better, never less.
FRONTIER_LAMBDAS = (0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42,
                    0.45, 0.48, 0.51, 0.54, 0.57, 0.60)


def table(lambdas=FRONTIER_LAMBDAS, **g):
    c.ensure_out(OUT)
    gg = {**GRID, **g}
    Fg, rg, _ = c.grids(gg["nF"], gg["nR"])
    print(f"grid {gg}   {N_PATHS} paths   seed {SEED}   entry = new-plan cohort")

    hdr = "  %-24s %8s %8s %9s %8s %8s %7s %7s" % (
        "design", "styRR", "leaRR", "cost", "joint", "avg%", "early", "late")
    fmt = "  %-24s %8.3f %8.3f %9.4f %8.4f %7.1f%% %7.1f %7.1f"
    print("\n  --- market designs (contribution set by the calendar) ---")
    print(hdr)
    mkt = []
    for lb, rate in DESIGNS:
        m = _run_design(rate, Fg, rg)
        mkt.append((lb, m))
        print(fmt % (lb, m["sty"], m["lea"], m["cost"], m["joint"], m["avg"], m["early"], m["late"]))

    print("\n  --- optimised (contribution set by the state) ---")
    print(hdr)
    dp_rows = [("DP banded %.0f-%.0f%%" % (BAND_PCT[0] * 100, BAND_PCT[1] * 100),
                _run_dp(Fg, rg, banded=True, **g)),
               ("DP unconstrained", _run_dp(Fg, rg, banded=False, **g))]
    for lb, m in dp_rows:
        print(fmt % (lb, m["sty"], m["lea"], m["cost"], m["joint"], m["avg"], m["early"], m["late"]))

    print(f"\n  --- gain from optimising, against the lambda-frontier ({len(lambdas)} points) ---")
    fr = _frontier(Fg, rg, lambdas, **g)
    print("  frontier spans cost %.4f-%.4f, styRR %.3f-%.3f"
          % (fr["cost"].min(), fr["cost"].max(), fr["sty"].min(), fr["sty"].max()))
    print("  %-24s %26s %26s" % ("design", "cost at matched adequacy", "adequacy at matched cost"))
    for lb, m in mkt:
        dp_cost, dp_sty = _matched(fr, m)
        a = ("%.4f vs %.4f  (%+.0f%%)" % (dp_cost, m["cost"], 100 * (dp_cost / m["cost"] - 1))
             if dp_cost is not None else "outside frontier range")
        b = ("%.3f vs %.3f  (%+.3f)" % (dp_sty, m["sty"], dp_sty - m["sty"])
             if dp_sty is not None else "outside frontier range")
        print("  %-24s %26s %26s" % (lb, a, b))
    print("\n  Read: a NEGATIVE cost figure is what the employer saves by optimising at the same")
    print("  adequacy; a POSITIVE adequacy figure is the extra replacement rate bought at the")
    print("  same budget. Both are one protocol, one entry cohort, one seed -- no error bars yet.")
    return mkt, dp_rows, fr


# ============ 2. the frontier with the designs on it ============
def frontier(lambdas=FRONTIER_LAMBDAS, **g):
    c.ensure_out(OUT)
    gg = {**GRID, **g}
    Fg, rg, _ = c.grids(gg["nF"], gg["nR"])
    fr = _frontier(Fg, rg, lambdas, **g)
    mkt = [(lb, _run_design(rate, Fg, rg)) for lb, rate in DESIGNS]
    for lb, m in mkt:
        print("  %-24s cost %.4f  styRR %.3f  benefit %+.4f" % (lb, m["cost"], m["sty"], m["benefit"]))

    mk = ["o", "s", "^", "D", "v"]
    cols = c.plt.cm.tab10(np.linspace(0, 0.9, len(mkt)))
    fig, (axL, axR) = c.plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axL.plot(fr["cost"], fr["benefit"], "-o", color="#1D9E75", lw=1.9, ms=5, zorder=3,
             label=r"optimised frontier (swept $\lambda$)")
    shown = []      # skip a label whose point coincides with one already drawn: on the
    for k in (0, len(lambdas) // 3, 2 * len(lambdas) // 3, len(lambdas) - 1):
        if any(abs(fr["cost"][k] / fr["cost"][j] - 1) < 0.02 for j in shown):
            continue                     # censored stretch several lambdas share one point
        shown.append(k)
        axL.annotate(rf"$\lambda={fr['lam'][k]:.2f}$", (fr["cost"][k], fr["benefit"][k]),
                     textcoords="offset points", xytext=(6, -12), fontsize=8.5, color="#146c50")
    for (lb, m), col, s in zip(mkt, cols, mk):
        axL.scatter([m["cost"]], [m["benefit"]], marker=s, s=62, color=col, zorder=4, label=lb)
    axL.set_xscale("log")
    axL.set_xlabel("employer cost per unit final salary (log; cheaper $\\leftarrow$)")
    axL.set_ylabel(r"employee value  $\mathbb{E}[u(\mathrm{RR})]$  (better $\uparrow$)")
    axL.set_title("Market designs against the optimised frontier")
    axL.legend(frameon=False, fontsize=8, loc="lower right"); axL.grid(True, which="both", alpha=0.22, lw=0.6)

    axR.plot(fr["cost"], fr["sty"], "-o", color="#1D9E75", lw=1.9, ms=5, zorder=3, label="optimised")
    for (lb, m), col, s in zip(mkt, cols, mk):
        axR.scatter([m["cost"]], [m["sty"]], marker=s, s=62, color=col, zorder=4, label=lb)
        _, dp_sty = _matched(fr, m)
        if dp_sty is not None:                 # vertical gap = adequacy bought at the same budget
            axR.annotate("", xy=(m["cost"], dp_sty), xytext=(m["cost"], m["sty"]),
                         arrowprops=dict(arrowstyle="->", color=col, lw=1.1, alpha=0.8))
    axR.axhline(c.dp.RR_TARGET, color="#C1121F", ls="--", lw=1.1, label=f"target {c.dp.RR_TARGET}")
    axR.axhline(c.dp.RR_LEGAL, color="#9aa0a6", ls=":", lw=1.0, label=f"legal {c.dp.RR_LEGAL}")
    axR.set_xscale("log"); axR.set_xlabel("employer cost per unit final salary (log)")
    axR.set_ylabel("stayer replacement rate")
    axR.set_title("Adequacy at the same budget (arrow = the gap)")
    axR.legend(frameon=False, fontsize=8, loc="lower right"); axR.grid(True, which="both", alpha=0.22, lw=0.6)
    fig.suptitle("How far inside the frontier a calendar-driven design sits", fontsize=12)
    fig.savefig(f"{OUT}/benchmark_frontier.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {OUT}/benchmark_frontier.png")


# ============ 3. the shapes ============
def schedules(**g):
    c.ensure_out(OUT)
    gg = {**GRID, **g}
    Fg, rg, _ = c.grids(gg["nF"], gg["nR"])
    yrs = np.arange(c.dp.T)
    dp_b = _run_dp(Fg, rg, banded=True, **g)
    fig, ax = c.plt.subplots(figsize=(7.6, 4.9), constrained_layout=True)
    cols = c.plt.cm.tab10(np.linspace(0, 0.9, len(DESIGNS)))
    for (lb, rate), col in zip(DESIGNS, cols):
        m = _run_design(rate, Fg, rg)
        ax.plot(yrs, m["c_by"], lw=1.8, color=col, label=f"{lb}  (styRR {m['sty']:.2f})")
    ax.plot(yrs, dp_b["c_by"], lw=3.4, color="white", zorder=4)
    ax.plot(yrs, dp_b["c_by"], lw=2.3, color="#111111", ls=(0, (5, 2)), zorder=5,
            label=f"OPTIMISED, banded  (styRR {dp_b['sty']:.2f})")
    ax.set_xlabel("career year $t$"); ax.set_ylabel("contribution (% of salary)")
    ax.set_title("Calendar-driven designs vs the state-driven optimum")
    ax.legend(frameon=False, fontsize=8.5); ax.grid(True, alpha=0.25, lw=0.6)
    fig.savefig(f"{OUT}/benchmark_schedules.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {OUT}/benchmark_schedules.png")


_ALL = {"table": table, "frontier": frontier, "schedules": schedules}


def main(which=None):
    c.ensure_out(OUT)
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
