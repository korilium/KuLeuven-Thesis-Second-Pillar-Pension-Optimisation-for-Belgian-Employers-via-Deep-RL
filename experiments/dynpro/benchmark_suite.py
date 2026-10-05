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
  visitation benchmark_visitation.png  where each plan drives the (F, rho) state
             benchmark_misfunding.png  a_design - a* on the states each design visits

Run:  python benchmark_suite.py [table|frontier|schedules|visitation]
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


def _run_design(rate_of_t, Fg, rg, visits=False):
    """A market design: contribution set by the calendar, not by the state.

    `visits=True` additionally returns the policy array and the (T, NF, NR) occupancy
    counts, so visitation() can reuse this construction rather than rebuilding it."""
    a_of_t = lambda t: min(rate_of_t(t) / c.P.GAMMA, 1.0)
    pol = c.schedule_policy(a_of_t, len(Fg), len(rg))
    r = c.simulate(pol, Fg, rg, **c.entry(N_PATHS, SEED), n_paths=N_PATHS, seed=SEED,
                   visits=visits)
    m = _metrics(r)
    if visits:
        m.update(pol=pol, visits=r["visits"], frac=r["frac"])
    return m


def _run_dp(Fg, rg, banded=True, lam=None, visits=False, **g):
    """The optimised schedule, banded to the protocol band or unconstrained.

    NOTE on what `pol` means when `visits=True`: simulate() CLIPS the looked-up action
    to the band, so the policy array alone is not what was applied. The banded action
    grid already lies inside the band, so for the banded case the two coincide; this is
    why visitation() compares against the banded optimum and not the unconstrained one."""
    gg = {**GRID, **g}
    with c.overrides(**({} if lam is None else dict(LAMBDA=float(lam)))):
        if banded:
            lo, hi = BAND_PCT[0] / c.P.GAMMA, min(BAND_PCT[1] / c.P.GAMMA, 1.0)
            ag, band = np.linspace(lo, hi, gg["na"]), (lo, hi)
        else:
            ag, band = c.dp.make_a_grid(n=gg["na"]), None
        pol = c.solve(Fg, rg, ag, gg["nq"])["policy"]
        r = c.simulate(pol, Fg, rg, **c.entry(N_PATHS, SEED), band=band,
                       n_paths=N_PATHS, seed=SEED, visits=visits)
    m = _metrics(r)
    if visits:
        m.update(pol=pol, visits=r["visits"], frac=r["frac"])
    return m


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
# wastes most of its points. Measured under the canonical objective (retirement-date
# money, DRIFT_CORRECTION, constant rates; protocol grid, 15000 paths; REVIEW M13):
#     lambda <= 0.30  pinned at the band FLOOR    (cost 0.3855, styRR 0.506, 2.0% of salary)
#     lambda >= 0.72  pinned at the band CEILING  (cost 2.8766, styRR 0.974, 15.0%)
# with the interior between (lambda 0.40: 3.2% of salary, 0.50: 7.3%, 0.60: 12.6%).
# (Under the old discounted objective the same stretch was lambda 0.15-0.60.)
# The grid below covers the interior, with 0.30 and 0.72 kept as brackets.
# Interpolation between the points CHORDS a concave frontier, so every matched figure
# stays conservative: the optimum is at least this much better, never less.
FRONTIER_LAMBDAS = (0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 0.57,
                    0.60, 0.63, 0.66, 0.69, 0.72)


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
    axR.axhline(c.P.RR_TARGET, color="#C1121F", ls="--", lw=1.1, label=f"target {c.P.RR_TARGET}")
    axR.axhline(c.P.RR_LEGAL, color="#9aa0a6", ls=":", lw=1.0, label=f"legal {c.P.RR_LEGAL}")
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
    yrs = np.arange(c.P.T)
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


# ============ 4. where each design drives the plan, and where it misfunds ============
MAP_DESIGNS = ["flat 5% of salary", "age scale 3% +1%/10y"]

# Occupancy is a DENSITY, and the protocol's 73x71 is coarse for one, so this section
# defaults to the finer grid state_visitation() uses. Nothing here is a cross-design
# VALUE comparison, so the difference from the protocol grid is a presentation choice --
# but a* is resolution-sensitive, so the scalars are checked across both (see the
# docstring note on grid stability).
MAP_GRID = dict(nF=145, nR=101, na=20, nq=5)


def _weighted(pol, V):
    """Visit-weighted mean of a policy, weighted JOINTLY over (year, cell).

    Pooling over years first would decouple the policy's time variation from the
    occupancy's -- both are strong here, so that average would answer no question."""
    tot = V.sum()
    return float((pol * V).sum() / tot) if tot else np.nan


def _contours(ax, d, Fg, rg, color="white"):
    """That year's occupancy at the 50/90/99th percentiles, with the degenerate case.

    Early on essentially all mass sits in one cell, so the quantiles coincide and
    contour() has nothing to draw; mark the modal cell instead."""
    pos = d[d > 0]
    if not pos.size:
        return
    levels = sorted({float(np.quantile(pos, q)) for q in (0.50, 0.90, 0.99)})
    if len(levels) > 1:
        ax.contour(Fg, rg, d.T, levels=levels, colors=color, linewidths=1.2)
    else:
        jF, jR = np.unravel_index(int(d.argmax()), d.shape)
        ax.plot(Fg[jF], rg[jR], "o", ms=9, mfc="none", mec=color, mew=1.8)


def visitation(designs=MAP_DESIGNS, years=(0, 10, 22, 44), **g):
    """Map the market designs onto the state space, and localise where they misfund.

    Two questions, two figures.

    WHERE THE PLAN GOES. A design that funds less keeps F lower, and keeps rho = S/L
    higher because L grows more slowly, so each design occupies a different part of
    (F, rho) than the optimum does. Figure A puts the three occupancy clouds side by
    side.

    WHERE IT MISFUNDS. For a calendar-driven design the policy surface is CONSTANT over
    (F, rho), so plotting it would colour a flat sheet. The informative quantity is
        Delta(t, F, rho) = a_design(t) - a*(t, F, rho)
    on the cells the design actually visits: what it contributes minus what the optimum
    would contribute THERE. Positive = over-funds that state, negative = under-funds it.

    OFF-POLICY CAVEAT. Delta reads a* at states the optimum's own simulation may rarely
    reach, because the design steers the plan elsewhere. a* is defined on the whole grid
    so this is legitimate, but it is extrapolation to the extent the two occupancies
    differ -- so the share of the design's mass landing in cells the optimum never
    visits is reported, and a large share means the map is not a clean attribution.
    Measured, that share is ~1%: the optimum's occupancy is a union over 45 years and
    covers nearly everything the designs reach, so the caveat turns out to be minor here.

    GRID SENSITIVITY. a* is resolution-sensitive, so the scalars move with the grid.
    Measured at 73x71 against the default 145x101: mean misfunding differs by 0.012
    (flat 5%) and 0.022 (age scale), a few percent of the value, with the sign and
    ordering unchanged; the state step is identical (+0.240) on both. The LEAST stable
    figure is the over-funded share (age scale: 33% at 145x101, 25% at 73x71), so read
    it as "roughly a third" rather than as a point estimate.
    """
    c.ensure_out(OUT)
    gg = {**MAP_GRID, **g}
    Fg, rg, _ = c.grids(gg["nF"], gg["nR"])
    by_label = dict(DESIGNS)
    missing = [d for d in designs if d not in by_label]
    assert not missing, f"not in DESIGNS: {missing}"

    star = _run_dp(Fg, rg, banded=True, visits=True, **gg)
    Vs = star["visits"]
    reach_star = Vs.sum(axis=0) > 0                 # cells the optimum ever occupies
    rows = [(lb, _run_design(by_label[lb], Fg, rg, visits=True)) for lb in designs]

    print(f"[visitation] {N_PATHS} paths on {gg['nF']}x{gg['nR']} "
          f"({gg['nF']*gg['nR']} cells), banded optimum as reference")
    # The visit weighting does TWO things at once, and they are worth separating:
    #   grid-mean -> year-weighted : re-weights YEARS by survival (churn thins later years)
    #   year-weighted -> visited   : re-weights STATES within each year
    # The second step is identically zero for any calendar design, because its policy does
    # not vary over (F, rho). So the optimum's state step is what state-dependence buys,
    # and it is not contaminated by the survival effect the designs also have.
    print("  %-24s %9s %9s %8s %10s %9s %9s %9s" % (
        "policy", "grid-mean", "yr-wtd", "visited", "a* there", "misfund",
        "over-fund", "off-policy"))
    stats = []
    for lb, m in [("OPTIMUM (banded)", star)] + rows:
        V, pol = m["visits"], m["pol"]
        w = V.sum(axis=(1, 2))                       # present path-years per year
        gm = float(pol.mean())
        ym = float((w * pol.mean(axis=(1, 2))).sum() / w.sum())
        vw = _weighted(pol, V)
        if lb.startswith("OPTIMUM"):
            star_steps = (gm, ym, vw)
            print("  %-24s %9.3f %9.3f %8.3f %10s %9s %9s %9s"
                  % (lb, gm, ym, vw, "-", "-", "-", "-"))
            continue
        vw_star = _weighted(star["pol"], V)          # a* on THIS design's occupancy
        D = pol - star["pol"]
        tot = V.sum()
        over = float((V * (D > 0)).sum() / tot)
        off = float((V * ~reach_star[None, :, :]).sum() / tot)
        stats.append((lb, m, D, gm, ym, vw, vw_star, over, off))
        print("  %-24s %9.3f %9.3f %8.3f %10.3f %+9.3f %8.0f%% %8.0f%%"
              % (lb, gm, ym, vw, vw_star, vw - vw_star, 100 * over, 100 * off))

    # Known-answer checks on the new code, both exact rather than eyeballed.
    for lb, m, _D, _gm, ym, vw, *_ in stats:
        # (1) a calendar design is constant over (F, rho), so the STATE step must vanish
        assert abs(ym - vw) < 1e-9, f"{lb}: state step must be 0 for a calendar design"
        # (2) the visit histogram must reproduce the independently-computed per-year
        #     present means: avg = sum_t c_by*frac / sum_t frac, in percent of salary
        assert abs(vw - m["avg"] / (100 * c.P.GAMMA)) < 1e-9, f"{lb}: avg cross-check"
    g0, y0, v0 = star_steps
    print(f"  year step (survival) {g0:+.3f} -> {y0:+.3f};  state step {y0:+.3f} -> {v0:+.3f}"
          f"  = {v0 - y0:+.3f} for the optimum, 0.000 for every calendar design")

    # ---- figure A: where each design drives the plan ----
    panels = [(lb, m["visits"].sum(axis=0)) for lb, m, *_ in stats]
    panels.append(("OPTIMUM (banded)", Vs.sum(axis=0)))
    fig, axes = c.plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.8),
                               sharey=True, constrained_layout=True)
    mm = None
    allpos = np.concatenate([d[d > 0].ravel() for _lb, d in panels])
    norm = c.mpl.colors.LogNorm(vmin=max(allpos.min(), 1), vmax=allpos.max())
    for ax, (lb, d) in zip(np.atleast_1d(axes), panels):
        mm = ax.pcolormesh(Fg, rg, np.where(d > 0, d, np.nan).T, cmap="magma",
                           norm=norm, shading="auto", rasterized=True)
        ax.set_yscale("log"); ax.set_xlim(Fg[0], Fg[-1]); ax.set_ylim(rg[0], rg[-1])
        ax.axvline(1.0, color="white", lw=0.9, ls=":", alpha=0.7)
        ax.set_xlabel(r"$F=R/L$")
        ax.set_title(f"{lb}\n{float(np.mean(d > 0)):.1%} of cells ever visited", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel(r"$\rho=S/L$ (log)")
    fig.colorbar(mm, ax=axes, shrink=0.9, pad=0.015).set_label("present path-years (log)")
    fig.suptitle("Where each plan drives the state (shared colour scale)", fontsize=12)
    fig.savefig(f"{OUT}/benchmark_visitation.png", dpi=c.DPI); c.plt.close(fig)
    print(f"  wrote {OUT}/benchmark_visitation.png")

    # ---- figure B: the misfunding map ----
    lim = max(float(np.nanmax(np.abs(np.where(m["visits"] > 0, D, np.nan))))
              for _lb, m, D, *_ in stats)
    # Careers occupy a thin ribbon of the grid, so plotting the full extent would leave
    # every panel mostly blank and hide the variation of Delta ACROSS F, which is the
    # part that is actually about the state. Crop to the cells shown, with a margin.
    shown = np.zeros((len(Fg), len(rg)), bool)
    for _lb, m, *_ in stats:
        shown |= m["visits"][list(years)].sum(axis=0) > 0
    iF, iR = np.where(shown)
    xlo, xhi = Fg[iF.min()], Fg[iF.max()]
    ylo, yhi = rg[iR.min()], rg[iR.max()]
    xpad = 0.05 * (xhi - xlo)
    xlo, xhi = max(Fg[0], xlo - xpad), min(Fg[-1], xhi + xpad)
    ylo, yhi = ylo / 1.6, yhi * 1.6                       # log axis: pad multiplicatively
    fig, axes = c.plt.subplots(len(stats), len(years),
                               figsize=(3.3 * len(years), 3.9 * len(stats)),
                               sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    mm = None
    for i, (lb, m, D, _gm, _ym, _vw, _vws, _ov, _off) in enumerate(stats):
        V = m["visits"]
        for j, t in enumerate(years):
            ax = axes[i, j]
            d = V[t]
            mm = ax.pcolormesh(Fg, rg, np.where(d > 0, D[t], np.nan).T, cmap="RdBu_r",
                               vmin=-lim, vmax=lim, shading="auto", rasterized=True)
            _contours(ax, d, Fg, rg, color="#222222")
            ax.set_yscale("log"); ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.axvline(1.0, color="#222222", lw=0.9, ls=":", alpha=0.6)
            w = _weighted(D[t:t + 1], d[None, :, :])
            ax.set_title(f"$t={t}$   mean $\\Delta$ {w:+.2f}", fontsize=9.5)
            if i == len(stats) - 1: ax.set_xlabel(r"$F=R/L$")
        axes[i, 0].set_ylabel(f"{lb}\n" + r"$\rho=S/L$ (log)", fontsize=9)
    fig.colorbar(mm, ax=axes, shrink=0.85, pad=0.015).set_label(
        r"$\Delta = a_{\rm design} - a^\star$   (red = over-funds, blue = under-funds)")
    fig.suptitle("Where each design departs from the optimum, on the states it actually visits\n"
                 "(blank = never visited; contours = 50/90/99th pct of that year's occupancy; "
                 "$a^\\star$ is read off-policy)", fontsize=11)
    fig.savefig(f"{OUT}/benchmark_misfunding.png", dpi=c.DPI); c.plt.close(fig)
    print(f"  wrote {OUT}/benchmark_misfunding.png")
    return stats


_ALL = {"table": table, "frontier": frontier, "schedules": schedules,
        "visitation": visitation}


def main(which=None):
    c.ensure_out(OUT)
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
