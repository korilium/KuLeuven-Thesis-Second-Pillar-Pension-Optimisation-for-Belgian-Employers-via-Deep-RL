"""Lambda-dial suite: how the employer/employee negotiation weight lambda
trades employer cost off against employee adequacy under the committed model
-- the funding threshold set by the legal floor, the Pareto frontier itself,
crowding-out by the legal pillar, and the frontier's robustness to the
plan-entry-state assumption.

Figures (-> figs/lambda/):
  lambda_threshold          lambda_threshold.png   grid-mean & path-weighted a* vs lambda
  pareto_frontier           lambda_frontier.png,    employer cost vs employee benefit, swept lambda,
                            lambda_outcomes.png     with constant-a plans as reference points
  crowding_out              crowding.png            a* against the first-pillar floor, and against lambda
  frontier_entry_robustness lambda_frontier_dist.png the frontier under two entry-state assumptions

Every lambda sweep here stops around 0.62. Above that the optimum is pinned at the
top of the action grid, so the curves are flat and a wider grid only resolves a
horizontal line -- measured: path-weighted a* is 0.349 at every lambda from 0.60
to 0.80, and employer cost 0.559 at 0.6 against 0.560 at 0.8.

Run:  python lambda_dial_suite.py [threshold|pareto|crowding|entrydist]
"""
import numpy as np
import common as c

OUT = f"{c.OUT}/lambda"   # this suite writes only here


# ============ 1. funding threshold in lambda (legal baseline) ============
@c.restores
def lambda_threshold(lambdas=np.linspace(0.15, 0.62, 16), nF=121, nR=91, na=25, nq=5,
                     n_paths=15000, seed=7):
    """Optimal funding against the employee weight.

    The sweep stops at 0.62 on purpose. Above lambda ~= 0.6 the optimum is pinned
    at the top of the action grid, so path-weighted a* is flat (measured: 0.349 at
    every lambda from 0.60 to 0.80) and a uniform 0.2-0.8 grid spent its top third
    resolving a horizontal line."""
    Fg, rg, _ = c.grids(nF, nR, na)
    rng = np.random.default_rng(seed); R0, L0, S0 = c.new_plan_init(n_paths, rng)
    gm, pw = [], []
    for lam in lambdas:
        c.restore(); c.update(LAMBDA=float(lam))
        pol = c.solve(Fg, rg, c.dp.make_a_grid(n=na), nq)["policy"]
        r = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n_paths, seed=seed)
        gm.append(pol.mean()); pw.append(r["mean_a"])
        print(f"  lambda={lam:.2f}: grid-mean a*={pol.mean():.3f}  path a*={r['mean_a']:.3f}")
    c.restore()
    fig, ax = c.plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    ax.plot(lambdas, gm, "-o", color="#1D9E75", lw=1.8, ms=4, label=r"grid-mean $a^\star$")
    ax.plot(lambdas, pw, "-s", color="#274690", lw=1.5, ms=4, label=r"path-weighted $a^\star$")
    ax.set_xlabel(r"employee weight $\lambda$"); ax.set_ylabel(r"optimal funding $a^\star$")
    # Report what was measured. The earlier title claimed a THRESHOLD; the curve is
    # a smooth, near-linear rise to the action ceiling, with no kink anywhere.
    # Keep the subtitle short enough to fit the axes: the long version ran off both
    # edges. The legal-floor effect belongs to crowding(), which actually varies it.
    ax.set_title(r"Optimal funding rises smoothly in $\lambda$, then saturates"
                 "\n"
                 rf"grid-mean {gm[0]:.2f}$\to${gm[-1]:.2f},  "
                 rf"path-weighted {pw[0]:.2f}$\to${pw[-1]:.2f}", fontsize=10)
    ax.legend(frameon=False); ax.grid(True, alpha=0.25, lw=0.6)
    fig.savefig(f"{OUT}/lambda_threshold.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {OUT}/lambda_threshold.png")


# ============ 2. Pareto frontier + naive-plan reference points ============
@c.restores
def pareto_frontier(lambdas=np.array([0.10, 0.18, 0.24, 0.30, 0.34, 0.38, 0.42,
                                      0.46, 0.50, 0.54, 0.58, 0.65]),
                    naive_a=(0.2, 0.5, 1.0), nF=121, nR=91, na=31, nq=5,
                    n_paths=30000, seed=7):
    """Employer cost against employee value, traced by sweeping lambda.

    Both axes are lambda-FREE -- simulate's `benefit` and `cost` contain no LAMBDA
    -- so sweeping it traces the locus of optima rather than re-scoring one policy.

    The grid is concentrated on 0.10-0.65 because the frontier is censored at both
    ends: below ~0.15 the optimum funds nothing, and from ~0.6 it is pinned at the
    top of the action grid (measured cost 0.559 at lambda=0.6 against 0.560 at
    0.8). A uniform 0.05-0.98 grid put half its points in those two clumps, and the
    annotations then overprinted inside them."""
    Fg, rg, ag = c.grids(nF, nR, na)
    fr_ben, fr_cost, fr_avg, fr_sty, fr_lea = [], [], [], [], []
    for lam in lambdas:
        c.restore(); c.update(LAMBDA=float(lam))
        pol = c.solve(Fg, rg, ag, nq)["policy"]
        r = c.simulate(pol, Fg, rg, **c.entry(n_paths, seed), n_paths=n_paths, seed=seed)
        fr_ben.append(r["benefit"]); fr_cost.append(r["cost"])
        # r["avg"], not r["mean_a"]: mean_a divides the summed action by T while
        # zeroing absent years, so churn dilutes it (~85% of path-years are absent
        # by T) and it read ~5% where the career average is ~10%. avg is the
        # present-only, survival-weighted career average -- what the axis claims.
        fr_avg.append(r["avg"]); fr_sty.append(r["sty"]); fr_lea.append(r["lea"])
        print(f"  lambda={lam:.2f}: benefit={r['benefit']:+.4f}  cost={r['cost']:.4f}  stayerRR={r['sty']:.3f}")
    c.restore()

    nv_ben, nv_cost = [], []
    for a_c in naive_a:
        pol = c.const_policy(a_c, c.P.T, len(Fg), len(rg))
        r = c.simulate(pol, Fg, rg, **c.entry(n_paths, seed), n_paths=n_paths, seed=seed)
        nv_ben.append(r["benefit"]); nv_cost.append(r["cost"])
        print(f"  naive a={a_c:.1f}: benefit={r['benefit']:+.4f}  cost={r['cost']:.4f}")

    fig, ax = c.plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    ax.plot(fr_cost, fr_ben, "-o", color="#1D9E75", lw=1.8, ms=5, zorder=3, label=r"optimal frontier (swept $\lambda$)")
    n_lam = len(lambdas)
    shown = []          # skip a label whose point coincides with one already drawn
    for k in (0, n_lam // 3, 2 * n_lam // 3, n_lam - 1):
        if any(abs(fr_cost[k] / fr_cost[j] - 1) < 0.02 for j in shown):
            continue
        shown.append(k)
        ax.annotate(rf"$\lambda={lambdas[k]:.2f}$", (fr_cost[k], fr_ben[k]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8.5, color="#146c50")
    ax.scatter(nv_cost, nv_ben, marker="s", s=60, color="#C1121F", zorder=4, label="constant-$a$ plans (reference)")
    for a_c, xc, yb in zip(naive_a, nv_cost, nv_ben):
        # to the LEFT: the frontier labels go right, and at a=0.2 the two collided
        ax.annotate(rf"$a={a_c:.1f}$", (xc, yb), textcoords="offset points",
                    xytext=(-30, 4), fontsize=9, color="#8a0d16")
    ax.set_xscale("log")
    ax.set_xlabel("employer cost per unit final salary (log; cheaper <-)")
    ax.set_ylabel(r"employee value  $\mathbb{E}[u(\mathrm{RR})]$  (better up)")
    # State the measured gap rather than asserting the conclusion: for each constant
    # plan, how much more it costs than the frontier at the same employee value.
    gaps = []
    for xc, yb in zip(nv_cost, nv_ben):
        if min(fr_ben) <= yb <= max(fr_ben):
            o = np.argsort(fr_ben)
            gaps.append(xc / float(np.interp(yb, np.array(fr_ben)[o],
                                             np.array(fr_cost)[o])) - 1)
    tag = (f"constant plans cost {100*min(gaps):.0f}-{100*max(gaps):.0f}% more than the "
           "frontier at equal employee value" if gaps else
           "constant plans fall outside the swept frontier")
    ax.set_title(r"Employer/employee frontier traced by the dial $\lambda$" + "\n" + tag,
                 fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9); ax.grid(True, which="both", alpha=0.22, lw=0.6)
    fig.savefig(f"{OUT}/lambda_frontier.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {OUT}/lambda_frontier.png")

    fig, ax = c.plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(lambdas, fr_avg, "-o", color="#146c50", lw=2, label="career-avg contrib %")
    ax2 = ax.twinx()
    ax2.plot(lambdas, fr_sty, "-s", color="#274690", lw=1.6, label="stayer RR")
    ax2.plot(lambdas, fr_lea, "-^", color="#E08D1C", lw=1.4, label="leaver RR")
    ax2.axhline(c.P.RR_TARGET, color="#274690", ls=":", lw=1, alpha=0.6); ax2.spines["top"].set_visible(False)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("career-avg contrib %", color="#146c50")
    ax2.set_ylabel("replacement rate", color="#274690")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=8)
    ax.set_title("Contribution & adequacy vs lambda")
    fig.savefig(f"{OUT}/lambda_outcomes.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {OUT}/lambda_outcomes.png")


# ============ 3. crowding-out by the legal floor ============
@c.restores
def crowding_out(RLs=np.array([0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60]),
                 lambdas=np.array([0.5, 0.65, 0.75, 0.80, 0.85, 0.90, 0.97]),
                 nF=121, nR=91, na=25, nq=5):
    Fg, rg, ag = c.grids(nF, nR, na)
    a_rl = []
    for rl in RLs:
        c.restore(); c.update(RR_LEGAL=float(rl))
        a_rl.append(float(c.solve(Fg, rg, ag, nq)["policy"].mean()))
        print(f"  RR_legal={rl:.2f}: grid-mean a*={a_rl[-1]:.3f}")
    c.restore()
    a_lam = []
    for lam in lambdas:
        c.restore(); c.update(LAMBDA=float(lam))
        a_lam.append(float(c.solve(Fg, rg, ag, nq)["policy"].mean()))
        print(f"  lambda={lam:.2f}: grid-mean a*={a_lam[-1]:.3f}")
    c.restore()
    fig, (a1, a2) = c.plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    a1.plot(RLs, a_rl, "-o", color="#C1121F", lw=1.8, ms=5)
    a1.axvline(getattr(c.BASE, "RR_LEGAL"), color="#274690", ls="--", lw=1.2,
               label=rf'committed $RR_{{legal}}$ = {getattr(c.BASE, "RR_LEGAL"):.2f}')
    a1.set_xlabel(r"first-pillar (legal) replacement $RR_{legal}$"); a1.set_ylabel(r"grid-mean $a^\star$")
    d_rl = (a_rl[-1] - a_rl[0]) / (RLs[-1] - RLs[0]) if len(RLs) > 1 else float("nan")
    a1.set_title(r"$a^\star$ vs the first pillar ($\lambda=%.2f$)" % getattr(c.BASE, "LAMBDA")
                 + "\n" + rf"slope {d_rl:+.2f} per unit $RR_{{legal}}$; "
                 rf"{a_rl[0]:.2f}$\to${a_rl[-1]:.2f}", fontsize=10)
    a1.legend(frameon=False, fontsize=9); a1.grid(True, alpha=0.25, lw=0.6)
    a2.plot(lambdas, a_lam, "-s", color="#1D9E75", lw=1.8, ms=5)
    a2.axvline(0.5, color="#9aa0a6", ls=":", lw=1.2, label=r"equal weight $\lambda=0.5$")
    a2.set_xlabel(r"employee weight $\lambda$"); a2.set_ylabel(r"grid-mean $a^\star$")
    a2.set_title(rf"$a^\star$ vs employee weight ($RR_{{legal}}={c.P.RR_LEGAL}$)"
                 + "\n" + rf"{a_lam[0]:.2f}$\to${a_lam[-1]:.2f} over "
                 rf"$\lambda$={lambdas[0]:.2f}-{lambdas[-1]:.2f}", fontsize=10)
    a2.legend(frameon=False, fontsize=9); a2.grid(True, alpha=0.25, lw=0.6)
    fig.suptitle("What moves optimal funding: the first-pillar floor (left) and the "
                 "employee weight (right)", fontsize=12)
    fig.savefig(f"{OUT}/crowding.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {OUT}/crowding.png")


# ============ 4. frontier robustness to the entry-state assumption ============
@c.restores
def frontier_entry_robustness(lambdas=np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.62]),
                              nF=121, nR=91, na=25, nq=5, n_paths=30000, seed=7):
    print("[frontier_entry_robustness] new-plan cohort vs the placeholder entry spread")
    Fg, rg, ag = c.grids(nF, nR, na)
    c_cost, c_ben, d_cost, d_ben = [], [], [], []
    for lam in lambdas:
        c.restore(); c.update(LAMBDA=float(lam))
        pol = c.solve(Fg, rg, ag, nq)["policy"]
        rc = c.simulate(pol, Fg, rg, **c.entry(n_paths, seed), n_paths=n_paths, seed=seed)
        rng = np.random.default_rng(seed + 1); R0, L0, S0 = c.sample_entry(rng, n_paths)
        rd = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n_paths, seed=seed)
        c_cost.append(rc["cost"]); c_ben.append(rc["benefit"])
        d_cost.append(rd["cost"]); d_ben.append(rd["benefit"])
        print(f"   lambda={lam:.2f}: corner (cost {rc['cost']:.3f}, ben {rc['benefit']:+.3f}) | "
              f"dist (cost {rd['cost']:.3f}, ben {rd['benefit']:+.3f})")
    c.restore()
    fig, ax = c.plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    # NOT a single corner any more: c.entry() is the new-plan cohort. The old label
    # predated that change and made this read as corner-vs-distribution, which it is not.
    ax.plot(c_cost, c_ben, "-o", color="#274690", lw=1.7, ms=5,
            label="new-plan cohort (the standard entry)")
    ax.plot(d_cost, d_ben, "-s", color="#1D9E75", lw=1.7, ms=5, label="entry distribution (placeholder)")
    ax.set_xscale("log")
    ax.set_xlabel("employer cost per unit final salary (log; cheaper <-)")
    ax.set_ylabel(r"employee value  $\mathbb{E}[u(\mathrm{RR})]$  (better up)")
    ax.set_title("Frontier robustness: two entry-state assumptions\n"
                 "both are distributions; the placeholder spread is uncalibrated",
                 fontsize=10)
    ax.legend(frameon=False, loc="lower right"); ax.grid(True, which="both", alpha=0.22, lw=0.6)
    fig.savefig(f"{OUT}/lambda_frontier_dist.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {OUT}/lambda_frontier_dist.png")


_ALL = {
    "threshold": lambda_threshold,
    "pareto": pareto_frontier,
    "crowding": crowding_out,
    "entrydist": frontier_entry_robustness,
}


def main(which=None):
    c.ensure_out(OUT)
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"--- {name} ---"); fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
