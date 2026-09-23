"""Lambda-dial suite: how the employer/employee negotiation weight lambda
trades employer cost off against employee adequacy under the committed model
-- the funding threshold set by the legal floor, the Pareto frontier itself,
crowding-out by the legal pillar, and the frontier's robustness to the
plan-entry-state assumption.

Figures (-> figs/):
  lambda_threshold          lambda_threshold.png   grid-mean & path-weighted a* vs lambda
  pareto_frontier           lambda_frontier.png,    employer cost vs employee benefit, swept lambda,
                            lambda_outcomes.png     with naive constant-a plans as reference points
  crowding_out              crowding.png            RR_legal sweep and lambda sweep: what crowds out funding
  frontier_entry_robustness lambda_frontier_dist.png single corner anchor vs sampled entry distribution

Run:  python lambda_dial_suite.py [threshold|pareto|crowding|entrydist]
"""
import numpy as np
import common as c


# ============ 1. funding threshold in lambda (legal baseline) ============
def lambda_threshold(lambdas=np.linspace(0.20, 0.80, 16), nF=121, nR=51, na=25, nq=5, n_paths=15000, seed=7):
    Fg, rg, _ = c.grids(nF, nR, na)
    rng = np.random.default_rng(seed); R0, L0, S0 = c.new_plan_init(n_paths, rng)
    gm, pw = [], []
    for lam in lambdas:
        c.restore(); c.dp.LAMBDA = float(lam)
        pol = c.solve(Fg, rg, c.dp.make_a_grid(n=na), nq)["policy"]
        r = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n_paths, seed=seed)
        gm.append(pol.mean()); pw.append(r["mean_a"])
        print(f"  lambda={lam:.2f}: grid-mean a*={pol.mean():.3f}  path a*={r['mean_a']:.3f}")
    c.restore()
    fig, ax = c.plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    ax.plot(lambdas, gm, "-o", color="#1D9E75", lw=1.8, ms=4, label=r"grid-mean $a^\star$")
    ax.plot(lambdas, pw, "-s", color="#274690", lw=1.5, ms=4, label=r"path-weighted $a^\star$")
    ax.set_xlabel(r"employee weight $\lambda$"); ax.set_ylabel(r"optimal funding $a^\star$")
    ax.set_title("Second-pillar funding is a THRESHOLD in lambda\n"
                 f"(legal {c.dp.RR_LEGAL:.0%} floor crowds out funding below a threshold)")
    ax.legend(frameon=False); ax.grid(True, alpha=0.25, lw=0.6)
    fig.savefig(f"{c.OUT}/lambda_threshold.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/lambda_threshold.png")


# ============ 2. Pareto frontier + naive-plan reference points ============
def pareto_frontier(lambdas=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]),
                    naive_a=(0.2, 0.5, 1.0), nF=121, nR=51, na=31, nq=5, n_paths=30000, seed=7):
    Fg, rg, ag = c.grids(nF, nR, na)
    fr_ben, fr_cost, fr_avg, fr_sty, fr_lea = [], [], [], [], []
    for lam in lambdas:
        c.restore(); c.dp.LAMBDA = float(lam)
        pol = c.solve(Fg, rg, ag, nq)["policy"]
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
        fr_ben.append(r["benefit"]); fr_cost.append(r["cost"])
        fr_avg.append(r["mean_a"] * c.dp.GAMMA * 100); fr_sty.append(r["sty"]); fr_lea.append(r["lea"])
        print(f"  lambda={lam:.2f}: benefit={r['benefit']:+.4f}  cost={r['cost']:.4f}  stayerRR={r['sty']:.3f}")
    c.restore()

    nv_ben, nv_cost = [], []
    for a_c in naive_a:
        pol = c.const_policy(a_c, c.dp.T, len(Fg), len(rg))
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
        nv_ben.append(r["benefit"]); nv_cost.append(r["cost"])
        print(f"  naive a={a_c:.1f}: benefit={r['benefit']:+.4f}  cost={r['cost']:.4f}")

    fig, ax = c.plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    ax.plot(fr_cost, fr_ben, "-o", color="#1D9E75", lw=1.8, ms=5, zorder=3, label=r"optimal frontier (swept $\lambda$)")
    n_lam = len(lambdas)
    for k in (0, n_lam // 3, 2 * n_lam // 3, n_lam - 1):
        ax.annotate(rf"$\lambda={lambdas[k]:.2f}$", (fr_cost[k], fr_ben[k]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8.5, color="#146c50")
    ax.scatter(nv_cost, nv_ben, marker="s", s=60, color="#C1121F", zorder=4, label="constant-$a$ plans (reference)")
    for a_c, xc, yb in zip(naive_a, nv_cost, nv_ben):
        ax.annotate(rf"$a={a_c:.1f}$", (xc, yb), textcoords="offset points", xytext=(7, 5), fontsize=9, color="#8a0d16")
    ax.set_xscale("log")
    ax.set_xlabel("employer cost per unit final salary (log; cheaper <-)")
    ax.set_ylabel(r"employee value  $\mathbb{E}[u(\mathrm{RR})]$  (better up)")
    ax.set_title("Employer/employee frontier traced by the negotiation dial lambda\n"
                 "a constant rule pays the frontier's cost only where it happens to match $a^\\star$")
    ax.legend(loc="lower right", frameon=False, fontsize=9); ax.grid(True, which="both", alpha=0.22, lw=0.6)
    fig.savefig(f"{c.OUT}/lambda_frontier.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/lambda_frontier.png")

    fig, ax = c.plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(lambdas, fr_avg, "-o", color="#146c50", lw=2, label="career-avg contrib %")
    ax2 = ax.twinx()
    ax2.plot(lambdas, fr_sty, "-s", color="#274690", lw=1.6, label="stayer RR")
    ax2.plot(lambdas, fr_lea, "-^", color="#E08D1C", lw=1.4, label="leaver RR")
    ax2.axhline(c.dp.RR_TARGET, color="#274690", ls=":", lw=1, alpha=0.6); ax2.spines["top"].set_visible(False)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("career-avg contrib %", color="#146c50")
    ax2.set_ylabel("replacement rate", color="#274690")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=8)
    ax.set_title("Contribution & adequacy vs lambda")
    fig.savefig(f"{c.OUT}/lambda_outcomes.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/lambda_outcomes.png")


# ============ 3. crowding-out by the legal floor ============
def crowding_out(RLs=np.array([0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60]),
                 lambdas=np.array([0.5, 0.65, 0.75, 0.80, 0.85, 0.90, 0.97]),
                 nF=121, nR=51, na=25, nq=5):
    Fg, rg, ag = c.grids(nF, nR, na)
    a_rl = []
    for rl in RLs:
        c.restore(); c.dp.RR_LEGAL = float(rl)
        a_rl.append(float(c.solve(Fg, rg, ag, nq)["policy"].mean()))
        print(f"  RR_legal={rl:.2f}: grid-mean a*={a_rl[-1]:.3f}")
    c.restore()
    a_lam = []
    for lam in lambdas:
        c.restore(); c.dp.LAMBDA = float(lam)
        a_lam.append(float(c.solve(Fg, rg, ag, nq)["policy"].mean()))
        print(f"  lambda={lam:.2f}: grid-mean a*={a_lam[-1]:.3f}")
    c.restore()
    fig, (a1, a2) = c.plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    a1.plot(RLs, a_rl, "-o", color="#C1121F", lw=1.8, ms=5)
    a1.axvline(0.45, color="#274690", ls="--", lw=1.2, label=r"Belgian $RR_{legal}\approx0.45$")
    a1.set_xlabel(r"first-pillar (legal) replacement $RR_{legal}$"); a1.set_ylabel(r"grid-mean $a^\star$")
    a1.set_title(r"Legal floor crowds out the 2nd pillar ($\lambda=0.5$)")
    a1.legend(frameon=False, fontsize=9); a1.grid(True, alpha=0.25, lw=0.6)
    a2.plot(lambdas, a_lam, "-s", color="#1D9E75", lw=1.8, ms=5)
    a2.axvline(0.5, color="#9aa0a6", ls=":", lw=1.2, label=r"equal weight $\lambda=0.5$")
    a2.set_xlabel(r"employee weight $\lambda$"); a2.set_ylabel(r"grid-mean $a^\star$")
    a2.set_title(rf"2nd pillar funds only at high $\lambda$ ($RR_{{legal}}={c.dp.RR_LEGAL}$)")
    a2.legend(frameon=False, fontsize=9); a2.grid(True, alpha=0.25, lw=0.6)
    fig.suptitle("The first pillar crowds out private second-pillar funding unless employee weight is high", fontsize=12)
    fig.savefig(f"{c.OUT}/crowding.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/crowding.png")


# ============ 4. frontier robustness to the entry-state assumption ============
def frontier_entry_robustness(lambdas=np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.92, 0.97]),
                              nF=121, nR=51, na=25, nq=5, n_paths=30000, seed=7):
    print("[frontier_entry_robustness] corner anchor vs sampled entry distribution (placeholder)")
    Fg, rg, ag = c.grids(nF, nR, na)
    c_cost, c_ben, d_cost, d_ben = [], [], [], []
    for lam in lambdas:
        c.restore(); c.dp.LAMBDA = float(lam)
        pol = c.solve(Fg, rg, ag, nq)["policy"]
        rc = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
        rng = np.random.default_rng(seed + 1); R0, L0, S0 = c.sample_entry(rng, n_paths)
        rd = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n_paths, seed=seed)
        c_cost.append(rc["cost"]); c_ben.append(rc["benefit"])
        d_cost.append(rd["cost"]); d_ben.append(rd["benefit"])
        print(f"   lambda={lam:.2f}: corner (cost {rc['cost']:.3f}, ben {rc['benefit']:+.3f}) | "
              f"dist (cost {rd['cost']:.3f}, ben {rd['benefit']:+.3f})")
    c.restore()
    fig, ax = c.plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    ax.plot(c_cost, c_ben, "-o", color="#274690", lw=1.7, ms=5, label="single corner anchor")
    ax.plot(d_cost, d_ben, "-s", color="#1D9E75", lw=1.7, ms=5, label="entry distribution (placeholder)")
    ax.set_xscale("log")
    ax.set_xlabel("employer cost per unit final salary (log; cheaper <-)")
    ax.set_ylabel(r"employee value  $\mathbb{E}[u(\mathrm{RR})]$  (better up)")
    ax.set_title("Frontier: single corner vs. expectation over entry states")
    ax.legend(frameon=False, loc="lower right"); ax.grid(True, which="both", alpha=0.22, lw=0.6)
    fig.savefig(f"{c.OUT}/lambda_frontier_dist.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/lambda_frontier_dist.png")


_ALL = {
    "threshold": lambda_threshold,
    "pareto": pareto_frontier,
    "crowding": crowding_out,
    "entrydist": frontier_entry_robustness,
}


def main(which=None):
    c.ensure_out()
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"--- {name} ---"); fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
