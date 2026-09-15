"""Sensitivity suite: how robust the committed-model results are to each
parameter, to grid resolution, to the choice of evaluation anchor, and to the
plan-entry-state assumption.

Figures (-> figs/):
  tornado                sens_tornado.png        +/- one-at-a-time sweep of every Tier-1/2 param
  interaction(pi,pj)     sens_inter_<tag>.png     pairwise (pi,pj) grid-mean a* heatmap
  policy_map_sweep(p)    sens_policy_<p>.png      policy-map row across values of one param
  frontier_shift(p)      sens_frontier_<p>.png    lambda-frontier overlaid across values of one param
  resolution_check(p)    sens_resolution_<p>.png  coarse vs fine mesh: is the trend mesh jitter?
  anchor_check(p)        sens_anchor_<p>.png      edge vs interior rho0: is the trend an edge artifact?
  entry_dist_check(p)    sens_entrydist_<p>.png   single-corner anchor vs sampled entry distribution
  schedule_grid          sens_schedule_grid.png   year-by-year schedule sensitivity, 3x3 param grid

Run:  python sensitivity_suite.py [tornado|interaction|policy|frontier|resolution|anchor|entrydist|schedule]
      python -c "import sensitivity_suite as s; s.policy_map_sweep('ETA',[2,3,5])"
"""
import numpy as np
import common as c

YEAR = 22


def _metrics(Fg, rg, ag, nq, n_paths=15000, seed=7):
    pol = c.solve(Fg, rg, ag, nq)["policy"]
    rng = np.random.default_rng(seed)
    R0, L0, S0 = c.new_plan_init(n_paths, rng)
    r = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n_paths, seed=seed)
    return dict(gm_a=float(pol.mean()), **r)


# ============ 1. tornado (one-at-a-time) ============
def tornado(pct=0.15, nF=73, nR=31, na=15, nq=5, params=None):
    params = params or c.PARAMS
    Fg, rg, ag = c.grids(nF, nR, na)
    c.restore(); base = _metrics(Fg, rg, ag, nq)
    print("baseline:", {k: round(v, 3) for k, v in base.items() if k in ("gm_a", "tot", "sty", "lea")})
    rows = []
    for p in params:
        b = c._BASE[p]
        c.restore(); setattr(c.dp, p, b * (1 - pct)); lo = _metrics(Fg, rg, ag, nq)
        c.restore(); setattr(c.dp, p, b * (1 + pct)); hi = _metrics(Fg, rg, ag, nq)
        rows.append((p, lo, hi))
        print(f"  {p:8}: RRtot {lo['tot']:.3f}->{hi['tot']:.3f}  gm_a {lo['gm_a']:.2f}->{hi['gm_a']:.2f}")
    c.restore()
    order = sorted(rows, key=lambda x: abs(x[2]["tot"] - x[1]["tot"]))
    fig, (ax1, ax2) = c.plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, key, ttl, bv in ((ax1, "tot", "total replacement rate (median)", base["tot"]),
                             (ax2, "gm_a", r"grid-mean $a^\star$", base["gm_a"])):
        y = np.arange(len(order))
        for i, (p, lo, hi) in enumerate(order):
            l, h = lo[key], hi[key]
            ax.barh(i, h - l, left=min(l, h), color="#1D9E75", alpha=0.8, height=0.6)
        ax.axvline(bv, color="#C1121F", ls="--", lw=1, label=f"baseline={bv:.3f}")
        ax.set_yticks(y); ax.set_yticklabels([c.LAB[p] for p, _, _ in order])
        ax.set_xlabel(ttl); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Tornado: sensitivity of results to each parameter (+/-{int(pct*100)}%)", fontsize=12)
    fig.savefig(f"{c.OUT}/sens_tornado.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/sens_tornado.png")


# ============ 2. pairwise interaction (grid-mean a*) ============
def interaction(pi, pj, vi, vj, nF=73, nR=31, na=15, nq=5, tag=""):
    Fg, rg, ag = c.grids(nF, nR, na)
    Z = np.zeros((len(vj), len(vi)))
    for a, vjj in enumerate(vj):
        for b, vii in enumerate(vi):
            c.restore(); setattr(c.dp, pi, float(vii)); setattr(c.dp, pj, float(vjj))
            Z[a, b] = c.solve(Fg, rg, ag, nq)["policy"].mean()
    c.restore()
    add = Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean()
    inter = np.abs(Z - add).mean()
    print(f"[{pi} x {pj}] grid-mean a* range {Z.min():.2f}-{Z.max():.2f}  interaction(non-additivity)={inter:.3f}")
    fig, ax = c.plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[vi[0], vi[-1], vj[0], vj[-1]])
    ax.set_xlabel(c.LAB[pi]); ax.set_ylabel(c.LAB[pj])
    fig.colorbar(im, ax=ax, label=r"grid-mean $a^\star$")
    ax.set_title(f"{c.LAB[pi]} x {c.LAB[pj]}: interaction={inter:.3f}\n(flat bands = separable; curved = they interact)")
    tag = tag or f"{pi}_{pj}"
    fig.savefig(f"{c.OUT}/sens_inter_{tag}.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/sens_inter_{tag}.png")
    return inter


# ============ 3. policy-map sweep (one param, several values) ============
def policy_map_sweep(param, values, year=YEAR, nF=121, nR=51, na=25, nq=5, n_paths=15000):
    Fg, rg, ag = c.grids(nF, nR, na); RR = c.iso_rr(Fg, rg)
    lev = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    fig, axes = c.plt.subplots(1, len(values), figsize=(4.5 * len(values), 4.3),
                               sharey=True, constrained_layout=True)
    if len(values) == 1: axes = [axes]
    mesh = None
    for ax, val in zip(axes, values):
        c.restore(); setattr(c.dp, param, float(val))
        pol = c.solve(Fg, rg, ag, nq)["policy"]
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=12345)
        print(f"   {param}={val}:  mean a*={pol.mean():.3f}   lambda={c.dp.LAMBDA:.2f}  "
              f"benefit={r['benefit']:+.4f}  cost={r['cost']:.4f}")
        mesh = ax.pcolormesh(Fg, rg, pol[year].T, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
        cs = ax.contour(Fg, rg, RR.T, levels=lev, colors="white", linewidths=0.8, alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=7, fmt="RR=%.1f")
        ax.axvline(1.0, color="white", lw=0.9, ls=":", alpha=0.7); ax.set_yscale("log")
        ax.set_xlim(Fg[0], Fg[-1]); ax.set_ylim(rg[0], rg[-1])
        ax.set_xlabel(r"$F=R/L$"); ax.set_title(f"{c.LAB[param]} = {val}")
    c.restore()
    axes[0].set_ylabel(r"$\rho=S/L$ (log)")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.015); cb.set_label(rf"$a^\star(t={year},F,\rho)$")
    fig.suptitle(rf"Sensitivity of the optimal rule to {c.LAB[param]} (year $t={year}$)", fontsize=12)
    path = f"{c.OUT}/sens_policy_{param}.png"
    fig.savefig(path, dpi=c.DPI); c.plt.close(fig); print(f"   wrote {path}")


# ============ 4. lambda-frontier shift under one param ============
def frontier_shift(param, values, lambdas=(0.1, 0.3, 0.5, 0.7, 0.9),
                   nF=100, nR=41, na=21, nq=5, n_paths=15000):
    Fg, rg, ag = c.grids(nF, nR, na)
    colors = ["#1D9E75", "#C1121F", "#274690", "#E08D1C"]
    fig, ax = c.plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    for val, col in zip(values, colors):
        ben, cost = [], []
        for lam in lambdas:
            c.restore(); setattr(c.dp, param, float(val)); c.dp.LAMBDA = float(lam)
            pol = c.solve(Fg, rg, ag, nq)["policy"]
            r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=12345)
            ben.append(r["benefit"]); cost.append(r["cost"])
        c.restore()
        ax.plot(cost, ben, "-o", color=col, lw=1.7, ms=4, label=rf"{c.LAB[param]} = {val}")
        print(f"   {param}={val}: costs {np.round(cost,3)}  benefits {np.round(ben,3)}")
    ax.set_xscale("log")
    ax.set_xlabel("employer cost per unit final salary (log; cheaper <-)")
    ax.set_ylabel(r"employee value  $\mathbb{E}[u(\mathrm{RR})]$  (better up)")
    ax.set_title(rf"Frontier shift under {c.LAB[param]} (swept $\lambda$)")
    ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.grid(True, which="both", alpha=0.22, lw=0.6)
    path = f"{c.OUT}/sens_frontier_{param}.png"
    fig.savefig(path, dpi=c.DPI); c.plt.close(fig); print(f"   wrote {path}")


# ============ 5. resolution: coarse vs fine mesh ============
def resolution_check(param="G", values=(0.0175, 0.025, 0.0375),
                     coarse=(121, 51, 25, 5), fine=(181, 61, 25, 7), n_paths=15000):
    print(f"[resolution_check] {param}  coarse={coarse}  fine={fine}")
    series = {}
    for tag, (nF, nR, na, nq) in (("coarse", coarse), ("fine", fine)):
        Fg, rg, ag = c.grids(nF, nR, na)
        ma, cst, ben = [], [], []
        for val in values:
            c.restore(); setattr(c.dp, param, float(val))
            pol = c.solve(Fg, rg, ag, nq)["policy"]
            r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=12345)
            ma.append(float(pol.mean())); cst.append(r["cost"]); ben.append(r["benefit"])
            print(f"   [{tag:6}] {param}={val:<7}: grid-mean a*={ma[-1]:.3f}  cost={r['cost']:.4f}  benefit={r['benefit']:+.4f}")
        series[tag] = dict(mean_a=ma, cost=cst, benefit=ben)
    c.restore()
    d = np.abs(np.array(series["fine"]["mean_a"]) - np.array(series["coarse"]["mean_a"]))
    print(f"   max |Delta grid-mean a*| coarse->fine = {d.max():.3f}")
    fig, ax = c.plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.plot(values, series["coarse"]["mean_a"], "-o", color="#C1121F", lw=1.7, ms=6, label="coarse mesh")
    ax.plot(values, series["fine"]["mean_a"], "-s", color="#1D9E75", lw=1.7, ms=6, label="fine mesh")
    ax.set_xlabel(c.LAB[param]); ax.set_ylabel(r"grid-mean $a^\star$")
    ax.set_title(f"Resolution check: grid-mean a* vs {c.LAB[param]}\n(overlapping lines = not mesh jitter)")
    ax.legend(frameon=False); ax.grid(True, alpha=0.25, lw=0.6)
    path = f"{c.OUT}/sens_resolution_{param}.png"
    fig.savefig(path, dpi=c.DPI); c.plt.close(fig); print(f"   wrote {path}")
    return series


# ============ 6. anchor: edge vs interior rho0 ============
def anchor_check(param="G", values=(0.0175, 0.025, 0.0375), nF=145, nR=61, na=31, nq=7,
                 anchors=(("edge (rho0=35)", 35.0), ("interior (rho0=3)", 3.0)), n_paths=15000):
    print(f"[anchor_check] {param}  grids=({nF},{nR},{na},q{nq})")
    Fg, rg, ag = c.grids(nF, nR, na)
    data = {name: dict(a=[], cost=[], ben=[]) for name, _ in anchors}
    for val in values:
        c.restore(); setattr(c.dp, param, float(val))
        pol = c.solve(Fg, rg, ag, nq)["policy"]
        for name, rho0 in anchors:
            r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=float(rho0), n_paths=n_paths, seed=12345)
            data[name]["a"].append(r["mean_a"]); data[name]["cost"].append(r["cost"]); data[name]["ben"].append(r["benefit"])
            print(f"   {param}={val:<7} {name:18}: path-mean a*={r['mean_a']:.3f}  cost={r['cost']:.4f}  benefit={r['benefit']:+.4f}")
    c.restore()
    fig, ax = c.plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    for (name, _), col, mk in zip(anchors, ("#274690", "#E08D1C"), ("o", "s")):
        ax.plot(values, data[name]["a"], "-" + mk, color=col, lw=1.7, ms=6, label=name)
    ax.set_xlabel(c.LAB[param]); ax.set_ylabel(r"path-weighted mean applied $a^\star$")
    ax.set_title(f"Anchor check: realized funding vs {c.LAB[param]}\n(both sloping the same way = not an edge artifact)")
    ax.legend(frameon=False); ax.grid(True, alpha=0.25, lw=0.6)
    path = f"{c.OUT}/sens_anchor_{param}.png"
    fig.savefig(path, dpi=c.DPI); c.plt.close(fig); print(f"   wrote {path}")
    return data


# ============ 7. entry-state distribution vs single corner anchor ============
def entry_dist_check(param="G", values=(0.0175, 0.025, 0.0375), nF=145, nR=61, na=31, nq=7, n_paths=15000):
    print(f"[entry_dist_check] {param}  (placeholder entry distribution, see common.sample_entry)")
    Fg, rg, ag = c.grids(nF, nR, na)
    corner_a, dist_a = [], []
    for val in values:
        c.restore(); setattr(c.dp, param, float(val))
        pol = c.solve(Fg, rg, ag, nq)["policy"]
        rc = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=7)
        rng = np.random.default_rng(8); R0, L0, S0 = c.sample_entry(rng, n_paths)
        rd = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n_paths, seed=7)
        corner_a.append(rc["mean_a"]); dist_a.append(rd["mean_a"])
        print(f"   {param}={val:<7}: corner a*={rc['mean_a']:.3f} (cost {rc['cost']:.3f}) | "
              f"dist a*={rd['mean_a']:.3f} (cost {rd['cost']:.3f}, benefit {rd['benefit']:+.3f})")
    c.restore()
    fig, ax = c.plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.plot(values, corner_a, "-o", color="#274690", lw=1.7, ms=6, label="single corner anchor")
    ax.plot(values, dist_a, "-s", color="#1D9E75", lw=1.7, ms=6, label="entry distribution (placeholder)")
    ax.set_xlabel(c.LAB[param]); ax.set_ylabel(r"path-weighted mean applied $a^\star$")
    ax.set_title(f"Realized funding vs {c.LAB[param]}: corner vs entry distribution\n(trend direction should be robust)")
    ax.legend(frameon=False); ax.grid(True, alpha=0.25, lw=0.6)
    path = f"{c.OUT}/sens_entrydist_{param}.png"
    fig.savefig(path, dpi=c.DPI); c.plt.close(fig); print(f"   wrote {path}")


# ============ 8. year-by-year schedule sensitivity, 3x3 param grid ============
def schedule_grid(nF=73, nR=31, band_pct=(0.02, 1.0), n_paths=15000, specs=None):
    specs = specs or [
        ("DISC_EMP", (0.02, 0.04), r"employee disc $\delta_e$"),
        ("DISC_ER", (0.03, 0.07), r"employer disc $\delta_f$"),
        ("MU", (0.02, 0.05), r"credited return $\mu$"),
        ("G", (0.02, 0.04), r"guarantee $G$"),
        ("GAMMA", (0.10, 0.20), r"ceiling $\Gamma$"),
        ("RR_TARGET", (0.60, 0.80), r"target $RR^\star$"),
        ("ETA", (1.5, 3.0), r"risk aversion $\eta$"),
        ("LAMBDA", (0.3, 0.7), r"weight $\lambda$"),
        ("ANNUITY", (12.0, 18.0), r"annuity $\ddot a$"),
    ]
    Fg, rg, _ = c.grids(nF, nR)

    def schedule():
        lo, hi = band_pct[0] / c.dp.GAMMA, band_pct[1] / c.dp.GAMMA
        ag = np.linspace(lo, min(hi, 1.0), 20)
        pol = c.solve(Fg, rg, ag, 5)["policy"]
        return c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], band=(lo, min(hi, 1.0)),
                          n_paths=n_paths, seed=7)["c_by"]

    c.restore(); c_base = schedule(); print("baseline schedule computed")
    yrs = np.arange(c.dp.T)
    fig, axes = c.plt.subplots(3, 3, figsize=(13.5, 10), constrained_layout=True)
    for ax, (pk, (lo, hi), lab) in zip(axes.ravel(), specs):
        c.restore(); setattr(c.dp, pk, lo); c_lo = schedule()
        c.restore(); setattr(c.dp, pk, hi); c_hi = schedule(); c.restore()
        ax.plot(yrs, c_base, color="#9aa0a6", lw=2.2, label=f"base {c._BASE[pk]:g}")
        ax.plot(yrs, c_lo, color="#274690", lw=1.8, label=f"low {lo:g}")
        ax.plot(yrs, c_hi, color="#C1121F", lw=1.8, label=f"high {hi:g}")
        ax.set_title(lab, fontsize=11); ax.set_xlabel("year $t$"); ax.set_ylabel("contrib %sal"); ax.set_ylim(0, 16)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        print(f"  {pk}: base_avg={c_base.mean():.1f} lo_avg={c_lo.mean():.1f} hi_avg={c_hi.mean():.1f}")
    fig.suptitle("Year-by-year sensitivity of the optimal contribution schedule to each parameter", fontsize=13)
    fig.savefig(f"{c.OUT}/sens_schedule_grid.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/sens_schedule_grid.png")


_ALL = {
    "tornado": lambda: tornado(),
    "interaction": lambda: interaction("ETA", "GAMMA", [2, 3, 5], [0.10, 0.15, 0.20], tag="ETA_GAMMA"),
    "policy": lambda: [policy_map_sweep(p, v) for p, v in
                       (("ETA", [2, 3, 5]), ("G", [0.0175, 0.025, 0.0375]), ("GAMMA", [0.25, 0.50, 0.75]))],
    "frontier": lambda: frontier_shift("ETA", [2, 3, 5]),
    "resolution": lambda: resolution_check("G"),
    "anchor": lambda: anchor_check("G"),
    "entrydist": lambda: entry_dist_check("G"),
    "schedule": lambda: schedule_grid(),
}


def main(which=None):
    c.ensure_out()
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"--- {name} ---"); fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
