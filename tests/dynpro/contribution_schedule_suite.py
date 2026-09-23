"""Contribution-schedule suite: shape and design of the employer's optimal
funding schedule a*(t) under the committed model -- unconstrained vs banded
(predictable) vs flat, new-plan front-loading, and macro-scenario robustness.

Figures (-> figs/):
  policy_map              policy_map.png            baseline a*(F,rho) heatmap w/ iso-RR contours
  baseline_schedule        baseline_schedule.png     banded-DCA schedule + stayer/leaver adequacy
  flat_design_curve        flat_design_curve.png     flat (fixed %) contribution design curve
  dca_predictability       dca_schedules.png,        unconstrained vs banded vs pure-DCA, and the
                           dca_cost.png              joint-value cost of imposing predictability
  backload_vs_baseline     backload_compare.png      preference-driven back-loading vs baseline
  schedule_vs_macro(p)     schedule_vs_<p>.png       banded schedule swept over mu or G
  new_plan_profile         new_plan_profile.png      front-load-then-coast funnel + terminal RR
  signal_schedule          signal_schedule.png       contribution read off V as a graded signal
                                                     (soft-argmax at temperature beta) + value gap
  scenario_grid            scenario_grid.png         (mu,G) x discount grid of schedules

Run:  python contribution_schedule_suite.py [policy|baseline|flat|dca|backload|macro|newplan|signal|scenario]
"""
import numpy as np
import common as c


# ============ 1. baseline policy map ============
def policy_map(nF=145, nR=61, na=41, nq=7, years=(5, 22, 40)):
    Fg, rg, ag = c.grids(nF, nR, na)
    pol = c.solve(Fg, rg, ag, nq)["policy"]
    print(f"[policy_map] bang-bang={np.all((pol==0)|(pol==1))}  mean a*={pol.mean():.3f}")
    RR = c.iso_rr(Fg, rg)
    fig, axes = c.plt.subplots(1, len(years), figsize=(4.5 * len(years), 4.4), sharey=True, constrained_layout=True)
    m = None
    for ax, t in zip(axes, years):
        m = ax.pcolormesh(Fg, rg, pol[t].T, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
        cs = ax.contour(Fg, rg, RR.T, levels=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
                        colors="white", linewidths=0.8, alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=7, fmt="RR=%.1f")
        ax.axvline(1.0, color="white", lw=0.9, ls=":", alpha=0.7); ax.set_yscale("log")
        ax.set_xlim(Fg[0], Fg[-1]); ax.set_ylim(rg[0], rg[-1])
        ax.set_xlabel(r"$F=R/L$"); ax.set_title(f"year $t={t}$")
    axes[0].set_ylabel(r"$\rho=S/L$ (log)")
    fig.colorbar(m, ax=axes, shrink=0.9, pad=0.015).set_label(r"optimal funding $a^\star$")
    fig.suptitle(r"Optimal funding rule (baseline) vs iso total-replacement contours ($\eta=%.0f$, $\lambda=%.2f$)"
                % (c.dp.ETA, c.dp.LAMBDA), fontsize=12)
    fig.savefig(f"{c.OUT}/policy_map.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/policy_map.png")


# ============ 2. baseline banded-DCA schedule + adequacy ============
def baseline_schedule(nF=145, nR=61, nq=7, band_pct=(0.02, 0.15), n_paths=40000, seed=7):
    Fg, rg, _ = c.grids(nF, nR)
    lo, hi = band_pct[0] / c.dp.GAMMA, band_pct[1] / c.dp.GAMMA
    ag = np.linspace(lo, hi, 26)
    pol = c.solve(Fg, rg, ag, nq)["policy"]
    r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], band=(lo, hi), n_paths=n_paths, seed=seed)
    print(f"[baseline_schedule] career-avg {r['avg']:.1f}%  stayer RR {r['sty']:.3f}  leaver RR {r['lea']:.3f}")
    yrs = np.arange(c.dp.T)
    fig, (axL, axR) = c.plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axL.bar(yrs, r["c_by"], color="#1D9E75", alpha=0.85, width=0.9)
    axL.axhline(r["avg"], color="#C1121F", ls="--", lw=1.2, label=f"career avg {r['avg']:.1f}%")
    axL.set_xlabel("career year $t$"); axL.set_ylabel("contribution (% of salary)")
    axL.legend(frameon=False, fontsize=9); axL.set_title("Banded-DCA optimal contribution (baseline)")
    axR.hist(r["RR_tot"][r["stay"]], bins=40, range=(0.42, np.quantile(r["RR_tot"], 0.995)),
             color="#274690", alpha=0.6, label=f"stayers (med {r['sty']:.2f})")
    axR.hist(r["RR_tot"][~r["stay"]], bins=40, range=(0.42, np.quantile(r["RR_tot"], 0.995)),
             color="#E08D1C", alpha=0.6, label=f"leavers (med {r['lea']:.2f})")
    axR.axvline(c.dp.RR_TARGET, color="#C1121F", ls=":", lw=1.2, label=f"target {c.dp.RR_TARGET}")
    axR.axvline(c.dp.RR_LEGAL, color="#9aa0a6", ls=":", lw=1)
    axR.set_xlabel("total replacement rate"); axR.set_ylabel("paths"); axR.legend(frameon=False, fontsize=8.5)
    axR.set_title("Adequacy: stayers on target, leavers proportional")
    fig.savefig(f"{c.OUT}/baseline_schedule.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/baseline_schedule.png")


# ============ 3. flat (fixed-cashflow) design curve ============
def flat_design_curve(rates_pct=np.linspace(2, 15, 14), n_paths=30000, seed=7, nF=145, nR=61):
    Fg, rg, _ = c.grids(nF, nR)
    tot, sty = [], []
    for cp in rates_pct:
        a = cp / 100.0 / c.dp.GAMMA
        pol = c.const_policy(a, c.dp.T, len(Fg), len(rg))
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
        tot.append(r["tot"]); sty.append(r["sty"])
    i = int(np.argmin(np.abs(np.array(sty) - c.dp.RR_TARGET)))
    fig, ax = c.plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    ax.plot(rates_pct, sty, "-o", color="#274690", lw=2, label="stayer RR")
    ax.plot(rates_pct, tot, "-s", color="#1D9E75", lw=2, label="median RR")
    ax.axhline(c.dp.RR_TARGET, color="#C1121F", ls="--", lw=1.2, label=f"target {c.dp.RR_TARGET}")
    ax.axhline(c.dp.RR_LEGAL, color="#9aa0a6", ls=":", lw=1, label=f"legal {c.dp.RR_LEGAL}")
    ax.axvline(rates_pct[i], color="#146c50", ls=":", lw=1)
    ax.set_xlabel("flat contribution (% of payroll, every year)"); ax.set_ylabel("replacement rate")
    ax.set_title(f"Fixed-cashflow design curve (baseline)\n~{rates_pct[i]:.1f}% flat lands stayers on target")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(f"{c.OUT}/flat_design_curve.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/flat_design_curve.png  | target-hitting flat rate ~{rates_pct[i]:.1f}%")


# ============ 4. banded-DCA vs unconstrained vs pure-DCA: cost of predictability ============
def dca_predictability(nF=121, nR=51, na=31, nq=5, eps_list=(0.0, 0.10, 0.25, 0.50, 1.0), n_paths=30000, seed=7):
    Fg, rg, _ = c.grids(nF, nR, na)
    unc = c.solve(Fg, rg, c.dp.make_a_grid(n=na), nq)["policy"]
    r_unc = c.simulate(unc, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
    print(f"unconstrained: joint={r_unc['joint']:+.3f}")

    best = None
    for ac in np.linspace(0.02, 0.6, 20):
        pol = c.const_policy(ac, c.dp.T, len(Fg), len(rg))
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
        if best is None or r["joint"] > best[1]: best = (ac, r["joint"])
    abar = best[0]
    print(f"optimal constant rate abar={abar:.3f} (contrib {abar*c.dp.GAMMA*100:.1f}% salary)  joint={best[1]:+.3f}")

    res = []
    for eps in eps_list:
        lo, hi = max(0.0, abar - eps), min(1.0, abar + eps)
        agb = np.array([abar]) if eps == 0 else np.linspace(lo, hi, max(3, int(na * (hi - lo) + 2)))
        pol = c.solve(Fg, rg, agb, nq)["policy"]
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], band=(lo, hi), n_paths=n_paths, seed=seed)
        rough = float(np.mean(np.diff(r["c_by"]) ** 2))
        res.append((eps, r, rough))
        print(f"  eps={eps:.2f}: joint={r['joint']:+.3f}  rough={rough:.4f}  totRR={r['tot']:.3f}")
    jd = res[0][1]["joint"]; ju = r_unc["joint"]
    print(f"COST OF PREDICTABILITY: pure DCA vs unconstrained = {100*(ju-jd)/abs(ju):.1f}% of joint value")

    yrs = np.arange(c.dp.T)
    fig, ax = c.plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(yrs, r_unc["c_by"], lw=2, color="#C1121F", label="unconstrained (front-loaded)")
    ax.plot(yrs, res[2][1]["c_by"], lw=2, color="#1D9E75", label=f"banded DCA (eps={res[2][0]:.2f})")
    ax.plot(yrs, res[0][1]["c_by"], lw=2, color="#274690", label="pure DCA (eps=0, flat)")
    ax.set_xlabel("career year $t$"); ax.set_ylabel("employer contribution (% of salary)")
    ax.set_title("Banded dollar-cost-averaging smooths the schedule")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(f"{c.OUT}/dca_schedules.png", dpi=c.DPI); c.plt.close(fig); print(f"wrote {c.OUT}/dca_schedules.png")

    fig, ax = c.plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    eps_ax = [e for e, _, _ in res]; jv = [r["joint"] for _, r, _ in res]
    ax.plot(eps_ax, jv, "-o", color="#1D9E75", lw=2, label="joint value (banded)")
    ax.axhline(ju, color="#C1121F", ls="--", lw=1.2, label="unconstrained")
    ax.axhline(jd, color="#274690", ls=":", lw=1.2, label="pure DCA")
    ax.set_xlabel(r"band half-width $\epsilon$ (0 = pure DCA, 1 = unconstrained)")
    ax.set_ylabel(r"joint value $\lambda V_{emp}+(1-\lambda)V_{empr}$")
    ax.set_title("Cost of predictability: joint value vs how tight the DCA band is")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(f"{c.OUT}/dca_cost.png", dpi=c.DPI); c.plt.close(fig); print(f"wrote {c.OUT}/dca_cost.png")


# ============ 5. preference-driven back-loading vs baseline ============
def backload_vs_baseline(de=0.05, lam=0.3, nF=121, nR=51, nq=5, n_paths=40000, seed=7):
    Fg, rg, _ = c.grids(nF, nR)
    lo, hi = 0.02 / c.dp.GAMMA, 1.0

    def run(de_, lam_):
        c.restore(); c.dp.DISC_EMP = de_; c.dp.LAMBDA = lam_
        pol = c.solve(Fg, rg, np.linspace(lo, hi, 26), nq)["policy"]
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], band=(lo, hi), n_paths=n_paths, seed=seed)
        c.restore()
        return r

    rb = run(c._BASE["DISC_EMP"], c._BASE["LAMBDA"]); rx = run(de, lam)
    yrs = np.arange(c.dp.T)
    fig, (axL, axR) = c.plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axL.plot(yrs, rb["c_by"], lw=2.4, color="#1D9E75", label=f"baseline avg {rb['avg']:.1f}%")
    axL.plot(yrs, rx["c_by"], lw=2.4, color="#C1121F", label=f"back-load avg {rx['avg']:.1f}%")
    axL.set_xlabel("career year $t$"); axL.set_ylabel("contribution (% of salary)"); axL.set_ylim(0, 16)
    axL.legend(frameon=False); axL.set_title(rf"Schedule: baseline vs back-load ($\delta_e$={de:.0%}, $\lambda$={lam})")
    xb = np.arange(2); w = 0.35
    axR.bar(xb - w / 2, [rb["sty"], rb["lea"]], w, color="#1D9E75", label="baseline")
    axR.bar(xb + w / 2, [rx["sty"], rx["lea"]], w, color="#C1121F", label="back-load")
    axR.axhline(c.dp.RR_TARGET, color="#274690", ls="--", lw=1.2, label=f"target {c.dp.RR_TARGET}")
    axR.axhline(c.dp.RR_LEGAL, color="#9aa0a6", ls=":", lw=1)
    axR.set_xticks(xb); axR.set_xticklabels(["stayer", "leaver"]); axR.set_ylabel("median total RR")
    axR.legend(frameon=False); axR.set_title("Adequacy cost of back-loading")
    fig.savefig(f"{c.OUT}/backload_compare.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/backload_compare.png")


# ============ 6. banded schedule vs a macro assumption (mu or G) ============
def schedule_vs_macro(param="G", values=(0.020, 0.025, 0.030, 0.035), fixed=0.025,
                      nF=121, nR=51, band_pct=(0.02, 0.15), n_paths=30000, seed=7):
    assert param in ("MU", "G")
    Fg, rg, _ = c.grids(nF, nR)
    lo, hi = band_pct[0] / c.dp.GAMMA, band_pct[1] / c.dp.GAMMA
    agb = np.linspace(lo, hi, 26)
    yrs = np.arange(c.dp.T); cols = c.plt.cm.viridis(np.linspace(0.15, 0.85, len(values)))
    res = []
    for v in values:
        c.restore()
        if param == "MU": c.dp.MU, c.dp.G = v, fixed
        else: c.dp.MU, c.dp.G = fixed, v
        pol = c.solve(Fg, rg, agb, 5)["policy"]
        r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], band=(lo, hi), n_paths=n_paths, seed=seed)
        nceil = int((r["c_by"] > (band_pct[1] * 100 - 0.5)).sum())
        res.append((r, nceil))
        print(f"  {param}={v:.3f}: avg={r['avg']:.1f}%  yrs@ceiling={nceil}  stayerRR={r['sty']:.3f}")
    c.restore()
    fig, (axL, axR) = c.plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for v, (r, _), col in zip(values, res, cols):
        axL.plot(yrs, r["c_by"], lw=2, color=col, label=f"{param}={v*100:.1f}%")
    axL.set_xlabel("career year $t$"); axL.set_ylabel("contribution (% salary)")
    axL.set_title(f"Optimal banded-DCA schedule vs {param} ({'G' if param=='MU' else 'mu'} fixed {fixed*100:.1f}%)")
    axL.legend(frameon=False, fontsize=8)
    ax2 = axR.twinx()
    axR.plot(values, [r["avg"] for r, _ in res], "-o", color="#1D9E75", lw=2, label="career-avg contrib (%)")
    ax2.plot(values, [r["sty"] for r, _ in res], "-s", color="#274690", lw=1.6, label="stayer RR")
    ax2.axhline(c.dp.RR_TARGET, color="#274690", ls=":", lw=1, alpha=0.6)
    axR.set_xlabel(param); axR.set_ylabel("career-avg contribution (%)", color="#146c50")
    ax2.set_ylabel("stayer RR", color="#274690"); ax2.spines["top"].set_visible(False)
    h1, l1 = axR.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=8)
    axR.set_title(f"Outcomes vs {param}")
    fig.savefig(f"{c.OUT}/schedule_vs_{param}.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/schedule_vs_{param}.png")


# ============ 7. new-plan funding profile ============
def new_plan_profile(nF=145, nR=61, na=31, nq=7, n_paths=40000, seed=7):
    Fg, rg, ag = c.grids(nF, nR, na)
    pol = c.solve(Fg, rg, ag, nq)["policy"]
    rng = np.random.default_rng(seed)
    R0, L0, S0 = c.new_plan_init(n_paths, rng)
    r = c.simulate(pol, Fg, rg, R0=R0, L0=L0, S0=S0, n_paths=n_paths, seed=seed, track=True)
    frac3 = np.nan_to_num(r["a_by_t"])[:3].sum() / np.nan_to_num(r["a_by_t"]).sum()
    print(f"[new_plan_profile] a*(0)={r['a_by_t'][0]:.3f}  first-3yr share={frac3:.2f}  "
          f"rho after yr0={r['rho_med'][1]:.2f} (1/Gamma={1/c.dp.GAMMA:.2f})")
    print(f"   TOTAL RR median={r['tot']:.3f} (legal={c.dp.RR_LEGAL})  stayer={r['sty']:.3f}  leaver={r['lea']:.3f}  "
          f"survival to T={c.dp.survival(c.dp.tenure_hazard)[c.dp.T]:.2f}")
    yrs = np.arange(c.dp.T)
    fig, (a1, a3) = c.plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    a1.bar(yrs, r["a_by_t"], color="#1D9E75", alpha=0.85, width=0.9, label=r"mean applied $a^\star$ (present)")
    a1.set_xlabel("career year $t$"); a1.set_ylabel(r"mean applied $a^\star$", color="#146c50"); a1.set_ylim(0, 1.02)
    a2 = a1.twinx(); a2.plot(yrs, r["rho_med"], color="#274690", lw=2, label=r"median $\rho(t)$")
    a2.axhline(1 / c.dp.GAMMA, color="#C1121F", ls="--", lw=1.2, label=r"$1/\Gamma$")
    a2.set_yscale("log"); a2.set_ylabel(r"median $\rho$ (log)", color="#274690"); a2.spines["top"].set_visible(False)
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=9)
    a1.set_title("New-plan funding: front-load then coast")
    a3.hist(r["RR_tot"][r["stay"]], bins=45, range=(c.dp.RR_LEGAL * 0.98, np.quantile(r["RR_tot"], 0.99)),
            color="#274690", alpha=0.6, label=f"stayers (med {r['sty']:.2f})")
    a3.hist(r["RR_tot"][~r["stay"]], bins=45, range=(c.dp.RR_LEGAL * 0.98, np.quantile(r["RR_tot"], 0.99)),
            color="#E08D1C", alpha=0.6, label=f"leavers (med {r['lea']:.2f})")
    a3.axvline(c.dp.RR_TARGET, color="#C1121F", ls=":", lw=1.2, label=f"target {c.dp.RR_TARGET}")
    a3.axvline(c.dp.RR_LEGAL, color="#9aa0a6", ls="--", lw=1.0)
    a3.set_xlabel("TOTAL annual replacement (legal + 2nd pillar)"); a3.set_ylabel("paths")
    a3.set_title("Terminal RR: stayers on target, leavers proportional"); a3.legend(frameon=False, fontsize=9)
    fig.suptitle("New-plan behaviour: front-loaded funding, service-pro-rated adequacy", fontsize=12)
    fig.savefig(f"{c.OUT}/new_plan_profile.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/new_plan_profile.png")


# ============ 8. value function as a signal (soft readout, objective unchanged) ============
def signal_schedule(betas=(0.002, 0.005, 0.02, 0.05), nF=145, nR=61, na=31, nq=7,
                    n_paths=30000, seed=7, years=(0, 5, 10, 15, 22, 30, 44)):
    """Read the contribution off the TRUE value function as a graded signal:
    a_soft = sum_a a*softmax(Q/beta), instead of the argmax. The objective and V
    are untouched -- beta only controls how sharply the contribution responds to
    Q, so the smoothness is a deliberate extraction choice and its value gap to
    the hard optimum is reported as the price of that legibility."""
    Fg, rg, ag = c.grids(nF, nR, na)
    out = c.solve(Fg, rg, ag, nq, betas=betas)
    hard = out["policy"]; soft = out["policy_soft"]

    rh = c.simulate(hard, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
    corner_h = float(np.mean((hard < 1e-9) | (hard > ag.max() - 1e-9)))
    rough_h = float(np.mean(np.diff(rh["c_by"]) ** 2))
    print(f"hard argmax : corner-frac={corner_h:.2f}  rough={rough_h:7.3f}  avg={rh['avg']:5.1f}%  "
          f"sty={rh['sty']:.3f}  joint={rh['joint']:+.4f}")
    rows = [("argmax", rh, corner_h, rough_h, 0.0)]
    for b in betas:
        r = c.simulate(soft[b], Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], n_paths=n_paths, seed=seed)
        corner = float(np.mean((soft[b] < 1e-9) | (soft[b] > ag.max() - 1e-9)))
        rough = float(np.mean(np.diff(r["c_by"]) ** 2))
        gap = 100 * (rh["joint"] - r["joint"]) / abs(rh["joint"])
        print(f"beta={b:<7}: corner-frac={corner:.2f}  rough={rough:7.3f}  avg={r['avg']:5.1f}%  "
              f"sty={r['sty']:.3f}  joint={r['joint']:+.4f}  value gap={gap:+.1f}%")
        rows.append((f"beta={b}", r, corner, rough, gap))

    print("\ncontribution (% of salary) by career year")
    print("  year  " + "".join(f"{n:>12}" for n, *_ in rows))
    for y in years:
        print(f"  {y:<5} " + "".join(f"{r['c_by'][y]:>12.1f}" for _, r, *_ in rows))

    yrs = np.arange(c.dp.T)
    fig, (axL, axR) = c.plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axL.plot(yrs, rh["c_by"], lw=2.4, color="#C1121F", label=f"argmax (hard, avg {rh['avg']:.1f}%)")
    cols = c.plt.cm.viridis(np.linspace(0.15, 0.8, len(betas)))
    for (b, col) in zip(betas, cols):
        r = rows[1 + list(betas).index(b)][1]
        axL.plot(yrs, r["c_by"], lw=2, color=col, label=rf"$\beta$={b} (avg {r['avg']:.1f}%)")
    axL.set_xlabel("career year $t$"); axL.set_ylabel("contribution (% of salary)")
    axL.set_title("Contribution read off the value function as a signal\n(same $V$; only the readout sharpness changes)")
    axL.legend(frameon=False, fontsize=8)
    gaps = [g for *_, g in rows[1:]]; roughs = [rr for _, _, _, rr, _ in rows[1:]]
    axR.plot(betas, gaps, "-o", color="#C1121F", lw=2, label="value gap vs argmax (%)")
    ax2 = axR.twinx(); ax2.plot(betas, roughs, "-s", color="#274690", lw=1.8, label="schedule roughness")
    ax2.axhline(rough_h, color="#274690", ls=":", lw=1.2, alpha=0.7, label="argmax roughness")
    ax2.spines["top"].set_visible(False); axR.set_xscale("log")
    axR.set_xlabel(r"readout temperature $\beta$ (log)"); axR.set_ylabel("value gap (% of joint)", color="#C1121F")
    ax2.set_ylabel("roughness (mean sq. yr-on-yr change)", color="#274690")
    h1, l1 = axR.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=8)
    axR.set_title("Price of legibility: smoother schedule costs joint value")
    fig.savefig(f"{c.OUT}/signal_schedule.png", dpi=c.DPI); c.plt.close(fig)
    print(f"\nwrote {c.OUT}/signal_schedule.png")


# ============ 9. (mu,G) x discount scenario grid ============
def scenario_grid(discounts=(0.025, 0.04), nF=121, nR=51, band_pct=(0.02, 0.15), n_paths=15000, seed=7):
    scen = [("B21 underwater", dict(MU=0.01, G=0.03)), ("neutral", dict(MU=0.02, G=0.02)),
            ("baseline", dict(MU=0.03, G=0.03)), ("B23", dict(MU=0.05, G=0.03))]
    Fg, rg, _ = c.grids(nF, nR)
    lo, hi = band_pct[0] / c.dp.GAMMA, band_pct[1] / c.dp.GAMMA
    agb = np.linspace(lo, hi, 26)
    yrs = np.arange(c.dp.T)
    fig, axes = c.plt.subplots(len(discounts), len(scen), figsize=(15, 6.5),
                               sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, de in enumerate(discounts):
        for j, (nm, ov) in enumerate(scen):
            c.restore(); c.dp.DISC_EMP = de
            for k, v in ov.items(): setattr(c.dp, k, v)
            pol = c.solve(Fg, rg, agb, 5)["policy"]
            r = c.simulate(pol, Fg, rg, R0=1.0, L0=1.0, S0=rg[-1], band=(lo, hi), n_paths=n_paths, seed=seed)
            c.restore()
            ax = axes[i, j]; ax.bar(yrs, r["c_by"], color="#1D9E75", alpha=0.85, width=0.9)
            ax.set_title(f"{nm}\n$\\delta_e$={de:.0%}  avg {r['avg']:.1f}%  sty {r['sty']:.2f}", fontsize=9)
            if i == len(discounts) - 1: ax.set_xlabel("year $t$")
            if j == 0: ax.set_ylabel("contrib %sal")
    fig.suptitle(r"Scenario grid: contribution schedule under ($\mu$,$G$) x employee discount", fontsize=12)
    fig.savefig(f"{c.OUT}/scenario_grid.png", dpi=c.DPI); c.plt.close(fig)
    print(f"wrote {c.OUT}/scenario_grid.png")


_ALL = {
    "policy": policy_map,
    "baseline": baseline_schedule,
    "flat": flat_design_curve,
    "dca": dca_predictability,
    "backload": backload_vs_baseline,
    "macro": lambda: [schedule_vs_macro("MU", (0.015, 0.025, 0.035, 0.045, 0.055), fixed=0.03),
                     schedule_vs_macro("G", (0.020, 0.025, 0.030, 0.035), fixed=0.025)],
    "newplan": new_plan_profile,
    "signal": signal_schedule,
    "scenario": scenario_grid,
}


def main(which=None):
    c.ensure_out()
    for name, fn in _ALL.items():
        if which in (None, name):
            print(f"--- {name} ---"); fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
