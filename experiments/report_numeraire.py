"""Reports for the retirement-numeraire change (Option 1), stage by stage.

    python experiments/report_numeraire.py stage1    # the real-world short-rate model
    python experiments/report_numeraire.py stage1b   # Hull-White under P, side by side
    python experiments/report_numeraire.py stage3    # verification of the retirement numeraire
"""
import sys

from pension import checks
from pension.params import DEFAULT


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def stage1():
    f = checks.vasicek_short_fit()
    print(f"Short-rate proxy: the {f['maturity']} OLO (shortest maturity in the NBB data)\n")
    print("| kappa | theta_P | sigma | t-stat of b | p-value | half-life | theta_Q | phi |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| {f['kappa']:.4f} | {pct(f['theta_P'])} | {pct(f['sigma'])} | {f['t_stat_b']:.2f} | "
          f"{f['p_val_b']:.3f} | {f['half_life']:.1f} y | {pct(f['theta_Q'])} | {f['phi']:.3f} |\n")
    print("In-sample fit of the reconstructed 10Y (from the observed short rate):\n")
    print("| RMSE | bias (0 by construction) | correlation | RMSE last 24 months | mean spread 10Y - short |")
    print("|---|---|---|---|---|")
    print(f"| {f['rmse'] * 1e4:.1f} bp | {f['bias'] * 1e4:.1e} bp | {f['corr']:.3f} | "
          f"{f['rmse_last24'] * 1e4:.1f} bp | {f['spread_obs'] * 1e4:.0f} bp |\n")
    print(f"Jump at t0: model 10Y at r0 = {pct(f['r0'], 3)} is {pct(f['model_10y_t0'], 3)} vs observed "
          f"{pct(f['observed_10y_t0'], 3)} -> {f['jump_t0'] * 1e4:+.1f} bp\n")

    p = DEFAULT.replace(RATE_MODEL="vasicek_short")
    st = checks.scenario_stats(p, "vasicek_short")
    print("Scenarios (2000 paths): 5th / median / 95th percentile\n")
    print("| year | short rate r_t | P(r_t < 0) | 10Y | G_t | mu_t | annual accrual |")
    print("|---|---|---|---|---|---|---|")
    fm = lambda q, d=2: " / ".join(pct(v, d) for v in q)
    for t, s in st.items():
        print(f"| {t} | {fm(s['r'])} | {s['r_neg']:.0%} | {fm(s['y10'])} | {fm(s['G'])} | "
              f"{fm(s['mu'])} | {' / '.join(f'{v:.4f}' for v in s['acc'])} |")


def stage1b(n=2000, n_acc=20000):
    import numpy as np
    from pension.economy import draw_rate_scenarios, hull_white_p_params, vasicek_short_params

    f = checks.vasicek_short_fit()
    print("vasicek_short t0 offset: e_t0 = %+.1f bp, AR(1) phi_e = %.4f (se %.4f), half-life %.1f y, "
          "t-stat of the decay %.1f vs |t| of kappa %.2f -> %s\n"
          % (f["offset_e_t0"] * 1e4, f["offset_phi_e"], f["offset_se"], f["offset_half_life_years"],
             f["offset_t_decay"], abs(f["t_stat_b"]),
             "kappa FALLBACK used" if f["offset_fallback_kappa"] else "AR(1) decay used"))
    hp = hull_white_p_params(DEFAULT)
    print("hull_white sigma: sigma_10Y = %s used as short-rate sigma; consistent sigma_short = %s; "
          "hull_white's 10Y volatility is understated by the factor B(10)/10 = %.3f (-%.0f%%)\n"
          % (pct(hp["sigma_10Y"], 3), pct(hp["sigma"], 3), hp["B10"] / 10, 100 * (1 - hp["B10"] / 10)))

    configs = [("hull_white (Q)", DEFAULT.replace(RATE_MODEL="hull_white"), lambda p: "0 (Q)"),
               ("hull_white_p, LONG_RATE_P=None", DEFAULT.replace(RATE_MODEL="hull_white_p"),
                lambda p: f"{hull_white_p_params(p)['phi']:+.3f}"),
               ("hull_white_p, LONG_RATE_P=2.25%", DEFAULT.replace(RATE_MODEL="hull_white_p", LONG_RATE_P=0.0225),
                lambda p: f"{hull_white_p_params(p)['phi']:+.3f}"),
               ("vasicek_short, OLS theta_P", DEFAULT.replace(RATE_MODEL="vasicek_short"),
                lambda p: f"{vasicek_short_params(p)['phi']:+.3f}"),
               ("vasicek_short, LONG_RATE_P=2.25%", DEFAULT.replace(RATE_MODEL="vasicek_short", LONG_RATE_P=0.0225),
                lambda p: f"{vasicek_short_params(p)['phi']:+.3f}")]
    print(f"Scenarios ({n} paths). Short rate as 5% / median / 95%.\n")
    print("| model | year | short rate r_t | P(r < 0) | 10Y median | G_t median | mu_t median | phi |")
    print("|---|---|---|---|---|---|---|---|")
    for name, p, phi in configs:
        sc = draw_rate_scenarios(n, p=p)
        for t in (0, 10, 44):
            r = sc["r"][t]
            print(f"| {name} | {t} | {' / '.join(pct(v) for v in np.percentile(r, [5, 50, 95]))} | "
                  f"{(r < 0).mean():.0%} | {pct(np.median(sc['y10'][t]))} | {pct(np.median(sc['G'][t]))} | "
                  f"{pct(np.median(sc['mu'][t]))} | {phi(p) if t == 0 else ''} |")

    print("\nChecks\n")
    a = checks.hw_p_link()
    print(f"(a) phi = 0 and legacy sigma vs hull_white, same seed: max |dr| = {a['r']:.1e}, "
          f"max |d10Y| = {a['y10']:.1e}, scenario r = {a['scenario_r']:.1e}")
    for lr in (None, 0.0225):
        p = DEFAULT.replace(RATE_MODEL="hull_white_p", LONG_RATE_P=lr)
        b = checks.hw_p_t0_fit(p)
        print(f"(b) LONG_RATE_P={lr}: model 10Y at t0 {pct(b['model_10y_t0'], 3)} = NSS 10Y "
              f"{pct(b['nss_10y'], 3)} (diff {abs(b['model_10y_t0'] - b['nss_10y']):.0e}); observed "
              f"{pct(b['observed_10y'], 3)}: {(b['observed_10y'] - b['model_10y_t0']) * 1e4:+.1f} bp "
              f"(NSS RMSE {b['nss_rmse'] * 1e4:.2f} bp)")
    print("\n(c) P-drift of hull_white_p: scenario mean of r_t (se) vs analytic E^P[r_t]\n")
    print("| LONG_RATE_P | " + " | ".join(f"t={t}" for t in (1, 5, 10, 20, 30, 44)) + " |")
    print("|---|" + "---|" * 6)
    for lr in (None, 0.0225):
        c = checks.hw_p_drift(DEFAULT.replace(RATE_MODEL="hull_white_p", LONG_RATE_P=lr))
        print(f"| {lr} | " + " | ".join(f"{pct(d['mc'], 3)} ({d['se'] * 1e4:.1f} bp) vs {pct(d['analytic'], 3)}"
                                         for d in c.values()) + " |")
    d = checks.vol_target(DEFAULT)
    print(f"\n(d) std of monthly 10Y changes: historical {d['historical'] * 1e4:.1f} bp, hull_white_p "
          f"{d['hull_white_p'] * 1e4:.1f} bp ({d['hull_white_p'] / d['historical']:.0%}), hull_white "
          f"{d['hull_white'] * 1e4:.1f} bp ({d['hull_white'] / d['historical']:.0%})")
    print(f"\n(e) accrual: MC mean of the realised accrual to T vs closed-form A(t,T), {n_acc} paths; "
          "relative error (se), in bp\n")
    print("| model | " + " | ".join(f"t={t}" for t in (0, 5, 10, 20, 30, 40, 44)) + " |")
    print("|---|" + "---|" * 7)
    for name, p, _ in configs[1:]:
        e = checks.accrual_accuracy(p, n=n_acc)
        print(f"| {name} | " + " | ".join(f"{v['rel_err'] * 1e4:+.1f} ({v['rel_se'] * 1e4:.1f})"
                                          for v in e.values()) + " |")


def stage3(n_paths=15000, n_env=300, n_acc=20000):
    import numpy as np
    import pension.dp as dp
    from pension.economy import vasicek_params
    Fg, rg, ag = dp.grids(nF=73, nR=71, na=20); nq = 5
    R0, L0, S0 = dp.new_plan_init(n_paths, np.random.default_rng(7))
    entry = dict(R0=R0, L0=L0, S0=S0)
    small = {k: v[:n_env] for k, v in entry.items()}
    new, old = DEFAULT, DEFAULT.replace(EMPLOYER_NUMERAIRE="discounted")

    print("(a) contracts\n")
    print("| check | result |\n|---|---|")
    for lab, p in (("constant, retirement", new), ("constant, discounted", old),
                   ("vasicek, retirement", new.replace(RATE_MODEL="vasicek"))):
        d = checks.env_contract(p, Fg, rg, ag, nq, small, n_env, 7)
        print(f"| env return vs simulate joint, {lab} ({n_env} careers) | {d:.1e} |")
    print(f"| dynamics.step vs dp.F_next / rho_next | {checks.reduced_form_gap(new):.1e} |")

    print("\n(b) retirement at lambda' vs discounted at lambda (constant rates)\n")
    print("| lambda | r = SHORT_RATE = DISC_ER | DISC_EMP | lambda' | max |policy diff| | max |V ratio - factor| / factor |")
    print("|---|---|---|---|---|---|")
    for lam, r, de in ((0.5, 0.03, 0.03), (0.5, 0.05, 0.03), (0.3, 0.02, 0.04), (0.7, 0.04, 0.01), (0.5, 0.0, 0.03)):
        dpol, dV, lp = checks.numeraire_equivalence(new, Fg, rg, ag, nq, lam, r, de)
        print(f"| {lam} | {pct(r)} | {pct(de)} | {lp:.4f} | {dpol:.1e} | {dV:.1e} |")

    f, d = checks.face_value_premiums(new, Fg, rg, entry, n_paths, 7)
    print(f"\n(c) SHORT_RATE = s = 0: max |premium or terminal factor - 1| = {f:.1e}; "
          f"cost/joint vs discounted at DISC_ER = DISC_EMP = 0: max difference {d:.1e}")

    early, late = checks.timing_neutrality(new, Fg, rg, nq, entry, n_paths, 7)
    print(f"\n(d) SHORT_RATE = MU = G: early (years 0-9) {early:.1f}% vs late (35-44) {late:.1f}% "
          f"of salary, tilt {abs(early - late) / max(early, late):.0%}")

    print(f"\n(e) accrual, vasicek (spread {vasicek_params(new)['spread'] * 1e4:.1f} bp), {n_acc} paths: "
          "MC mean of the realised accrual to T vs closed-form A(t,T), relative error (se) in bp\n")
    e = checks.accrual_accuracy(new.replace(RATE_MODEL="vasicek"), n=n_acc)
    print("| t | " + " | ".join(str(t) for t in e) + " |\n|---|" + "---|" * len(e))
    print("| A(t,T) closed form | " + " | ".join(f"{v['closed']:.4f}" for v in e.values()) + " |")
    print("| rel. error (se) | " + " | ".join(f"{v['rel_err'] * 1e4:+.1f} ({v['rel_se'] * 1e4:.1f})" for v in e.values()) + " |")

    print("\n(f) headline, constant rates, banded 2-15% optimum, new-plan cohort\n")
    rows = [("old default (discounted), scored discounted", old, old),
            ("new default (retirement), scored retirement", new, new),
            ("old-default policy, scored retirement", old, new),
            ("new-default policy, scored discounted", new, old)]
    print("| policy / scoring | employer cost | joint | stayer RR | leaver RR | contribution % by decade (0-9, 10-19, 20-29, 30-39, 40-44) |")
    print("|---|---|---|---|---|---|")
    for lab, pp, ps in rows:
        h = checks.headline(pp, Fg, rg, nq, entry, n_paths, 7, score_p=ps)
        print(f"| {lab} | {h['cost']:.4f} | {h['joint']:+.4f} | {h['sty']:.3f} | {h['lea']:.3f} | "
              + " / ".join(f"{x:.1f}" for x in h["by_decade"]) + " |")
    print("\nvasicek (rate mode): not run -- no headline in rate mode until LONG_RATE_P is set.")


if __name__ == "__main__":
    {"stage1": stage1, "stage1b": stage1b, "stage3": stage3}[sys.argv[1] if len(sys.argv) > 1 else "stage1"]()
