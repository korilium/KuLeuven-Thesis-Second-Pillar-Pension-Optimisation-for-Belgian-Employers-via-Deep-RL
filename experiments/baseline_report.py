"""The numbers of the baseline report (docs/REVIEW.md, "Baseline v2").

    python experiments/baseline_report.py

1. Rates per year (0, 5, 20, 44): mean G_t, share at the 1.75% floor, mean mu_t,
   mean i_t - G_t, mean r_t.
2. Constant policies alpha = 1, 0, 0.5: J, both legs, mean shortfall and its
   frequency, mean premium cost.
3. Removing sigma_R: legacy_v1 at the same premium, all in Branch 21, with and
   without its reserve shock, next to the baseline at alpha = 1 with the tariff
   floor switched off (I_TARIFF = -1, so i_t = mu_t as in legacy_v1).
4. The DP (pension/baseline_dp.py, 20 000 training paths): its policy scored out
   of sample, its in-sample V_0, and its mean alpha by year.
"""
import dataclasses

import numpy as np

from pension import baseline as b
from pension import baseline_dp as bdp
import pension.legacy_v1.dp as ldp
from pension.legacy_v1.params import DEFAULT as LEGACY

N = 20000


def const(a):
    return lambda obs: np.full(len(obs), a)


def rates_table():
    d = b.simulate(const(0.5), n=N)["by_year"]
    print("year  G      floor  mu     i-G     r")
    for t in (0, 5, 20, 44):
        print(f"{t:>4}  {d['G'][t]:.4f} {d['G_floor'][t]:.3f}  {d['mu'][t]:.4f} {d['i_minus_G'][t]:+.4f} {d['r'][t]:.4f}")


def policies():
    print("alpha  J        employee  shortfall  P(short>0)  premium cost")
    for a in (1.0, 0.0, 0.5):
        r = b.simulate(const(a), n=N)
        print(f"{a:<5}  {r['joint']:+.4f}  {r['employee']:+.4f}   {r['shortfall']:.5f}    "
              f"{r['shortfall_freq']:.4f}      {r['premium_cost']:.4f}")


def legacy_at_fixed_premium(sigma_r):
    """legacy_v1, vasicek, constant contribution a = C / GAMMA (all in Branch 21), a
    near-empty entry (R0 = L0 = 1e-9, S0 = 1), its own book yield and horizontal ledger."""
    p = LEGACY.replace(RATE_MODEL="vasicek", SIGMA_R_RATES=sigma_r)
    Fg, rg = ldp.make_F_grid(n=31), ldp.make_rho_grid(n=25)
    pol = ldp.const_policy(b.BASELINE.C / p.GAMMA, p.T, len(Fg), len(rg))
    s = ldp.simulate(pol, Fg, rg, R0=1e-9, L0=1e-9, S0=1.0, n_paths=N, seed=b.BASELINE.EVAL_SEED, p=p)
    ST = (1 + p.W) ** p.T
    short = np.maximum(s["L_T"] - s["R_T"], 0.0) / ST
    return s["benefit"], short.mean(), (short > 0).mean()


def sigma_r_effect():
    print("model                                   employee  shortfall  P(short>0)")
    for name, sig in (("legacy_v1, sigma_R = 0.05 (as frozen)", 0.05), ("legacy_v1, sigma_R = 0", 0.0)):
        e, sh, f = legacy_at_fixed_premium(sig)
        print(f"{name:<40}{e:+.4f}   {sh:.5f}    {f:.4f}")
    r = b.simulate(const(1.0), n=N, p=dataclasses.replace(b.BASELINE, I_TARIFF=-1.0))
    print(f"{'baseline, alpha = 1, no tariff floor':<40}{r['employee']:+.4f}   {r['shortfall']:.5f}    "
          f"{r['shortfall_freq']:.4f}")


def dp_section():
    sol = bdp.solve(n=20000)
    pol = bdp.policy(sol)
    r = b.simulate(pol, n=N)
    print(f"DP policy: J {r['joint']:+.4f}, employee {r['employee']:+.4f}, shortfall {r['shortfall']:.5f}, "
          f"P(short>0) {r['shortfall_freq']:.4f}; in-sample V_0 {sol['V0']:+.4f}")
    noise = b.draw_noise(N, b.BASELINE.EVAL_SEED)
    s = b.initial_state(noise.eps[:b.premonths()], b.BASELINE)
    print("year  mean alpha (in force)")
    for t in range(b.BASELINE.T):
        a = pol(s.obs(b.BASELINE))
        if t in (0, 5, 10, 20, 30, 44):
            print(f"{t:>4}  {a[s.inforce].mean():.3f}")
        s = b.step(s, a, *noise.year(t), b.BASELINE)


if __name__ == "__main__":
    rates_table(); print()
    policies(); print()
    sigma_r_effect(); print()
    dp_section()
