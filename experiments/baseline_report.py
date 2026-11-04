"""The numbers of the baseline report (docs/REVIEW.md, "Baseline v2").

    python experiments/baseline_report.py

1. Rates per year (0, 5, 20, 44): mean G_t, share at the 1.75% floor, mean mu_t,
   mean i_t - G_t, mean r_t.
2. Constant policies alpha = 1, 0, 0.5: J, both legs, mean shortfall and its
   frequency, mean premium cost.
3. The DP (pension/baseline_dp.py, 20 000 training paths): its policy scored out
   of sample, its in-sample V_0, and its mean alpha by year.
"""
import numpy as np

from pension import baseline as b
from pension import baseline_dp as bdp

N = 20000

#alway same action 50% 21 
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
    dp_section()
