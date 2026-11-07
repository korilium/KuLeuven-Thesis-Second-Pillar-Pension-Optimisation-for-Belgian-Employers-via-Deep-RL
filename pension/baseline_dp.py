"""
baseline_dp.py -- dynamic programming for the baseline MDP (pension/baseline.py) on
its FULL exact state, with the action values fitted by regression.

A grid over the state (eight variables and nine liability buckets) cannot run, so
Q is fitted instead (regression Monte Carlo, Longstaff-Schwartz style):

    Q_t(s, a_k) = E[ (1 - h(t)) V_{t+1}(s') + h(t) Phi_{t+1}(s') | s, a_k ]       Eq. (12), MODEL.md
    V_t(s)      = Q_t(s, pi_t(s)),  pi_t = argmax_k of the fitted Q_t,  V_T = the reward at T

- s' is the state after year t, from baseline.step itself.
- Phi_j is the value of leaving at tau = j. Nothing is decided after leaving, so
  it is rolled forward exactly: the liability buckets are fixed in money, the
  units decay by the fund fee, and each Branch 21 bucket accrues at its bucket
  rate on the path's mu_s (tariff plus profit-sharing).
- The regression target is NOT the fitted V_{t+1}: it is what the path actually
  earns from t + 1 on under the already-fitted policy pi_{t+1}, ..., pi_{T-1}.
  Unrolling Eq. (12), that is the survival-weighted mix of the leaver values along
  the realised rate path plus the reward at T if the member stays,
      sum_{j=t+1}^{T} P(exit at j) Phi_j + P(stay to T) reward_T.
  The fit then only ranks actions. Fitting max_k of noisy fitted Q's instead
  selects upward errors, E[max_k Q_hat] >= max_k E[Q], and backward induction
  passes the overshoot back year by year; with realised targets the error stays in
  the ranking, and a suboptimal policy is a lower bound, not an overshoot.
- Q_t(., a_k) is a least-squares fit, on features of s, of that target realised on
  simulated states with common noise across the actions.

The state is exact, so this is a true MDP: the only error is the fit of Q, and
evaluating the policy with baseline.simulate on fresh noise measures it. Exit
does not depend on the state, so the training states are simulated in force
(u = 1) and exit enters through h(t) only, as exact weights, never sampled.
"""

import time

import numpy as np

from pension import baseline as b


def features(obs, t, p):
    """Regression basis at year t, from the observation (the exact state) only.

    The buckets enter through summaries, not one by one, to keep the quadratic
    basis small: the total L_t; its weighted rate Gbar_t; the projected terminal
    liability Lp_t if the member stays (each bucket at its own g, future mid-year
    premiums at the current G_t); the total R21_t; the projected reserve Rp_t (each
    Branch 21 bucket at its current bucket rate, units net of the fee); and the two
    spreads that drive the choice, y_long - c_fee - i_t (ladder vs Branch 21, i_t the
    January proxy new_premium_rate) and y_long - G_t (ladder vs new guarantee), with
    their positive parts. Then the reward legs at (Lp, Rp) and all products of pairs
    of the base list."""
    y, G, Hm, Km, mu, N = obs[:, 1:7].T
    Lg = obs[:, 7:16].T                                   # (9, n), order of wap_grid()
    R21k = obs[:, 16:].T                                  # (16, n), order of tariff_grid()
    grid = b.wap_grid()
    m = p.T - t
    L = Lg.sum(axis=0)
    gL = sum(g * Lg[k] for k, g in enumerate(grid))
    Gbar = np.divide(gL, L, out=np.zeros_like(L), where=L > 0)
    Lp = sum(Lg[k] * (np.exp(g) / (1 + p.W)) ** m for k, g in enumerate(grid))
    Lp = Lp + p.C * sum(np.exp(G * (k - 0.5)) / (1 + p.W) ** k for k in range(1, m + 1))
    R21 = R21k.sum(axis=0)
    Rp = sum(R21k[k] * (np.exp(b.bucket_rate(i_k, mu, p)) / (1 + p.W)) ** m
             for k, i_k in enumerate(b.tariff_grid(p)))
    Rp = Rp + N * (1 - p.FUND_FEE) ** m * (1 + p.W) ** -m
    i = b.new_premium_rate(y, mu, p)
    y_long = b.locked_yield(y, t, b.vasicek_params(p), p)
    s21, sG = y_long - p.FUND_FEE - i, y_long - G
    emp, short = b.reward_legs(Lp, Rp, p.T, p)
    base = np.stack([y, G, Hm, Km, mu, L, R21, N, Gbar, Lp, Rp,
                     s21, sG, np.maximum(s21, 0), np.maximum(sG, 0)], axis=1)
    iu = np.triu_indices(base.shape[1])
    quad = (base[:, :, None] * base[:, None, :])[:, iu[0], iu[1]]
    return np.column_stack([np.ones(len(obs)), base, emp, short, quad])


def paidup_growth(states, p):
    """Per path and Branch 21 bucket, the growth (in units of S_T) from January of
    year tau to T after leaving at tau: prod_{s >= tau} e^{r_k,s} / (1 + W), with
    r_k,s = bucket_rate(i_k, mu_s). (T + 1, 16, n); row T is 1. The rates are
    exogenous, so the in-force training states supply mu_s."""
    grid = b.tariff_grid(p)
    r = np.array([[b.bucket_rate(i_k, s.mu, p) for i_k in grid] for s in states[:p.T]])  # (T, 16, n)
    g = np.exp(np.cumsum(r[::-1], axis=0)[::-1]) / (1 + p.W) ** (p.T - np.arange(p.T))[:, None, None]
    return np.concatenate([g, np.ones((1,) + r.shape[1:])])


def _fit(X, target):
    """Least squares with columns scaled to unit spread; returns (beta, scale)."""
    scale = X.std(axis=0)
    # a constant column has a std of rounding noise (~1e-18), not 0; dividing by it
    # would blow the column up to ~1e16 and lstsq would then drop every other column
    scale[scale <= 1e-12 * np.abs(X).max(axis=0)] = 1.0
    scale[0] = 1.0
    beta = np.linalg.lstsq(X / scale, target, rcond=None)[0]
    return beta, scale


def _reward(L, R, tau, p):
    emp, short = b.reward_legs(L, R, tau, p)
    return p.LAMBDA * emp - (1 - p.LAMBDA) * short


def leaver_value(s, grow21, p):
    """Phi_tau, Eq. (12): the reward at T of leaving at tau = s.t, rolled forward
    exactly from the January state s: liability buckets fixed in money, units net of
    the fee, each Branch 21 bucket by its paid-up growth (grow21 = paidup_growth,
    row tau)."""
    tau = s.t
    d = (1 + p.W) ** -(p.T - tau)
    units = s.N * (1 - p.FUND_FEE) ** (p.T - tau) * d
    return _reward(s.L * d, (s.R21k * grow21[tau]).sum(axis=0) + units, tau, p)


def rollout_target(s2, t, fits, actions, noise, grow21, p):
    """The regression target of Q_t, Eq. (12) unrolled: from s2 = the state after
    year t, follow the fitted greedy policy of years t+1 .. T-1 on the training
    noise, in force, and weight each year's leaver value by its exit probability:
        h(t) Phi_{t+1} + sum_{j>t} prod_{m=t}^{j-1}(1 - h(m)) h(j) Phi_{j+1}
                       + prod_{m=t}^{T-1}(1 - h(m)) reward_T.
    Leaving at T is staying (tau = T), so the last two terms are one reward."""
    n = len(s2.y)
    target = b.hazard(t, p) * leaver_value(s2, grow21, p)
    surv, s = 1 - b.hazard(t, p), s2
    for j in range(t + 1, p.T):
        a = actions[q_values(fits[j], s.obs(p), j, p).argmax(axis=1)]
        s = b.step(s, a, noise.year(j)[0], np.ones(n), p)
        h = b.hazard(j, p)
        target = target + surv * h * leaver_value(s, grow21, p)
        surv = surv * (1 - h)
    return target + surv * _reward(s.L, s.R21 + s.N, p.T, p)


def solve(p=b.BASELINE, n=10000, actions=np.linspace(0.0, 1.0, 11), seed=None):
    """Backward induction, Eq. (12), with realised rollout targets. Returns the
    fitted Q per year and action, the action grid, V0 (the in-sample mean, over
    the starting states, of the realised target of the greedy action at t = 0), and
    timings: seconds spent on the forward simulation, the rollouts and the fits.
    Training noise: seed (default p.TRAIN_SEED), never p.EVAL_SEED."""
    seed = p.TRAIN_SEED if seed is None else seed
    timings = dict(forward=0.0, rollouts=0.0, fits=0.0)  # seconds per phase
    clock = time.perf_counter()
    noise = b.draw_noise(n, seed, p)
    stay = np.ones_like(noise.u)                          # train in force; exit via h(t)
    rng = np.random.default_rng(seed + 1)

    # forward: training states, exploring the action grid
    states = [b.initial_state(noise.eps[:b.premonths()], p)]
    for t in range(p.T):
        a = rng.choice(actions, size=n)
        states.append(b.step(states[-1], a, noise.year(t)[0], stay[t], p))

    grow21 = paidup_growth(states, p)
    timings["forward"] = time.perf_counter() - clock

    actions = np.asarray(actions, float)
    fits = [None] * p.T
    for t in range(p.T - 1, -1, -1):
        s = states[t]
        clock = time.perf_counter()
        targets = []
        for a in actions:                                 # common noise across the actions
            s2 = b.step(s, np.full(n, a), noise.year(t)[0], stay[t], p)
            targets.append(rollout_target(s2, t, fits, actions, noise, grow21, p))
        timings["rollouts"] += time.perf_counter() - clock
        clock = time.perf_counter()
        X = features(s.obs(p), t, p)
        betas = [_fit(X, tg) for tg in targets]
        fits[t] = dict(beta=np.array([bt for bt, _ in betas]), scale=np.array([sc for _, sc in betas]))
        timings["fits"] += time.perf_counter() - clock
    greedy = q_values(fits[0], states[0].obs(p), 0, p).argmax(axis=1)
    V0 = float(np.array(targets)[greedy, np.arange(n)].mean())
    return dict(fits=fits, actions=actions, V0=V0, timings=timings)


def q_values(fit, obs, t, p):
    """Q_t(s, a_k) for every action, (n, K)."""
    X = features(obs, t, p)
    return np.stack([(X / sc) @ bt for bt, sc in zip(fit["beta"], fit["scale"])], axis=1)


def value(fit, obs, t, p):
    """V_t(s) = max_k Q_t(s, a_k)."""
    return q_values(fit, obs, t, p).max(axis=1)


def policy(sol, p=b.BASELINE):
    """The greedy policy obs -> alpha, for baseline.simulate and PensionEnv."""
    def act(obs):
        t = int(round(obs[0, 0] * p.T))
        return sol["actions"][q_values(sol["fits"][t], obs, t, p).argmax(axis=1)]
    return act


def interior_share(sol, p=b.BASELINE, n=10000, seed=None):
    """Share of the in-force decisions of the greedy policy with 0 < alpha < 1, on
    evaluation noise (seed: EVAL_SEED). The share is 0 when the policy is bang-bang."""
    noise = b.draw_noise(n, p.EVAL_SEED if seed is None else seed, p)
    act = policy(sol, p)
    s = b.initial_state(noise.eps[:b.premonths()], p)
    interior = decisions = 0
    for t in range(p.T):
        a = act(s.obs(p))
        interior += int(((a > 0) & (a < 1) & s.inforce).sum())
        decisions += int(s.inforce.sum())
        s = b.step(s, a, *noise.year(t), p)
    return interior / decisions
