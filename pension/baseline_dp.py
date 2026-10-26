"""
baseline_dp.py -- dynamic programming for the baseline MDP (pension/baseline.py) on
its FULL exact state, with the value function fitted by regression.

A grid over the nine state variables cannot run (about 1e8 points per year), so
the value function is fitted instead (regression Monte Carlo):

    Q_t(s, a_k) = E[ (1 - h(t)) V_{t+1}(s') + h(t) Phi_{t+1}(s') | s, a_k ]       Eq. (12), MODEL.md
    V_t(s)      = max_k Q_t(s, a_k),   V_T = the reward at T (exact)

- s' is the state after year t, from baseline.step itself.
- Phi_{t+1} is the value of leaving at tau = t + 1. Nothing is decided after
  leaving, so it is rolled forward exactly: L and N are fixed in money, and R21
  accrues at the path's i_s.
- Q_t(., a_k) is a least-squares fit, on features of s, of that target realised on
  simulated states with common noise across the actions.

The state is exact, so this is a true MDP: the only error is the fit of V, and
evaluating the policy with baseline.simulate on fresh noise measures it. Exit
does not depend on the state, so the training states are simulated in force
(u = 1) and exit enters through h(t) only.
"""

import numpy as np

from pension import baseline as b


def features(obs, t, p):
    """Regression basis at year t, from the observation (the exact state) only.

    The basis is the state; the projected terminal liability and reserve if
    the member stays (current G_t and i_t held fixed, future premiums all to
    Branch 21 for the projection only); the reward legs at those projections;
    and all products of pairs of the state variables and the projections."""
    y, G, Hm, Km, mu, L, R21, N = obs[:, 1:].T
    m = p.T - t
    gl = np.exp(G) / (1 + p.W)
    gi = np.exp(b.credited_rate(mu, p)) / (1 + p.W)
    Lp = L * gl ** m + p.C * sum(gl ** k for k in range(1, m + 1))
    Rp = R21 * gi ** m + N * (1 + p.W) ** -m
    emp, short = b.reward_legs(Lp, Rp, p.T, p)
    base = np.stack([y, G, Hm, Km, mu, L, R21, N, Lp, Rp], axis=1)
    iu = np.triu_indices(base.shape[1])
    quad = (base[:, :, None] * base[:, None, :])[:, iu[0], iu[1]]
    return np.column_stack([np.ones(len(obs)), base, emp, short, quad])


def paidup_growth(states, p):
    """Per path, the growth of R21 (in units of S_T) from January of year tau to T
    after leaving at tau: prod_{s >= tau} e^{i_s} / (1 + W). (T + 1, n); row T is 1.
    The rates are exogenous, so the in-force training states supply i_s."""
    i = np.array([b.credited_rate(s.mu, p) for s in states[:p.T]])
    g = np.exp(np.cumsum(i[::-1], axis=0)[::-1]) / (1 + p.W) ** (p.T - np.arange(p.T))[:, None]
    return np.vstack([g, np.ones(i.shape[1])])


def _fit(X, target):
    """Least squares with columns scaled to unit spread; returns (beta, scale)."""
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    scale[0] = 1.0
    beta = np.linalg.lstsq(X / scale, target, rcond=None)[0]
    return beta, scale


def _reward(L, R, tau, p):
    emp, short = b.reward_legs(L, R, tau, p)
    return p.LAMBDA * emp - (1 - p.LAMBDA) * short


def solve(p=b.BASELINE, n=10000, actions=np.linspace(0.0, 1.0, 11), seed=None):
    """Backward induction, Eq. (12). Returns the fitted Q per year and action, the
    action grid, and V0 (the in-sample mean of V_0 over the starting states).
    Training noise: seed (default p.TRAIN_SEED), never p.EVAL_SEED."""
    seed = p.TRAIN_SEED if seed is None else seed
    noise = b.draw_noise(n, seed, p)
    stay = np.ones_like(noise.u)                          # train in force; exit via h(t)
    rng = np.random.default_rng(seed + 1)

    # forward: training states, exploring the action grid
    states = [b.initial_state(noise.eps[:b.premonths()], p)]
    for t in range(p.T):
        a = rng.choice(actions, size=n)
        states.append(b.step(states[-1], a, noise.year(t)[0], stay[t], p))

    grow21 = paidup_growth(states, p)

    fits = [None] * p.T
    for t in range(p.T - 1, -1, -1):
        s = states[t]
        X = features(s.obs(p), t, p)
        eps, u = noise.year(t)[0], stay[t]
        h = b.hazard(t, p)
        betas = []
        for a in actions:
            s2 = b.step(s, np.full(n, a), eps, u, p)
            if t + 1 == p.T:
                v_next = _reward(s2.L, s2.R21 + s2.N, p.T, p)
            else:
                v_next = value(fits[t + 1], s2.obs(p), t + 1, p)
            d = (1 + p.W) ** -(p.T - t - 1)
            leave = _reward(s2.L * d, s2.R21 * grow21[t + 1] + s2.N * d, t + 1, p)
            betas.append(_fit(X, (1 - h) * v_next + h * leave))
        fits[t] = dict(beta=np.array([bt for bt, _ in betas]), scale=np.array([sc for _, sc in betas]))
    V0 = float(value(fits[0], states[0].obs(p), 0, p).mean())
    return dict(fits=fits, actions=np.asarray(actions, float), V0=V0)


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
