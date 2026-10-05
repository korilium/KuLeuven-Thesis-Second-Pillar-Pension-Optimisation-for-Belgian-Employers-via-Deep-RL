"""
pension_env.py -- the funding problem as a gymnasium environment (the RL rung).

One episode is one member's career with one employer, stepped by the SAME
pension.dynamics.step() that dp.simulate uses and scored by the SAME
pension.objective.Objective that dp.solve optimises. The return of an episode is
therefore exactly that path's contribution to simulate(...)["joint"], so an RL
agent and the DP oracle are measured on one scale (tests/test_env.py pins this).

Observation (float64, 5 -- float64, like the action, so a policy read off it sees
exactly the state simulate() sees and its actions are not rounded):
    t / T          career progress
    F = R / L      funding ratio
    log rho        log(S / L), salary relative to the guarantee
    G_t            the WAP rate a contribution made now is locked at
    mu_t           the credited return this year
  Under constant rates G_t = G and mu_t = MU throughout; under a rate model they
  are the episode's path from the scenario pool.

Action: Box([0], [1]) -- a, the fraction of contribution capacity used this year
  (contribution = a * GAMMA * salary), clipped to `band` if given.

Reward (per unit of final salary, discounted to t = 0, as in the objective):
    each year in force   -w_er * contribution(a, t) * exp(-DISC_ER t)
    at retirement        w_emp * exp(-DISC_EMP T) * employee(RR_tot, target)
                         - w_er * exp(-DISC_ER T) * shortfall((L - R)+ / S_T)
  If the member leaves, the contract goes paid-up: the remaining years need no
  decisions, so the environment rolls them forward and pays the terminal reward
  in the same step. gamma = 1 -- the discounting is in the reward.

Randomness: entry state, shocks and churn come from the episode's own RNG
(seeded via reset), the rate path is drawn from a fixed pool of `rate_pool`
scenarios (memoised, seeded by p.RATE_SEED). reset(options={"exogenous": exo,
"entry": (R0, L0, S0)}) replays a given path instead -- this is how the contract
test feeds the environment the exact paths simulate() saw.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pension.economy as _economy
from pension.dp import _objective, bilinear, new_plan_init, tenure_hazard
from pension.dynamics import Exogenous, State, settle, step
from pension.params import DEFAULT


class PensionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, p=DEFAULT, objective=None, band=None, hazard=tenure_hazard,
                 entry=None, rate_pool=2000):
        """p: Params; objective: name or Objective (None -> p.OBJECTIVE);
        band: (lo, hi) clip on a; entry: callable rng -> (R0, L0, S0) for a new
        career (default: dp.new_plan_init, the new-plan cohort of the suites)."""
        self.p = p
        self.obj = _objective(objective, p)
        self.w_emp, self.w_er = self.obj.weights(p)
        self.band = (0.0, 1.0) if band is None else band
        self.hazard = hazard
        self.entry = entry or (lambda rng: tuple(x[0] for x in new_plan_init(1, rng)))
        self.rate_pool = rate_pool
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float64)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64)
        self._pool = None

    # --- episode setup ------------------------------------------------------------
    def _rate_path(self):
        if self.p.RATE_MODEL == "constant":
            return None
        if self._pool is None:
            self._pool = _economy.draw_rate_scenarios(self.rate_pool, p=self.p)
        j = int(self.np_random.integers(self.rate_pool))
        return dict(G=self._pool["G"][:, j:j + 1], mu=self._pool["mu"][:, j:j + 1])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        p = self.p
        self.exo = options.get("exogenous")
        if self.exo is None:
            self.exo = Exogenous.draw(p, 1, int(self.np_random.integers(2**63 - 1)), self._rate_path())
        R0, L0, S0 = options.get("entry") or self.entry(self.np_random)
        self.state = State.initial(p, self.exo, R0, L0, S0)
        self.ST = self.state.S * (1.0 + p.W) ** p.T
        return self._obs(), {}

    # --- dynamics -------------------------------------------------------------------
    def _rates_now(self):
        t = min(self.state.t, self.p.T - 1)
        if self.exo.rates:
            return float(self.exo.G[t, 0]), float(self.exo.mu[t, 0])
        return self.p.G, self.p.MU

    def _obs(self):
        s = self.state
        G_t, mu_t = self._rates_now()
        return np.array([s.t / self.p.T, s.F[0], np.log(s.rho[0]), G_t, mu_t], dtype=np.float64)

    def _terminal_reward(self):
        p, end = self.p, settle(self.state, self.p)
        RR2 = end["payout"] / (p.ANNUITY * self.ST)
        target = p.RR_LEGAL + end["svc"] * (p.RR_TARGET - p.RR_LEGAL)
        emp = np.exp(-p.DISC_EMP * p.T) * self.obj.employee(p.RR_LEGAL + RR2, target, p)
        short = self.obj.shortfall(end["short"] / self.ST, p) * np.exp(-p.DISC_ER * p.T)
        return float(self.w_emp * emp[0] - self.w_er * short[0]), RR2[0]

    def step(self, action):
        p, s = self.p, self.state
        t = s.t
        a = np.clip(np.asarray(action, dtype=float).reshape(1), *self.band)
        reward = float(-self.w_er * self.obj.contribution(a, t, p)[0] * np.exp(-p.DISC_ER * t))
        s, _ = step(s, a, self.exo, p, self.hazard)
        while not s.present[0] and s.t < p.T:            # paid-up: nothing left to decide
            s, _ = step(s, np.zeros(1), self.exo, p, self.hazard)
        self.state = s
        terminated = s.t >= p.T
        info = {}
        if terminated:
            r_T, RR2 = self._terminal_reward()
            reward += r_T
            info = dict(RR_tot=p.RR_LEGAL + RR2, stayed=bool(s.leave_t[0] >= p.T))
        return self._obs(), reward, terminated, False, info


class DPPolicyAgent:
    """Runs a solved DP policy a*(t, F, rho) inside the environment: the oracle as
    an agent, for benchmarking learned policies on identical episodes."""

    def __init__(self, policy, Fg, rg, T):
        self.policy, self.Fg, self.lrg, self.T = policy, Fg, np.log(rg), T

    def __call__(self, obs):
        t = int(round(obs[0] * self.T))
        a = bilinear(self.Fg, self.lrg, self.policy[t], np.array([obs[1]], float),
                     np.array([obs[2]], float))
        return np.asarray(a, dtype=np.float64)


def make_env(**kw):
    """PensionEnv(**kw) -- convenience for gymnasium.vector / RL libraries."""
    return PensionEnv(**kw)
