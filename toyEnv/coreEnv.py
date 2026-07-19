"""Gymnasium environment for rung 2: funding ratio enters the state.

Wraps the SAME recursions as basicEnv.run_episode — no rewrite. The economy
(G, MU, DISC, KAPPA, SIGMA) is read from the basicEnv module namespace at
call time, so testBasicEnv.apply_config() configures this env too.

State (all O(1) for the MLP):
    s_t = [ t/T,  F_t,  L_t / L^max_t ]
    F_t = R_t / L_t, with F = 1.0 when L = 0 (no liability = fully funded)
    L^max_t = liability under always-contribute (deterministic, precomputed)
This is the exact-Markov state: the terminal max()/shortfall scales with
the LEVEL of L, not just the ratio, so (t, F) alone aliases states.

Reward: identical emission to rung 1 — per-year discounted cost, terminal
payout/shortfall — already discounted at DISC inside the reward.
=> PPO must use gamma = 1.0 (anything else double-discounts; frozen decision).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import basicEnv as env


class PensionEnv(gym.Env):
    """One employee career. Binary action: contribute this year or not."""

    metadata = {"render_modes": []}

    def __init__(self, plan, seed=None):
        # guard: economy must be configured (apply_config) before use
        for k in ("G", "MU", "DISC", "KAPPA", "SIGMA"):
            assert hasattr(env, k), \
                f"basicEnv.{k} not set — call apply_config(CONFIGS[...]) first"

        self.plan = plan
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # L^max_t: always-contribute liability AFTER t steps (index 0..T).
        # L_max[0] = 1.0 is a divide-by-zero guard only: at t=0 the true L
        # is 0, so L_tilde = 0/1 = 0 regardless.
        self.L_max = np.empty(env.T + 1)
        self.L_max[0] = 1.0
        L, S = 0.0, env.S0
        for t in range(env.T):
            L = (L + plan(t, S)) * (1.0 + env.G)
            S *= (1.0 + env.W)
            self.L_max[t + 1] = L

    # -- observation at the START of step t: R, L reflect t completed steps
    def _obs(self):
        F = self.R / self.L if self.L > 0 else 1.0
        L_tilde = self.L / self.L_max[self.t]
        return np.array([self.t / env.T, F, L_tilde], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # shock path: injected (consistency checks, CRN evaluation) or drawn
        if options is not None and "shocks" in options and options["shocks"] is not None:
            self.shocks = np.asarray(options["shocks"], dtype=float)
            assert self.shocks.shape == (env.T,)
        elif env.SIGMA > 0:
            self.shocks = self.rng.standard_normal(env.T)
        else:
            self.shocks = np.zeros(env.T)

        self.t = 0
        self.R = 0.0
        self.L = 0.0
        self.S = env.S0
        return self._obs(), {}

    def step(self, action):
        a = int(action)
        t = self.t
        c = self.plan(t, self.S) * a

        # identical to run_episode, line for line
        reward = -env.KAPPA * c * np.exp(-env.DISC * t)
        self.R = (self.R + c) * np.exp(env.MU + env.SIGMA * self.shocks[t])
        self.L = (self.L + c) * (1.0 + env.G)
        self.S *= (1.0 + env.W)
        self.t += 1

        terminated = self.t == env.T
        if terminated:
            payout = max(self.R, self.L)
            shortfall = max(self.L - self.R, 0.0)
            reward += (payout - env.KAPPA * shortfall) * np.exp(-env.DISC * env.T)

        return self._obs(), float(reward), terminated, False, {}


# ---------------------------------------------------------------------------
# Consistency check: the bridge that carries rung-1 certification forward.
# Plays PensionEnv with fixed open-loop policies against run_episode on the
# SAME shock paths; totals must agree to 1e-10. Same spirit as
# check_batch_consistency — a second implementation demands an equality test.
# ---------------------------------------------------------------------------

def check_gym_consistency(plan, policies, n_paths=5, seed=99):
    rng = np.random.default_rng(seed)
    genv = PensionEnv(plan)
    for p_idx, policy in enumerate(policies):
        policy = np.asarray(policy, dtype=np.int64)
        for i in range(n_paths):
            shocks = rng.standard_normal(env.T) if env.SIGMA > 0 else None
            _, r = env.run_episode(lambda t: int(policy[t]), plan, shocks)

            obs, _ = genv.reset(options={"shocks": shocks})
            total, term = 0.0, False
            for t in range(env.T):
                obs, rew, term, _, _ = genv.step(int(policy[t]))
                total += rew
            assert term, "episode did not terminate at T"
            assert abs(total - r.sum()) < 1e-10, \
                f"policy {p_idx}, path {i}: gym {total} vs run_episode {r.sum()}"
    print(f"gym consistency: {len(policies)} policies x {n_paths} paths OK")


if __name__ == "__main__":
    # run the bridge against the FROZEN rung-1 artifacts
    from testBasicEnv import CONFIGS, apply_config
    apply_config(CONFIGS["rung1_stoch"])

    plans = {"fixed": env.plan_fixed(), "step": env.plan_step(), "age": env.plan_age()}
    for name, plan in plans.items():
        d = np.load(f"tests/rung1_stoch/mc_{name}.npz")
        policies = [d["bench_policy"],
                    np.zeros(env.T, dtype=np.int64),
                    np.ones(env.T, dtype=np.int64)]
        check_gym_consistency(plan, policies)
    print("PensionEnv certified against run_episode.")