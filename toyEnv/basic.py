import os
 
import numpy as np


"""Tabular Monte Carlo control on the most basic pension contribution toy.
 
No neural networks, no Stable Baselines, no Gymnasium. One Q-table, one loop.
 
Environment (stripped to the bone)
----------------------------------
- T = 20 annual steps, binary action: contribute c_max (a=1) or nothing (a=0).
- Vintage-exact aggregate recursions under a frozen WAP-style guarantee g:
      R <- (R + c) * (1 + X_t)     credited reserve leg, X_t lognormal
      L <- (L + c) * (1 + g)       liability leg at the guarantee rate
- Reward: zero until the end, then the joint value
      J = max(R, L) - kappa * [ (1 + tau) * C + max(L - R, 0) ]
 
Agent
-----
- State: the year t only (policy class = corner policies, which is where the
  true optimum lives; honest caveat: this state is non-Markov w.r.t. R and L,
  which is fine here but will NOT be fine once the WAP rate is stochastic).
- Q-table of shape (T, 2). Every-visit Monte Carlo: each episode's terminal
  reward is credited to all (t, a) pairs visited (with gamma = 1 and only a
  terminal reward, the return at every step IS the terminal reward).
- Epsilon-greedy exploration with linear decay.
 
Ground truth
------------
Contribute at year t iff exp(mu * (T - t)) > kappa * (1 + tau)
  => switch date t* = T - ln(kappa * (1 + tau)) / mu   (~12.5 with defaults;
     the shortfall put nudges the exact switch slightly earlier).
"""
 
# --------------------------------------------------------------------------- 
# Parameters
# ---------------------------------------------------------------------------
T = 45          # career length in years
C_MAX = 1.0     # contribution when a = 1
G = 0.0175      # frozen WAP guarantee rate
MU = 0.03       # tarrif on mathematical reserve 
KAPPA = 0.5    # employer-cost weight (> 1, else the shortfall cancels) sensitivy 
SCALE = C_MAX * T * np.exp(MU * T)   # keeps rewards O(1)
 
N_EPISODES = 100_000

 
 # ---------------------------------------------------------------------------
# Environment: one function that plays a full episode given an action rule
# ---------------------------------------------------------------------------
 
def run_episode(choose_action, rng):
    """Play one career. choose_action(t) -> 0 or 1. Returns (actions, reward)."""
    R = L = C = 0.0
    actions = np.empty(T, dtype=np.int64)
    for t in range(T):
        a = choose_action(t)
        actions[t] = a
        c = C_MAX * a
        x = np.exp(MU)
        R = (R + c) * x
        L = (L + c) * (1.0 + G)
        C += c
    payout = max(R, L)
    shortfall = max(L - R, 0.0)
    reward = (payout - KAPPA * (C + shortfall)) / SCALE
    return actions, reward


# ---------------------------------------------------------------------------
# Monte Carlo control
# ---------------------------------------------------------------------------
 
def mc_control(n_episodes=N_EPISODES, seed=0, epsStart =1.0, epsEnd =0.01 ):
    rng = np.random.default_rng(seed)
    Q = np.zeros((T, 2))            # value estimates
    visits = np.zeros((T, 2))       # visit counts for running averages
    reward_trace = np.empty(n_episodes)
 
    for ep in range(n_episodes):
        eps = epsStart + (epsEnd - epsStart) * ep / n_episodes
 
        def choose_action(t):
            if rng.random() < eps:
                return int(rng.integers(2))          # explore
            return int(np.argmax(Q[t]))              # exploit
 
        actions, reward = run_episode(choose_action, rng)
        reward_trace[ep] = reward
 
        # Every-visit MC update: the terminal reward is the return at every t.
        for t in range(T):
            a = actions[t]
            visits[t, a] += 1
            Q[t, a] += (reward - Q[t, a]) / visits[t, a]   # incremental mean
 
    return {
        "Q": Q,
        "policy": np.argmax(Q, axis=1),
        "reward_trace": reward_trace,
    }
    

def analytic_switch():
    return T - np.log(KAPPA) / MU
 


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def plot_results(result, path="tests/mc_basic2.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Q, policy, trace = result["Q"], result["policy"], result["reward_trace"]
    t_star = analytic_switch()
    ts = np.arange(T)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
 
    axes[0].plot(ts, Q[:, 1] - Q[:, 0], marker="o")
    axes[0].axhline(0.0, color="grey", lw=0.8)
    axes[0].axvline(t_star, ls="--", color="grey",
                    label=f"analytic t* = {t_star:.1f}")
    axes[0].set_title("Q(t, contribute) - Q(t, don't)")
    axes[0].set_xlabel("year t")
    axes[0].legend()
 
    axes[1].step(ts, policy, where="mid")
    axes[1].axvline(t_star, ls="--", color="grey")
    axes[1].set_yticks([0, 1], ["don't", "contribute"])
    axes[1].set_title("Learned greedy policy")
    axes[1].set_xlabel("year t")
 
    window = 2000
    smoothed = np.convolve(trace, np.ones(window) / window, mode="valid")
    axes[2].plot(smoothed)
    axes[2].set_title(f"Episode reward ({window}-episode moving average)")
    axes[2].set_xlabel("episode")
 
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
 
 
if __name__ == "__main__":
    print(f"analytic switch date (put ignored): t* = {analytic_switch():.2f}")
    result = mc_control()
    print("learned policy (year: action):")
    print({t: int(a) for t, a in enumerate(result["policy"])})
    plot_results(result)
 


