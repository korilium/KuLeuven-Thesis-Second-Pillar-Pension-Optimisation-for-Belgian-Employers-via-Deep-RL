import os
 
import numpy as np


"""Tabular Monte Carlo control on a deterministic pension contribution toy.

No neural networks, no libraries beyond numpy. One Q-table, one loop.

Environment
-----------
- T = 45 annual steps, binary action: contribute this year's premium or not.
- Premium is salary-linked: c_t = C_RATE * S_t, with S_t = S0 * (1+W)^t.
- Deterministic aggregate recursions (vintage-exact) under a frozen guarantee:
      R <- (R + c) * exp(MU)       reserve credited at the tariff MU
      L <- (L + c) * (1 + G)       WAP liability leg at the guarantee rate
- Rewards are emitted when cash flows occur, already discounted to t=0
  at rate DISC (one numeraire, owned by the environment):
      per year:  -KAPPA * c_t * exp(-DISC * t)
      terminal:  (max(R,L) - KAPPA * max(L-R, 0)) * exp(-DISC * T)
  KAPPA is the employer-cost weight; KAPPA = 1 makes the shortfall a pure
  transfer that cancels from the joint objective.
  Current simplification: DISC = MU (timing-neutral economy).

Agent
-----
- State: the year t only. Sufficient here because salary and rates are
  deterministic functions of time; must grow once anything stochastic
  enters (WAP rate, credited return).
- On-policy every-visit Monte Carlo control, epsilon-greedy with linear
  decay, gamma = 1 (financial discounting lives in the reward; gamma < 1
  would double-discount — do not tune).
- Update target is the return-from-t (reverse cumsum of rewards), so
  sunk costs never pollute later decisions' estimates.

Validation
----------
No closed forms: the benchmark evaluates the environment directly, so it
cannot go stale when run_episode changes.
- sweep_benchmark(): best switch policy by direct evaluation.
- numeric_gap(): per-year marginal value of contributing.
Pass criteria: learned policy matches the sweep's switch; MC Q-gaps track
numeric_gap(); reward trace plateaus at the benchmark value.
Caveats: single-episode evaluation and per-year decomposition are exact
only while the environment is deterministic and the reward is linear in
contributions — both need Monte Carlo averaging at the stochastic rung.
"""
 
# --------------------------------------------------------------------------- 
# Parameters
# ---------------------------------------------------------------------------
T = 45          # career length in years
G = 0.0175      # frozen WAP guarantee rate
MU = 0.03       # tarrif on mathematical reserve 
KAPPA = 2    # employer-cost weight (> 1, else the shortfall cancels) sensitivy 
S0 = 1.0        #starting salary
W = 0.02        # deterministic salary growth 
C_RATE = 0.05   # contribution as a fraction of the salary 
SCALE = C_RATE * S0 * T * np.exp(MU * T) # keeps rewards O(1)
DISC= MU

ALPHA= 0.02 #WEIGHT DECAY


 
 # ---------------------------------------------------------------------------
# Environment: one function that plays a full episode given an action rule
# ---------------------------------------------------------------------------
 
def run_episode(choose_action):
    """Play one career. choose_action(t) -> 0 or 1. Returns (actions, reward)."""
    R = L = 0.0
    actions = np.empty(T, dtype=np.int64)
    S = S0
    rewards = np.zeros(T) 
    for t in range(T):
        a = choose_action(t)
        actions[t] = a
        c = C_RATE * S *  a
        rewards[t] = -KAPPA * c * np.exp(-DISC * t) / SCALE
        R = (R + c) * np.exp(MU)
        L = (L + c) * (1.0 + G)
        S *= (1.0+ W)
    payout = max(R, L)
    shortfall = max(L - R, 0.0)
    rewards[-1] += (payout - KAPPA * shortfall)*np.exp(-DISC*T) / SCALE
    return actions, rewards


# ---------------------------------------------------------------------------
# Monte Carlo control
# ---------------------------------------------------------------------------
 
def mc_control(n_episodes=100000, seed=0, epsStart =1.0, epsEnd =0.05 ):
    rng = np.random.default_rng(seed)
    Q = np.zeros((T, 2))            # value estimates
    reward_trace = np.empty(n_episodes)
 
    for ep in range(n_episodes):
        eps = epsStart + (epsEnd - epsStart) * ep / n_episodes
 
        def choose_action(t):
            if rng.random() < eps:
                return int(rng.integers(2))          # explore
            return int(np.argmax(Q[t]))              # exploit
 
        actions, rewards = run_episode(choose_action)
        reward_trace[ep] = rewards.sum()   # G_0, comparable to before
        returns = np.cumsum(rewards[::-1])[::-1]  # G_t = Σ_{s≥t} r_s  (γ = 1) as already discounted with NPV 
        # Every-visit MC update: the terminal reward is the return at every t.
        for t in range(T):
            a = actions[t]
            Q[t, a] += ALPHA * (returns[t] - Q[t, a]) #apply weight decay 
    
    return {
        "Q": Q,
        "policy": np.argmax(Q, axis=1),
        "reward_trace": reward_trace,
    }

# ----------------------------------------------------------------------
# Environment-based benchmark (no learning) 
# ----------------------------------------------------------------------

def sweep_benchmark(n_grid=T + 1):
    """Best switch policy by direct evaluation — no learning, no formulas.
    Automatically stays correct under any change to run_episode."""
    best = {"switch": 0, "value": -np.inf}
    for k in range(n_grid):                    # contribute years 0..k-1
        actions, rewards = run_episode(lambda t: int(t < k))
        value = rewards.sum()                  # deterministic env: one episode suffices
        if value > best["value"]:
            best = {"switch": k, "value": value}
    return best


def numeric_gap():
    """Marginal value of contributing in each single year, measured directly
    from the environment. No formulas: stays correct under env changes.
    (Exact per-year decomposition only while the env is deterministic and
    the reward linear in contributions — revisit at the stochastic rung.)"""
    _, r_none = run_episode(lambda t: 0)
    base = r_none.sum()
    gaps = np.empty(T)
    for k in range(T):
        _, r = run_episode(lambda t, k=k: int(t == k))
        gaps[k] = r.sum() - base
    return gaps

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def plot_results(result, path="tests/mc_basic.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Q, policy, trace = result["Q"], result["policy"], result["reward_trace"]
    ts = np.arange(T)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bench = sweep_benchmark()
    # switch line only if the best policy is interior
    if bench is not None and 0 < bench["switch"] < T:
        for ax in (axes[0], axes[1]):
            ax.axvline(bench["switch"], ls="--", color="grey",
                   label=f"benchmark switch = {bench['switch']}")
    gap_numeric = numeric_gap()

    axes[0].plot(ts, gap_numeric, color="black", lw=1, label="numeric marginal gap")
    axes[0].plot(ts, Q[:, 1] - Q[:, 0], marker="o", label="MC estimate")
    axes[0].axhline(0.0, color="grey", lw=0.8)
    axes[0].legend()

    
    axes[1].step(ts, policy, where="mid")
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
    bench = sweep_benchmark()
    if bench["switch"] == T:
        print(f"benchmark: always contribute (value {bench['value']:.4f})")
    elif bench["switch"] == 0:
        print(f"benchmark: never contribute (value {bench['value']:.4f})")
    else:
        print(f"benchmark: contribute until year {bench['switch']} (value {bench['value']:.4f})")
    result = mc_control()
    print("learned policy (year: action):")
    print({t: int(a) for t, a in enumerate(result["policy"])})
    plot_results(result)
 


