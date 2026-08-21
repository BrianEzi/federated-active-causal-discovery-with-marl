"""GATE 4 -- the one that decides whether the two-agent case is a real problem.

The gates in `ma/gates.py` established that confounding is devastating: a confounded agent
identifies its own induced DAG from observation alone 0% of the time, with mean posterior
mass on the truth of 7.5e-08. No amount of its OWN data fixes that, because under
confounding no DAG over its window is correct.

The design claims the other agent can fix it: if B intervenes on the `z_B` responsible for
A's bidirected edge, the association breaks IN A'S OWN DATA, with no disclosure at all
(MA_DESIGN section 4). That claim has never been tested. This script tests it.

Three arms, identical episodes and identical true graphs:

    solo        A acts, B passes.        What A reaches alone.
    partner     A acts, B acts greedily. What A reaches when someone else is also
                                         experimenting on the shared system.
    oracle      A acts, B intervenes on its OWN PRIVATE node every round.
                                         The upper bound on rescue: B doing the one thing
                                         that can break A's confounding, deliberately.

The comparison is restricted to CONFOUNDED episodes, since those are the only ones where
the mechanism can possibly matter. Unconfounded episodes are reported alongside as a
control -- there, the three arms should be roughly equal, and if they are not, the effect
is not confounding-specific and the interpretation is wrong.

PRE-REGISTERED PREDICTION, before the numbers exist:
    oracle > partner > solo on confounded episodes, with solo near zero.
    On unconfounded episodes, all three within noise of each other.

    If solo == partner == oracle on confounded episodes, B cannot rescue A, the
    coordination story is false, and the two-agent design needs rethinking rather than
    training. That is a stop condition, not a tuning target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legacy.ma_v1.baselines import GreedyAgentPolicy, PassPolicy, RandomAgentPolicy
from legacy.ma_v1.env import CLAMP, PASS_ACTION, VARY, MAConfig, TwoAgentEnv
from ma.projection import bidirected_pairs
from ma.topology import Topology


class PrivateOnlyPolicy:
    """Intervene only on own private nodes. The deliberate-rescue arm.

    This is not a policy an agent would run for its own benefit -- its private nodes are
    often not what IT most needs to test. It exists to upper-bound what B's actions can do
    for A.
    """

    def __init__(self, name: str, env: TwoAgentEnv, seed: int = 0, mode: str = CLAMP):
        self.name = name
        view = env.views[name]
        # CLAMP, not VARY. Randomising the confounder replaces one latent common cause
        # with another and cuts nothing -- measured 2026-08-16, 0.0% rescue at scale 2.0
        # and 1.0 against ~18% at scale 0.1 and 0.0.
        self.actions = [view.actions.index((node, mode)) for node in view.private]
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __call__(self, env, result) -> int:
        return int(self.rng.choice(self.actions))


def run_arm(config: MAConfig, arm: str, episodes: int, seed: int) -> list:
    env = TwoAgentEnv(config, seed=seed)
    policy_a = GreedyAgentPolicy("A", env, seed=seed)
    if arm == "solo":
        policy_b = PassPolicy("B")
    elif arm == "partner":
        policy_b = GreedyAgentPolicy("B", env, seed=seed + 1)
    elif arm == "oracle":
        policy_b = PrivateOnlyPolicy("B", env, seed=seed + 1, mode=CLAMP)
    elif arm == "oracle_vary":
        # The same deliberate-rescue arm but randomising instead of clamping. Included so
        # the mode effect is visible in the gate itself rather than only in a diagnostic.
        policy_b = PrivateOnlyPolicy("B", env, seed=seed + 1, mode=VARY)
    else:
        raise ValueError(arm)

    rows = []
    for ep in range(episodes):
        # Same seed across arms => same true graph and same SCM. The arms differ only in
        # what B does.
        result = env.reset(seed=seed * 1_000_000 + ep)
        policy_a.reset(seed=seed * 7919 + ep)
        policy_b.reset(seed=seed * 6271 + ep)

        confounded = len(bidirected_pairs(env.true_adjacency, env.views["A"].nodes)) > 0
        steps = 0
        while not result.done and steps < config.budget:
            result = env.step(policy_a(env, result), policy_b(env, result))
            steps += 1

        rows.append({
            "episode": ep,
            "arm": arm,
            "confounded_A": bool(confounded),
            "identified_A": bool(result.identified["A"]),
            "true_mass_A": float(result.info["true_mass"]["A"]),
            "steps": steps,
        })
    return rows


def summarise(rows, arm, confounded):
    subset = [r for r in rows if r["arm"] == arm and r["confounded_A"] == confounded]
    if not subset:
        return None
    ident = np.array([r["identified_A"] for r in subset], dtype=float)
    mass = np.array([r["true_mass_A"] for r in subset])
    n = len(subset)
    z = 1.96
    p = ident.mean()
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {
        "n": n,
        "identified_rate": float(p),
        "ci": [float(max(0.0, centre - half)), float(min(1.0, centre + half))],
        "mean_true_mass": float(mass.mean()),
        "median_true_mass": float(np.median(mass)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--n_obs", type=int, default=2000)
    ap.add_argument("--n_int", type=int, default=200)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ma/coordination_gate.json")
    args = ap.parse_args()

    topology = Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                      budget=args.budget)

    rows = []
    for arm in ("solo", "partner", "oracle", "oracle_vary"):
        rows.extend(run_arm(config, arm, args.episodes, args.seed))
        print(f"{arm} done", flush=True)

    report = {"args": vars(args), "confounded": {}, "unconfounded": {}}
    for arm in ("solo", "partner", "oracle", "oracle_vary"):
        report["confounded"][arm] = summarise(rows, arm, True)
        report["unconfounded"][arm] = summarise(rows, arm, False)

    print("\nA's identification rate, CONFOUNDED episodes:")
    for arm in ("solo", "partner", "oracle"):
        s = report["confounded"][arm]
        if s:
            print(f"  {arm:>8}: {s['identified_rate']:.3f} "
                  f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  n={s['n']}  "
                  f"mean mass {s['mean_true_mass']:.4f}")
    print("A's identification rate, UNCONFOUNDED episodes (control):")
    for arm in ("solo", "partner", "oracle", "oracle_vary"):
        s = report["unconfounded"][arm]
        if s:
            print(f"  {arm:>8}: {s['identified_rate']:.3f} "
                  f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  n={s['n']}  "
                  f"mean mass {s['mean_true_mass']:.4f}")

    solo = report["confounded"]["solo"]
    oracle = report["confounded"]["oracle"]
    if solo and oracle:
        report["gate4_passed"] = bool(oracle["ci"][0] > solo["ci"][1])
        print(f"\nGATE 4 {'PASSED' if report['gate4_passed'] else 'FAILED'}"
              f" -- oracle rescue {'is' if report['gate4_passed'] else 'is NOT'}"
              f" measurably above solo on confounded episodes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
