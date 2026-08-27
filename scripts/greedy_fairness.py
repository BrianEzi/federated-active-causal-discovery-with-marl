"""Is the greedy baseline a fair opponent, or is it handicapped?

THE SUSPICION. `UncertaintyGreedyAgent` takes `bar=0.7` and every construction in the
repository uses that default, while the deterministic and attributed backends grade at
`claim_bar=1.0`. So greedy treats a claim that 70% of surviving hypotheses agree on as
SETTLED and stops scoring it -- and when nothing clears its own bar it returns
`window.pass_index` and forfeits the round. The environment meanwhile still counts that
claim open. If that is biting, every learned-vs-greedy margin in this project is inflated
by a bookkeeping mismatch rather than earned.

WHAT THIS MEASURES. The same greedy rule at its default bar and at the bar the task is
actually graded on, plus a variant that is forbidden to pass, all on IDENTICAL episodes
against the learned checkpoint and the computable ceiling. If the bar is the problem,
greedy at 1.0 closes most of the gap and the honest margin is much smaller.

The ceiling matters as much as the baseline: a margin over greedy says nothing about how
much of the ACHIEVABLE the policy captured, and for this configuration the ceiling is
known exactly (results/vs_strict/reference_4a_b8.json: 1.0).
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np

from ma.baselines import RandomAgent, UncertaintyGreedyAgent
from ma.env import PASS_ACTION
from ma.evaluate import run_arm
from ma.policy import IndependentPPO
from scripts.vs_evaluate import build_env


class NeverPassGreedy(UncertaintyGreedyAgent):
    """Greedy that spends the round on its best node even when nothing clears the bar.

    Separates two things the default conflates: WHICH node greedy picks, and WHETHER it
    acts at all. Passing is free only if the belief is genuinely finished; under a 1.0
    grading bar it usually is not.
    """

    def __call__(self, env, result) -> int:
        window = env.windows[self.agent]
        belief = window.belief.last
        if belief is None:
            return int(self.rng.integers(0, window.n_actions - 1))
        counts = self._unsure_touching(belief, window.k)
        scores = {node: counts[window.pos[node]] for node in window.authority}
        best = max(scores.values())
        if best <= 0:
            # The only change: pick uniformly among own nodes instead of forfeiting.
            node = int(self.rng.choice(sorted(scores)))
        else:
            candidates = [n for n, s in scores.items() if s == best]
            node = int(self.rng.choice(candidates))
        from ma.env import VARY
        return window.action_index(node, prefer=VARY)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph_model", default="sf", choices=["er", "sf"])
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--private_size", type=int, default=1)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    env = build_env(args.n_agents, args.budget, channels=True, partner_counts=True,
                    private_size=args.private_size, graph_model=args.graph_model,
                    seed=args.seed)
    agents = env.topology.agents

    arms: Dict[str, dict] = {}
    if args.policy and pathlib.Path(args.policy).exists():
        ppo = IndependentPPO.load(args.policy, env)
        arms["learned"] = ppo.policies(deterministic=False)
    # The bar sweep. 0.7 is what every call site in the repository actually builds.
    for bar in (0.7, 0.9, 1.0):
        arms[f"greedy_bar{bar}"] = {a: UncertaintyGreedyAgent(a, args.seed, bar=bar)
                                    for a in agents}
    arms["greedy_bar1.0_nopass"] = {a: NeverPassGreedy(a, args.seed, bar=1.0)
                                    for a in agents}
    arms["random_vary"] = {a: RandomAgent(a, args.seed, allow_clamp=False) for a in agents}

    report = {"config": vars(args), "arms": {}}
    print(f"\n=== {args.n_agents} agents, private {args.private_size}, budget "
          f"{args.budget}, {args.graph_model}, {args.episodes} identical episodes ===")
    print(f"{'arm':24s} {'joint success':>14s} {'rounds used':>12s} {'forfeits/agent':>15s}")
    for label, policies in arms.items():
        row = run_arm(env, policies, args.episodes, seed=args.seed)
        forfeits = row.get("forfeits_mean")
        if forfeits is None:
            forfeits = float("nan")
        report["arms"][label] = {k: v for k, v in row.items()
                                 if isinstance(v, (int, float, str))}
        print(f"{label:24s} {row['success']:14.3f} {row['mean_rounds']:12.2f} "
              f"{forfeits:15}")
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
