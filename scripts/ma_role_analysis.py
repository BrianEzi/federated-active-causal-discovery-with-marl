"""Do the agents DIFFERENTIATE, or do they both clamp at once?

I reported the 2026-08-17 result as "learned to clamp but not when", treating the 84-96%
clamp rate as indiscriminate. That framing is probably wrong, and this measures the thing it
missed.

Reconsider what clamping does. From the scoring sweep, p(clamp)=1.0 was best for agent A on
BOTH confounded and unconfounded episodes -- so from A's point of view, having its partner
clamp always is genuinely optimal, and B clamping constantly is not irrational. But a
clamping agent is NOT experimenting: it spends its budget holding a variable still for
someone else's benefit and learns nothing about its own graph. If both agents clamp in the
same round, the round is wasted for both.

So the coordinated solution is not "clamp less". It is ROLE DIFFERENTIATION: in each round
one agent clamps while the other experiments, and they swap. Seed 2's failure -- clamp
0.957, solve 0.165, below random -- looks exactly like both agents clamping at once.

That makes the diagnostic a joint distribution over the pair's actions, not a per-agent
rate:

    both clamp        wasted round, nobody learns
    exactly one       the coordinated pattern
    neither           ordinary parallel experimentation

PRE-REGISTERED, before the numbers exist:
    If role differentiation is being learned, P(exactly one clamps) should exceed what
    independent clamping at the observed marginal rates would produce. That independence
    baseline is the right null: two agents clamping 90% of the time each, with no
    coordination, give P(exactly one) = 2 x 0.9 x 0.1 = 0.18. Anything near that is
    "both clamp a lot", not "they take turns".

    I expect the good seeds (0 and 1) to sit ABOVE the independence baseline and seed 2 to
    sit at or below it. If ALL seeds sit at the baseline, no differentiation is being
    learned and the coordination claim is weaker than I have been stating.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ma.env import MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO, MAPPOConfig
from ma.projection import bidirected_pairs
from ma.topology import Topology


def analyse(agent, config, episodes, seed):
    env = TwoAgentEnv(config, seed=seed)
    counts = {"both": 0, "one": 0, "neither": 0}
    conf_counts = {"both": 0, "one": 0, "neither": 0}
    clamp_a = clamp_b = rounds = conf_rounds = 0
    alternation = 0          # rounds where the clamping agent differs from last round
    last_clamper = None
    solved = []

    for ep in range(episodes):
        result = env.reset(seed=seed * 500_000 + ep)
        confounded = any(
            len(bidirected_pairs(env.true_adjacency, env.views[n].nodes)) > 0
            for n in ("A", "B"))
        steps = 0
        while not result.done and steps < config.budget:
            actions, clamping = {}, {}
            for name in ("A", "B"):
                index, _, _ = agent._act(name, env.observation(name), deterministic=True)
                actions[name] = index
                target, mode = env.views[name].actions[index]
                clamping[name] = (target != -1 and mode == "clamp")

            n_clamp = sum(clamping.values())
            key = "both" if n_clamp == 2 else "one" if n_clamp == 1 else "neither"
            counts[key] += 1
            if confounded:
                conf_counts[key] += 1
                conf_rounds += 1
            clamp_a += clamping["A"]
            clamp_b += clamping["B"]
            rounds += 1

            if n_clamp == 1:
                who = "A" if clamping["A"] else "B"
                if last_clamper is not None and who != last_clamper:
                    alternation += 1
                last_clamper = who

            result = env.step(actions["A"], actions["B"])
            steps += 1
        solved.append(float(result.info["both_identified"]))

    rate_a = clamp_a / max(rounds, 1)
    rate_b = clamp_b / max(rounds, 1)
    # Null model: the two agents clamp independently at their own observed marginal rates.
    independent_one = rate_a * (1 - rate_b) + rate_b * (1 - rate_a)
    observed_one = counts["one"] / max(rounds, 1)

    return {
        "rounds": rounds,
        "clamp_rate_A": rate_a,
        "clamp_rate_B": rate_b,
        "p_both": counts["both"] / max(rounds, 1),
        "p_one": observed_one,
        "p_neither": counts["neither"] / max(rounds, 1),
        "p_one_if_independent": independent_one,
        "differentiation": observed_one - independent_one,
        "alternation_rate": alternation / max(counts["one"], 1),
        "p_both_confounded": conf_counts["both"] / max(conf_rounds, 1),
        "p_one_confounded": conf_counts["one"] / max(conf_rounds, 1),
        "solve_rate": float(np.mean(solved)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="results/ma/checkpoints")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--out", default="results/ma/role_analysis.json")
    args = ap.parse_args()

    topology = Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    checkpoints = sorted(Path(args.checkpoint_dir).glob("*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {args.checkpoint_dir}")

    rows = []
    print(f"{'policy':>24} {'clampA':>7} {'clampB':>7} {'P(both)':>8} {'P(one)':>7} "
          f"{'indep':>7} {'diff':>7} {'alt':>6} {'solve':>6}")
    for path in checkpoints:
        placeholder = MAConfig(topology=topology, budget=args.budget)
        agent = IndependentPPO(placeholder, MAPPOConfig(total_episodes=0, seed=0))
        meta = agent.load(path)
        config = MAConfig(topology=topology, budget=args.budget,
                          score_rule=meta["trained_under_rule"])
        stats = analyse(agent, config, args.episodes, seed=99)
        stats.update({"policy": path.stem, "trained_under": meta["trained_under_rule"]})
        rows.append(stats)
        print(f"{path.stem:>24} {stats['clamp_rate_A']:>7.3f} {stats['clamp_rate_B']:>7.3f} "
              f"{stats['p_both']:>8.3f} {stats['p_one']:>7.3f} "
              f"{stats['p_one_if_independent']:>7.3f} {stats['differentiation']:>+7.3f} "
              f"{stats['alternation_rate']:>6.3f} {stats['solve_rate']:>6.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    positive = [r for r in rows if r["differentiation"] > 0.05]
    print(f"{len(positive)}/{len(rows)} policies show differentiation above the "
          f"independence baseline by more than 0.05")


if __name__ == "__main__":
    main()
