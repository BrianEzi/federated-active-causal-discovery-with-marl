"""Can the oracle's choice be decoded from what the agent actually sees?

This separates two explanations that look identical from the outside, and which the RL
results alone cannot distinguish:

  A. The observation carries the answer, but PPO is failing to find it.
  B. The observation does not carry a decodable answer, so no amount of RL tuning helps.

The test is supervised and deliberately generous. Collect (observation, oracle's best
target) pairs from real episodes, then train the SAME network architecture the agent uses
on them, with abundant labels, full supervision, and no exploration problem. If a network
of that capacity cannot learn the mapping even under these conditions, the agent was never
going to learn it from a reward signal, and the finding is about the representation rather
than the algorithm.

The comparison that matters is against two baselines, because raw accuracy means little:

  * `majority` -- always predict the most common target. Beats a broken probe.
  * `chance`   -- 1/d.

A probe that lands near either has learned nothing.

Run for both observation kinds: `edge_marginals` is what scales, `posterior` is the exact
sufficient statistic. A gap between them localises the loss to the compression, which is
precisely what the A-vs-B comparison exists to measure.

Note the label is the oracle's *set* of tied-best targets, not a single index: ties are
genuine, and scoring a tied choice as wrong would measure enumeration order rather than
the probe. Steps where nothing is informative are dropped, for the same reason they are
excluded from the agent's optimality rate -- every choice is vacuously correct there.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from sa.baselines import GreedyOraclePolicy, RandomPolicy
from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.graphs import build_graph_space
from sa.oracle import InterventionOracle
from sa.policy import ActorCritic, PerNodeActorCritic


def collect(env, oracle, n_episodes, kind, rng, explore) -> tuple:
    """Observations paired with the oracle's tied-best target mask.

    Episodes are driven by a mix of the oracle and a random policy: following only the
    oracle would visit just the states a good policy reaches, and the agent needs to act
    correctly from the states it actually finds itself in.
    """
    greedy = GreedyOraclePolicy(env.space, seed=0)
    random_policy = RandomPolicy(seed=1)

    observations, masks = [], []
    for episode in range(n_episodes):
        result = env.reset(seed=int(rng.integers(1 << 31)))
        while not result.done:
            scores, best = oracle.best_targets(result.posterior)
            if scores.max() > 1e-9:          # skip uninformative steps
                observations.append(env.observation(kind))
                masks.append(best.astype(np.float32))
            driver = random_policy if rng.random() < explore else greedy
            action = driver(env, result)
            if action == PASS_ACTION:
                break
            result = env.step(action)
    return np.array(observations, dtype=np.float32), np.array(masks, dtype=np.float32)


def train_probe(x, y, d, hidden, epochs, lr, seed, arch="flat",
                include_counts=False) -> dict:
    """Train the agent's own architecture to predict the oracle's best target."""
    torch.manual_seed(seed)
    n_train = int(len(x) * 0.8)
    xtr, ytr = torch.as_tensor(x[:n_train]), torch.as_tensor(y[:n_train])
    xte, yte = torch.as_tensor(x[n_train:]), torch.as_tensor(y[n_train:])

    # Same network the agent would use, so the probe measures the architecture's ability to
    # express the mapping rather than some other model's.
    if arch == "pernode":
        net = PerNodeActorCritic(d, hidden=hidden, include_counts=include_counts,
                                 allow_pass=False)
    else:
        net = ActorCritic(obs_dim=x.shape[1], n_actions=d, hidden=hidden)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)

    for _ in range(epochs):
        order = torch.randperm(len(xtr))
        for start in range(0, len(xtr), 256):
            idx = order[start:start + 256]
            logits, _ = net(xtr[idx])
            # Cross-entropy against the tied-best SET: maximise the total probability the
            # probe puts on any optimal target.
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(log_probs * ytr[idx]).sum(1).div(ytr[idx].sum(1)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    with torch.no_grad():
        chosen = net(xte)[0].argmax(1)
        hit = yte[torch.arange(len(yte)), chosen]
        accuracy = float(hit.mean())

    # Baselines. `majority` is the honest floor: a probe that has learned nothing but the
    # label distribution can still look good when one target dominates.
    counts = y[:n_train].sum(0)
    majority = int(np.argmax(counts))
    majority_accuracy = float(y[n_train:][:, majority].mean())
    # A tied-best set of size k makes chance better than 1/d; account for it honestly.
    chance = float(y[n_train:].mean())

    return {"probe_accuracy": accuracy, "majority_accuracy": majority_accuracy,
            "chance_accuracy": chance, "n_train": n_train, "n_test": len(xte),
            "mean_tied_best": float(y.sum(1).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--explore", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="flat",
                        choices=["flat", "pernode", "both"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    space = build_graph_space(args.d)
    oracle = InterventionOracle(space)
    env = CausalDiscoveryEnv(EnvConfig(d=args.d, budget=20), space=space)

    results = {"d": args.d, "episodes": args.episodes, "conditions": {}}
    print(f"d={args.d}  {space.n_dags} DAGs / {space.n_mecs} classes")

    architectures = ["flat", "pernode"] if args.arch == "both" else [args.arch]
    for kind in ("edge_marginals", "posterior"):
        rng = np.random.default_rng(args.seed)
        x, y = collect(env, oracle, args.episodes, kind, rng, args.explore)
        if len(x) < 100:
            print(f"  {kind}: too few informative steps ({len(x)})")
            continue
        for arch in architectures:
            # The per-node scorer reads a [d, d] marginal matrix out of the observation,
            # which the exact posterior simply is not.
            if arch == "pernode" and kind != "edge_marginals":
                continue
            stats = train_probe(x, y, args.d, args.hidden, args.epochs, args.lr,
                                args.seed, arch=arch)
            results["conditions"][f"{kind}/{arch}"] = stats
            print(f"  {kind:<16} {arch:<8} probe {stats['probe_accuracy']:.3f}  "
                  f"majority {stats['majority_accuracy']:.3f}  "
                  f"chance {stats['chance_accuracy']:.3f}  "
                  f"(n={stats['n_train']}+{stats['n_test']}, "
                  f"mean tied-best {stats['mean_tied_best']:.2f})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
