"""Pretrain ONE policy across a mixture of federation shapes, then test it zero-shot.

WHY THIS SCRIPT EXISTS. Every policy trained before today was locked to a single window
size and a single agent count, because the network's partner features were concatenated in
agent order and its role vector was a saved buffer. `PortableRoleActorCritic` removes both,
so one set of weights can act in a 4-agent k=4 federation and an 8-agent k=5 one. That is
the prerequisite for the thesis's framing -- pretrain in the deterministic idealisation on
DIVERSE topologies, then transfer -- and until now the diversity half was not expressible.

THE CONTROL THAT DECIDES IT, and it is run here rather than left for later: for every
held-out shape the script reports the pretrained policy zero-shot AND a policy trained on
that shape alone for the same number of episodes, against greedy and random. Without the
scratch arm, a mixture policy that merely does as well as greedy proves nothing about
transfer -- greedy needs no training at all.

WHAT IS SHARED AND WHAT IS NOT. One network serves every agent (parameter sharing), which
is necessary because the number of agents varies and there is no stable agent identity to
give a network to. Execution stays decentralised: each agent acts on its own observation,
no critic sees the joint state, and nothing pools observations. That is not CTDE, but it is
a departure from "one independent learner per agent" and is reported as one.

Pooling partners also costs something real, and it is stated in `PortableRoleActorCritic`:
the policy sees HOW MANY partners did what, never WHICH. Per-partner attribution is not
expressible by this variant.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from ma.baselines import make_baselines
from ma.env import ROUND_ROBIN, VARY, MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import ER, SF, federated_topology
from scripts.vs_evaluate import run_ceiling, run_policy

# (n_agents, private_size, n_shared, budget). Budgets are TWICE the agent count, which is
# what the strict grading needs: at one round per agent the optimum is censored in most
# episodes (measured 2026-08-26, 25/40 unsolvable at 4 agents), so a comparison there runs
# partly inside a regime where no arm can win.
TRAIN_MIX = ((3, 1, 3, 6), (4, 1, 3, 8), (5, 1, 3, 10), (4, 2, 3, 10))
HELD_OUT = ((6, 1, 3, 12), (4, 3, 3, 14), (8, 1, 3, 16))

# RANGES for `--sample_shapes`, which draws a fresh federation every batch instead of
# cycling a fixed list. Student's call, 2026-08-26: a general policy cannot come out of a
# fixed topology, so the topology should be part of what varies. Evaluation stays on the
# fixed ladder above so that rungs remain comparable to each other and to earlier runs.
AGENT_RANGE = (3, 6)
PRIVATE_RANGE = (1, 3)
SHARED_RANGE = (2, 4)
_ENV_CACHE: dict = {}


def build(shape, arch="gnn_portable", seed=0, graph_model=ER, sf_m=2) -> TwoAgentEnv:
    n_agents, private_size, n_shared, budget = shape
    topology = federated_topology(n_agents, private_size, n_shared)
    config = MAConfig(topology=topology, n_obs=60, n_int=20, budget=budget,
                      disclose_regime=True, turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="version_space", policy_arch=arch,
                      episode_mix="confounded", reward_criterion="claims", claim_bar=1.0,
                      per_agent_reward=True, observe_belief_channels=True,
                      observe_partner_counts=True, graph_model=graph_model, sf_m=sf_m)
    return TwoAgentEnv(config, seed=seed)


def cached(shape, **kw) -> TwoAgentEnv:
    """One environment per distinct shape, reused across batches.

    Building an environment is cheap but not free, and a randomised run draws hundreds of
    batches from a handful of distinct shapes. Reuse is safe because `reset` rebuilds
    everything an episode depends on -- the graph, the parameters, and the version space.
    """
    key = (shape, tuple(sorted(kw.items())))
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = build(shape, **kw)
    return _ENV_CACHE[key]


def sample_shape(rng, max_window: int = 6) -> tuple:
    """Draw a federation. Window size is capped because the version space enumerates
    3^(edges in a window) and k > 6 is past the measured usable range -- a draw that
    exceeded it would not be a harder episode, it would be a hung one."""
    while True:
        n_agents = int(rng.integers(AGENT_RANGE[0], AGENT_RANGE[1] + 1))
        private_size = int(rng.integers(PRIVATE_RANGE[0], PRIVATE_RANGE[1] + 1))
        n_shared = int(rng.integers(SHARED_RANGE[0], SHARED_RANGE[1] + 1))
        if private_size + n_shared <= max_window:
            return (n_agents, private_size, n_shared, 2 * n_agents)


def label(shape) -> str:
    n_agents, private_size, n_shared, budget = shape
    return f"{n_agents}a x{private_size}p +{n_shared}s (k={private_size + n_shared}, b={budget})"


def train_mixture(shapes, episodes, config: PPOConfig, verbose=True, sampler=None,
                  graph_model=ER, sf_m=2) -> IndependentPPO:
    """Train one shared network over many federation shapes, one batch per shape.

    Rebinding BETWEEN batches rather than interleaving within one: an update pools the
    batch into a single tensor, and observations of different widths cannot be pooled. Each
    batch is therefore homogeneous and the mixture happens across batches, which is what
    makes the gradient well defined.

    `sampler`, when given, draws a fresh shape for every batch instead of cycling `shapes`.
    That is the general-policy setting: the topology becomes part of what the policy has to
    be robust to, rather than a constant it can overfit. `shapes` is then used only to seed
    the first environment.
    """
    kw = {"graph_model": graph_model, "sf_m": sf_m}
    learner = IndependentPPO(cached(shapes[0], **kw), config)
    batches = max(1, episodes // config.episodes_per_update)
    seen: dict = {}
    for batch_index in range(batches):
        shape = sampler() if sampler else shapes[batch_index % len(shapes)]
        learner.bind(cached(shape, **kw))
        batch = learner.collect(config.episodes_per_update,
                                batch_index * config.episodes_per_update, mask_pass=False)
        learner.update(batch["buffers"])
        seen[label(shape)] = seen.get(label(shape), 0) + config.episodes_per_update
        learner.history.append({"batch": batch_index, "shape": label(shape),
                                "entropy": batch["entropy"],
                                "solve_rate": batch["solve_rate"],
                                "window_rate": batch["window_rate"]})
        if verbose and batch_index % 50 == 0:
            recent = learner.history[-25:]
            mean = float(np.mean([r["window_rate"] for r in recent]))
            print(f"  batch {batch_index:5d} / {batches}   window rate (last 25 batches, "
                  f"mixed shapes) {mean:.3f}   shapes seen {len(seen)}", flush=True)
    if verbose:
        print("  episodes per shape: "
              + ", ".join(f"{k} {v}" for k, v in sorted(seen.items())), flush=True)
    learner.shapes_seen = seen
    return learner


def evaluate(env, arms, episodes, ceiling_episodes) -> dict:
    ceil, _ = run_ceiling(env, ceiling_episodes)
    out = {"ceiling": float(ceil.mean())}
    for name, policies in arms.items():
        rates, rounds, duplicates = run_policy(env, policies, episodes)
        out[name] = {
            "window_rate": float(rates.mean()),
            "window_stderr": float(rates.std(ddof=1) / np.sqrt(len(rates))),
            "rounds": float(rounds.mean()),
            "rounds_stderr": float(rounds.std(ddof=1) / np.sqrt(len(rounds))),
            "duplicate_coverage": float(duplicates.mean()),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_episodes", type=int, default=12000)
    ap.add_argument("--scratch_episodes", type=int, default=None,
                    help="episodes for the per-shape control; default: train_episodes")
    ap.add_argument("--eval_episodes", type=int, default=150)
    ap.add_argument("--ceiling_episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_shapes", action="store_true",
                    help="draw a fresh federation every batch instead of cycling the fixed "
                         "mixture -- the general-policy setting. Evaluation stays on the "
                         "fixed ladder either way")
    ap.add_argument("--graph_model", default=ER, choices=[ER, SF])
    ap.add_argument("--sf_m", type=int, default=2)
    ap.add_argument("--no_scratch", action="store_true",
                    help="skip the trained-from-scratch control (it is the arm that "
                         "decides whether pretraining did anything -- skip deliberately)")
    ap.add_argument("--out", default="results/vs_portable/mixture.json")
    args = ap.parse_args(argv)

    started = time.time()
    ppo_config = PPOConfig(total_episodes=args.train_episodes, seed=args.seed,
                           episodes_per_update=16, gnn_layers=2)
    print(f"pretraining on {len(TRAIN_MIX)} shapes: "
          + ", ".join(label(s) for s in TRAIN_MIX), flush=True)
    rng = np.random.default_rng(args.seed)
    sampler = (lambda: sample_shape(rng)) if args.sample_shapes else None
    if args.sample_shapes:
        print(f"  sampling shapes: {AGENT_RANGE[0]}-{AGENT_RANGE[1]} agents, "
              f"{PRIVATE_RANGE[0]}-{PRIVATE_RANGE[1]} private, "
              f"{SHARED_RANGE[0]}-{SHARED_RANGE[1]} shared, window <= 6", flush=True)
    learner = train_mixture(TRAIN_MIX, args.train_episodes, ppo_config, sampler=sampler,
                            graph_model=args.graph_model, sf_m=args.sf_m)
    train_seconds = time.time() - started
    print(f"  pretrained in {train_seconds / 60:.1f} min", flush=True)

    report = {"train_mix": ("sampled" if args.sample_shapes
                            else [label(s) for s in TRAIN_MIX]),
              "shapes_seen": getattr(learner, "shapes_seen", None),
              "graph_model": args.graph_model,
              "train_episodes": args.train_episodes, "seed": args.seed,
              "train_seconds": train_seconds, "shapes": {}}

    scratch_episodes = args.scratch_episodes or args.train_episodes
    for shape in TRAIN_MIX + HELD_OUT:
        env = build(shape, seed=args.seed, graph_model=args.graph_model, sf_m=args.sf_m)
        agents = list(env.topology.agents)
        reference = {a: make_baselines(env, a, seed=args.seed) for a in agents}
        arms = {
            "pretrained": learner.bind(env).policies(deterministic=False),
            "greedy_uncertainty": {a: reference[a]["greedy_uncertainty"] for a in agents},
            "random_vary": {a: reference[a]["random_vary"] for a in agents},
        }
        if not args.no_scratch:
            # Same architecture, same episode count, this shape only. Anything the
            # pretrained arm gains over THIS is attributable to the mixture rather than to
            # the architecture -- which is the confound the control exists to remove.
            scratch_env = build(shape, seed=args.seed + 1,
                                graph_model=args.graph_model, sf_m=args.sf_m)
            scratch = IndependentPPO(scratch_env, PPOConfig(
                total_episodes=scratch_episodes, seed=args.seed,
                episodes_per_update=16, gnn_layers=2))
            scratch.train(verbose=False)
            arms["scratch"] = scratch.bind(env).policies(deterministic=False)

        held_out = shape in HELD_OUT
        row = evaluate(env, arms, args.eval_episodes, args.ceiling_episodes)
        row["held_out"] = held_out
        report["shapes"][label(shape)] = row
        tag = "HELD OUT" if held_out else "in mixture"
        print(f"\n{label(shape)}  [{tag}]   ceiling {row['ceiling']:.3f}", flush=True)
        for name in ("pretrained", "scratch", "greedy_uncertainty", "random_vary"):
            if name not in row:
                continue
            cell = row[name]
            print(f"  {name:20s} {cell['window_rate']:.3f} +/- {cell['window_stderr']:.3f}"
                  f"   rounds {cell['rounds']:.2f}"
                  f"   duplicates {cell['duplicate_coverage']:.3f}", flush=True)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    checkpoint = path.with_suffix(".pt")
    learner.save(checkpoint)
    print(f"\nwrote {path} and {checkpoint}")


if __name__ == "__main__":
    main()
