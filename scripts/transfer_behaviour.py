"""Transfer, with the BEHAVIOURAL metric that says why it failed.

The standing result is a bare null: a policy trained in the deterministic idealisation gets
no reliable benefit on noisy data. A null is a weak thing to report, because two completely
different explanations produce it and the outcome metric cannot separate them:

  A. THE BEHAVIOUR DID NOT TRANSFER. The policy stops dividing the work -- it sees a belief
     shaped unlike anything it trained on and its choices degrade toward random.
  B. THE BEHAVIOUR TRANSFERRED AND STOPPED PAYING. The policy still divides the work
     perfectly, and the noisy engine cannot convert good experiment selection into settled
     claims.

They imply opposite next steps. A says fix the policy -- domain randomisation over belief
sharpness, finetuning. B says the policy is fine and the INFERENCE ENGINE is the bottleneck,
in which case no amount of policy work helps.

DUPLICATE COVERAGE separates them, and it is the reason it was built to be belief-free: it
counts how often two agents spend rounds on the same shared variable, which is a fact about
what agents DID and means exactly the same thing in both environments. Identification rates
do not -- they are not comparable across engines at all.

  transferred duplicates ~ deterministic duplicates, identification flat  =>  B
  transferred duplicates ~ random duplicates                             =>  A

Reported with the deterministic-environment value of the SAME policy alongside, because the
question is whether the behaviour changed, and that needs both ends.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from cb.claims import score_window
from ma.baselines import make_baselines
from ma.env import ROUND_ROBIN, VARY, MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO
from ma.topology import federated_topology


def statistical_env(n_agents, private_size, n_shared, budget, n_obs, n_int, n_boot,
                    seed=0, channels=True, counts=True):
    config = MAConfig(topology=federated_topology(n_agents, private_size, n_shared),
                      n_obs=n_obs, n_int=n_int, budget=budget, disclose_regime=True,
                      turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="constraint", cb_n_boot=n_boot, policy_arch="gnn",
                      episode_mix="confounded", reward_criterion="claims",
                      observe_belief_channels=channels, observe_partner_counts=counts)
    return TwoAgentEnv(config, seed=seed)


def deterministic_env(n_agents, private_size, n_shared, budget, seed=0,
                      channels=True, counts=True):
    config = MAConfig(topology=federated_topology(n_agents, private_size, n_shared),
                      n_obs=60, n_int=20, budget=budget, disclose_regime=True,
                      turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="version_space", policy_arch="gnn",
                      episode_mix="confounded", reward_criterion="claims", claim_bar=1.0,
                      per_agent_reward=True, observe_belief_channels=channels,
                      observe_partner_counts=counts)
    return TwoAgentEnv(config, seed=seed)


def measure(env, policies, episodes, seed_base=70_000):
    """Identification AND duplicate coverage, per episode, on the same episodes."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(0)
    rates, duplicates = [], []
    for episode in range(episodes):
        result = env.reset(seed=seed_base + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
        identified = []
        for agent in env.topology.agents:
            window = env.windows[agent]
            identified.append(float(score_window(
                window.belief.last, env._true_mag(agent),
                [window.pos[n] for n in window.private],
                bar=env.config.claim_bar).identified))
        rates.append(float(np.mean(identified)))
        duplicates.append(env.duplicate_coverage())
    return np.array(rates, float), np.array(duplicates, float)


def cell(name, rates, duplicates):
    return {"name": name,
            "identified": float(rates.mean()),
            "identified_se": float(rates.std(ddof=1) / np.sqrt(len(rates))),
            "duplicate_coverage": float(duplicates.mean()),
            "duplicate_se": float(duplicates.std(ddof=1) / np.sqrt(len(duplicates)))}


def show(row, indent="  "):
    print(f"{indent}{row['name']:26s} identified {row['identified']:.3f} "
          f"+/- {row['identified_se']:.3f}    duplicates {row['duplicate_coverage']:.3f} "
          f"+/- {row['duplicate_se']:.3f}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policies", nargs="+", required=True,
                    help="deterministic-trained checkpoints to transfer")
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--private_size", type=int, default=1)
    ap.add_argument("--n_shared", type=int, default=3)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=1000)
    ap.add_argument("--cb_n_boot", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--out", default="results/transfer/behaviour.json")
    args = ap.parse_args(argv)

    shape = (args.n_agents, args.private_size, args.n_shared, args.budget)
    noisy = statistical_env(*shape, args.n_obs, args.n_int, args.cb_n_boot)
    clean = deterministic_env(*shape)
    agents = list(noisy.topology.agents)
    report = {"shape": {"n_agents": args.n_agents, "private_size": args.private_size,
                        "n_shared": args.n_shared, "budget": args.budget},
              "episodes": args.episodes, "noisy": [], "clean": []}

    print(f"{args.n_agents} agents x {args.private_size} private + {args.n_shared} shared, "
          f"budget {args.budget}, {args.episodes} episodes\n")

    print("DETERMINISTIC environment -- where the policies were trained")
    reference = {a: make_baselines(clean, a, seed=0) for a in agents}
    for label in ("greedy_uncertainty", "random_vary"):
        row = cell(label, *measure(clean, {a: reference[a][label] for a in agents},
                                   args.episodes))
        report["clean"].append(row)
        show(row)
    for path in args.policies:
        ppo = IndependentPPO.load(path, clean)
        row = cell(f"learned ({pathlib.Path(path).stem})",
                   *measure(clean, ppo.policies(deterministic=False), args.episodes))
        report["clean"].append(row)
        show(row)

    print("\nSTATISTICAL environment -- transferred, never trained here")
    reference = {a: make_baselines(noisy, a, seed=0) for a in agents}
    for label in ("greedy_uncertainty", "random_vary"):
        row = cell(label, *measure(noisy, {a: reference[a][label] for a in agents},
                                   args.episodes))
        report["noisy"].append(row)
        show(row)
    for path in args.policies:
        # `allow_backend_transfer` is the whole experiment: performance normally belongs to
        # the (policy, backend) pair, and this asks whether experiment SELECTION is a
        # structural skill that survives the change of engine.
        ppo = IndependentPPO.load(path, noisy, allow_backend_transfer=True)
        row = cell(f"transferred ({pathlib.Path(path).stem})",
                   *measure(noisy, ppo.policies(deterministic=False), args.episodes))
        report["noisy"].append(row)
        show(row)

    # The verdict, stated in the terms the metric was built for.
    def find(rows, prefix):
        hits = [r for r in rows if r["name"].startswith(prefix)]
        return float(np.mean([r["duplicate_coverage"] for r in hits])) if hits else None

    learned_clean = find(report["clean"], "learned")
    transferred = find(report["noisy"], "transferred")
    noisy_random = find(report["noisy"], "random")
    if None not in (learned_clean, transferred, noisy_random):
        to_trained = abs(transferred - learned_clean)
        to_random = abs(transferred - noisy_random)
        verdict = ("BEHAVIOUR TRANSFERRED (duplicate coverage stayed near its trained "
                   "value) -- the engine, not the policy, is the bottleneck"
                   if to_trained < to_random else
                   "BEHAVIOUR DID NOT TRANSFER (duplicate coverage drifted toward random) "
                   "-- the policy degraded, so finetuning or randomisation is the lever")
        print(f"\nduplicate coverage: trained {learned_clean:.3f} -> transferred "
              f"{transferred:.3f}, random here {noisy_random:.3f}")
        print(verdict)
        report["verdict"] = verdict

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
