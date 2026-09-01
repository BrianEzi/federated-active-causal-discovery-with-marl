"""Does power-limited ORACLE evidence produce belief trajectories that resemble genuine
SAMPLED (CI-test) evidence, or is it a different shape of degradation wearing the same name?

Every use of `evidence_power` on this project has assumed the answer is yes, without
checking. This checks it directly, isolated from policy behaviour: same graph, same SCM
params, same observational data, same intervention SEQUENCE (a fixed, belief-independent
`RandomAgent`, so the two conditions cannot diverge because one condition made a smarter
choice) -- the only thing that differs is which evidence rule reads the interventions.

Tracks the pooled RESOLVED FRACTION (share of pairs the belief has settled on one mark) round
by round, since that is the quantity `evidence_power` is supposed to be standing in for: an
unsettled belief that only gradually resolves as evidence accumulates. If power-limiting
produces a similar SHAPE of resolution curve to genuine sampled evidence at some power value,
that value is defensible as a cheap proxy. If no power value matches the shape (e.g. sampled
resolves smoothly while power-limiting resolves in a step, or vice versa), the two are
different processes that happen to share a final accuracy number, and using one to stand in
for the other needs to be argued, not assumed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cb.versionspace import marks_from_mag, pairs
from ma.baselines import RandomAgent
from ma.env import MAConfig, ROUND_ROBIN, VARY, TwoAgentEnv
from ma.topology import federated_topology


def build_env(vs_evidence: str, evidence_power: float, n_agents, private, shared, budget,
             n_obs, n_int) -> TwoAgentEnv:
    topology = federated_topology(n_agents, private, shared)
    config = MAConfig(topology=topology, n_obs=n_obs, n_int=n_int, budget=budget,
                      turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="factored", policy_arch="gnn_portable",
                      episode_mix="confounded", reward_criterion="claims", claim_bar=1.0,
                      per_agent_reward=True, graph_model="sf", sf_m=2,
                      vs_evidence=vs_evidence, vs_evidence_power=evidence_power)
    return TwoAgentEnv(config)


def resolved_trajectory(env: TwoAgentEnv, episodes: int, seed_base: int, budget: int):
    """Pooled resolved-fraction AND pooled ERROR-fraction (settled-and-WRONG pairs) after
    each of the first `budget` rounds, averaged over episodes and agents. Uses RandomAgent
    (belief-independent) so both conditions play the SAME sequence of interventions when
    given the same seed.

    THE ERROR CHECK, per Agent A's 1 Sep 23:00 note. Resolved fraction alone cannot tell
    apart two processes that trace the same curve while one of them is quietly settling on
    a FALSE mark -- power-limited evidence withholds (sound, can be asked again) while a
    real CI test can settle wrong (Fisher-z error rate ~0.8% at alpha=1e-3, per the ledger).
    `evidence_power` can never produce a settled-wrong pair by construction, so if sampled
    evidence's error curve is non-trivial, the resolved-fraction match is real but partial.
    """
    agents = env.topology.agents
    resolved_traj = np.zeros(budget)
    error_traj = np.zeros(budget)
    counted = np.zeros(budget)
    for ep in range(episodes):
        policies = {a: RandomAgent(a, seed_base + ep, allow_clamp=False) for a in agents}
        result = env.reset(seed=seed_base + ep)
        true_marks = {a: marks_from_mag(env._true_mag(a)) for a in agents}
        round_idx = 0
        while not result.done and round_idx < budget:
            actions = {a: policies[a](env, result) for a in agents}
            result = env.step(actions)
            resolved = []
            errors = []
            for a in agents:
                belief = env.windows[a].belief.last
                k = env.windows[a].k
                truth = true_marks[a]
                settled = 0
                wrong = 0
                for index, key in enumerate(pairs(k)):
                    marks = belief.possible[key]
                    if len(marks) == 1:
                        settled += 1
                        if next(iter(marks)) != truth[index]:
                            wrong += 1
                total = max(len(belief.possible), 1)
                resolved.append(settled / total)
                errors.append(wrong / total)
            resolved_traj[round_idx] += float(np.mean(resolved))
            error_traj[round_idx] += float(np.mean(errors))
            counted[round_idx] += 1
            round_idx += 1
    denom = np.maximum(counted, 1)
    return resolved_traj / denom, error_traj / denom


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--private_size", type=int, default=4)
    ap.add_argument("--n_shared", type=int, default=4)
    ap.add_argument("--budget", type=int, default=35)
    ap.add_argument("--n_obs", type=int, default=60)
    ap.add_argument("--n_int_sampled", type=int, default=200)
    ap.add_argument("--powers", default="1.0,0.95,0.9,0.85,0.8,0.7,0.5")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="results/power/dist_compare.json")
    args = ap.parse_args(argv)

    print("Building SAMPLED-evidence reference trajectory (n_int={})...".format(args.n_int_sampled))
    env_sampled = build_env("sampled", 1.0, args.n_agents, args.private_size, args.n_shared,
                            args.budget, args.n_obs, args.n_int_sampled)
    sampled_traj, sampled_err = resolved_trajectory(env_sampled, args.episodes, seed_base=500_000,
                                                     budget=args.budget)
    print("sampled resolved:", np.round(sampled_traj, 3))
    print("sampled error   :", np.round(sampled_err, 4))

    results = {"sampled": sampled_traj.tolist(), "sampled_error": sampled_err.tolist()}
    for p in [float(x) for x in args.powers.split(",")]:
        env_power = build_env("oracle", p, args.n_agents, args.private_size, args.n_shared,
                              args.budget, args.n_obs, 20)
        traj, err = resolved_trajectory(env_power, args.episodes, seed_base=500_000,
                                        budget=args.budget)
        mad = float(np.mean(np.abs(traj - sampled_traj)))
        print(f"power={p:.2f}  mad_vs_sampled={mad:.4f}  error(mean)={float(np.mean(err)):.4f}  traj={np.round(traj, 3)}")
        results[f"power_{p}"] = {"trajectory": traj.tolist(), "mad_vs_sampled": mad,
                                 "error_trajectory": err.tolist()}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
