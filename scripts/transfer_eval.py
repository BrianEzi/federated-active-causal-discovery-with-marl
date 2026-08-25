"""Does a policy trained on PERFECT data still choose good experiments on NOISY data?

The hypothesis worth testing, in the student's words: an intervention should be a good
intervention regardless of how noisy the measurement is. WHICH node to intervene on is a
structural question -- noise changes how much an experiment reveals, not which experiment is
worth running -- so a policy that learned "cover the shared variable nobody else is covering"
should carry over unchanged.

If it transfers, a 7-hour statistical training loop collapses into a 2-minute deterministic
one. If it does not, the statistical environment is rewarding something other than
experiment selection, which is worth knowing and goes straight in the log.

Reports per-WINDOW identification, the metric the headroom was measured in, with a standard
error and against the same baselines in the same environment.
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
from ma.topology import Topology


def build_env(n_agents, budget, n_obs, n_int, n_boot, oracle, seed=0):
    topology = Topology(name=f"T_{n_agents}agent_1each",
                        private=tuple((i,) for i in range(n_agents)),
                        exposed=tuple(range(n_agents, n_agents + 3)))
    config = MAConfig(topology=topology, n_obs=n_obs, n_int=n_int, budget=budget,
                      disclose_regime=True, turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="constraint", cb_n_boot=n_boot, policy_arch="gnn",
                      episode_mix="confounded", reward_criterion="claims",
                      oracle_obs_structure=oracle)
    return TwoAgentEnv(config, seed=seed)


def window_rates(env, policies, episodes, seed_base=70_000):
    rates = []
    for episode in range(episodes):
        result = env.reset(seed=seed_base + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
        identified = []
        for agent in env.topology.agents:
            window = env.windows[agent]
            score = score_window(window.belief.last, env._true_mag(agent),
                                 [window.pos[n] for n in window.private],
                                 bar=env.config.claim_bar)
            identified.append(float(score.identified))
        rates.append(float(np.mean(identified)))
    return np.array(rates, float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="a .pt trained in ANY backend")
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=1000)
    ap.add_argument("--cb_n_boot", type=int, default=12)
    ap.add_argument("--oracle_obs", action="store_true")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = build_env(args.n_agents, args.budget, args.n_obs, args.n_int,
                    args.cb_n_boot, args.oracle_obs)
    agents = list(env.topology.agents)
    reference = {a: make_baselines(env, a, seed=0) for a in agents}

    ppo = IndependentPPO.load(args.policy, env, allow_backend_transfer=True)
    arms = {
        "transferred": ppo.policies(deterministic=False),
        "greedy_uncertainty": {a: reference[a]["greedy_uncertainty"] for a in agents},
        "random_vary": {a: reference[a]["random_vary"] for a in agents},
    }

    print(f"statistical env: {args.n_agents} agents, budget {args.budget}, "
          f"n_int {args.n_int}, oracle_obs {args.oracle_obs}, {args.episodes} episodes")
    print(f"  policy trained on: version_space (deterministic), evaluated here on noisy data")
    results = {}
    for name, policies in arms.items():
        rates = window_rates(env, policies, args.episodes)
        results[name] = {"mean": float(rates.mean()),
                         "stderr": float(rates.std(ddof=1) / np.sqrt(len(rates)))}
        print(f"  {name:20s} per-window identified {results[name]['mean']:.3f} "
              f"+/- {results[name]['stderr']:.3f}", flush=True)

    gap = results["transferred"]["mean"] - results["greedy_uncertainty"]["mean"]
    combined = np.hypot(results["transferred"]["stderr"],
                        results["greedy_uncertainty"]["stderr"])
    print(f"  transferred MINUS greedy: {gap:+.3f} +/- {combined:.3f}")

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"policy": args.policy, "config": vars(args),
                                    "arms": results, "gap_vs_greedy": gap}, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
