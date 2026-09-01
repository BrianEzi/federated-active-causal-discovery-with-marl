"""Lead 1 (docs/AGENT_B_INBOX.md, 1 Sep 20:40): is the power-limited-evidence gate the wrong
gate?

The power work gated on `arms.greedy_uncertainty.success >= 0.85` -- the ALL-agents
conjunction, which the project demoted from primary this week because it saturates and
amplifies seed variance. The rest of the project gates on `window_rate >= 0.70`
(`scripts/sweep_report.py`'s WINDOW_FLOOR), a per-window pooled rate that does not have that
problem.

This recomputes greedy's window_rate (via `scripts/transfer_eval.py::window_rates`, the
existing pooled-per-window metric) for every result file passed on the command line, using
each file's OWN config and OWN seed so the replayed episodes match what actually ran.

Falsification, stated up front per the handover: if the seeds that failed the 0.85-on-success
gate sit at window_rate < 0.70 too, the gate was fine and the environment really is starved.
If they sit above 0.70, the 1-of-6 replication was a measurement artefact of using the wrong
gate, not evidence the environment is broken.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from ma.baselines import UncertaintyGreedyAgent                      # noqa: E402
from ma.env import MAConfig, ROUND_ROBIN, VARY, TwoAgentEnv          # noqa: E402
from ma.topology import Topology                                     # noqa: E402
from scripts.transfer_eval import window_rates                       # noqa: E402


def build_env(cfg: dict) -> TwoAgentEnv:
    t = cfg["topology"]
    topology = Topology(name=t["name"], private=tuple(tuple(p) for p in t["private"]),
                        exposed=tuple(t["exposed"]))
    config = MAConfig(topology=topology, n_obs=cfg["n_obs"], n_int=cfg["n_int"],
                      budget=cfg["budget"], turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend=cfg.get("backend", "factored"),
                      policy_arch=cfg.get("policy_arch", "gnn_portable"),
                      episode_mix="confounded", reward_criterion="claims", claim_bar=1.0,
                      per_agent_reward=True, graph_model=cfg.get("graph_model", "sf"),
                      sf_m=cfg.get("sf_m", 2), vs_evidence=cfg.get("vs_evidence", "oracle"),
                      vs_evidence_power=cfg.get("evidence_power", cfg.get("vs_evidence_power", 1.0)),
                      distance_weighted_power=cfg.get("distance_weighted_power", False))
    return TwoAgentEnv(config)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--episodes", type=int, default=60)
    args = ap.parse_args(argv)

    print(f"{'file':45s} {'success(0.85 gate)':>20s} {'window_rate(0.70 gate)':>24s}")
    for path in args.files:
        d = json.loads(open(path).read())
        cfg = d["config"]
        seed = d.get("seed", 0)
        env = build_env(cfg)
        agent = UncertaintyGreedyAgent
        policies = {a: agent(a, seed, bar=1.0) for a in env.topology.agents}
        rates = window_rates(env, policies, args.episodes, seed_base=seed * 100_000)
        wr = float(np.mean(rates))
        success = d["arms"]["greedy_uncertainty"]["success"]
        g_old = "PASS" if success >= 0.85 else "FAIL"
        g_new = "PASS" if wr >= 0.70 else "FAIL"
        print(f"{path:45s} success={success:.3f}[{g_old}]      window_rate={wr:.3f}[{g_new}]")


if __name__ == "__main__":
    main()
