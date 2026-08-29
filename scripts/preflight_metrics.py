"""PRE-FLIGHT: is every metric actually live in the dict a real run writes out?

RUN THIS BEFORE LAUNCHING ANYTHING LONG. The unit tests cover each metric in isolation and
all of them passed while THREE fields were silently missing from `run_arm`'s output --
`global_soft_shd`, `duplicate_coverage` and `identified_round` were computed per episode and
dropped during aggregation, so they would have reached no result file at all. A fourth,
`time_to_identification`, was present but entirely nan because the censor was never applied.
None of that is visible from a unit test; it is only visible end to end.

Checks the fields are PRESENT and SANE in what `run_arm` returns, at a window size the old
enumerating metrics could not reach, on both turn orders.

  .venv/bin/python scripts/preflight_metrics.py

Exits non-zero on failure, so it can gate a launch script.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ma.env import MAConfig, TwoAgentEnv
from ma.topology import federated_topology
from ma.evaluate import run_arm
from ma.baselines import UncertaintyGreedyAgent

FAIL = []
def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond: FAIL.append(msg)

for turn in ("round_robin", "random"):
    env = TwoAgentEnv(MAConfig(
        topology=federated_topology(4, 10, 10), n_obs=60, n_int=20, budget=20,
        turn_order=turn, belief_backend="factored", action_modes=("vary",), claim_bar=1.0,
        reward_criterion="claims", policy_arch="gnn_portable", graph_model="sf", sf_m=2,
        episode_mix="confounded", vs_evidence="oracle"))
    pol = {a: UncertaintyGreedyAgent(a, 0, bar=1.0) for a in env.topology.agents}
    out = run_arm(env, pol, episodes=15, seed=0)
    print(f"\n=== turn_order={turn}  k=20  4 agents  budget 20 ===")

    check("some_agent_never_acted" in out, "some_agent_never_acted present")
    check("effort_evenness" in out and "effort_evenness_null" in out,
          "effort_evenness + null present")
    check("success_feasible" in out, "success_feasible present")
    check("global_soft_shd" in out, "global_soft_shd present (replaces union_*)")
    check("time_to_identification" in out, "time_to_identification present")
    check("duplicate_coverage_floor" in out or "duplicate_coverage" in out,
          "duplicate coverage reported")
    check("free_rider_index" in out and "union_acyclic" in out,
          "legacy fields kept, so old files stay reproducible")

    null = out["effort_evenness_null"]
    # The null depends on agents AND budget: 4 agents over 20 rounds is far more even than
    # 8 over 24. Assert the PROPERTY (round-robin is exactly fair, random is not), not a
    # constant lifted from a different rung.
    check(null == 1.0 if turn == "round_robin" else 0.0 < null < 1.0,
          f"effort_evenness_null = {null:.3f} matches the protocol")
    check(0.0 <= out["global_soft_shd"] <= 1.0, f"global_soft_shd = {out['global_soft_shd']:.4f} in range")
    check(out["global_contradiction"] == 0.0,
          f"global_contradiction = {out['global_contradiction']:.3f} (soundness holds)")
    check(out["global_pairs"] == 625, f"global_pairs = {out['global_pairs']} = covered pairs at w20")
    surv = out["time_to_identification"][0]
    check(0.0 <= surv["censored_fraction"] <= 1.0 and surv["restricted_mean"] <= 21,
          f"survival bounded: censored {surv['censored_fraction']:.2f}, "
          f"restricted_mean {surv['restricted_mean']:.2f} <= budget+1")
    if turn == "random":
        # At 4 agents over 20 rounds, P(some agent draws nothing) = 4*(3/4)^20 ~= 1.3%, so
        # zero over 15 episodes is expected. The field earns its keep at 8 agents / budget
        # 24, where the rate is 29.9%. Assert it is DEFINED and a probability, not that it
        # fires at a rung where it should not.
        check(0.0 <= out["some_agent_never_acted"] <= 1.0,
              f"some_agent_never_acted = {out['some_agent_never_acted']:.3f} (well-defined)")
        check(out["never_acted_episodes"] == 0.0,
              "legacy never_acted_episodes still 0.0 -- which is exactly why it was useless")

print("\n" + ("ALL METRIC CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
