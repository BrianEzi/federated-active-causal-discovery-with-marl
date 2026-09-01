"""Prove `_act_many` (batched round forward) is behaviourally identical to the per-agent loop.

WHY BIT-FOR-BIT ON THE FLOATS IS THE WRONG BAR, and what this checks instead. A batched
matmul and N single-row matmuls take different BLAS paths and may differ in the last ulp, so
demanding bitwise-identical logits would fail a CORRECT implementation. What must match
exactly is the part that decides behaviour and consumes randomness: the ACTIONS. `_act_many`
batches only `forward`; sampling stays in a per-agent loop in agent order, so the global torch
RNG is consumed in the same order and the same number of times as before. If a logit
difference in the last ulp ever flipped a sample, this would catch it as an action mismatch.

Reports max absolute difference on value/logp (expected ~1e-6, never asserted at 0) and an
EXACT match requirement on every action taken.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ma.policy import IndependentPPO, PPOConfig                      # noqa: E402
from scripts.rescore_from_config import env_from_config              # noqa: E402


def run(ppo, episodes, batched: bool, rng_state, torch_state):
    """One `collect` from a fixed RNG state, with batching on or off."""
    ppo.rng.bit_generator.state = rng_state
    torch.set_rng_state(torch_state)
    shared = ppo.shared_net
    if not batched:
        ppo.shared_net = None          # `_act_many` falls back to the per-agent loop
    start = time.time()
    try:
        buffers = ppo.collect(episodes, 0, mask_pass=False)
    finally:
        ppo.shared_net = shared
    return buffers, time.time() - start


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="a run json whose config defines the environment")
    ap.add_argument("--episodes", type=int, default=8)
    args = ap.parse_args(argv)

    config = json.loads(pathlib.Path(args.result).read_text())["config"]
    env = env_from_config(config, seed=0)
    ppo = IndependentPPO(env, PPOConfig(seed=0))
    if ppo.shared_net is None:
        print(f"!! policy_arch={config.get('policy_arch')} has no shared net; "
              f"`_act_many` falls back by construction. Nothing to verify.")
        return 0

    rng_state = ppo.rng.bit_generator.state
    torch_state = torch.get_rng_state()

    old_batch, t_old = run(ppo, args.episodes, False, rng_state, torch_state)
    new_batch, t_new = run(ppo, args.episodes, True, rng_state, torch_state)
    old, new = old_batch["buffers"], new_batch["buffers"]
    for key in ("entropy", "solve_rate", "window_rate"):
        # Tolerance, not equality: these are means over floats that reassociate.
        same = "" if abs(old_batch[key] - new_batch[key]) < 1e-6 else "   <-- DIFFERS"
        print(f"  {key:12s} loop {old_batch[key]:.6f}   batched {new_batch[key]:.6f}{same}")

    print(f"{args.episodes} episodes, {len(env.topology.agents)} agents, "
          f"arch={config.get('policy_arch')}, k={config.get('k')}")
    print(f"  per-agent loop {t_old:.2f}s     batched round {t_new:.2f}s     "
          f"speedup {t_old / t_new:.2f}x")

    failures = 0
    for agent in old:
        for key in ("action", "logp", "value", "reward", "done"):
            a = np.asarray(old[agent][key], dtype=np.float64)
            b = np.asarray(new[agent][key], dtype=np.float64)
            if a.shape != b.shape:
                print(f"  !! agent {agent} {key}: SHAPE {a.shape} vs {b.shape}")
                failures += 1
                continue
            if key in ("action", "done", "reward"):
                # Behaviour and environment response: must be identical, not close.
                bad = int((a != b).sum())
                print(f"  agent {agent} {key:7s} n={a.size:5d}  mismatches {bad}"
                      f"{'   <-- FAIL' if bad else ''}")
                failures += bad
            else:
                delta = float(np.abs(a - b).max()) if a.size else 0.0
                print(f"  agent {agent} {key:7s} n={a.size:5d}  max|diff| {delta:.3e}")
                if delta > 1e-4:
                    print("       <-- FAIL: beyond float-reassociation noise")
                    failures += 1

    print("\nPASS -- identical behaviour" if not failures
          else f"\nFAIL -- {failures} mismatches")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
