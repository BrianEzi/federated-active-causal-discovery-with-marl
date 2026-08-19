"""PHASE 0 -- freeze the enumerated two-agent posteriors as a regression fixture.

The subset-DP belief (Phase 1) replaces exact enumeration over the window's 543 DAGs. The
enumerated posterior is the ONLY ground truth the DP can be checked against, and it stops
existing the moment the enumeration is deleted. So it gets captured first.

What is stored, per (episode, round, agent):

  samples   the agent's own columns of the data seen so far
  known     the [n, k] hard-intervention indicator the score rule consumes
  clean     the [n] regime mask -- was this row drawn while everything hidden was clamped
  post_*    the full 543-vector posterior under each of the four scoring rules

Storing the INPUTS as well as the outputs is the point. A fixture holding only posteriors
would let a Phase 1 bug hide behind a differently-sampled dataset; holding the inputs means
the DP is fed byte-identical data and any discrepancy is the DP's.

Actions come from a seeded uniform policy rather than greedy. Greedy never clamps
(measured 2026-08-19: clamp_fraction 0.000), so a greedy-driven fixture would contain no
clean rows at all and would exercise exactly one branch of the four rules.

Usage:
    python scripts/ma_freeze_reference.py --episodes 200
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import time

import numpy as np

from ma.env import MAConfig, TwoAgentEnv
from ma.score_regimes import RULES
from ma.topology import Topology

AGENTS = ("A", "B")


def capture(env: TwoAgentEnv, episodes: int, rounds: int, seed: int) -> dict:
    """Run seeded episodes and record every belief input and output along the way."""
    rng = np.random.default_rng(seed)
    store: dict = {}
    for episode in range(episodes):
        env.reset(seed=seed * 100_000 + episode)
        for rnd in range(rounds):
            # Stored ONCE per round, not per agent: both windows are column slices of the
            # same matrix, and duplicating it doubled the fixture for no extra coverage.
            # Must stay float64 -- the DP has to be fed byte-identical values or the 1e-10
            # comparison is meaningless.
            store[f"e{episode}_r{rnd}_samples"] = env.samples.astype(np.float64)
            for name in AGENTS:
                view = env.views[name]
                tag = f"e{episode}_r{rnd}_{name}"
                store[f"{tag}_nodes"] = np.asarray(view.nodes, dtype=np.int64)
                store[f"{tag}_known"] = np.asarray(env.known[name], dtype=np.float64)
                store[f"{tag}_clean"] = np.asarray(env.clean[name], dtype=bool)
                for rule in RULES:
                    store[f"{tag}_post_{rule}"] = view.posterior(
                        env.samples[:, view.nodes], env.known[name], env.clean[name],
                        rule=rule).astype(np.float64)
            # Uniform over real actions, excluding PASS, so every round adds data.
            actions = {n: int(rng.integers(env.views[n].n_actions - 1)) for n in AGENTS}
            result = env.step(actions["A"], actions["B"])
            if result.done:
                break
        store[f"e{episode}_truth"] = env.true_adjacency.astype(np.int8)
        store[f"e{episode}_true_index"] = np.array(
            [env.true_index[n] for n in AGENTS], dtype=np.int64)
    return store


def digest(store: dict) -> str:
    """Order-independent hash over the whole fixture, for the determinism check."""
    h = hashlib.sha256()
    for key in sorted(store):
        h.update(key.encode())
        h.update(np.ascontiguousarray(store[key]).tobytes())
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n_obs", type=int, default=100)
    ap.add_argument("--n_int", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tests/fixtures/ma_reference_posteriors.npz")
    args = ap.parse_args(argv)

    topology = Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                      budget=args.rounds, identify_threshold=0.7)

    started = time.time()
    first = capture(TwoAgentEnv(config), args.episodes, args.rounds, args.seed)
    d1 = digest(first)
    print(f"capture 1: {len(first)} arrays, sha256 {d1[:16]}  [{time.time()-started:.0f}s]")

    # GATE: the fixture is worthless if the environment is not reproducible under a fixed
    # seed, and non-determinism would surface later as a phantom Phase 1 mismatch.
    second = capture(TwoAgentEnv(config), args.episodes, args.rounds, args.seed)
    d2 = digest(second)
    print(f"capture 2: sha256 {d2[:16]}")
    if d1 != d2:
        raise SystemExit("PHASE 0 GATE FAILED -- environment is not deterministic under a "
                         "fixed seed. Fix that before building the DP.")
    print("PHASE 0 GATE PASSED -- bit-identical across two independent captures.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **first, _digest=np.array([d1]))
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size:.1f} MB, {args.episodes} episodes)")


if __name__ == "__main__":
    main()
