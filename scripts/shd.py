"""Structural Hamming distance between each agent's belief and the true window MAG.

WHY THIS EXISTS. `cb.claims.score_window` answers a binary question -- identified or not,
zero tolerance on anything settled wrong -- which is the right criterion to TRAIN on but
throws away exactly the information "how close did it get" needs: an episode that is one
wrong orientation away from identified scores the same 0 as one that is completely
unstructured. SHD is the standard graph-distance metric and answers that question directly.

WHAT COUNTS AS ONE UNIT OF DISTANCE, and why it reduces to the textbook definition. Per pair
(u, v), each mark -- NONE, FWD (u->v), BACK (v->u), BI (confounded) -- is mutually exclusive,
so a mismatch decomposes as: a true non-edge estimated as any edge costs 1 (an EXTRA edge); a
true edge estimated as no edge costs 1 (a MISSING edge); a true edge estimated as the wrong
type or orientation ALSO costs 1, and costs nothing on top of that for the adjacency call,
which was correct. That is exactly Tsamardinos et al.'s SHD (missing + extra + misoriented,
each counted once per pair) -- it falls out of scoring adjacency and type as independent
questions rather than being imposed.

TWO VERSIONS, because the belief is a distribution and "how close" has a soft and a hard
reading.

  SOFT (expected) SHD = sum over pairs of (1 - P(true mark)), reading P directly off the
  belief's own frequency matrices (`adjacency`, `directed`, `bidirected` -- these partition
  probability mass exactly by construction, see `cb/factored.py::FactoredBelief.__init__`).
  Meaningful even when nothing has crossed a confidence bar, which is most of an episode.

  HARD SHD takes the MAP mark per pair -- argmax over the four mass values, NONE first so a
  genuine tie (an unsettled pair) defaults to "no edge" rather than to whichever mark numpy's
  argmax happens to see first -- and counts a mismatch as exactly 1, giving the textbook
  integer-valued metric people mean when they say "SHD".

  Soft is the one to trust when comparing arms: it does not need an arbitrary tie-break rule
  and does not throw away partial credit.

REPORTED PER WINDOW AND NORMALISED BY PAIR COUNT (C(k,2)), so window sizes are comparable
-- an SHD of 3 means something different at k=4 (6 pairs) than at k=30 (435 pairs).

SCOPE. `belief_backend` must be one of `ma.env.CLAIM_BACKENDS` -- the frequency matrices this
reads are that family's interface. Attribution (whose latent) is a SEPARATE axis, scored by
`cb.attribution.score_groups`; SHD here is purely structural and does not know or care who
owns a confounded pair.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from ma.baselines import ProbeThenWorkAgent, UncertaintyGreedyAgent
from ma.env import ATTRIBUTED, CLAIM_BACKENDS
from ma.policy import IndependentPPO
from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED
from scripts.rescore_from_config import env_from_config


def window_shd(belief, true_mag: np.ndarray) -> Tuple[float, int, int]:
    """(soft SHD, hard SHD, n_pairs) for one window."""
    mag = np.asarray(true_mag)
    k = mag.shape[0]
    adjacency = np.asarray(belief.adjacency)
    directed = np.asarray(belief.directed)
    bidirected = np.asarray(belief.bidirected)

    soft = 0.0
    hard = 0
    for u, v in combinations(range(k), 2):
        p_none = 1.0 - float(adjacency[u, v])
        p_fwd = float(directed[u, v])
        p_back = float(directed[v, u])
        p_bi = float(bidirected[u, v])
        masses = np.array([p_none, p_fwd, p_back, p_bi])   # NONE first: ties default to it

        if mag[u, v] == MAG_BIDIRECTED:
            truth_index, p_truth = 3, p_bi
        elif mag[u, v] == MAG_DIRECTED:
            truth_index, p_truth = 1, p_fwd
        elif mag[v, u] == MAG_DIRECTED:
            truth_index, p_truth = 2, p_back
        else:
            truth_index, p_truth = 0, p_none

        soft += 1.0 - p_truth
        hard += int(np.argmax(masses) != truth_index)
    n_pairs = k * (k - 1) // 2
    return soft, hard, n_pairs


def _factories(env, seed: int) -> Dict[str, object]:
    labels = {
        "random_vary": lambda a: __import__("ma.baselines", fromlist=["RandomAgent"])
                                 .RandomAgent(a, seed, allow_clamp=False),
        "greedy": lambda a: UncertaintyGreedyAgent(a, seed, bar=1.0),
    }
    if env.config.belief_backend == ATTRIBUTED:
        labels["probe_then_work"] = lambda a: ProbeThenWorkAgent(a, seed)
    return labels


def measure(result_path: pathlib.Path, episodes: int, seed: int, arms: List[str]) -> dict:
    report = json.loads(result_path.read_text())
    config = report["config"]
    if config["belief_backend"] not in CLAIM_BACKENDS:
        raise SystemExit(f"{result_path.name}: belief_backend={config['belief_backend']!r} "
                         "is not a claims backend -- SHD needs adjacency/directed/bidirected")
    use_seed = seed if seed is not None else report.get("seed", 0)
    env = env_from_config(config, seed=use_seed)

    built: Dict[str, dict] = {}
    checkpoint = result_path.with_suffix(".pt")
    if "learned" in arms and checkpoint.exists():
        built["learned"] = IndependentPPO.load(str(checkpoint), env).policies(
            deterministic=False)
    factories = _factories(env, use_seed)
    for label in arms:
        if label != "learned" and label in factories:
            built[label] = {a: factories[label](a) for a in env.topology.agents}

    out = {"source": str(result_path), "seed": use_seed, "arms": {}}
    for label, policies in built.items():
        for policy in policies.values():
            if hasattr(policy, "reset"):
                policy.reset(use_seed)
        soft_rows, hard_rows, norm_rows = [], [], []
        for episode in range(episodes):
            result = env.reset(seed=use_seed * 100_000 + episode)
            while not result.done:
                result = env.step({a: policies[a](env, result)
                                   for a in env.topology.agents})
            soft_total = hard_total = pairs_total = 0
            for agent, window in env.windows.items():
                soft, hard, n_pairs = window_shd(window.belief.last, env._true_mag(agent))
                soft_total += soft
                hard_total += hard
                pairs_total += n_pairs
            soft_rows.append(soft_total / max(pairs_total, 1))
            hard_rows.append(hard_total / max(pairs_total, 1))
            norm_rows.append(pairs_total)
        out["arms"][label] = {
            "soft_shd_mean": float(np.mean(soft_rows)),
            "soft_shd_sd": float(np.std(soft_rows)),
            "hard_shd_mean": float(np.mean(hard_rows)),
            "hard_shd_sd": float(np.std(hard_rows)),
            "soft_rows": soft_rows, "hard_rows": hard_rows,
        }
    return out


def paired(a: List[float], b: List[float]) -> str:
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    flag = "" if abs(d.mean()) > 2 * se else "  (inside 2 se)"
    return f"{d.mean():+.4f} +/- {se:.4f}{flag}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arms", default="learned,greedy,random_vary")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    arms = args.arms.split(",")
    payload = []
    for path in args.results:
        row = measure(pathlib.Path(path), args.episodes, args.seed, arms)
        payload.append(row)
        print(f"\n=== {pathlib.Path(path).stem}  ({args.episodes} episodes, "
             f"normalised per pair, k unknown -- see source config) ===")
        print(f"{'arm':14s} {'soft SHD/pair':>14s} {'hard SHD/pair':>14s}")
        for label, data in row["arms"].items():
            print(f"{label:14s} {data['soft_shd_mean']:14.4f} {data['hard_shd_mean']:14.4f}")
        if "learned" in row["arms"]:
            for other in ("greedy", "probe_then_work", "random_vary"):
                if other in row["arms"]:
                    print(f"  PAIRED learned - {other:15s} (soft) "
                         f"{paired(row['arms']['learned']['soft_rows'], row['arms'][other]['soft_rows'])}")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
