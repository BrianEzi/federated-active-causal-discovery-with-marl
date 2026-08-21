"""The acyclicity exchange, weighted by a real posterior instead of a uniform prior.

`scripts/exchange_value.py` measures the structural version: over all legal joint
hypotheses, counted uniformly, how many does the `|X|^2`-bit disclosure remove? The answer
came out tiny -- 0.014 to 0.045 bits against 4 to 9 disclosed, pruning 1-3% of hypotheses.

That is not the end of the question, because counting is not the same as weighting. The
jointly-cyclic combinations could be rare but hold disproportionate POSTERIOR mass,
precisely in the situations where the agents are individually uncertain and each has
routed a path through its own private nodes. Uniform counting cannot see that; only a
data-conditioned measurement can.

Method, per episode:
  1. Draw a true global DAG under the topology mask, draw an SCM, generate observational
     data for the single shared system (MA_DESIGN section 6).
  2. Each agent scores every DAG over its OWN window, seeing only its own columns. BGe,
     the same score the single-agent work uses.
  3. Form the joint posterior over pairs that AGREE on the induced subgraph on `X`, as the
     normalised product of the two local posteriors. Agreement is applied first, so the
     exchange is never credited with pruning that agreement already did.
  4. Report the posterior mass sitting on jointly-cyclic pairs, and the entropy reduction
     from deleting it.

PRE-REGISTERED PREDICTION, before the numbers exist:
    I expect the posterior-weighted mass on cyclic pairs to EXCEED the uniform 1-3%,
    because a cyclic combination requires both agents to posit private routings, and
    private routings are exactly what each agent's own data cannot rule out. But I do not
    expect it to be large in absolute terms. If it lands under ~5% then the exchange is
    a correctness device and not an inference device, my "pruning" reframing was wrong,
    and the original safety-net framing in MA_DESIGN section 5 was right.

    Recording that explicitly because I talked myself out of the safety-net reading
    earlier today on an argument that this measurement can falsify.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ma.projection import ancestor_matrix
from ma.topology import Topology
from sa.graphs import build_graph_space
from sa.scm import sample as scm_sample, sample_scm_params
from sa.score import BGeScore

from legacy.scripts.exchange_value import induced_order_on_shared, jointly_acyclic


def local_hypotheses(nodes, shared):
    """Every DAG over one agent's window, with its `X`-signature and induced order."""
    space = build_graph_space(len(nodes))
    pos = {node: i for i, node in enumerate(nodes)}
    shared_pos = [pos[s] for s in shared]
    k = len(shared)
    out = []
    for adjacency in space.dags:
        adjacency = np.asarray(adjacency, dtype=np.int8)
        sig = tuple(int(adjacency[shared_pos[a], shared_pos[b]])
                    for a in range(k) for b in range(k) if a != b)
        out.append((adjacency, sig, induced_order_on_shared(adjacency, nodes, shared_pos)))
    return out


def log_posterior(hypotheses, samples, score):
    """Local posterior over an agent's own DAGs, from its own columns only.

    Observational data only, so no interventional bookkeeping is needed -- every row
    contributes a likelihood term for every node.
    """
    cache = {}
    logs = np.empty(len(hypotheses))
    for i, (adjacency, _, _) in enumerate(hypotheses):
        total = 0.0
        for node in range(adjacency.shape[0]):
            parents = tuple(np.flatnonzero(adjacency[:, node]).tolist())
            key = (node, parents)
            if key not in cache:
                cache[key] = score.local_score(node, parents, samples)
            total += cache[key]
        logs[i] = total
    logs -= logs.max()
    weights = np.exp(logs)
    return weights / weights.sum()


def run(topology: Topology, episodes: int, n_obs: int, seed: int) -> dict:
    shared = list(topology.exposed)
    k = len(shared)
    obs = {a: list(topology.observed_by(a)) for a in ("A", "B")}
    hyps = {a: local_hypotheses(obs[a], shared) for a in ("A", "B")}
    score = BGeScore(len(obs["A"]))
    rng = np.random.default_rng(seed)

    cyclic_mass, bits_gained, agree_mass = [], [], []

    for _ in range(episodes):
        truth = topology.sample_dag(rng, p=0.5)
        params = sample_scm_params(truth, rng)
        samples, _ = scm_sample(params, n_obs, rng)

        post = {}
        for agent in ("A", "B"):
            cols = obs[agent]
            post[agent] = log_posterior(hyps[agent], samples[:, cols], score)

        # Joint posterior over pairs agreeing on X, as the normalised product.
        by_sig = {}
        for agent in ("A", "B"):
            for idx, (_, sig, bits) in enumerate(hyps[agent]):
                by_sig.setdefault(sig, {"A": [], "B": []})[agent].append(
                    (post[agent][idx], bits))

        joint, cyclic = [], []
        for sig, sides in by_sig.items():
            if not sides["A"] or not sides["B"]:
                continue
            for pa, ba in sides["A"]:
                if pa < 1e-12:
                    continue
                for pb, bb in sides["B"]:
                    weight = pa * pb
                    if weight < 1e-15:
                        continue
                    joint.append(weight)
                    cyclic.append(not jointly_acyclic(ba, bb, k))

        joint = np.asarray(joint)
        cyclic = np.asarray(cyclic)
        agree_mass.append(float(joint.sum()))
        joint = joint / joint.sum()

        bad = float(joint[cyclic].sum()) if cyclic.any() else 0.0
        cyclic_mass.append(bad)

        # Entropy reduction from deleting the cyclic mass and renormalising.
        before = -float((joint * np.log2(np.clip(joint, 1e-300, None))).sum())
        keep = joint[~cyclic]
        if keep.sum() > 0:
            keep = keep / keep.sum()
            after = -float((keep * np.log2(np.clip(keep, 1e-300, None))).sum())
        else:
            after = before
        bits_gained.append(before - after)

    def stat(v):
        arr = np.asarray(v, dtype=float)
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                "se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
                "median": float(np.median(arr)), "max": float(arr.max())}

    return {
        "topology": topology.name,
        "episodes": episodes,
        "n_obs": n_obs,
        "cyclic_posterior_mass": stat(cyclic_mass),
        "bits_gained": stat(bits_gained),
        "bits_disclosed_worst_case": k * k,
        "agreement_mass_before_normalising": stat(agree_mass),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n_obs", type=int, nargs="+",
                    default=[50, 100, 200, 500, 1000, 5000],
                    help="Swept DOWNWARD deliberately. The prediction above is about the "
                         "regime where each agent is individually uncertain, and at "
                         "n_obs=1000 with a 3-4 node window each local posterior is "
                         "already concentrated -- a smoke test there returned ~0 cyclic "
                         "mass, which does not test the claim.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ma/exchange_value_data.json")
    args = ap.parse_args()

    topologies = [
        Topology("(1,1,2)", a_private=(0,), b_private=(1,), exposed=(2, 3)),
        Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4)),
    ]
    rows = []
    for n_obs in args.n_obs:
      for topology in topologies:
        row = run(topology, args.episodes, n_obs, args.seed)
        rows.append(row)
        print(f"n_obs={row['n_obs']:>5} {row['topology']}: cyclic_mass={row['cyclic_posterior_mass']['mean']:.4f}"
              f" +/- {row['cyclic_posterior_mass']['se']:.4f}"
              f"  (median {row['cyclic_posterior_mass']['median']:.4f},"
              f" max {row['cyclic_posterior_mass']['max']:.4f})"
              f"  bits_gained={row['bits_gained']['mean']:.4f}"
              f" of {row['bits_disclosed_worst_case']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
