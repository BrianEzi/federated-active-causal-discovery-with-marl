"""Does confounding stay confined to the shared set when there are more than two agents?

**Everything in the scaling plan rests on the answer.** For two agents it is proved and
exhaustively verified (`tests/test_projection.py`): every bidirected edge in an agent's
latent projection has BOTH endpoints in the shared set. That is what lets a belief be "a DAG
over my window plus one flag per shared pair", keeps the score decomposable, and lets the
subset DP carry over untouched. If it fails for `n > 2`, the belief needs full MAG machinery
and the score stops decomposing.

**The prediction, stated before the measurement so it can be falsified.** It should still
hold, and for a reason that does not care how many agents there are: cross-private edges are
forbidden, so every parent of a node private to agent `i` is itself visible to `i`. A node
with no hidden parent cannot be the endpoint of a bidirected edge, because the first
non-endpoint on a confounding path must be a hidden parent. Only shared nodes are allowed
hidden parents, so only shared nodes can be confounded.

If that argument has a hole, an exhaustive enumeration will find it.

Two topology families are checked, because the second is where it could plausibly break:

  DISJOINT   one exposed set visible to everyone, plus disjoint private sets. This is what
             `ma/topology.py` implements and what the scaling plan assumes.
  OVERLAP    each node carries an arbitrary VISIBILITY SET. A node visible to agents {1,2}
             but not {3} is neither private nor fully shared, and it is hidden from agent 3
             while being a legitimate parent of things agent 1 can see. This is the natural
             next generalisation, and it is the one to test before relying on it.

Bidirected edges come from `ma.projection.bidirected_pairs` -- the existing verified
implementation, deliberately reused rather than reimplemented, so this cannot pass by
disagreeing with the two-agent result.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ma.projection import bidirected_pairs
from sa.graphs import is_acyclic


CROSS_PRIVATE = "cross_private"
JOINTLY_VISIBLE = "jointly_visible"
EDGE_RULES = (CROSS_PRIVATE, JOINTLY_VISIBLE)


def allowed_edge_mask(visibility: Sequence[frozenset], n_agents: int,
                      rule: str = JOINTLY_VISIBLE) -> np.ndarray:
    """`[d, d]` bool: may `i -> j` exist?

    Two rules, because the difference turns out to decide the whole question.

    `cross_private` -- the literal two-agent rule: forbid an edge between two nodes that are
    each visible to exactly one agent, when those agents differ. This is what
    `ma/topology.py` implements, and under partial overlap it is TOO PERMISSIVE: a node
    visible to {0, 2} is not private to anyone, so this rule lets it parent a node private
    to agent 1 -- an edge NO agent can see.

    `jointly_visible` -- the principle the two-agent rule was an instance of: an edge may
    exist only if SOME agent sees both of its endpoints. An edge no one can observe is not
    learnable by anyone, so admitting it into the hypothesis space only adds structure that
    no data can ever bear on. Under the disjoint topology the two rules coincide exactly.
    """
    d = len(visibility)
    mask = np.ones((d, d), dtype=bool)
    np.fill_diagonal(mask, False)
    for u in range(d):
        for v in range(d):
            if u == v:
                continue
            vu, vv = visibility[u], visibility[v]
            if rule == JOINTLY_VISIBLE:
                if not (vu & vv):
                    mask[u, v] = False
            elif len(vu) == 1 and len(vv) == 1 and vu != vv:
                mask[u, v] = False
    return mask


def enumerate_dags(mask: np.ndarray, limit: int):
    """Every acyclic orientation permitted by `mask`, or `None` if there are too many."""
    slots = [(u, v) for u in range(mask.shape[0]) for v in range(mask.shape[0]) if mask[u, v]]
    if 2 ** len(slots) > limit:
        return None
    out = []
    for bits in range(2 ** len(slots)):
        adjacency = np.zeros(mask.shape, dtype=bool)
        for k, (u, v) in enumerate(slots):
            if bits >> k & 1:
                adjacency[u, v] = True
        if is_acyclic(adjacency):
            out.append(adjacency)
    return out


def sample_dags(mask: np.ndarray, count: int, rng: np.random.Generator):
    """Random DAGs under the mask, via a random topological order."""
    d = mask.shape[0]
    out = []
    for _ in range(count):
        order = rng.permutation(d)
        adjacency = np.zeros((d, d), dtype=bool)
        for a in range(d):
            for b in range(a + 1, d):
                u, v = int(order[a]), int(order[b])
                if mask[u, v] and rng.random() < 0.5:
                    adjacency[u, v] = True
        out.append(adjacency)
    return out


def check(visibility: Sequence[frozenset], n_agents: int, limit: int,
          samples: int, seed: int, rule: str = JOINTLY_VISIBLE) -> dict:
    """Every agent, every DAG: is every bidirected pair confined to that agent's SHARED set?

    "Shared" from agent `i`'s point of view means a node it can see that at least one OTHER
    agent can also see. A node visible only to `i` is private to `i`, and the claim under
    test is that such a node is never an endpoint of a bidirected edge.
    """
    mask = allowed_edge_mask(visibility, n_agents, rule)
    dags = enumerate_dags(mask, limit)
    exhaustive = dags is not None
    if not exhaustive:
        dags = sample_dags(mask, samples, np.random.default_rng(seed))

    violations = []
    checked = 0
    for adjacency in dags:
        for agent in range(n_agents):
            observed = [i for i, vis in enumerate(visibility) if agent in vis]
            private = {i for i in observed if visibility[i] == frozenset({agent})}
            checked += 1
            for u, v in bidirected_pairs(adjacency, observed):
                if u in private or v in private:
                    violations.append({
                        "agent": agent, "pair": [int(u), int(v)],
                        "private_endpoint": [int(x) for x in (u, v) if x in private],
                        "adjacency": adjacency.astype(int).tolist(),
                        "visibility": [sorted(s) for s in visibility]})
                    break
        if violations:
            break
    return {"n_agents": n_agents, "d": len(visibility), "edge_rule": rule,
            "visibility": [sorted(s) for s in visibility],
            "exhaustive": exhaustive, "n_dags": len(dags), "projections_checked": checked,
            "violations": violations, "holds": not violations}


def disjoint_family(n_agents: int, n_private: int, n_shared: int):
    """The topology `ma/topology.py` implements: private sets plus one common exposed set."""
    visibility = []
    for agent in range(n_agents):
        visibility += [frozenset({agent})] * n_private
    visibility += [frozenset(range(n_agents))] * n_shared
    return visibility


def overlap_family(n_agents: int, n_private: int, pair_shared: int, all_shared: int):
    """The generalisation: some nodes visible to only SOME agents.

    A node visible to {0, 1} but not {2} is hidden from agent 2 while being a legitimate
    parent of nodes agents 0 and 1 can see. If confinement breaks anywhere, here is where.
    """
    visibility = []
    for agent in range(n_agents):
        visibility += [frozenset({agent})] * n_private
    for a, b in itertools.combinations(range(n_agents), 2):
        visibility += [frozenset({a, b})] * pair_shared
    visibility += [frozenset(range(n_agents))] * all_shared
    return visibility


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=2 ** 22,
                    help="enumerate exhaustively below this many orientations")
    ap.add_argument("--samples", type=int, default=20000,
                    help="random DAGs per configuration when exhaustive is impossible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", default=JOINTLY_VISIBLE, choices=list(EDGE_RULES))
    ap.add_argument("--out", default="results/confinement_n_agents.json")
    args = ap.parse_args(argv)

    configs = []
    for n in (2, 3, 4):
        for n_private in (1, 2):
            for n_shared in (2, 3):
                configs.append((f"disjoint n={n} priv={n_private} shared={n_shared}",
                                disjoint_family(n, n_private, n_shared), n))
    for n in (3, 4):
        configs.append((f"overlap n={n} priv=1 pairshared=1 allshared=1",
                        overlap_family(n, 1, 1, 1), n))
        configs.append((f"overlap n={n} priv=1 pairshared=1 allshared=0",
                        overlap_family(n, 1, 1, 0), n))

    report = {"args": vars(args), "configs": []}
    any_violation = False
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for label, visibility, n_agents in configs:
        started = time.time()
        result = check(visibility, n_agents, args.limit, args.samples, args.seed,
                       rule=args.rule)
        result["label"] = label
        result["seconds"] = time.time() - started
        report["configs"].append(result)
        out.write_text(json.dumps(report, indent=1))     # incremental: survive a kill
        kind = "exhaustive" if result["exhaustive"] else f"sampled {result['n_dags']}"
        status = "HOLDS" if result["holds"] else "*** VIOLATED ***"
        print(f"{label:44s} d={result['d']:2d}  {kind:18s}  "
              f"{result['projections_checked']:8d} projections  {status}  "
              f"[{result['seconds']:.1f}s]", flush=True)
        if not result["holds"]:
            any_violation = True
            print("   counterexample:", json.dumps(result["violations"][0]["pair"]),
                  "private endpoint", result["violations"][0]["private_endpoint"], flush=True)

    report["holds_everywhere"] = not any_violation
    out.write_text(json.dumps(report, indent=1))
    print()
    print("CONFINEMENT HOLDS EVERYWHERE" if not any_violation
          else "CONFINEMENT FAILS -- the scaling plan changes shape")
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
