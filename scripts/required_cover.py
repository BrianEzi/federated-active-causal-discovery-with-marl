"""The intervention set a window REQUIRES, derived from the belief mechanism itself.

WHY THIS EXISTS. The main scaling figure holds interventions-per-node fixed across window
sizes, and the standing objection is that this is the wrong normalisation: if the number of
experiments a window actually NEEDS grows sublinearly in k, then an iso-budget curve hands
the large windows a more generous budget and the decline it shows is a budget effect wearing
a window-size costume. `docs/PLAN_2026_08_28.md` F1 states the required set is closed-form --
a directed edge needs its TAIL, a bidirected pair needs BOTH endpoints -- at 0.757k for k=4
falling to 0.542k at k=30.

WHY IT DERIVES THE RULE INSTEAD OF APPLYING IT. That closed form is an assertion about
`cb.factored._apply_ancestry`, and the whole figure would rest on it. So `forced_positions`
does not implement the rule: it asks the backend which positions settle which pair, by
replay, and then `--check_rule` compares the answer against the stated closed form. If they
agree the rule is confirmed on real draws and can be used at scale; where they disagree the
mechanism wins and the disagreement is printed.

WHAT "REQUIRED" MEANS HERE. Exactly what `cb.claims.score_window` grades: the pairs whose
claims carry `required=True` -- every type claim under `claims_require_all_types`, plus every
adjacency claim. A cover that settles those is a cover that identifies the window, which is
the event the figure plots.

ORACLE EVIDENCE ONLY. Under sampled evidence the belief is not a function of the intervened
SET alone, so no set is sufficient with certainty and "the required cover" is not defined.
The script refuses rather than returning a number that would be quietly meaningless.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from itertools import combinations
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

from cb.factored import FactoredBackend, marks_from_mag, pairs
from cb.versionspace import reveal
from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED


def settled_pairs(true_mag: np.ndarray, k: int, positions) -> Set[Tuple[int, int]]:
    """The pairs whose mark is pinned to the truth after intervening on exactly `positions`.

    A replay, not a rollback: `_apply_ancestry` only ever narrows a pair, so the belief
    after a set is rebuilt from scratch. Mirrors `cb.factored.credit_for_set`, which is the
    same computation reported as a fraction.
    """
    backend = FactoredBackend(k)
    backend.reset(np.asarray(true_mag))
    backend.reset_marks()
    for x in sorted(positions):
        backend._apply_ancestry(x, reveal(backend.truth, k, x))
    truth = marks_from_mag(np.asarray(true_mag))
    return {key for index, key in enumerate(pairs(k))
            if backend._possible[key] == frozenset({truth[index]})}


def minimal_covers(true_mag: np.ndarray, k: int,
                   max_size: int = 2) -> Dict[Tuple[int, int], List[FrozenSet[int]]]:
    """For each pair, every MINIMAL position set of size <= max_size that settles it.

    Minimal in the subset sense: a set is recorded only if no proper subset already settles
    the pair. More than one entry means the pair offers a CHOICE and the cover is a set-cover
    problem rather than a forced union -- which is the assumption `--check_rule` tests.
    """
    covers: Dict[Tuple[int, int], List[FrozenSet[int]]] = {key: [] for key in pairs(k)}
    settled_by: Dict[FrozenSet[int], Set[Tuple[int, int]]] = {}
    for size in range(0, max_size + 1):
        for subset in combinations(range(k), size):
            key = frozenset(subset)
            settled_by[key] = settled_pairs(true_mag, k, subset)
    for pair_key, entries in covers.items():
        for size in range(0, max_size + 1):
            for subset in combinations(range(k), size):
                candidate = frozenset(subset)
                if pair_key not in settled_by[candidate]:
                    continue
                if any(smaller < candidate for smaller in entries):
                    continue
                entries.append(candidate)
    return covers


def required_pairs(true_mag: np.ndarray, k: int) -> Set[Tuple[int, int]]:
    """The pairs `cb.claims` grades as required, under `claims_require_all_types=True`.

    Every pair carries an adjacency claim, so every pair is required. Kept as a function
    because the criterion has moved twice and a future one may not be "all of them".
    """
    return set(pairs(k))


def closed_form(true_mag: np.ndarray, k: int) -> Set[int]:
    """The rule `docs/PLAN_2026_08_28.md` F1 asserts: tails of directed edges, both endpoints
    of confounded pairs."""
    mag = np.asarray(true_mag)
    needed: Set[int] = set()
    for u, v in pairs(k):
        if mag[u, v] == MAG_BIDIRECTED:
            needed |= {u, v}
        elif mag[u, v] == MAG_DIRECTED:
            needed.add(u)
        elif mag[v, u] == MAG_DIRECTED:
            needed.add(v)
    return needed


def forced_positions(true_mag: np.ndarray, k: int,
                     max_size: int = 2) -> Tuple[Set[int], List[str]]:
    """The union of every required pair's unique minimal cover, plus any complaints.

    Returns the positions and a list of the pairs that were NOT forced -- either because no
    set of size <= max_size settles them, or because more than one does. An empty complaint
    list is what licenses calling the result "the required cover" rather than "a cover".
    """
    covers = minimal_covers(true_mag, k, max_size)
    needed: Set[int] = set()
    complaints: List[str] = []
    for pair_key in sorted(required_pairs(true_mag, k)):
        entries = covers[pair_key]
        if not entries:
            complaints.append(f"{pair_key}: no cover of size <= {max_size}")
            continue
        smallest = min(len(e) for e in entries)
        minima = [e for e in entries if len(e) == smallest]
        if smallest == 0:
            continue                      # settled by observation alone; costs nothing
        if len(minima) > 1:
            complaints.append(f"{pair_key}: {len(minima)} minimal covers {minima}")
            continue
        needed |= set(minima[0])
    return needed, complaints


def _windows_of(env):
    return [(agent, env.windows[agent]) for agent in env.topology.agents]


def measure(config: dict, episodes: int, seed_base: int, max_size: int,
            check_rule: bool, closed_form_only: bool = False) -> dict:
    """Per episode, the required cover of every window AND of the system.

    TWO NUMBERS, AND ONLY THE SECOND PAIRS WITH THE BUDGET. `forced_positions` works in
    window POSITIONS, so summing it over agents double-counts every shared node -- and the
    windows overlap on the shared set by construction. `budget` is a pool of ROUNDS for the
    whole system (`docs/TURN_BUDGET_SPEC.md` section 2), one intervention per round, so the
    quantity a budget must cover is the union of the per-window sets mapped back to GLOBAL
    node ids. That union is `required_system`; `required` stays per window because the
    identification event is per window.
    """
    from scripts.rescore_from_config import env_from_config

    if config.get("vs_evidence", "oracle") != "oracle":
        raise SystemExit("required cover is defined under ORACLE evidence only -- under "
                         "sampled evidence the belief is not a function of the intervened "
                         "set, so no set is sufficient with certainty")
    env = env_from_config(config, seed=seed_base)
    rows, episode_rows, disagreements, complaints_total = [], [], 0, 0
    for episode in range(episodes):
        env.reset(seed=seed_base + episode)
        union: Set[int] = set()
        for agent, window in _windows_of(env):
            mag = np.asarray(env._true_mag(agent))
            k = window.k
            if closed_form_only:
                needed, complaints = closed_form(mag, k), []
            else:
                needed, complaints = forced_positions(mag, k, max_size)
            complaints_total += len(complaints)
            union |= {int(window.nodes[p]) for p in needed}
            row = {"episode": episode, "agent": int(agent), "k": int(k),
                   "required": len(needed), "complaints": len(complaints)}
            if check_rule and not closed_form_only:
                predicted = closed_form(mag, k)
                row["closed_form"] = len(predicted)
                if predicted != needed:
                    disagreements += 1
                    row["disagreement"] = {"mechanism": sorted(needed),
                                           "closed_form": sorted(predicted)}
            rows.append(row)
        episode_rows.append({"episode": episode, "required_system": len(union)})
    ks = sorted({r["k"] for r in rows})
    system = [r["required_system"] for r in episode_rows]
    summary = {"episodes": episodes, "budget": config["budget"],
               "n_agents": len(config["topology"]["private"]),
               "windows": len(rows), "unforced_claims": complaints_total,
               "closed_form_only": bool(closed_form_only),
               "mean_required_system": float(np.mean(system)),
               "sd_required_system": float(np.std(system)),
               "budget_over_required": config["budget"] / float(np.mean(system)),
               "by_k": {}}
    for k in ks:
        sel = [r["required"] for r in rows if r["k"] == k]
        summary["by_k"][str(k)] = {"mean_required": float(np.mean(sel)),
                                   "sd": float(np.std(sel)),
                                   "per_node": float(np.mean(sel)) / k,
                                   "n": len(sel)}
    if check_rule and not closed_form_only:
        summary["closed_form_disagreements"] = disagreements
    return {"summary": summary, "rows": rows, "episodes_rows": episode_rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a result .json whose config defines the episodes")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=90_000)
    ap.add_argument("--max_size", type=int, default=2,
                    help="largest position set searched per pair")
    ap.add_argument("--check_rule", action="store_true",
                    help="compare the mechanism against PLAN F1's closed form")
    ap.add_argument("--closed_form_only", action="store_true",
                    help="skip the exhaustive per-pair search and apply the closed form. "
                         "Only legitimate because --check_rule agreed with the mechanism "
                         "on every window at k=4, 6 and 8; the search is O(k^2) replays "
                         "per window and does not scale to k=30.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    config = json.loads(pathlib.Path(args.source).read_text())["config"]
    result = measure(config, args.episodes, args.seed_base, args.max_size,
                     args.check_rule, args.closed_form_only)
    summary = result["summary"]
    print(json.dumps(summary, indent=1))
    if summary["unforced_claims"]:
        print(f"\n  {summary['unforced_claims']} claim(s) were NOT forced -- the union is a "
              "LOWER BOUND on the cover, not the cover")
    if args.check_rule:
        print(f"\n  closed form disagreed with the mechanism on "
              f"{summary['closed_form_disagreements']} of {summary['windows']} windows")
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
