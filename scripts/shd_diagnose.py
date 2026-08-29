"""Why is the learned arm's soft SHD worse than greedy's? Four checks, one entry point.

WHY THIS EXISTS. `scripts/shd.py` reports soft SHD per pair and the learned arm loses at
every window rung above k=4. Read as written -- "greedy's belief is closer to the true MAG"
-- that contradicts the learned arm's win on identification. It turns out both the reading
and the comparison are wrong, in ways that only a decomposition shows. See
`docs/FINDINGS_SHD_2026_08_29.md`.

  decompose  Split each pair's contribution into WRONG (confident and wrong), UNSETTLED
             (nothing reaches the bar) and RESIDUAL (right, mass left over), alongside node
             coverage, repeats and both success criteria. Adds an ARGMAX arm, which
             `shd.py` does not evaluate -- it loads with deterministic=False.

  identity   Falsify `soft SHD per pair == 1 - 1/|surviving marks|`. `FactoredBelief`
             spreads mass uniformly over survivors and each update is individually sound, so
             under oracle evidence the true mark never leaves the set. If that holds, the
             metric can only take the values {0, 1/2, 2/3, 3/4} and can NEVER register a
             structural error -- it is a count of residual ambiguity wearing a distance's
             name.

  descent    Falsify `greedy's node score == the number of nonzero-SHD pairs incident to
             that node`. At bar=1.0 `_unsure_touching` counts a pair iff more than one mark
             survives, which is exactly the nonzero-SHD condition. If the two agree, the
             baseline is one-step steepest descent on the support of the metric it is being
             scored on, and the comparison is not neutral between the arms.

  spend      SHARED versus PRIVATE allocation. A move on a shared node lands in every
             agent's window at once; a move on a private node helps only the mover. Soft SHD
             is averaged over windows, so it prices a shared move at roughly n times a
             private one. Reports the share of moves on shared nodes, the resulting union
             coverage per window, and duplicate coverage of the shared surface.

  targeting  WHERE the budget goes: mean true within-window degree of intervened nodes
             against the window mean, and residual mass split by how many of a pair's
             endpoints anyone intervened on. The second is the forced-cover rule as a
             measurement -- a directed edge needs its TAIL, a bidirected pair needs BOTH.

Evaluation only. Rebuilds each environment from the run's OWN config block via
`env_from_config`, for the reason given in `scripts/rescore_from_config.py`.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
from itertools import combinations
from typing import Dict

import numpy as np

from cb.claims import score_window
from cb.versionspace import BACK, BI, FWD, NONE
from ma.baselines import RandomAgent, UncertaintyGreedyAgent
from ma.env import CLAMP, PASS_ACTION
from ma.policy import IndependentPPO
from ma.projection import BIDIRECTED as MAG_BI
from ma.projection import DIRECTED as MAG_DIR
from scripts.rescore_from_config import env_from_config

# The bucket bar for `decompose` only. It separates "committed" from "diffuse" and is a
# DIAGNOSTIC choice -- the environment grades at `config.claim_bar`, which the solve-rate
# columns use. Kept at 0.7 so a near-certain-but-wrong mark would still land in WRONG.
BUCKET_BAR = 0.7
GRID = (0.0, 0.5, round(2 / 3, 6), 0.75)
MARK_OF = {0: NONE, 1: FWD, 2: BACK, 3: BI}


def _truth_index(mag: np.ndarray, u: int, v: int) -> int:
    if mag[u, v] == MAG_BI:
        return 3
    if mag[u, v] == MAG_DIR:
        return 1
    if mag[v, u] == MAG_DIR:
        return 2
    return 0


def _masses(belief, u: int, v: int):
    """(NONE, FWD, BACK, BI) mass for one pair. Order matches `_truth_index`."""
    return (1.0 - float(belief.adjacency[u, v]), float(belief.directed[u, v]),
            float(belief.directed[v, u]), float(belief.bidirected[u, v]))


def _survivors(belief, u: int, v: int):
    return belief.possible.get((u, v), belief.possible.get((v, u)))


def _load(path: pathlib.Path, seed: int):
    config = json.loads(path.read_text())["config"]
    env = env_from_config(config, seed=seed)
    ppo = IndependentPPO.load(str(path.with_suffix(".pt")), env)
    return config, env, ppo


def _arms(env, ppo, seed: int, with_sample: bool = True) -> Dict[str, dict]:
    arms = {}
    if with_sample:
        arms["learned_sample"] = ppo.policies(deterministic=False)
    arms["learned_argmax"] = ppo.policies(deterministic=True)
    arms["greedy"] = {a: UncertaintyGreedyAgent(a, seed, bar=1.0) for a in env.topology.agents}
    arms["random_vary"] = {a: RandomAgent(a, seed, allow_clamp=False) for a in env.topology.agents}
    return arms


def _rollout(env, policies, episodes: int, seed: int):
    """Run one arm and yield the finished env per episode, plus the move tally."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        touched = {a: set() for a in env.topology.agents}
        moves = repeats = passes = clamps = 0
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
            for agent, (node, mode) in env.last_chosen.items():
                if node == PASS_ACTION:
                    passes += 1
                    continue
                moves += 1
                clamps += int(mode == CLAMP)
                repeats += int(node in touched[agent])
                touched[agent].add(node)
        yield dict(moves=moves, repeats=repeats, passes=passes, clamps=clamps)


def _paired(a, b) -> str:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return f"{d.mean():+.4f} +/- {se:.4f}  {'SIG' if abs(d.mean()) > 2 * se else 'ns '}"


# --------------------------------------------------------------------------- decompose

def decompose(path: pathlib.Path, episodes: int, seed: int) -> None:
    config, env, ppo = _load(path, seed)
    k = len(env.windows[env.topology.agents[0]].nodes)
    print(f"\n=== {path.stem}  k={k} agents={len(env.topology.agents)} "
          f"budget={config['budget']} claim_bar={config['claim_bar']} ===")
    header = (f"{'arm':16s} {'softSHD':>8s} {'wrong':>8s} {'unsett':>8s} {'resid':>8s} "
              f"{'cover':>7s} {'repeat':>7s} {'joint':>7s} {'perwin':>7s}")
    print(header)
    print("-" * len(header))
    out: Dict[str, dict] = {}
    for name, policies in _arms(env, ppo, seed).items():
        acc = collections.defaultdict(list)
        for tally in _rollout(env, policies, episodes, seed):
            wrong = unsettled = residual = 0.0
            pairs = covered = nodes = solved = agents = 0
            for agent, window in env.windows.items():
                mag = np.asarray(env._true_mag(agent))
                belief = window.belief.last
                for u, v in combinations(range(window.k), 2):
                    masses = _masses(belief, u, v)
                    truth = _truth_index(mag, u, v)
                    gap = 1.0 - masses[truth]
                    top = int(np.argmax(masses))
                    if masses[top] < BUCKET_BAR:
                        unsettled += gap
                    elif top != truth:
                        wrong += gap
                    else:
                        residual += gap
                    pairs += 1
                covered += int((np.asarray(env.own_counts[agent]) > 0).sum())
                nodes += window.k
                score = score_window(
                    belief, mag,
                    private_positions=[window.pos[n] for n in window.private],
                    bar=env.config.claim_bar,
                    require_all_types=env.config.claims_require_all_types)
                solved += int(score.required_right == score.required_total
                              and score.required_wrong == 0)
                agents += 1
            acc["soft"].append((wrong + unsettled + residual) / pairs)
            acc["wrong"].append(wrong / pairs)
            acc["unsettled"].append(unsettled / pairs)
            acc["residual"].append(residual / pairs)
            acc["cover"].append(covered / nodes)
            acc["repeat"].append(tally["repeats"] / max(tally["moves"], 1))
            acc["joint"].append(float(solved == agents))
            acc["perwin"].append(solved / agents)
        out[name] = {key: np.asarray(v) for key, v in acc.items()}
        m = out[name]
        print(f"{name:16s} {m['soft'].mean():8.4f} {m['wrong'].mean():8.4f} "
              f"{m['unsettled'].mean():8.4f} {m['residual'].mean():8.4f} "
              f"{m['cover'].mean():7.3f} {m['repeat'].mean():7.3f} "
              f"{m['joint'].mean():7.3f} {m['perwin'].mean():7.3f}")
    print("\n  paired vs greedy (+ means the learned arm is worse on SHD terms)")
    for name in ("learned_sample", "learned_argmax"):
        for field in ("soft", "wrong", "cover", "joint", "perwin"):
            print(f"    {name:15s} {field:9s} {_paired(out[name][field], out['greedy'][field])}")


# ---------------------------------------------------------------------------- identity

def identity(path: pathlib.Path, episodes: int, seed: int) -> None:
    """Every per-pair soft value must land in GRID, and the truth must never leave `possible`."""
    _, env, ppo = _load(path, seed)
    print(f"\n=== {path.stem}: soft SHD per pair == 1 - 1/|surviving marks|? ===")
    for name, policies in _arms(env, ppo, seed, with_sample=False).items():
        counts: collections.Counter = collections.Counter()
        off_grid = truth_lost = total = 0
        for _ in _rollout(env, policies, episodes, seed):
            for agent, window in env.windows.items():
                mag = np.asarray(env._true_mag(agent))
                belief = window.belief.last
                for u, v in combinations(range(window.k), 2):
                    truth = _truth_index(mag, u, v)
                    soft = round(1.0 - _masses(belief, u, v)[truth], 6)
                    counts[soft] += 1
                    total += 1
                    off_grid += int(soft not in GRID)
                    surviving = _survivors(belief, u, v)
                    if surviving is not None and MARK_OF[truth] not in surviving:
                        truth_lost += 1
        print(f"\n  {name}: {total} pairs")
        for value, n in sorted(counts.items()):
            marks = round(1 / (1 - value)) if value < 1 else "-"
            print(f"    soft={value:<9} n={n:6d} ({n / total:6.2%})  => {marks} surviving marks")
        print(f"    off-grid: {off_grid}    true mark not in survivors: {truth_lost}")
        print(f"    VERDICT: {'identity holds' if not (off_grid or truth_lost) else 'REFUTED'}")


# ----------------------------------------------------------------------------- descent

def descent(path: pathlib.Path, episodes: int, seed: int) -> None:
    """Greedy's node score must equal the count of nonzero-SHD pairs incident to that node."""
    _, env, ppo = _load(path, seed)
    policies = ppo.policies(deterministic=True)
    probe = UncertaintyGreedyAgent(0, seed, bar=1.0)
    disagree = total = 0
    # Checked at EVERY step, not just at episode end: the intermediate belief states are
    # the ones greedy actually chooses from, so they are where the claim has to hold.
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
            for agent, window in env.windows.items():
                belief = window.belief.last
                greedy_score = probe._unsure_touching(belief, window.k)
                shd_score = np.zeros(window.k)
                for u, v in combinations(range(window.k), 2):
                    surviving = _survivors(belief, u, v)
                    if surviving is not None and len(surviving) > 1:
                        shd_score[u] += 1
                        shd_score[v] += 1
                total += window.k
                disagree += int((greedy_score != shd_score).sum())
    print(f"\n=== {path.stem}: is greedy descent on the SHD support? ===")
    print(f"  node-scores compared: {total}   disagreements: {disagree}")
    print(f"  VERDICT: {'IDENTICAL' if disagree == 0 else 'NOT identical -- claim refuted'}")


# --------------------------------------------------------------------------- targeting

def targeting(path: pathlib.Path, episodes: int, seed: int) -> None:
    config, env, ppo = _load(path, seed)
    k = len(env.windows[env.topology.agents[0]].nodes)
    print(f"\n=== {path.stem}  k={k} budget={config['budget']}: where does the budget go? ===")
    header = (f"{'arm':16s} {'deg_hit':>8s} {'deg_all':>8s} {'ratio':>7s} {'hubshr':>7s} "
              f"{'resid_0':>8s} {'resid_1':>8s} {'resid_2':>8s} | {'n0':>6s} {'n1':>6s} {'n2':>6s}")
    print(header)
    print("-" * len(header))
    out: Dict[str, dict] = {}
    for name, policies in _arms(env, ppo, seed, with_sample=False).items():
        acc = collections.defaultdict(list)
        for _ in _rollout(env, policies, episodes, seed):
            touched_any = set(env._touched_by)
            hit_deg, all_deg, hub = [], [], []
            mass = {0: 0.0, 1: 0.0, 2: 0.0}
            count = {0: 0, 1: 0, 2: 0}
            for agent, window in env.windows.items():
                mag = np.asarray(env._true_mag(agent))
                adjacent = (mag != 0) | (mag.T != 0)
                np.fill_diagonal(adjacent, False)
                degree = adjacent.sum(axis=1).astype(float)
                hit = np.flatnonzero(np.asarray(env.own_counts[agent]) > 0)
                if len(hit):
                    hit_deg.append(degree[hit].mean())
                    hub.append(float((degree[hit] >= max(np.quantile(degree, 0.75), 1)).mean()))
                all_deg.append(degree.mean())
                covered = np.zeros(window.k, dtype=bool)
                for node in window.nodes:
                    if node in touched_any:
                        covered[window.pos[node]] = True
                belief = window.belief.last
                for u, v in combinations(range(window.k), 2):
                    gap = 1.0 - _masses(belief, u, v)[_truth_index(mag, u, v)]
                    bucket = int(covered[u]) + int(covered[v])
                    mass[bucket] += gap
                    count[bucket] += 1
            total = max(sum(count.values()), 1)
            acc["deg_hit"].append(np.mean(hit_deg) if hit_deg else np.nan)
            acc["deg_all"].append(np.mean(all_deg))
            acc["hub"].append(np.mean(hub) if hub else np.nan)
            for bucket in (0, 1, 2):
                acc[f"resid{bucket}"].append(mass[bucket] / total)
                acc[f"n{bucket}"].append(count[bucket] / total)
        out[name] = {key: np.asarray(v, dtype=float) for key, v in acc.items()}
        m = out[name]
        print(f"{name:16s} {np.nanmean(m['deg_hit']):8.3f} {m['deg_all'].mean():8.3f} "
              f"{np.nanmean(m['deg_hit']) / m['deg_all'].mean():7.3f} {np.nanmean(m['hub']):7.3f} "
              f"{m['resid0'].mean():8.4f} {m['resid1'].mean():8.4f} {m['resid2'].mean():8.4f} | "
              f"{m['n0'].mean():6.3f} {m['n1'].mean():6.3f} {m['n2'].mean():6.3f}")
    print("\n  paired learned_argmax - greedy")
    for field in ("deg_hit", "hub", "resid0", "resid1", "resid2", "n0", "n2"):
        print(f"    {field:9s} {_paired(out['learned_argmax'][field], out['greedy'][field])}")


# ------------------------------------------------------------------------------- spend

def spend(path: pathlib.Path, episodes: int, seed: int) -> None:
    config, env, ppo = _load(path, seed)
    k = len(env.windows[env.topology.agents[0]].nodes)
    print(f"\n=== {path.stem}  k={k} agents={len(env.topology.agents)} "
          f"budget={config['budget']}: shared vs private spend ===")
    header = f"{'arm':16s} {'shared%':>8s} {'own_cov':>8s} {'union_cov':>10s} {'duplicate':>10s}"
    print(header)
    print("-" * len(header))
    out: Dict[str, dict] = {}
    exposed = set(env.topology.exposed)
    for name, policies in _arms(env, ppo, seed, with_sample=False).items():
        acc = collections.defaultdict(list)
        for policy in policies.values():
            if hasattr(policy, "reset"):
                policy.reset(seed)
        for episode in range(episodes):
            result = env.reset(seed=seed * 100_000 + episode)
            on_shared = moves = 0
            while not result.done:
                result = env.step({a: policies[a](env, result) for a in env.topology.agents})
                for agent, (node, _mode) in env.last_chosen.items():
                    if node == PASS_ACTION:
                        continue
                    moves += 1
                    on_shared += int(node in env.windows[agent].shared)
            touched = set(env._touched_by)
            own, union = [], []
            for agent, window in env.windows.items():
                own.append(float((np.asarray(env.own_counts[agent]) > 0).sum()) / window.k)
                union.append(sum(n in touched for n in window.nodes) / window.k)
            hit = [n for n in exposed if n in env._touched_by]
            acc["shared"].append(on_shared / max(moves, 1))
            acc["own"].append(float(np.mean(own)))
            acc["union"].append(float(np.mean(union)))
            acc["dup"].append(float(np.mean([len(env._touched_by[n]) > 1 for n in hit]))
                              if hit else 0.0)
        out[name] = {key: np.asarray(v) for key, v in acc.items()}
        m = out[name]
        print(f"{name:16s} {m['shared'].mean():8.3f} {m['own'].mean():8.3f} "
              f"{m['union'].mean():10.3f} {m['dup'].mean():10.3f}")
    print("\n  paired learned_argmax - greedy")
    for field in ("shared", "own", "union", "dup"):
        print(f"    {field:8s} {_paired(out['learned_argmax'][field], out['greedy'][field])}")


# ------------------------------------------------------------------------------ global

def global_shd(path: pathlib.Path, episodes: int, seed: int) -> None:
    """SHD over the union of covered pairs, each pair counted ONCE.

    WHY. `scripts/shd.py` sums over windows and divides by the total window-pair count, so a
    pair of SHARED nodes is counted once PER AGENT -- n times. Soft SHD therefore prices a
    shared move at roughly n private ones, which is exactly the channel greedy exploits
    (`spend`: greedy puts 44-71% of its moves on the shared surface against the learner's
    35-59%). If the metric is de-duplicated the channel should close.

    Cross-private pairs -- one agent's private node against another's -- are in NOBODY's
    window and are excluded. That is not a gap in the measurement, it is the federated
    constraint: no agent can hold a belief about them, so no arm can differ on them. They
    are about half of all pairs, so including them would only add a constant.

    THREE NUMBERS.
      per_window  the existing metric, reproduced as a control.
      dedup       each covered pair counted once; a pair in several windows contributes the
                  MEAN of its per-window soft values. Each window is still scored against
                  its OWN MAG, so there is no projection ambiguity. This is the clean test.
      pooled      additionally intersects the survivor sets across windows before scoring --
                  the federated belief. Sound because under oracle evidence every agent's
                  survivor set contains the truth, so the intersection does too.

    THE ONE SUBTLETY, reported rather than hidden. A shared pair's true mark is a LATENT
    PROJECTION into a window, and different windows project out different private nodes, so
    two agents can legitimately hold different true marks for the same shared pair. The
    `mark_disagree` column counts it. Where it happens, `pooled` cannot intersect and falls
    back to the per-window mean, same as `dedup`.
    """
    _, env, ppo = _load(path, seed)
    print(f"\n=== {path.stem}: SHD de-duplicated over covered pairs ===")
    header = (f"{'arm':16s} {'per_window':>11s} {'dedup':>9s} {'pooled':>9s} "
              f"{'mark_disagree':>14s}")
    print(header)
    print("-" * len(header))
    out: Dict[str, dict] = {}
    for name, policies in _arms(env, ppo, seed, with_sample=False).items():
        acc = collections.defaultdict(list)
        for _ in _rollout(env, policies, episodes, seed):
            per_pair: Dict[tuple, list] = collections.defaultdict(list)
            window_soft = window_pairs = 0.0, 0
            window_soft, window_pairs = 0.0, 0
            for agent, window in env.windows.items():
                mag = np.asarray(env._true_mag(agent))
                belief = window.belief.last
                for u, v in combinations(range(window.k), 2):
                    truth = _truth_index(mag, u, v)
                    soft = 1.0 - _masses(belief, u, v)[truth]
                    window_soft += soft
                    window_pairs += 1
                    key = tuple(sorted((window.nodes[u], window.nodes[v])))
                    per_pair[key].append((soft, truth, _survivors(belief, u, v)))
            dedup = pooled = 0.0
            disagree = 0
            for entries in per_pair.values():
                dedup += float(np.mean([e[0] for e in entries]))
                marks = {e[1] for e in entries}
                if len(marks) > 1:
                    disagree += 1
                    pooled += float(np.mean([e[0] for e in entries]))
                    continue
                sets = [e[2] for e in entries if e[2] is not None]
                if not sets:
                    pooled += float(np.mean([e[0] for e in entries]))
                    continue
                merged = set(sets[0])
                for other in sets[1:]:
                    merged &= set(other)
                truth_mark = MARK_OF[entries[0][1]]
                pooled += (1.0 - 1.0 / len(merged)) if merged and truth_mark in merged \
                    else float(np.mean([e[0] for e in entries]))
            n = max(len(per_pair), 1)
            acc["per_window"].append(window_soft / max(window_pairs, 1))
            acc["dedup"].append(dedup / n)
            acc["pooled"].append(pooled / n)
            acc["disagree"].append(disagree / n)
        out[name] = {key: np.asarray(v) for key, v in acc.items()}
        m = out[name]
        print(f"{name:16s} {m['per_window'].mean():11.4f} {m['dedup'].mean():9.4f} "
              f"{m['pooled'].mean():9.4f} {m['disagree'].mean():14.4f}")
    print("\n  paired learned_argmax - greedy (+ means the learned arm is worse)")
    for field in ("per_window", "dedup", "pooled"):
        print(f"    {field:11s} {_paired(out['learned_argmax'][field], out['greedy'][field])}")


CHECKS = {"decompose": decompose, "identity": identity, "descent": descent,
          "targeting": targeting, "spend": spend, "global": global_shd}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="ladder result JSONs, with .pt beside them")
    ap.add_argument("--check", default="decompose", choices=[*CHECKS, "all"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    names = list(CHECKS) if args.check == "all" else [args.check]
    for result in args.results:
        for name in names:
            episodes = min(args.episodes, 25) if name in ("identity", "descent") else args.episodes
            CHECKS[name](pathlib.Path(result), episodes, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
