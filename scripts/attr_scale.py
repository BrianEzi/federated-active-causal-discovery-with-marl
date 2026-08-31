"""How far attribution reaches, measured, for the two factored backends.

WHAT QUESTION THIS ANSWERS. Attribution was cut from the thesis on the belief that it capped
at k~5. `cb/factored_attribution.py` moved the cap to k=12 by factoring the STRUCTURE half.
`cb/component_attribution.py` factors the OWNERSHIP half over connected components of the
bidirected graph. This script measures what each actually delivers at a given window size,
on identical episodes, so "it scales" is a table rather than an argument.

WHAT IT REPORTS, and why each column is here rather than only the headline:

    right/wrong/unsure   the claim verdict per TRUE latent group at bar 1.0. WRONG must be
                         zero: a confident misattribution is worse than no attribution, and
                         it is the failure mode both backends are designed against.
    scope                mean share of settled bidirected pairs the belief may speak about.
                         The enumerated-ownership backend caps this GLOBALLY, so on a large
                         window most pairs are unattributable by construction and its
                         `unsure` is dominated by pairs it never looked at. The component
                         backend caps PER COMPONENT. Comparing `right` without this column
                         compares two different denominators.
    pairs / comp / max   settled pairs, components they fall into, largest component. The
                         component backend's cost is set by the LAST of these, not the first.
    cands                candidates held. For the component backend this is the size of a
                         product that is never built, so it is printed in scientific form
                         and routinely exceeds what could be enumerated.
    s/ep                 wall clock, which is the thing that decides whether an eval pass
                         over the sweep's checkpoints is affordable before the freeze.

The driving policy is a deterministic round-robin sweep of each window, NOT a learned one.
That is deliberate: this measures the BELIEF's reach, and a learned policy would confound it
with how well that policy happens to probe partners.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cb.attribution import response_signature, score_groups
from cb.component_attribution import ComponentAttributedBackend
from cb.factored_attribution import FactoredAttributedBackend
from ma.env import MAConfig, TwoAgentEnv
from ma.topology import federated_topology

BACKENDS = {"component": ComponentAttributedBackend, "factored": FactoredAttributedBackend}


def build_env(k: int, sigma: float, n_agents: int, budget: int, n_obs: int, n_int: int):
    private = max(1, round(k * (1 - sigma)))
    shared = max(1, k - private)
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(n_agents, private, shared), n_obs=n_obs, n_int=n_int,
        budget=budget, turn_order="round_robin", belief_backend="factored",
        action_modes=("vary",), claim_bar=1.0, reward_criterion="claims",
        policy_arch="gnn_portable", graph_model="sf", sf_m=2, episode_mix="confounded",
        vs_evidence="oracle"))


def drive(env, backends, episode: int) -> None:
    """One episode, sweeping every window position, feeding every backend the same evidence.

    The partner channel is the ORACLE signature -- which of the agent's true groups the
    actor's private node sits above -- because this measures the belief, not the detector.
    Under sampled evidence the same message would come from `estimated_moved`.
    """
    result = env.reset(seed=episode)
    for agent, backend in backends.items():
        backend.reset(env._true_mag(agent), adjacency=env.true_adjacency,
                      topology=env.topology)
    turns = {a: 0 for a in env.topology.agents}
    while not result.done:
        active = env.active_agent()
        actions = {a: env.windows[a].action_index(
                       env.windows[a].nodes[turns[a] % env.windows[a].k], "vary")
                   for a in env.topology.agents}
        result = env.step(actions)
        for agent, backend in backends.items():
            backend.edge_marginals(env.samples[:, env.windows[agent].nodes], env.known[agent])
        if active is not None:
            node, _ = env.last_chosen[active]
            if node is not None and node not in env.topology.exposed:
                for agent, backend in backends.items():
                    if agent == active or not backend.true_groups:
                        continue
                    hit = response_signature(env.true_adjacency, env.topology, agent,
                                             backend.true_groups, node)
                    moved = frozenset(p for g, h in zip(backend.true_groups, hit) if h
                                      for p in g.pairs())
                    if moved:
                        backend.observe_partner(active, moved)
            turns[active] += 1


def run_cell(name: str, k: int, sigma: float, n_agents: int, budget: int, episodes: int,
             n_obs: int, n_int: int, which: List[str], cap: int, ref_cap: int,
             max_candidates: int, local_disturbance: bool) -> List[Dict]:
    rows = []
    for label in which:
        env = build_env(k, sigma, n_agents, budget, n_obs, n_int)
        agents = list(env.topology.agents)
        kwargs = dict(n_agents=len(agents), evidence="oracle")
        tally = {"right": 0, "wrong": 0, "unsure": 0, "total": 0}
        scope_share, pairs, comps, largest, cands = [], [], [], [], []
        contradictions = out_of_scope = violations = 0
        msg_single = msg_cross = vio_single = vio_cross = 0
        start = time.time()
        for episode in range(episodes):
            if label == "component":
                backends = {a: ComponentAttributedBackend(
                                env.windows[a].k, agent=a, max_component_pairs=cap,
                                max_component_candidates=max_candidates,
                                local_disturbance=local_disturbance, **kwargs)
                            for a in agents}
            else:
                backends = {a: FactoredAttributedBackend(
                                env.windows[a].k, agent=a, max_attribution_pairs=ref_cap,
                                local_disturbance=local_disturbance, **kwargs)
                            for a in agents}
            drive(env, backends, episode)
            for agent, backend in backends.items():
                score = score_groups(backend.last, backend.true_groups, bar=1.0)
                for key in ("right", "wrong", "unsure", "total"):
                    tally[key] += score[key]
                settled = backend.settled_bidirected()
                pairs.append(len(settled))
                scope_share.append(len(backend.last.scope) / max(len(settled), 1))
                contradictions += backend.contradictions
                out_of_scope += getattr(backend, "out_of_scope", 0)
                # WHERE A `wrong` COMES FROM. Under oracle evidence the truth cannot leave a
                # sound candidate set, so a confident misattribution can only mean a message
                # refuted the TRUE attribution -- which is the local-disturbance assumption
                # of rule 1 failing, counted here rather than inferred. Reporting `wrong`
                # without it invites reading engine error as attribution error.
                violations += backend.assumption_violations
                msg_single += getattr(backend, "messages_single", 0)
                msg_cross += getattr(backend, "messages_cross", 0)
                vio_single += getattr(backend, "violations_single", 0)
                vio_cross += getattr(backend, "violations_cross", 0)
                if label == "component":
                    comps.append(backend.n_components)
                    largest.append(backend.largest_component)
                else:
                    comps.append(1 if settled else 0)
                    largest.append(len(backend.last.scope))
                cands.append(float(backend.n_candidates))
        seconds = (time.time() - start) / max(episodes, 1)
        rows.append({"cell": name, "backend": label, "k": k, "sigma": sigma,
                     "n_agents": n_agents, "budget": budget, "episodes": episodes,
                     "seconds_per_episode": seconds, "contradictions": contradictions,
                     "out_of_scope": out_of_scope,
                     "assumption_violations": violations,
                     "messages_single": msg_single, "messages_cross": msg_cross,
                     "violations_single": vio_single, "violations_cross": vio_cross,
                     "max_component_candidates": max_candidates,
                     "local_disturbance": local_disturbance,
                     "scope_share": float(np.mean(scope_share)),
                     "settled_pairs": float(np.mean(pairs)),
                     "components": float(np.mean(comps)),
                     "largest_component": float(np.mean(largest)),
                     "candidates": float(np.mean(cands)), **tally})
    return rows


def _rate(bad: int, total: int) -> str:
    """`bad/total (pct)` -- the raw counts stay visible because a rate over five messages is
    not a rate, and a reader must be able to see that for themselves."""
    if not total:
        return "-"
    return f"{bad}/{total} {100.0 * bad / total:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="6:0.5:3:24,8:0.5:3:32,12:0.5:4:60,20:0.5:4:80,"
                                       "30:0.5:4:100",
                    help="k:sigma:n_agents:budget, comma separated")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--n_obs", type=int, default=50)
    ap.add_argument("--n_int", type=int, default=10)
    # SEPARATE CAPS ON PURPOSE. The two are not the same quantity: the enumerated-ownership
    # backend caps the TOTAL settled pairs it will attribute, and 5 is its measured floor for
    # soundness (at 4 it settled two attributions WRONG). The component backend caps pairs
    # PER COMPONENT, which on a sparse window binds on nothing. Forcing one number on both
    # would either cripple the component backend or run the other for hours.
    ap.add_argument("--cap", type=int, default=8,
                    help="max_component_pairs, the component backend's per-component cap")
    ap.add_argument("--ref_cap", type=int, default=5,
                    help="max_attribution_pairs, the enumerated-ownership global cap")
    # THE SENSITIVITY ARM. Rule 1 is a modelling assumption, not a theorem; this switches it
    # off and keeps only atomicity, which is sound unconditionally. The gap between the two
    # runs IS what the assumption buys, measured rather than argued.
    ap.add_argument("--no_local_disturbance", dest="local_disturbance",
                    action="store_false", default=True)
    # Raising this lets a dense component be enumerated rather than truncated, buying SCOPE
    # back at the cost of enumeration time. At 3 partners the ladder is 7^p, so 50k admits a
    # 5-pair component and 400k admits 6.
    ap.add_argument("--max_component_candidates", type=int, default=50_000)
    ap.add_argument("--backends", default="factored,component")
    ap.add_argument("--out", default="results/attr_scale.json")
    args = ap.parse_args()

    which = [b.strip() for b in args.backends.split(",") if b.strip()]
    for label in which:
        if label not in BACKENDS:
            raise SystemExit(f"unknown backend {label!r}; choose from {sorted(BACKENDS)}")

    rows: List[Dict] = []
    header = (f"{'cell':>16s} {'backend':>10s} {'right':>6s} {'wrong':>6s} {'unsure':>7s} "
              f"{'total':>6s} {'scope':>6s} {'pairs':>6s} {'comp':>5s} {'max':>4s} "
              f"{'cands':>10s} {'viol':>5s} {'1comp':>12s} {'xcomp':>12s} {'s/ep':>7s}")
    print(header)
    print("-" * len(header))
    for spec in args.cells.split(","):
        k, sigma, n_agents, budget = spec.split(":")
        name = f"k{int(k)}s{int(float(sigma) * 100):02d}n{int(n_agents):02d}"
        for row in run_cell(name, int(k), float(sigma), int(n_agents), int(budget),
                            args.episodes, args.n_obs, args.n_int, which, args.cap,
                            args.ref_cap, args.max_component_candidates,
                            args.local_disturbance):
            rows.append(row)
            print(f"{row['cell']:>16s} {row['backend']:>10s} {row['right']:6d} "
                  f"{row['wrong']:6d} {row['unsure']:7d} {row['total']:6d} "
                  f"{row['scope_share']:6.2f} {row['settled_pairs']:6.1f} "
                  f"{row['components']:5.1f} {row['largest_component']:4.1f} "
                  f"{row['candidates']:10.2e} {row['assumption_violations']:5d} "
                  f"{_rate(row['violations_single'], row['messages_single']):>12s} "
                  f"{_rate(row['violations_cross'], row['messages_cross']):>12s} "
                  f"{row['seconds_per_episode']:7.2f}",
                  flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
