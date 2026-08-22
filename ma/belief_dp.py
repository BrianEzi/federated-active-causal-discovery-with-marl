"""PHASE 1 -- an agent's window belief by subset DP instead of enumeration.

The enumerated path (`ma/env.py:AgentView.posterior`) holds all 543 DAGs of a 4-node window
explicitly. That dies at k=6 and is the reason the two-agent case could not scale. This
module computes the same quantities through `sa/dp.py`'s Robinson sink recurrence, which is
O(k 2^k) and carries to k ~ 15-20 without approximation.

WHAT TRANSFERS CLEANLY, AND WHAT DOES NOT
=========================================

The subset DP requires the score to be MODULAR: the log score of a DAG must be a sum over
nodes of a term depending only on (node, its parent set). Three of the four rules are:

  POOLED   local(node, pa) scored on all rows                       -> modular
  SUBSET   local(node, pa) scored on clean rows only                -> modular
  JOINT    local_clean(node, pa) + local_dirty(node, pa)            -> modular, just a sum
                                                                        of two tables

JOINT_CONF is NOT, and the reason is worth stating precisely because it exposes a wart in
the existing implementation rather than a limitation of the DP.

`RegimeScorer._dirty_parents` adds a confounding edge for each marked shared pair and
orients it "along the DAG's topological order". But a DAG has many topological orders, and
`_topological_order` picks one by an arbitrary tie-break (lowest available index). For two
shared nodes that are INCOMPARABLE in the DAG -- neither an ancestor of the other -- the
orientation of their confounding edge is therefore decided by node numbering, not by
anything structural. Those two orientations give different BGe scores, so the hypothesis
being scored depends on an implementation detail.

It is also exactly what breaks modularity: node v's dirty parent set depends on the DAG's
global order, not on v's own parents, so no per-(node, parent-set) table can express it.

THE REPLACEMENT, AND WHY IT IS BETTER RATHER THAN MERELY DIFFERENT
==================================================================

Make the orientation part of the hypothesis and marginalise over it.

A hypothesis is (DAG H over the window, set P of ordered pairs declared confounding), with
P's edges required to be present in H. The clean regime scores H minus P's edges; the dirty
regime scores H entire. Then:

  local(node, pa) = clean(node, pa \\ P_into_node) + dirty(node, pa)   [pa must contain
                                                                        P_into_node]

which is modular for FIXED P, so the whole thing is `3^(pairs)` DP passes -- 27 at |X|=3,
one per assignment of each shared pair to {absent, u->v, v->u}.

Three properties this gains:

  1. Acyclicity is free. H is produced by the DP, which only ever emits DAGs, so the
     augmented structure cannot be cyclic. The old formulation had to argue this.
  2. No arbitrary tie-break. Both orientations of an incomparable pair are scored and
     marginalised, which is what "we do not know which way it goes" should mean.
  3. It still respects confinement -- P ranges only over shared pairs, per the proved
     result in `ma/projection.py`.

The cost is `3^(pairs)` rather than `2^(pairs)`: 27 against 8 at |X|=3, 729 against 64 at
|X|=4. Still exponential in the SHARED SET only, not in the window.

**This means the Phase 1 equivalence gate splits.** POOLED, SUBSET and JOINT are held to
1e-10 against the frozen enumerated fixture. JOINT_CONF cannot be, because it is
deliberately a different hypothesis space; it is checked for internal consistency instead,
and compared to the old rule by measurement rather than by identity.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sa.dp import DPPosterior
from sa.score import BGeScore

POOLED = "pooled"
SUBSET = "subset"
JOINT = "joint"
JOINT_CONF = "joint_conf"
MODULAR_RULES = (POOLED, SUBSET, JOINT)
RULES = (POOLED, SUBSET, JOINT, JOINT_CONF)

NEG_INF = -np.inf

# An assignment whose partition function is this far below the largest cannot change the
# mixture at float64 resolution, so its marginals are never computed.
#
# MEASURED: at |X| = 3 this prunes NOTHING -- all 25 assignments carry more than 1e-14 of
# the mass (top weights 1.0, 0.500, 0.500, 0.025, 0.018, 0.012...). The guard is kept
# because it is nearly free and because at |X| = 4 there are 729 assignments where most
# should fall away, but that is an expectation and has NOT been measured. Do not cite this
# as a speedup at the current topology; it is not one.
NEGLIGIBLE_WEIGHT = 1e-14


def _log_sum_exp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return NEG_INF
    top = finite.max()
    return float(top + np.log(np.exp(finite - top).sum()))


class WindowBeliefDP:
    """Exact belief over one agent's window, without enumerating DAGs.

    `shared_positions` are indices INTO THE WINDOW (not global node ids) of the shared
    variables -- the only places a confounding edge may appear.
    """

    def __init__(self, k: int, shared_positions: Sequence[int]):
        self.k = k
        self.shared_positions = list(shared_positions)
        self.pairs: List[Tuple[int, int]] = list(combinations(self.shared_positions, 2))
        self.score = BGeScore(k)
        # Uniform over DAGs, matching `AgentView.log_prior = zeros`. Passing
        # "erdos_renyi" with p=0.5 would ALSO be uniform over edges but not over DAGs,
        # and the two differ once graphs of different density are compared.
        self.dp = DPPosterior.for_prior(k, self.score, kind="uniform")
        self.scorer = self.dp.scorer

        # Ordered confounding assignments: each pair is absent, u->v, or v->u.
        # Each shared pair (u, v) independently takes one of three states: no confounding,
        # u -> v, or v -> u. 3^pairs assignments -- 27 at |X| = 3.
        per_pair = [(None, (u, v), (v, u)) for u, v in self.pairs]
        candidates = ([tuple(combo) for combo in product(*per_pair)]
                      if per_pair else [()])
        # An assignment whose forced edges already contain a cycle admits NO acyclic
        # completion, so its hypothesis class is empty. Left in, the DP's alternating
        # inclusion-exclusion cancels to an exactly zero partition function and raises.
        # At |X| = 3 the three shared pairs form a triangle, and exactly 2 of the 27
        # assignments are the two cyclic orientations of it -- so 25 survive.
        self.assignments: List[Tuple[Optional[Tuple[int, int]], ...]] = [
            a for a in candidates if not self._forces_a_cycle(a)]
        self._table_key: Optional[Tuple[int, int]] = None
        self._table_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # Per-assignment (log_w, log_z), keyed like the tables. Three callers need these
        # for the same data in the same step -- marginals, single-DAG mass, and set mass --
        # and log_partition was being recomputed 25 times per caller.
        self._assign_key: Optional[Tuple[int, int]] = None
        self._assign_cache: Optional[List[Tuple[np.ndarray, float]]] = None

    def _forces_a_cycle(self, assignment) -> bool:
        """Kahn's algorithm over just the forced confounding edges."""
        edges = [e for e in assignment if e is not None]
        if not edges:
            return False
        nodes = {n for e in edges for n in e}
        indegree = {n: 0 for n in nodes}
        succ: Dict[int, List[int]] = {n: [] for n in nodes}
        for u, v in edges:
            succ[u].append(v)
            indegree[v] += 1
        queue = [n for n in nodes if indegree[n] == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for m in succ[n]:
                indegree[m] -= 1
                if indegree[m] == 0:
                    queue.append(m)
        return seen != len(nodes)

    # -- local score tables -------------------------------------------------------------

    # -- local score tables -------------------------------------------------------------

    def regime_tables(self, samples: np.ndarray, known_intervened: np.ndarray,
                      clean: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """List of `(clean_fraction, table)` across all distinct regimes in `clean`, memoised."""
        clean = np.asarray(clean, dtype=float)
        unique_f, counts = np.unique(clean, return_counts=True)
        key = (samples.shape[0], tuple(float(f) for f in unique_f), tuple(int(c) for c in counts))
        if self._table_key == key and self._table_cache is not None:
            return self._table_cache

        tables = []
        for f in unique_f:
            f_float = float(f)
            rows = (clean == f)
            table_f = self.local_table(samples, known_intervened, rows)
            tables.append((f_float, table_f))

        self._table_key, self._table_cache = key, tables
        return tables

    def tables(self, samples: np.ndarray, known_intervened: np.ndarray,
               clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """`(clean_table, dirty_table)`, memoised for binary 2-regime data."""
        clean = np.asarray(clean, dtype=float)
        clean_rows = clean > 0.0
        return (self.local_table(samples, known_intervened, clean_rows),
                self.local_table(samples, known_intervened, ~clean_rows))

    def local_table(self, samples: np.ndarray, known_intervened: np.ndarray,
                    rows: np.ndarray) -> np.ndarray:
        """`[k, n_parent_sets]` local scores over one regime's rows.

        Deliberately mirrors `RegimeScorer.log_posterior.local` line for line, including
        the small-sample guard, because Phase 1's gate is exact agreement with it. The
        batched `LocalScorer.table` is the intended fast replacement, but it must not be
        swapped in until the equivalence test is green -- it does not implement the guard.
        """
        table = np.zeros((self.k, self.scorer.n_parent_sets))
        sub = samples[rows]
        known_sub = known_intervened[rows]
        for node in range(self.k):
            # Cooper & Yoo: a row where this node was hard-intervened contributes no
            # likelihood term for the node itself.
            keep = known_sub[:, node] < 0.5
            node_rows = sub[keep]
            # ONE O(n k^2) pass per node, reused across all 2^(k-1) of its parent sets.
            # `local_score` recomputes the sufficient statistics on every call, so the
            # previous version made 16 full passes over the data per node -- 64 per table
            # at k=4 -- where 4 suffice. The statistics depend only on the node's row
            # subset, never on which parents are being scored.
            stats = self.score.sufficient_stats(node_rows) if len(node_rows) else None
            for i, parents in enumerate(self.scorer.parent_sets[node]):
                if len(node_rows) <= len(parents) + 2 or stats is None:
                    # A regime carrying no usable evidence contributes 0.0, the neutral
                    # element -- NOT a penalty. Matches the enumerated path exactly.
                    table[node, i] = 0.0
                else:
                    table[node, i] = self.score.local_score_from_stats(node, parents, stats)
        return table

    def assignment_weights_and_z(self, samples: np.ndarray, known_intervened: np.ndarray,
                                 clean: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """`(log_w, log_z)` per surviving assignment, computed once per belief update."""
        clean = np.asarray(clean, dtype=float)
        unique_f, counts = np.unique(clean, return_counts=True)
        key = (samples.shape[0], tuple(float(f) for f in unique_f), tuple(int(c) for c in counts))
        if self._assign_key == key and self._assign_cache is not None:
            return self._assign_cache
        reg_tables = self.regime_tables(samples, known_intervened, clean)
        out: List[Tuple[np.ndarray, float]] = []
        for assignment in self.assignments:
            log_w = self._assignment_weights(reg_tables, assignment)
            if not np.isfinite(log_w).any():
                out.append((log_w, NEG_INF))
                continue
            out.append((log_w, float(self.dp.log_partition(log_w))))
        self._assign_key, self._assign_cache = key, out
        return out

    # -- modular rules ------------------------------------------------------------------

    def log_weights(self, samples: np.ndarray, known_intervened: np.ndarray,
                    clean: np.ndarray, rule: str) -> np.ndarray:
        """`[k, n_parent_sets]` weights for a modular rule, ready for the DP."""
        if rule not in MODULAR_RULES:
            raise ValueError(f"{rule!r} is not modular; use `joint_conf_marginals`")
        clean = np.asarray(clean, dtype=float)
        all_rows = np.ones(len(samples), dtype=bool)
        clean_rows = clean > 0.0
        has_clean = bool(clean_rows.any())

        # Same fallback as the enumerated path: with no clean rows there is no split to
        # exploit and all three rules reduce to scoring everything once.
        if not has_clean or rule == POOLED:
            return self.local_table(samples, known_intervened, all_rows)
        if rule == SUBSET:
            return self.local_table(samples, known_intervened, clean_rows)
        return (self.local_table(samples, known_intervened, clean_rows)
                + self.local_table(samples, known_intervened, ~clean_rows))

    def edge_marginals(self, samples: np.ndarray, known_intervened: np.ndarray,
                       clean: np.ndarray, rule: str) -> np.ndarray:
        """`[k, k]` posterior edge probabilities -- what the policy actually observes."""
        if rule == JOINT_CONF:
            return self.joint_conf_marginals(samples, known_intervened, clean)
        log_w = self.log_weights(samples, known_intervened, clean, rule)
        return self.dp.edge_marginals_onepass(log_w)

    def log_prob_dag(self, samples: np.ndarray, known_intervened: np.ndarray,
                     clean: np.ndarray, rule: str, adjacency: np.ndarray) -> float:
        log_w = self.log_weights(samples, known_intervened, clean, rule)
        # Third argument is log_z, NOT k. Passing k here silently subtracts a
        # constant instead of normalising, which looks like a plausible probability.
        return float(self.dp.log_prob_dag(log_w, np.asarray(adjacency)))

    # -- joint_conf, reformulated -------------------------------------------------------

    def _assignment_weights(self, regime_tables: List[Tuple[float, np.ndarray]],
                            assignment: Sequence[Optional[Tuple[int, int]]]
                            ) -> np.ndarray:
        """Weights for one fixed confounding assignment across all data regimes.

        For each declared confounding edge `u -> v`: v's parent set MUST contain u (that
        edge is part of the hypothesis).
        - In fully clean regime (f=1.0, q=0.0): clean term is read at `pa \\ {u}` (stripped).
        - In fully dirty regime (f=0.0, q=1.0): dirty term is read at `pa` (full).
        - In partially clean regime (0 < f < 1.0): mixture over active confounding states.
        """
        required: List[int] = [0] * self.k          # bitmask of forced parents per node
        for edge in assignment:
            if edge is None:
                continue
            u, v = edge
            required[v] |= 1 << u

        out = np.zeros((self.k, self.scorer.n_parent_sets))
        for node in range(self.k):
            need = required[node]
            if need == 0:
                for _, table in regime_tables:
                    out[node, :] += table[node, :]
            else:
                for i, parents in enumerate(self.scorer.parent_sets[node]):
                    mask = 0
                    for p in parents:
                        mask |= 1 << p
                    if mask & need != need:
                        out[node, i] = NEG_INF      # hypothesis requires an absent edge
                        continue
                    stripped = tuple(p for p in parents if not (need >> p) & 1)
                    j = self.scorer.lookup[node][stripped]
                    for f, table in regime_tables:
                        q = 1.0 - f
                        if f == 1.0:
                            out[node, i] += table[node, j]
                        elif f == 0.0:
                            out[node, i] += table[node, i]
                        else:
                            v_clean = table[node, j]
                            v_dirty = table[node, i]
                            out[node, i] += np.logaddexp(np.log(1.0 - q) + v_clean,
                                                         np.log(q) + v_dirty)
        return out

    def joint_conf_marginals(self, samples: np.ndarray, known_intervened: np.ndarray,
                             clean: np.ndarray) -> np.ndarray:
        """Edge marginals with confounding marginalised out over all assignments.

        Assignments are combined by their log partition functions, which is the correct
        weighting: Z(P) is the total unnormalised mass of every DAG compatible with that
        confounding pattern, so mixing the per-assignment marginals in proportion to Z(P)
        is exactly marginalising P out of the joint.
        """
        clean = np.asarray(clean, dtype=bool)
        clean_table, dirty_table = self.tables(samples, known_intervened, clean)

        # TWO PASSES, and the split is the optimisation. The mixture weight of an
        # assignment is fixed by its partition function alone, and log_partition is far
        # cheaper than edge_marginals_onepass. So compute every Z first, then compute
        # marginals ONLY for assignments that carry non-negligible weight. Profiling put
        # ~70% of an episode in edge_marginals_onepass at 25 calls per belief update;
        # in practice a handful of assignments hold essentially all the mass.
        #
        # The threshold is far below float64 resolution against a weight of 1, so a
        # dropped assignment cannot move the result -- this is exact to the precision the
        # arithmetic already has, not an approximation with a knob.
        prepared = [(z, w) for w, z in
                    self.assignment_weights_and_z(samples, known_intervened, clean)
                    if np.isfinite(z)]
        if not prepared:
            return np.zeros((self.k, self.k))

        all_z = np.asarray([z for z, _ in prepared])
        keep_from = all_z.max() + np.log(NEGLIGIBLE_WEIGHT)
        log_zs: List[float] = []
        marginals: List[np.ndarray] = []
        for log_z, log_w in prepared:
            if log_z < keep_from:
                continue
            log_zs.append(log_z)
            marginals.append(self.dp.edge_marginals_onepass(log_w))

        log_zs_arr = np.asarray(log_zs)
        # Per-row shift, never a global one: a single shift underflows the weaker
        # assignments to zero and deletes them silently. That exact bug cost a day.
        shift = log_zs_arr.max()
        weights = np.exp(log_zs_arr - shift)
        weights /= weights.sum()
        return np.tensordot(weights, np.asarray(marginals), axes=(0, 0))

    def joint_conf_dag_probability(self, samples: np.ndarray, known_intervened: np.ndarray,
                                   clean: np.ndarray, adjacency: np.ndarray,
                                   confounded_pairs: Sequence[Tuple[int, int]] = ()
                                   ) -> float:
        r"""P(the agent has the causal structure AND the confounding right | data).

        Two wrong versions were tried first, and the criterion sits between them.

        P(H == truth) was too HARSH. A hypothesis is (DAG H, confounding set P) with P's
        edges required present in H, so a confounded pair appears as an edge in H labelled
        as confounding. The true DAG contains no such edge, so it took mass only under the
        empty assignment -- the one hypothesis that refuses to model the confounding.
        Measured: EXACTLY 0.000 for the affected agent on every confounded episode at every
        budget, which read as a GATE 3 coordination failure.

        P(H \ P == truth), marginalising the confounding away, was too GENEROUS. With no
        clean rows the confounding label is UNFALSIFIABLE: any extra edge can be added and
        called confounding, paying only the BGe complexity penalty, and every such superset
        maps back to the same base graph. Measured: observational-only identification jumped
        to 0.2387 against a singleton-MEC target of 0.0402 -- a GATE 1 leak.

        The criterion here asks for both: the base structure AND the correct set of
        confounded pairs. That is also what [U14] actually requires -- an agent that cannot
        tell a confounded pair from a causal edge has not recovered the structure.
        Orientation of the modelling edge is not part of the claim, since both orientations
        express the same "u and v share a hidden cause".
        """
        clean = np.asarray(clean, dtype=float)
        adjacency = np.asarray(adjacency) > 0.5
        # The agent is credited only for assignments that name the TRUE confounded pairs.
        # Orientation is not part of the claim -- "u and v share a hidden cause" is one
        # statement, and the two orientations of the modelling edge express it equally --
        # so both are accepted for a truly confounded pair.
        truth_pairs = {frozenset(pair) for pair in confounded_pairs}

        log_zs: List[float] = []
        log_ps: List[float] = []
        prepared = self.assignment_weights_and_z(samples, known_intervened, clean)
        for assignment, (log_w, log_z) in zip(self.assignments, prepared):
            if not np.isfinite(log_z):
                continue
            log_zs.append(log_z)
            named = {frozenset(edge) for edge in assignment if edge is not None}
            if named != truth_pairs:
                continue                    # wrong confounding claim -> no credit
            candidate = adjacency.copy()
            cyclic = False
            for edge in assignment:
                if edge is None:
                    continue
                u, v = edge
                if candidate[v, u]:            # reversing an existing edge -> cycle
                    cyclic = True
                    break
                candidate[u, v] = True
            if not cyclic:
                log_ps.append(self._log_dag_weight(log_w, candidate))

        if not log_zs or not log_ps:
            return 0.0
        numerator = _log_sum_exp(np.asarray(log_ps))
        denominator = _log_sum_exp(np.asarray(log_zs))
        if not np.isfinite(denominator) or not np.isfinite(numerator):
            return 0.0
        return float(np.exp(min(numerator - denominator, 0.0)))

    def joint_conf_set_probability(self, samples: np.ndarray,
                                   known_intervened: np.ndarray, clean: np.ndarray,
                                   candidates: Sequence[np.ndarray],
                                   confounded_pairs: Sequence[Tuple[int, int]] = ()
                                   ) -> float:
        """P(the CAUSAL graph is one of `candidates`, and the confounding is right | data).

        The set-valued form of `joint_conf_dag_probability`, and it exists because the
        reported success criterion credits a SET -- every graph Markov equivalent to the
        truth that also matches it on the private-incident edges.

        THE BUG THIS REPLACES. `evaluate2` was computing credit mass from a posterior
        indexed by H, the AUGMENTED graph, and comparing H against the true CAUSAL graph.
        On a confounded episode the truth contains no confounding edge, so it matched only
        under the empty assignment -- the one hypothesis that refuses to model the
        confounding -- and every other assignment produced an H with an extra edge, a
        different skeleton, and therefore a different equivalence class. Measured result:
        reported success EXACTLY 0.000 on confounded episodes, against ~0.59 unconfounded.
        The metric could not score the case the whole design exists to study.

        Candidates are CAUSAL graphs. Each is augmented with the assignment's edges before
        scoring, which is the same correction already applied to the single-graph form.

        Cost is `len(candidates) x len(assignments)` cheap weight lookups plus one partition
        function per assignment. The candidate set ranges only over the SHARED subgraph --
        the private-incident edges are pinned by the criterion -- so it is exponential in
        |X| and not in the window size, the same axis the confounding enumeration already
        costs. Nothing here reintroduces window enumeration.
        """
        clean = np.asarray(clean, dtype=float)
        truth_pairs = {frozenset(pair) for pair in confounded_pairs}
        candidates = [np.asarray(c) > 0.5 for c in candidates]

        log_zs: List[float] = []
        log_ps: List[float] = []
        prepared = self.assignment_weights_and_z(samples, known_intervened, clean)
        for assignment, (log_w, log_z) in zip(self.assignments, prepared):
            if not np.isfinite(log_z):
                continue
            log_zs.append(log_z)
            named = {frozenset(edge) for edge in assignment if edge is not None}
            if named != truth_pairs:
                continue                    # wrong confounding claim -> no credit
            for base in candidates:
                augmented = base.copy()
                cyclic = False
                for edge in assignment:
                    if edge is None:
                        continue
                    u, v = edge
                    if augmented[v, u]:
                        cyclic = True
                        break
                    augmented[u, v] = True
                if not cyclic:
                    log_ps.append(self._log_dag_weight(log_w, augmented))

        if not log_zs or not log_ps:
            return 0.0
        numerator = _log_sum_exp(np.asarray(log_ps))
        denominator = _log_sum_exp(np.asarray(log_zs))
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return 0.0
        return float(np.exp(min(numerator - denominator, 0.0)))

    def _log_dag_weight(self, log_w: np.ndarray, adjacency: np.ndarray) -> float:
        """Unnormalised log weight of one DAG under a weight table."""
        total = 0.0
        for node in range(self.k):
            mask = int(np.dot(adjacency[:, node], 1 << np.arange(self.k)))
            index = self.dp._mask_to_index[node, mask]
            if index < 0:
                return NEG_INF
            value = log_w[node, index]
            if not np.isfinite(value):
                return NEG_INF
            total += float(value)
        return total

    # -- diagnostics --------------------------------------------------------------------

    @property
    def n_assignments(self) -> int:
        return len(self.assignments)
