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
            for i, parents in enumerate(self.scorer.parent_sets[node]):
                if len(node_rows) <= len(parents) + 2:
                    # A regime carrying no usable evidence contributes 0.0, the neutral
                    # element -- NOT a penalty. Matches the enumerated path exactly.
                    table[node, i] = 0.0
                else:
                    table[node, i] = self.score.local_score(node, parents, node_rows)
        return table

    # -- modular rules ------------------------------------------------------------------

    def log_weights(self, samples: np.ndarray, known_intervened: np.ndarray,
                    clean: np.ndarray, rule: str) -> np.ndarray:
        """`[k, n_parent_sets]` weights for a modular rule, ready for the DP."""
        if rule not in MODULAR_RULES:
            raise ValueError(f"{rule!r} is not modular; use `joint_conf_marginals`")
        clean = np.asarray(clean, dtype=bool)
        all_rows = np.ones(len(samples), dtype=bool)
        has_clean = bool(clean.any())

        # Same fallback as the enumerated path: with no clean rows there is no split to
        # exploit and all three rules reduce to scoring everything once.
        if not has_clean or rule == POOLED:
            return self.local_table(samples, known_intervened, all_rows)
        if rule == SUBSET:
            return self.local_table(samples, known_intervened, clean)
        return (self.local_table(samples, known_intervened, clean)
                + self.local_table(samples, known_intervened, ~clean))

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

    def _assignment_weights(self, clean_table: np.ndarray, dirty_table: np.ndarray,
                            assignment: Sequence[Optional[Tuple[int, int]]]
                            ) -> np.ndarray:
        """Weights for one fixed confounding assignment.

        For each declared confounding edge `u -> v`: v's parent set MUST contain u (that
        edge is part of the hypothesis), and the CLEAN regime must not be credited for it,
        so the clean term is read at `pa \\ {u}` while the dirty term is read at `pa`.
        """
        required: List[int] = [0] * self.k          # bitmask of forced parents per node
        for edge in assignment:
            if edge is None:
                continue
            u, v = edge
            required[v] |= 1 << u

        out = np.full((self.k, self.scorer.n_parent_sets), NEG_INF)
        for node in range(self.k):
            need = required[node]
            for i, parents in enumerate(self.scorer.parent_sets[node]):
                mask = 0
                for p in parents:
                    mask |= 1 << p
                if mask & need != need:
                    continue                        # hypothesis requires an absent edge
                stripped = tuple(p for p in parents if not (need >> p) & 1)
                j = self.scorer.lookup[node][stripped]
                out[node, i] = clean_table[node, j] + dirty_table[node, i]
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
        clean_table = self.local_table(samples, known_intervened, clean)
        dirty_table = self.local_table(samples, known_intervened, ~clean)

        log_zs: List[float] = []
        marginals: List[np.ndarray] = []
        for assignment in self.assignments:
            log_w = self._assignment_weights(clean_table, dirty_table, assignment)
            if not np.isfinite(log_w).any():
                continue
            log_zs.append(float(self.dp.log_partition(log_w)))
            marginals.append(self.dp.edge_marginals_onepass(log_w))

        log_zs_arr = np.asarray(log_zs)
        # Per-row shift, never a global one: a single shift underflows the weaker
        # assignments to zero and deletes them silently. That exact bug cost a day.
        shift = log_zs_arr.max()
        weights = np.exp(log_zs_arr - shift)
        weights /= weights.sum()
        return np.tensordot(weights, np.asarray(marginals), axes=(0, 0))

    def joint_conf_dag_probability(self, samples: np.ndarray, known_intervened: np.ndarray,
                                   clean: np.ndarray, adjacency: np.ndarray) -> float:
        """P(this DAG | data) with confounding marginalised out.

        Not obtainable from `log_prob_dag`, which needs a single weight table: joint_conf is
        a MIXTURE over confounding assignments, so the DAG's mass is the Z-weighted average
        of its mass under each. Written out rather than approximated because the true DAG's
        mass is exactly what the identification threshold reads.
        """
        clean = np.asarray(clean, dtype=bool)
        clean_table = self.local_table(samples, known_intervened, clean)
        dirty_table = self.local_table(samples, known_intervened, ~clean)
        adjacency = np.asarray(adjacency) > 0.5

        log_zs: List[float] = []
        log_ps: List[float] = []
        for assignment in self.assignments:
            # THE CAUSAL ANSWER IS H MINUS THE CONFOUNDING EDGES, NOT H.
            #
            # In this formulation a hypothesis is (DAG H, ordered set P declared
            # confounding), where P's edges must be PRESENT in H and are stripped again
            # for the clean regime. So H is the augmented structure and the causal claim
            # is `H \ P`. Asking for P(H == truth) therefore asks the wrong question: a
            # confounded pair is not a real edge, so the true DAG contains none of P's
            # edges and picks up mass ONLY under the empty assignment -- exactly the
            # hypothesis that refuses to model the confounding.
            #
            # Measured consequence before this fix: on confounded episodes the affected
            # agent's true mass was EXACTLY 0.000 at every budget, which read as a failure
            # of coordination (GATE 3) when it was a failure of bookkeeping.
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
            log_w = self._assignment_weights(clean_table, dirty_table, assignment)
            if not np.isfinite(log_w).any():
                continue
            log_zs.append(float(self.dp.log_partition(log_w)))
            # UNNORMALISED weight of the candidate, not its per-assignment probability.
            #
            # Combining per-assignment probabilities and then reweighting by Z means
            # dividing by each Z and multiplying it straight back, which throws away
            # precision and -- worse -- trusts every individual Z. The signed sink
            # recurrence is least reliable exactly on the heavily masked tables an
            # assignment produces, and a single underestimated Z makes log_prob_dag come
            # out POSITIVE. Measured before this fix: "probabilities" of 1e131.
            #
            # Ratio of sums, computed once in log space, needs only the TOTAL of the Zs.
            log_ps.append(NEG_INF if cyclic
                          else self._log_dag_weight(log_w, candidate))

        if not log_zs:
            return 0.0
        numerator = _log_sum_exp(np.asarray(log_ps))
        denominator = _log_sum_exp(np.asarray(log_zs))
        if not np.isfinite(denominator) or not np.isfinite(numerator):
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
