"""Scoring an agent's window when its data spans two regimes.

Last night's failure: the belief rule "where clean rows exist, use only those" creates a
valley. Occasional clamping by the partner makes the agent discard thousands of good
observational rows for a few hundred clean ones, which is a loss on the ~85% of episodes
where the agent is not confounded. The learner sees a negative gradient at low clamp rates
and never crosses to the large payoff on the far side.

Four scoring rules, so the choice is made by measurement rather than argument:

  POOLED     One dataset, one score. Ignores the regime bit entirely. This is what the
             environment did before disclosure existed, and it cannot identify a confounded
             agent's graph at all -- no DAG fits a mixture of two regimes.

  SUBSET     Clean rows only when any exist. Correct inference, and what the environment
             does today. It is what creates the valley.

  JOINT      Same structure in both regimes, INDEPENDENT parameters: score each regime with
             its own BGe term and add the logs. Clean rows now ADD information instead of
             replacing it, so the valley should disappear. But the dirty regime still
             prefers a structure that mimics the confounding, so this may not fix the
             confounded case -- it fixes the gradient, not necessarily the target.

  JOINT_CONF The theoretically right one, and it is only tractable because of the
             confinement result: every bidirected edge has BOTH endpoints in the shared set
             (verified exhaustively, tests/test_projection.py). So a hypothesis is a DAG
             over the window PLUS a subset S of the shared PAIRS marked as confounded --
             at |X|=3 that is 8 subsets, and the space is 543 x 8 = 4344, still exact.

             A DAG-only model mimics latent confounding between two nodes with an edge
             between them. So S is applied to the DIRTY regime only: for each confounded
             pair, the dirty-regime parent sets gain that edge, oriented to agree with the
             DAG's own topological order so acyclicity is free. The clean regime is scored
             against the bare DAG, because the confounding is switched off there.

             Marginalising S out leaves a posterior over DAGs. The clean regime is what
             disambiguates "real edge" from "confounding artefact" -- exactly the job the
             partner's clamp is supposed to do.

None of these share data between agents. The regime bit is the only thing crossing the
boundary, and it names nothing.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

POOLED = "pooled"
SUBSET = "subset"
JOINT = "joint"
JOINT_CONF = "joint_conf"
RULES = (POOLED, SUBSET, JOINT, JOINT_CONF)


def _topological_order(adjacency: np.ndarray) -> List[int]:
    """Any topological order of a DAG. Used to orient an added confounding edge so that it
    can never create a cycle."""
    k = adjacency.shape[0]
    remaining = set(range(k))
    order = []
    while remaining:
        for node in sorted(remaining):
            if not any(adjacency[p, node] for p in remaining):
                order.append(node)
                remaining.discard(node)
                break
        else:                                   # pragma: no cover -- input is a DAG
            raise RuntimeError("cycle in a supposed DAG")
    return order


class RegimeScorer:
    """Scores one agent's hypothesis space under a chosen rule.

    Local scores are cached per (regime, node, parent-set), so the cost is the number of
    DISTINCT parent sets -- k * 2^(k-1), i.e. 32 at k=4 -- not the number of hypotheses.
    """

    def __init__(self, view, shared_positions: Sequence[int]):
        self.view = view
        self.k = view.k
        self.shared_positions = list(shared_positions)
        self.pairs: List[Tuple[int, int]] = list(combinations(self.shared_positions, 2))
        self.n_subsets = 1 << len(self.pairs)
        self.orders = [_topological_order(dag) for dag in view.dags]

    # -- parent sets ---------------------------------------------------------------

    def _dirty_parents(self, index: int, subset: int) -> List[Tuple[int, ...]]:
        """Parent sets for the dirty regime: the DAG's own, plus an edge for each pair
        marked confounded, oriented along the DAG's topological order."""
        parents = [set(p) for p in self.view.parents[index]]
        if subset:
            rank = {node: i for i, node in enumerate(self.orders[index])}
            for bit, (u, v) in enumerate(self.pairs):
                if (subset >> bit) & 1:
                    if rank[u] < rank[v]:
                        parents[v].add(u)
                    else:
                        parents[u].add(v)
        return [tuple(sorted(p)) for p in parents]

    # -- scoring -------------------------------------------------------------------

    def log_posterior(self, samples: np.ndarray, known_intervened: np.ndarray,
                      clean: np.ndarray, rule: str) -> np.ndarray:
        """Posterior over the agent's DAGs. `clean` is the per-row regime bit."""
        clean = np.asarray(clean, dtype=bool)
        score = self.view.score
        cache: Dict[Tuple[str, int, Tuple[int, ...]], float] = {}

        def local(tag: str, rows: np.ndarray, node: int, parents: Tuple[int, ...]) -> float:
            key = (tag, node, parents)
            if key not in cache:
                keep = known_intervened[rows][:, node] < 0.5
                subset_rows = samples[rows][keep]
                # Guard: a regime can be empty, or too small for the marginal likelihood
                # to be defined. Contributing 0.0 is the right neutral element -- it means
                # this regime carries no evidence, not that it carries bad evidence.
                if len(subset_rows) <= parents.__len__() + 2:
                    cache[key] = 0.0
                else:
                    cache[key] = score.local_score(node, parents, subset_rows)
            return cache[key]

        if rule not in RULES:
            raise ValueError(rule)

        all_rows = np.ones(len(samples), dtype=bool)
        dirty_rows = ~clean
        has_clean = bool(clean.any())

        # With no clean rows there is no regime split to exploit, so POOLED, SUBSET and
        # JOINT all reduce to scoring everything once.
        #
        # JOINT_CONF deliberately does NOT reduce. Without clean data an agent genuinely
        # cannot tell a real shared edge from a confounding artefact, and the whole point
        # of the rule is to represent that rather than assume it away. The consequence is
        # visible and intended: JOINT_CONF starts LOWER than the others at p(clamp)=0,
        # because it is spreading mass over hypotheses the others silently exclude. What
        # it buys is that clamping then resolves the ambiguity.
        if rule in (POOLED, SUBSET, JOINT) and not has_clean:
            groups = [("all", all_rows)]
        elif rule == POOLED:
            groups = [("all", all_rows)]
        elif rule == SUBSET:
            groups = [("clean", clean)]
        elif rule == JOINT:
            groups = [("clean", clean), ("dirty", dirty_rows)]
        else:
            groups = None                        # JOINT_CONF handled below

        n_dags = self.view.n_dags
        if groups is not None:
            log_post = np.zeros(n_dags)
            for i in range(n_dags):
                total = 0.0
                for tag, rows in groups:
                    for node in range(self.k):
                        total += local(tag, rows, node, self.view.parents[i][node])
                log_post[i] = total
            return _normalise(log_post)

        # JOINT_CONF: score every (DAG, confounded-subset) pair, then marginalise S out.
        table = np.full((n_dags, self.n_subsets), -np.inf)
        for i in range(n_dags):
            clean_term = sum(local("clean", clean, node, self.view.parents[i][node])
                             for node in range(self.k))
            for subset in range(self.n_subsets):
                dirty_parents = self._dirty_parents(i, subset)
                dirty_term = sum(local("dirty", dirty_rows, node, dirty_parents[node])
                                 for node in range(self.k))
                # Uniform prior over subsets. A sparsity prior favouring fewer confounded
                # pairs would be defensible and is NOT applied, because it would bias the
                # comparison towards finding no confounding, which is the thing being
                # measured.
                table[i, subset] = clean_term + dirty_term

        # Row-wise log-sum-exp. A single global shift underflows every entry of the
        # weaker rows to zero and then log(0) = -inf, which silently deletes hypotheses
        # rather than ranking them.
        shift = table.max(axis=1, keepdims=True)
        marginal = np.log(np.exp(table - shift).sum(axis=1)) + shift.ravel()
        return _normalise(marginal)


def _normalise(log_post: np.ndarray) -> np.ndarray:
    log_post = log_post - log_post.max()
    weights = np.exp(log_post)
    return weights / weights.sum()
