"""Two principled replacements for the under-mixed structure-MCMC sampler.

The shipped `SamplingOracle` uses structure MCMC with a stopgap burn-in. Acceptance is 5.8%
regardless of settings, because single-edge moves are the wrong proposal for a posterior
whose effective support is ~172 graphs. Two real fixes, both implemented here.

--------------------------------------------------------------------------------------
1. EXACT SAMPLING BY LAYER DECOMPOSITION  (`LayeredExactSampler`)
--------------------------------------------------------------------------------------
Talvitie, Vuoksenmaa & Koivisto, "Exact Sampling of Directed Acyclic Graphs from Modular
Distributions", UAI 2019. Their bounds are O~(3^n) preprocessing and O~(2^n) per sample,
and the precondition is that the distribution be MODULAR -- P(G) proportional to a product
of per-node factors depending only on the node and its parents. That is exactly the form
our BGe + modular prior already has, which is why the subset DP works at all.

The recurrence used here decomposes a DAG by its SOURCE LAYERS, which is the construction
that avoids signed terms:

    L1 = nodes with no parents
    Li = nodes whose parents all lie in L1..L(i-1), with at least one in L(i-1)

Every DAG has exactly one such decomposition, so the layers partition the DAG space with no
double counting and no inclusion-exclusion. Writing alpha_v(U) for the sum of weights of v's
parent sets contained in U -- precisely `log_zeta` of the weight table -- the weight of
adding layer M after placed set U whose last layer was L is

    prod over v in M of [ alpha_v(U) - alpha_v(U \\ L) ]

The subtraction enforces "at least one parent in the previous layer". It is a difference of
two sums over nested sets, so it is NON-NEGATIVE by construction: alpha is monotone in U.
That is the whole point. The Robinson sink recurrence the DP uses for the partition function
is an alternating inclusion-exclusion whose terms can be negative, and negative terms cannot
be sampled from; this decomposition trades a slightly larger state space for terms that are
all valid probabilities.

Sampling is then a forward walk: from (U, L) draw the next layer M with probability
proportional to its term times the value of the remaining subproblem.

--------------------------------------------------------------------------------------
2. PARTITION MCMC  (`PartitionMCMC`)
--------------------------------------------------------------------------------------
Kuipers & Moffa, "Partition MCMC for Inference on Acyclic Digraphs", JASA 2017
(arXiv:1504.05006). The chain moves in the space of ordered partitions -- the same layer
objects as above -- with split and join moves, and a compatible DAG is drawn given the
partition. It converges better than structure MCMC and, unlike order MCMC, introduces NO
bias: order MCMC weights each DAG by its number of compatible topological orders, partition
MCMC does not.

Use it when n is past the exact sampler's reach. It is the fallback, not the default.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sa.dp import log_zeta

NEG_INF = -np.inf


def _log_sum_exp(values: np.ndarray) -> float:
    if len(values) == 0:
        return NEG_INF
    top = values.max()
    if not np.isfinite(top):
        return NEG_INF
    return float(top + np.log(np.exp(values - top).sum()))


def _log_diff_exp(big: np.ndarray, small: np.ndarray) -> np.ndarray:
    """log(e^big - e^small), elementwise, for big >= small.

    Clipped at zero rather than allowed to go negative: `big` and `small` are alpha over
    nested sets so the true difference cannot be negative, and anything below is floating
    point noise. Letting it through would produce a NaN log and look like an algorithm bug.
    """
    delta = np.where(big > small, big + np.log1p(-np.exp(np.minimum(small - big, -1e-12))),
                     NEG_INF)
    return np.where(np.isfinite(delta), delta, NEG_INF)


class LayeredExactSampler:
    """Exact DAG sampling from a modular distribution, via the source-layer recurrence."""

    def __init__(self, dp, log_w: np.ndarray):
        self.dp = dp
        self.d = dp.d
        self.full = (1 << self.d) - 1
        # alpha[v, U] = log sum over parent sets P subset U of w_v(P).
        self._masked_cache = dp._log_weights_masked(log_w)
        self.alpha = log_zeta(self._masked_cache, self.d)
        self._memo: Dict[Tuple[int, int], float] = {}
        self._node_cache: Dict[int, List[Tuple[int, float]]] = {}

    # -- recurrence ---------------------------------------------------------------------

    def _layer_weight(self, layer: int, placed: int, previous: int) -> float:
        """log weight of adding `layer` after `placed`, whose last layer was `previous`."""
        without = placed & ~previous
        nodes = [v for v in range(self.d) if (layer >> v) & 1]
        if not nodes:
            return NEG_INF
        big = self.alpha[nodes, placed]
        small = self.alpha[nodes, without] if previous else np.full(len(nodes), NEG_INF)
        terms = _log_diff_exp(big, small) if previous else big
        if not np.isfinite(terms).all():
            return NEG_INF
        return float(terms.sum())

    def _remaining(self, placed: int, previous: int) -> float:
        """log total weight of every way to finish, given `placed` and its last layer."""
        if placed == self.full:
            return 0.0
        key = (placed, previous)
        if key in self._memo:
            return self._memo[key]
        free = self.full & ~placed
        total: List[float] = []
        layer = free
        # Iterate every non-empty subset of `free`, the standard subset-enumeration trick.
        while layer:
            weight = self._layer_weight(layer, placed, previous)
            if np.isfinite(weight):
                total.append(weight + self._remaining(placed | layer, layer))
            layer = (layer - 1) & free
        value = _log_sum_exp(np.asarray(total))
        self._memo[key] = value
        return value

    def log_partition(self) -> float:
        """Total log weight over all DAGs -- an independent check on the DP's own Z."""
        return self._remaining(0, 0)

    # -- sampling -----------------------------------------------------------------------

    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None
               ) -> np.ndarray:
        """`[n_samples, d, d]` DAGs drawn EXACTLY from the modular distribution.

        No burn-in, no thinning, no autocorrelation, no acceptance rate -- draws are
        independent by construction. That is the entire point of doing this.
        """
        rng = np.random.default_rng() if rng is None else rng
        out = np.zeros((n_samples, self.d, self.d), dtype=bool)
        for s in range(n_samples):
            out[s] = self._sample_one(rng)
        return out

    def _sample_one(self, rng: np.random.Generator) -> np.ndarray:
        adjacency = np.zeros((self.d, self.d), dtype=bool)
        placed, previous = 0, 0
        layers: List[int] = []
        while placed != self.full:
            free = self.full & ~placed
            candidates, weights = [], []
            layer = free
            while layer:
                weight = self._layer_weight(layer, placed, previous)
                if np.isfinite(weight):
                    candidates.append(layer)
                    weights.append(weight + self._remaining(placed | layer, layer))
                layer = (layer - 1) & free
            weights = np.asarray(weights)
            probabilities = np.exp(weights - weights.max())
            probabilities /= probabilities.sum()
            layer = candidates[int(rng.choice(len(candidates), p=probabilities))]
            layers.append(layer)
            previous, placed = layer, placed | layer

        # Given the layers, each node's parent set is drawn independently: parents must lie
        # in earlier layers, with at least one in the immediately preceding layer.
        placed, previous = 0, 0
        for layer in layers:
            for v in range(self.d):
                if not (layer >> v) & 1:
                    continue
                parents = self._sample_parents(v, placed, previous, rng)
                for p in range(self.d):
                    if (parents >> p) & 1:
                        adjacency[p, v] = True
            previous, placed = layer, placed | layer
        return adjacency

    def _sample_parents(self, node: int, placed: int, previous: int,
                        rng: np.random.Generator) -> int:
        """Draw one parent set for `node` from those allowed by its layer position."""
        raw = self._node_weights(node)
        allowed = []
        values = []
        for mask, value in raw:
            if mask & ~placed:
                continue                                    # a parent not yet placed
            if previous and not (mask & previous):
                continue                                    # needs one in the last layer
            if not previous and mask:
                continue                                    # first layer has no parents
            allowed.append(mask)
            values.append(value)
        values = np.asarray(values)
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        return int(allowed[int(rng.choice(len(allowed), p=probabilities))])

    def _node_weights(self, node: int) -> List[Tuple[int, float]]:
        if node not in self._node_cache:
            masked = self._masked_cache
            self._node_cache[node] = [
                (mask, float(masked[node, mask]))
                for mask in range(1 << self.d)
                if np.isfinite(masked[node, mask])]
        return self._node_cache[node]


class PartitionMCMC:
    """Kuipers & Moffa partition MCMC -- the fallback past the exact sampler's reach.

    State is an ordered partition (the same source layers). Moves are split, join and swap;
    a compatible DAG is drawn per retained sample using the exact conditional above, so the
    only approximation is the chain's mixing, never the DAG draw.
    """

    def __init__(self, dp, log_w: np.ndarray, seed: int = 0):
        self.core = LayeredExactSampler(dp, log_w)
        self.dp = dp
        self.d = dp.d
        self.rng = np.random.default_rng(seed)
        self.accepted = 0
        self.proposed = 0

    def _score(self, layers: Sequence[int]) -> float:
        placed, previous, total = 0, 0, 0.0
        for layer in layers:
            weight = self.core._layer_weight(layer, placed, previous)
            if not np.isfinite(weight):
                return NEG_INF
            total += weight
            previous, placed = layer, placed | layer
        return total if placed == self.core.full else NEG_INF

    def _initial(self) -> List[int]:
        """Start from the finest partition -- one node per layer, in a random order. Always
        a valid DAG partition, so the chain never begins in an impossible state."""
        order = self.rng.permutation(self.d)
        return [1 << int(v) for v in order]

    def _propose(self, layers: List[int]) -> Tuple[List[int], float]:
        """Return a candidate partition and the log Hastings ratio log q(back)/q(forward).

        THE RATIO IS NOT OPTIONAL. Split and join are not symmetric -- a join has one way to
        merge a given adjacent pair, while the reverse split must pick one of 2^s - 2
        non-empty proper subsets -- so omitting it targets the wrong distribution entirely.
        Measured before this was added: max edge-marginal error 0.60 against exact, i.e. the
        chain was converging confidently to the wrong answer rather than mixing slowly.

        Proposal, chosen uniformly among three moves:
          split  pick a layer uniformly among m, then a uniform non-empty proper subset of
                 it. log ratio = +log(2^s - 2).
          join   pick one of the m-1 adjacent pairs uniformly. log ratio = -log(2^s - 2),
                 where s is the merged size.
          swap   exchange one node between adjacent layers. Self-inverse with equal
                 probability, so the ratio is 0.
        """
        move = ("split", "join", "swap")[int(self.rng.integers(3))]
        m = len(layers)
        layers = list(layers)

        if move == "split":
            i = int(self.rng.integers(m))
            nodes = [v for v in range(self.d) if (layers[i] >> v) & 1]
            size = len(nodes)
            if size < 2:
                return layers, NEG_INF
            n_subsets = (1 << size) - 2
            pick = 1 + int(self.rng.integers(n_subsets))   # skip empty and full
            first = sum(1 << nodes[b] for b in range(size) if (pick >> b) & 1)
            if first == 0 or first == layers[i]:
                return layers, NEG_INF
            layers[i:i + 1] = [first, layers[i] & ~first]
            return layers, float(np.log(n_subsets))

        if move == "join":
            if m < 2:
                return layers, NEG_INF
            i = int(self.rng.integers(m - 1))
            merged = layers[i] | layers[i + 1]
            size = bin(merged).count("1")
            layers[i:i + 2] = [merged]
            return layers, -float(np.log((1 << size) - 2))

        if m < 2:
            return layers, NEG_INF
        i = int(self.rng.integers(m - 1))
        a = [v for v in range(self.d) if (layers[i] >> v) & 1]
        b = [v for v in range(self.d) if (layers[i + 1] >> v) & 1]
        u, w = int(self.rng.choice(a)), int(self.rng.choice(b))
        layers[i] = (layers[i] & ~(1 << u)) | (1 << w)
        layers[i + 1] = (layers[i + 1] & ~(1 << w)) | (1 << u)
        return layers, 0.0

    def sample(self, n_samples: int, burn_in: int = 2000, thin: int = 5
               ) -> Tuple[np.ndarray, float]:
        layers = self._initial()
        score = self._score(layers)
        draws: List[np.ndarray] = []
        total = burn_in + n_samples * thin
        for step in range(total):
            candidate, log_ratio = self._propose(layers)
            candidate_score = self._score(candidate)
            self.proposed += 1
            accept = candidate_score - score + log_ratio
            if (candidate_score > NEG_INF and np.isfinite(log_ratio)
                    and (accept >= 0 or np.log(self.rng.random()) < accept)):
                layers, score = candidate, candidate_score
                self.accepted += 1
            if step >= burn_in and (step - burn_in) % thin == 0:
                draws.append(self._dag_given_layers(layers))
        return np.asarray(draws), self.accepted / max(self.proposed, 1)

    def _dag_given_layers(self, layers: Sequence[int]) -> np.ndarray:
        adjacency = np.zeros((self.d, self.d), dtype=bool)
        placed, previous = 0, 0
        for layer in layers:
            for v in range(self.d):
                if not (layer >> v) & 1:
                    continue
                parents = self.core._sample_parents(v, placed, previous, self.rng)
                for p in range(self.d):
                    if (parents >> p) & 1:
                        adjacency[p, v] = True
            previous, placed = layer, placed | layer
        return adjacency
