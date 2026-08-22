"""Graph priors: how the true DAG is drawn, and what the estimator assumes.

The first version sampled uniformly over all enumerated DAGs. That is unrealistic in a
specific direction -- there are far more dense DAGs than sparse ones, so uniform sampling
skews dense (measured mean edge count 3.71 of a possible 6 at d=4), while real causal
graphs are sparse. It is also off the research standard: the causal discovery literature
samples Erdos-Renyi or scale-free graphs, so uniform-over-DAGs makes results hard to
compare against published work.

Both priors here are defined as **weights over the fully enumerated space**, not as
samplers. That keeps everything else exact: the posterior uses the same weights as its
prior (so generator and inference agree), and GATE 1's target becomes the prior-weighted
singleton fraction -- still computed in closed form rather than estimated.

  `uniform`     -- every DAG equally likely.
  `erdos_renyi` -- each of the C(d,2) possible adjacencies present independently with
                   probability p, conditioned on acyclicity. Sparsity controlled by p.
  `scale_free`  -- weights DAGs by how concentrated their degree distribution is, favouring
                   hub structures in the spirit of preferential attachment.

**Uniform is not a separate family: uniform-over-DAGs IS Erdos-Renyi with p = 0.5.**
`P(G) ~ p^|E| (1-p)^(pairs-|E|)` reduces to the constant `0.5^pairs` at p=0.5, so every
DAG gets equal weight. Verified numerically at d=4, where p=0.5 reproduces the uniform
figures exactly (E[edges] 3.71, E[MEC] 5.46, singleton fraction 0.1087). So the original
setup was never off the research standard -- it was ER at an unrealistically high p.

**Sparsity is close to vacuous at small d, and this is worth knowing before reading too
much into any ER-vs-scale-free comparison.** The literature's ER-1 convention (expected
edges = d) gives p = 2/(d-1), which is *denser* than uniform for d <= 5: p=1.0 at d=3,
0.667 at d=4, 0.5 at d=5. There is simply no room to be sparse when C(d,2) is close to d.
Measured expected class sizes at expected-edges = d-1: 3.97 (d=3), 5.46 (d=4), 5.96 (d=5)
-- barely distinguishable from uniform. Sparsity becomes a real lever around d >= 8, where
C(d,2) >> d, and that is also where the exact posterior is unavailable and graphs must be
sampled rather than enumerated. The same applies to `scale_free`: five nodes cannot host a
hub. Both belong to the large-d, edge-marginal phase.

Honest limitation on `scale_free`: preferential attachment is a *generative process*, and
what is implemented here is a degree-concentration weighting over an enumerated space,
which is not the same object. At d <= 5 the distinction is largely academic -- you cannot
have a meaningful hub among 5 nodes -- so the ER/scale-free comparison only becomes
informative at larger d, where the exact posterior is unavailable anyway and graphs must
be sampled directly. Treat it as a placeholder that keeps the interface honest, not as a
faithful Barabasi-Albert prior.

Sparsity trades against difficulty, knowingly: sparse graphs sit in small Markov
equivalence classes (a 1-edge graph in a class of 2, the empty graph alone), and class
size is what drives how many interventions are needed. So a realistic prior makes the task
easier. The agreed response is to recover difficulty by raising `d` rather than by keeping
an unrealistic prior.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sa.graphs import GraphSpace


def uniform_prior(space: GraphSpace) -> np.ndarray:
    return np.full(space.n_dags, 1.0 / space.n_dags)


def erdos_renyi_prior(space: GraphSpace, p: float = 0.3) -> np.ndarray:
    """P(G) proportional to p^|E| (1-p)^(C(d,2) - |E|), restricted to DAGs.

    `p` is the probability that any given pair of nodes is adjacent. Expected edge count
    before conditioning on acyclicity is p * C(d,2), so to hold expected *degree* roughly
    constant as d grows, p should fall like 1/d -- see `expected_degree_p`.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    n_pairs = space.d * (space.d - 1) // 2
    edges = space.dags.reshape(space.n_dags, -1).sum(axis=1)
    log_w = edges * np.log(p) + (n_pairs - edges) * np.log1p(-p)
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def expected_degree_p(d: int, degree: float = 2.0) -> float:
    """The `p` giving an expected degree of `degree` on d nodes.

    Each node has d-1 possible neighbours, so expected degree is p*(d-1). Holding degree
    fixed while d grows is what "sparse" means at scale -- a fixed p makes graphs denser
    and denser as d grows, which is neither realistic nor tractable.

    **Only meaningful once d is reasonably large.** At d=3 a degree of 2 means p=1, the
    complete graph, because every node is adjacent to every other. See the module
    docstring on why sparsity is close to vacuous below about d=8.
    """
    return float(np.clip(degree / (d - 1), 1e-6, 1 - 1e-6))


def connectivity_prior_p(d: int) -> float:
    """`p = 2 ln(d) / d` -- the default edge probability, scaled with `d`.

    Two DIFFERENT thresholds get conflated here and the distinction decides the answer.
    The GIANT-COMPONENT (percolation) threshold is `1/d`; above it one component holds a
    constant fraction of the nodes. The FULL-CONNECTIVITY threshold is `ln(d)/d`; above it
    every node is in one component with no isolated stragglers.

    **We want the second**, because a disconnected graph splits the agents into independent
    subproblems: no path crosses the private/shared boundary, so there is no latent
    confounding and nothing to coordinate about. Those episodes cannot test what this
    project is building.

    The factor of two is empirical, not asymptotic. `ln(d)/d` is the threshold at which
    connectivity becomes *typical* as `d -> infinity`; at the finite sizes here it is far
    too weak, measured at 28-51% connected across d=5..30. Doubling it clears the constant.

    **This does NOT guarantee a single component**, and nothing short of rejection sampling
    would -- rejection would distort the prior away from the generator, which is the exact
    misspecification `ma/topology.py` is written to avoid. Measured over 400 draws per size
    (`scripts/sa_graph_density.py`, `results/graph_density.json`):

        d      5      8     10     15     20     30
        p   .644   .520   .461   .361   .300   .227
        connected  .92   .95   .97    .97    .97    .99
        degree    2.58  3.63  4.18   5.08   5.77   6.53

    So connectivity stays in the 92-99% band, and mean degree drifts up from 2.6 to 6.5 --
    inside the ER-2..ER-6 band the literature uses, at the dense end of it.

    **THOSE ARE UNMASKED NUMBERS, and the two-agent environment never draws such a graph.**
    `ma/topology.py` forbids every pair no single agent observes, which removes 10-33% of
    the possible edges, and removing edges can only reduce connectivity. Re-measured under
    the mask (`scripts/ma_graph_density_masked.py`, 2026-08-22): 86% at the current 1-1-3
    shape, 74% at 2-2-2, and 64% at three agents with two private nodes each. Treat the
    table above as an UPPER BOUND. The rule is still the right one -- the alternatives are
    far worse -- and connectivity is reported per episode with every headline split by it.

    For contrast, at the sizes this project targets: fixed `p = 0.5` gives degree 14.5 at
    d=30, and ER-2 gives **1% connected** at d=30 and is unusable. Connectivity is reported
    per episode either way, and every headline is split by it rather than pooled.
    """
    if d < 2:
        return 0.0
    return float(np.clip(2.0 * np.log(d) / d, 1e-6, 1 - 1e-6))


def expected_edges_p(d: int, edges: float) -> float:
    """The `p` giving `edges` expected adjacencies before conditioning on acyclicity."""
    n_pairs = d * (d - 1) // 2
    return float(np.clip(edges / n_pairs, 1e-6, 1 - 1e-6))


def scale_free_prior(space: GraphSpace, p: float = 0.3, gamma: float = 1.0) -> np.ndarray:
    """Erdos-Renyi reweighted toward concentrated degree distributions.

    Multiplies the ER weight by exp(gamma * (normalised degree concentration)), measured
    as the Gini coefficient of the total-degree sequence. `gamma = 0` recovers ER exactly;
    larger values favour graphs where a few nodes carry most of the edges.

    See the module docstring: this is a degree-concentration weighting, not preferential
    attachment, and it is only meaningfully different from ER once d is large enough for
    hubs to exist.
    """
    base = erdos_renyi_prior(space, p)
    if gamma == 0.0:
        return base

    adjacency = space.dags.astype(np.float64)
    degree = adjacency.sum(axis=1) + adjacency.sum(axis=2)  # [N, d] total degree per node

    # Gini coefficient of each degree sequence: 0 when perfectly even, higher when
    # concentrated. Empty graphs have no degree at all and are left at concentration 0.
    sorted_degree = np.sort(degree, axis=1)
    n = space.d
    index = np.arange(1, n + 1)[None, :]
    total = sorted_degree.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gini = (2 * (index * sorted_degree).sum(axis=1)) / (n * total) - (n + 1) / n
    gini = np.nan_to_num(gini, nan=0.0)

    w = base * np.exp(gamma * gini)
    return w / w.sum()


def build_prior(space: GraphSpace, kind: str = "erdos_renyi",
                p: Optional[float] = 0.5, expected_deg: Optional[float] = None,
                gamma: float = 1.0) -> np.ndarray:
    """Factory.

    `p` is the primary knob and defaults to 0.5, which is exactly the uniform-over-DAGs
    prior (see the module docstring). Pass `expected_deg` instead to hold expected degree
    fixed as d grows, which is the right parameterisation once d is large enough for
    sparsity to be meaningful.
    """
    if kind == "uniform":
        return uniform_prior(space)
    if expected_deg is not None:
        p = expected_degree_p(space.d, expected_deg)
    if p is None:
        p = 0.5
    if kind == "erdos_renyi":
        return erdos_renyi_prior(space, p)
    if kind == "scale_free":
        return scale_free_prior(space, p, gamma)
    raise ValueError(f"unknown prior {kind!r}; expected 'uniform', 'erdos_renyi', 'scale_free'")


def prior_singleton_fraction(space: GraphSpace, prior: np.ndarray) -> float:
    """GATE 1's target under a non-uniform prior.

    The probability that a graph drawn from this prior is alone in its Markov equivalence
    class, hence identifiable without intervening. Exact, not estimated.
    """
    return float(prior[space.is_singleton].sum())


def prior_mean_mec_size(space: GraphSpace, prior: np.ndarray) -> float:
    """Expected Markov equivalence class size under the prior.

    The single best summary of how hard the prior's instances are: class size correlates
    with interventions needed (0.56 at d=4) far more strongly than edge count does (0.29).
    """
    return float((prior * space.mec_sizes[space.mec_id]).sum())
