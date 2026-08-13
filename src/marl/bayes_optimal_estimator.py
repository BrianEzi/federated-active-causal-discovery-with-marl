"""Bayes-optimal graph estimator over the 8 known candidate topologies.

Unlike `analytic`/`avici`/`learned` (each a continuous edge-probability heuristic
with no guarantee of optimality), this estimator computes an exact posterior over
the small, fully-enumerable hypothesis space this environment actually uses (see
`src/generators.py::get_all_4node_topologies`) -- a genuine comparison ceiling for
how well a graph predictor *could* do here, given the same data the agents'
interventions produce, if graph structure were the only unknown.

Scope and assumptions (spelled out explicitly rather than silently baked in):
- Assumes the `hard` intervention regime (the project default as of 2026-08-13):
  a sample where node i was itself intervened on does not follow node i's own
  structural equation at all, so those samples are excluded from node i's own
  regression fit (but remain valid as *parent* values for other nodes' fits --
  a hard-intervened node's realized value is exactly what do-calculus wants
  downstream nodes conditioned on). Not correct under `soft_shift` (where the
  equation is preserved, just shifted) -- would need a per-hypothesis,
  per-intervention-type likelihood adjustment to extend there.
- Uses the environment's true, known `noise_scale` -- a fixed config
  hyperparameter of the simulated environment, not something private to any
  agent (the same sense in which `true_adjacency` is available for computing
  reward/SHD but never fed to the policy). This gives an exact Gaussian
  likelihood rather than one hostage to a small-sample noise-variance estimate.
- The likelihood is a *profile* (plug-in MLE) likelihood, not a fully
  marginalized one: per hypothesis, per node, fits the maximum-likelihood
  linear regression of that node on its hypothesis-specified parents
  (closed-form OLS -- for Gaussian noise, MLE and OLS coincide), then
  evaluates the Gaussian likelihood at that fit. A fully Bayesian marginal
  likelihood would instead integrate the edge weights out against their true
  prior (uniform on [-2,-0.5] u [0.5,2]) -- intractable in closed form since
  that prior isn't conjugate to the Gaussian likelihood. Profile likelihood is
  the standard, well-justified fallback here, documented as an approximation
  rather than overclaimed as exact.
- All 8 hypotheses have exactly 3 edges each (every candidate is a spanning
  tree over 4 nodes), so they're equally complex -- no model-complexity
  penalty (e.g. BIC) is needed to compare them fairly; a uniform prior over
  the 8 hypotheses plus raw likelihood comparison is sufficient.
"""
import numpy as np
from typing import Tuple


def _fit_node_log_likelihood(
    node_idx: int,
    parents: np.ndarray,
    samples: np.ndarray,
    self_intervened: np.ndarray,
    noise_scale: float,
) -> float:
    """Profile log-likelihood of node `node_idx`'s samples under a linear-Gaussian
    fit on its hypothesis-specified parents, using only samples where node_idx
    itself was not intervened on (see module docstring's hard-intervention scope)."""
    valid = ~self_intervened
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return 0.0  # No information yet -- contributes nothing to any hypothesis's score.

    x_i = samples[valid, node_idx]

    if len(parents) == 0:
        residuals = x_i  # Root node under this hypothesis: X_i = noise, mean 0 by construction.
    else:
        x_pa = samples[valid][:, parents]
        # Closed-form OLS (== MLE under Gaussian noise). Ridge-stabilized since X'X can
        # be near-singular with very few samples early in an episode.
        gram = x_pa.T @ x_pa + 1e-6 * np.eye(len(parents))
        beta_hat = np.linalg.solve(gram, x_pa.T @ x_i)
        residuals = x_i - x_pa @ beta_hat

    var = max(noise_scale ** 2, 1e-8)
    log_lik = -0.5 * np.sum(residuals ** 2) / var - 0.5 * n_valid * np.log(2 * np.pi * var)
    return float(log_lik)


def compute_hypothesis_posterior(
    raw_samples: np.ndarray,
    raw_interv: np.ndarray,
    candidate_adjacencies: np.ndarray,
    noise_scale: float,
) -> np.ndarray:
    """Posterior probability [H] over the candidate hypotheses (uniform prior),
    given the accumulated (already valid-sliced) samples/intervention labels so far."""
    candidate_adjacencies = np.asarray(candidate_adjacencies)
    H, d, _ = candidate_adjacencies.shape
    log_liks = np.zeros(H)

    for h in range(H):
        adj = candidate_adjacencies[h]
        total = 0.0
        for node_idx in range(d):
            parents = np.where(adj[:, node_idx] > 0.5)[0]
            self_intervened = raw_interv[:, node_idx] > 0.5
            total += _fit_node_log_likelihood(node_idx, parents, raw_samples, self_intervened, noise_scale)
        log_liks[h] = total

    log_liks = log_liks - np.max(log_liks)  # numerical stability before exponentiating
    unnorm = np.exp(log_liks)
    return unnorm / np.sum(unnorm)


def bayes_optimal_predict(
    raw_samples: np.ndarray,
    raw_interv: np.ndarray,
    n_valid: int,
    candidate_adjacencies: np.ndarray,
    noise_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (edge_probability_matrix [d,d], posterior [H]).
    edge_probability_matrix[i,j] = sum_h posterior[h] * candidate_adjacencies[h,i,j] --
    the posterior-weighted mixture over the known hypothesis space, in the same [d,d]
    probability-matrix format the other three estimators already produce."""
    candidate_adjacencies = np.asarray(candidate_adjacencies)
    H = candidate_adjacencies.shape[0]
    if n_valid == 0:
        posterior = np.full(H, 1.0 / H)
    else:
        posterior = compute_hypothesis_posterior(
            raw_samples[:n_valid], raw_interv[:n_valid], candidate_adjacencies, noise_scale
        )
    prob = np.tensordot(posterior, candidate_adjacencies, axes=(0, 0))
    return prob.astype(np.float32), posterior
