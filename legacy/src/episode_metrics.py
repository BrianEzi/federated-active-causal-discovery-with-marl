"""Pure-function helpers for the agent-vs-estimator-learning evaluation metrics.

These are deliberately kept out of compute_ippo_rewards / the reward pipeline -- they exist
to measure whether the intervention-selection policy is learning real skill, independent of
whether the graph estimator itself is improving (e.g. by memorizing the small set of training
topologies). See docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md's "Morning session" section for
the full rationale.
"""
import numpy as np


def gaussian_entropy(cov: np.ndarray, d: int, eps: float = 1e-6) -> float:
    """Differential entropy of a d-dimensional multivariate Gaussian with the given
    covariance: H = 0.5 * (d*log(2*pi*e) + logdet(Cov)). Exact under this environment's
    default MechanismType.LINEAR + Gaussian noise, where the SCM's stationary distribution
    is exactly multivariate Gaussian. Uses slogdet (not log(det(...))) for numerical
    stability -- det() alone over/underflows for the near-singular covariances common early
    in an episode. `running_covariance` starts at a literal zero matrix at episode start, so
    a small eps*I floor is applied unconditionally, not just defensively.
    """
    cov = np.asarray(cov, dtype=np.float64)
    cov = cov + np.eye(d) * eps
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)


def shd_trajectory_auc(shd_trajectory, max_shd: float) -> float:
    """Trapezoidal AUC of a within-episode SHD trajectory, normalized to [0, 1] by the
    worst-case achievable SHD (max_shd) times trajectory length. Lower = faster/more
    efficient convergence to low SHD -- an inversion of typical AUC conventions (where higher
    is usually better), so callers should not assume "higher AUC is better" here.
    """
    if len(shd_trajectory) < 2:
        return 0.0
    raw_auc = float(np.trapz(shd_trajectory))
    denom = max(1e-8, max_shd * (len(shd_trajectory) - 1))
    return raw_auc / denom


def shd_reduction_auc(shd_trajectory, max_shd: float) -> float:
    """AUC of (shd[0] - shd[t]) instead of raw shd[t] -- isolates the policy's own
    contribution to SHD reduction from the estimator's zero-intervention starting-guess
    quality, which raw shd_trajectory_auc conflates. Higher = more cumulative reduction
    achieved (a normal-orientation AUC, unlike shd_trajectory_auc).
    """
    if len(shd_trajectory) < 2:
        return 0.0
    baseline = shd_trajectory[0]
    reduction = [baseline - s for s in shd_trajectory]
    raw_auc = float(np.trapz(reduction))
    denom = max(1e-8, max_shd * (len(shd_trajectory) - 1))
    return raw_auc / denom


def normalized_target_entropy(node_intervention_counts: dict, d: int) -> float:
    """Shannon entropy of the empirical distribution of which nodes got intervened on
    across an episode, normalized by log(d) to [0, 1]. Low entropy = agents repeatedly
    targeting the same node(s) despite having a broader legal target set; high entropy =
    broad, even exploration across targets. Returns 0.0 if no interventions occurred at all
    (entropy of an empty/all-zero distribution is undefined, not "maximally broad").
    """
    counts = np.array([node_intervention_counts.get(i, 0) for i in range(d)], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(d)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0
