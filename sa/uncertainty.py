"""Splitting the agent's uncertainty into the part observation can remove and the part
only intervention can remove.

Why the earlier attempt was replaced. The step-0 diagnostic reported a "skeleton error" and
an "orientation error" side by side as though they partitioned the belief. They did not:
they were a SUM over pairs and a MEAN over edges, on different scales, summing to nothing.
Worse, orientation error has an irreducible floor -- 16.4% of edges are reversible within
their Markov equivalence class and can never be oriented from observational data -- so
comparing how much each moved compared a free quantity against a pinned one.

This is the honest version, and the split is dictated by what the data can actually do.
Observational data identifies a DAG only up to its Markov equivalence class. So partition
the graphs by that class and apply the chain rule:

    H(G)  =  H(E)  +  H(G | E)

    H(E)      which class are we in.    Observation reduces this.
    H(G | E)  which member of the class. Observation can NEVER reduce this. Only
                                         interventions can.

`H(G | E)` at step 0 is therefore exactly the intervention-addressable uncertainty, in bits.
Watching it fall through an episode is watching interventions do the one job that is theirs.

Two properties make this checkable rather than merely plausible:

  * The chain rule is an identity, so the three numbers must agree to floating point. Any
    disagreement is a bug, not noise.
  * BGe is score-equivalent: every DAG in a class has identical likelihood. With
    observational data only and a uniform prior, the within-class posterior is therefore
    UNIFORM, giving the closed form

        H(G | E)  =  SUM_c  p_c * log |c|

    That is an exact prediction for step 0, independent of this code, and it is asserted in
    tests/test_uncertainty.py. It also fails loudly if score equivalence is ever broken.

Everything here is in BITS.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_EPS = 1e-300


def _entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def class_masses(posterior: np.ndarray, mec_id: np.ndarray, n_mecs: int) -> np.ndarray:
    """Total posterior mass on each equivalence class."""
    return np.bincount(mec_id, weights=np.asarray(posterior, dtype=np.float64),
                       minlength=n_mecs)


def decompose(posterior: np.ndarray, space) -> Dict[str, float]:
    """Split the posterior's entropy into class-level and within-class parts.

    Returns bits, plus the closed-form step-0 prediction for comparison and the residual of
    the chain rule, so a caller can assert on both rather than trust them.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    mec_id = np.asarray(space.mec_id)
    n_mecs = int(space.mec_sizes.shape[0])

    h_total = _entropy_bits(posterior)
    masses = class_masses(posterior, mec_id, n_mecs)
    h_class = _entropy_bits(masses)

    # H(G|E) = SUM_c p_c H(within c). Computed directly rather than as H(G) - H(E), so that
    # the chain rule becomes an independent check instead of a tautology.
    within = 0.0
    for c in np.flatnonzero(masses > 0):
        block = posterior[mec_id == c]
        block = block / block.sum()
        within += masses[c] * _entropy_bits(block)

    # Closed form that must hold when the belief is observational-only: score equivalence
    # makes every member of a class equally likely, so within-class entropy is log |c|.
    uniform_within = float((masses * np.log2(np.asarray(space.mec_sizes,
                                                        dtype=np.float64))).sum())

    return {
        "h_total": h_total,
        "h_class": h_class,
        "h_within": within,
        "chain_rule_residual": h_total - (h_class + within),
        "h_within_if_uniform": uniform_within,
        "n_classes_with_mass": int((masses > 1e-12).sum()),
        "top_class_mass": float(masses.max()),
    }


def episode_trace(env, policy, seed: int, budget: Optional[int] = None) -> Dict[str, list]:
    """Run one episode, recording the decomposition after every intervention.

    The record starts at step 0, before any action, so `trace[0]` is the observational
    belief and every later entry shows what an intervention bought.
    """
    space = env.space
    budget = env.config.budget if budget is None else budget
    result = env.reset(seed=seed)
    rows = [decompose(result.posterior, space)]
    actions = []

    if hasattr(policy, "reset"):
        policy.reset(seed=seed)

    steps = 0
    while not result.done and steps < budget:
        action = policy(env, result)
        actions.append(int(action))
        result = env.step(action)
        rows.append(decompose(result.posterior, space))
        steps += 1

    return {
        "rows": rows,
        "actions": actions,
        "identified": bool(result.identified),
        "interventions": steps,
        "mec_size": int(result.info["mec_size"]),
    }


def summarise_trace(trace: Dict[str, list]) -> Dict[str, float]:
    """Bits removed over the episode, split by which kind of uncertainty they came from."""
    first, last = trace["rows"][0], trace["rows"][-1]
    steps = max(trace["interventions"], 1)
    return {
        "interventions": trace["interventions"],
        "identified": trace["identified"],
        # The quantity of interest: intervention-addressable uncertainty present at the
        # start, and how much of it was actually removed.
        "addressable_bits_at_start": first["h_within"],
        "addressable_bits_removed": first["h_within"] - last["h_within"],
        "addressable_bits_per_intervention": (first["h_within"] - last["h_within"]) / steps,
        # Class-level uncertainty should be nearly gone already if n_obs is calibrated. If
        # an episode removes a lot of it, the agent is doing structure discovery that the
        # observational data was supposed to have done.
        "class_bits_at_start": first["h_class"],
        "class_bits_removed": first["h_class"] - last["h_class"],
        "total_bits_removed": first["h_total"] - last["h_total"],
    }
