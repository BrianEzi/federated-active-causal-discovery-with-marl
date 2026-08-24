"""Three-outcome claim scoring: settled-right, unsure, settled-wrong.

WHY THREE OUTCOMES AND NOT TWO. A claim decided 7-of-12 is a coin flip; majority voting
launders it into a confident assertion, and twelve such launderings make the joint answer
almost certainly wrong somewhere. "We did not determine it" and "we determined it wrongly"
are different failures with different costs -- for a thesis about confounding, the second
is the one that can least be afforded -- so they are counted separately and never summed.

WHY PER CLAIM AND NOT PER GRAPH. The previous criterion demanded each bootstrap replicate
be simultaneously perfect everywhere. Errors scatter across replicates, so 95%-per-claim
accuracy multiplied down to ~36% episode success -- measured, 2026-08-24. Aggregating per
claim is also the honest analogue of the exact engine's criterion, which scores posterior
MASS on a set of acceptable graphs rather than any single graph.

WHAT A CLAIM IS. For one agent's window against its true MAG:
  - one ADJACENCY claim per node pair: are these two connected at all?
  - one TYPE claim per true-MAG edge: directed the right way, or confounded?
Type claims exist only for true edges: on a correctly-absent pair there is no further
question to ask, and inventing one would double-count the adjacency claim.

REWARD USE. `fraction = (n_right - penalty * n_wrong) / n_claims`. The environment pays
the per-step CHANGE in this, which is dense, moves whenever an experiment settles
anything, and punishes a claim that settles wrongly harder than one that stays open.
Truth enters here exactly as it entered the old reward (scoring is oracle-side);
observations remain truth-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED


@dataclass(frozen=True)
class ClaimScore:
    n_right: int
    n_wrong: int
    n_unsure: int
    required_right: int
    required_total: int
    required_wrong: int

    @property
    def n_claims(self) -> int:
        return self.n_right + self.n_wrong + self.n_unsure

    def fraction(self, penalty: float = 1.0) -> float:
        if self.n_claims == 0:
            return 0.0
        return (self.n_right - penalty * self.n_wrong) / self.n_claims

    @property
    def identified(self) -> bool:
        """All REQUIRED claims settled right, and nothing anywhere settled wrong.

        Zero-tolerance on settled-wrong is deliberate: an agent confidently wrong about
        any part of its window has not identified it, whatever else it got right.
        """
        return (self.required_right == self.required_total
                and self.n_wrong == 0)


def _outcome(freq_correct: float, freq_wrong: float, bar: float):
    if freq_correct >= bar:
        return "right"
    if freq_wrong >= bar:
        return "wrong"
    return "unsure"


def score_window(belief, true_mag: np.ndarray, private_positions: Sequence[int] = (),
                 bar: float = 0.7) -> ClaimScore:
    """Score one window's belief against its true MAG.

    `belief` is a `BootstrapBelief` (needs `.adjacency`, `.directed`, `.bidirected`
    frequency matrices). `bar` is the confidence bar per claim -- 0.7, matching the
    identify threshold's meaning of "confident", NOT a bare majority.

    REQUIRED claims (the identification set, mirroring [U14]): every adjacency claim,
    every confounding type claim, and the type claim of every private-incident directed
    edge. A shared-block directed edge may stay unsure without blocking identification --
    Markov equivalence leaves such edges unorientable for want of information.
    """
    mag = np.asarray(true_mag)
    k = mag.shape[0]
    private = set(int(p) for p in private_positions)
    adjacency = np.asarray(belief.adjacency)
    directed = np.asarray(belief.directed)
    bidirected = np.asarray(belief.bidirected)

    n_right = n_wrong = n_unsure = 0
    required_right = required_total = required_wrong = 0

    def tally(outcome: str, required: bool):
        nonlocal n_right, n_wrong, n_unsure
        nonlocal required_right, required_total, required_wrong
        if outcome == "right":
            n_right += 1
        elif outcome == "wrong":
            n_wrong += 1
        else:
            n_unsure += 1
        if required:
            required_total += 1
            required_right += outcome == "right"
            required_wrong += outcome == "wrong"

    for u, v in combinations(range(k), 2):
        truly_adjacent = mag[u, v] != 0 or mag[v, u] != 0
        f_adj = float(adjacency[u, v])
        # -- adjacency claim, always required ------------------------------------------
        if truly_adjacent:
            tally(_outcome(f_adj, 1.0 - f_adj, bar), required=True)
        else:
            tally(_outcome(1.0 - f_adj, f_adj, bar), required=True)
            continue                       # no type claim on a true non-edge
        # -- type claim for the true edge ----------------------------------------------
        f_bi = float(bidirected[u, v])
        if mag[u, v] == MAG_BIDIRECTED:
            correct, wrong = f_bi, max(float(directed[u, v]), float(directed[v, u]))
            required = True                # confounding is the thesis: always required
        elif mag[u, v] == MAG_DIRECTED:
            correct = float(directed[u, v])
            wrong = max(float(directed[v, u]), f_bi)
            required = u in private or v in private
        else:                              # mag[v, u] == MAG_DIRECTED
            correct = float(directed[v, u])
            wrong = max(float(directed[u, v]), f_bi)
            required = u in private or v in private
        tally(_outcome(correct, wrong, bar), required=required)

    return ClaimScore(n_right, n_wrong, n_unsure,
                      required_right, required_total, required_wrong)
