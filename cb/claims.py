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


@dataclass(frozen=True)
class Claim:
    """One scored question about one node pair -- the unit `score_window` counts.

    Exists so a trace can show WHICH claim went wrong rather than only how many did.
    `freq_correct` / `freq_wrong` are the bootstrap frequencies the outcome was read off,
    so a claim that missed the bar by 0.02 is distinguishable from one that was never
    close.
    """
    kind: str               # "adjacency" | "type"
    u: int
    v: int
    required: bool
    outcome: str            # "right" | "wrong" | "unsure"
    truth: str              # what the true MAG says, human-readable
    freq_correct: float
    freq_wrong: float


def _outcome(freq_correct: float, freq_wrong: float, bar: float):
    if freq_correct >= bar:
        return "right"
    if freq_wrong >= bar:
        return "wrong"
    return "unsure"


def enumerate_claims(belief, true_mag: np.ndarray, private_positions: Sequence[int] = (),
                     bar: float = 0.7, require_all_types: bool = True,
                     confounding_claims: bool = True):
    """Every claim in one window, with its outcome and the frequencies behind it.

    `score_window` is a tally over exactly this list -- the two cannot disagree because
    the second is defined in terms of the first.

    `require_all_types` (default since 2026-08-26) makes EVERY type claim required, not
    only the private-incident ones. See the note in `score_window`.
    """
    mag = np.asarray(true_mag)
    k = mag.shape[0]
    private = set(int(p) for p in private_positions)
    adjacency = np.asarray(belief.adjacency)
    directed = np.asarray(belief.directed)
    bidirected = np.asarray(belief.bidirected)

    claims = []
    for u, v in combinations(range(k), 2):
        truly_adjacent = mag[u, v] != 0 or mag[v, u] != 0
        f_adj = float(adjacency[u, v])
        if truly_adjacent:
            correct, wrong = f_adj, 1.0 - f_adj
            truth = "adjacent"
        else:
            correct, wrong = 1.0 - f_adj, f_adj
            truth = "not adjacent"
        claims.append(Claim("adjacency", u, v, True,
                            _outcome(correct, wrong, bar), truth, correct, wrong))
        if not truly_adjacent:
            continue                       # no type claim on a true non-edge

        f_bi = float(bidirected[u, v])
        if mag[u, v] == MAG_BIDIRECTED and not confounding_claims:
            # The ATTRIBUTION claim replaces this one outright (2026-08-26). Scoring both
            # would double-count the same fact and, worse, let an agent bank the easy half
            # -- "something confounds these" -- while failing the half that carries the
            # thesis, which is whose it is. See cb/attribution.py.
            continue
        if mag[u, v] == MAG_BIDIRECTED:
            correct, wrong = f_bi, max(float(directed[u, v]), float(directed[v, u]))
            required = True                # confounding is the thesis: always required
            truth = "confounded"
        elif mag[u, v] == MAG_DIRECTED:
            correct = float(directed[u, v])
            wrong = max(float(directed[v, u]), f_bi)
            required = require_all_types or u in private or v in private
            truth = "directed"
        else:                              # mag[v, u] == MAG_DIRECTED
            correct = float(directed[v, u])
            wrong = max(float(directed[u, v]), f_bi)
            required = require_all_types or u in private or v in private
            truth = "reverse directed"
        claims.append(Claim("type", u, v, required,
                            _outcome(correct, wrong, bar), truth, correct, wrong))
    return claims


def score_window(belief, true_mag: np.ndarray, private_positions: Sequence[int] = (),
                 bar: float = 0.7, require_all_types: bool = True,
                 confounding_claims: bool = True) -> ClaimScore:
    """Score one window's belief against its true MAG.

    `belief` is a `BootstrapBelief` (needs `.adjacency`, `.directed`, `.bidirected`
    frequency matrices). `bar` is the confidence bar per claim -- 0.7, matching the
    identify threshold's meaning of "confident", NOT a bare majority.

    REQUIRED claims (the identification set, mirroring [U14]): every adjacency claim and
    every type claim. Identification therefore means "recovered the window".

    THE EXEMPTION THAT USED TO LIVE HERE, and why it is gone (2026-08-26). Shared-block
    directed edges were exempt, on the stated ground that "Markov equivalence leaves such
    edges unorientable". THAT WAS WRONG, and it was wrong in a way that made the task
    easier than it should be. Markov equivalence constrains what OBSERVATION can settle;
    the interventional reveal channel is pairwise ancestry, so for an adjacent pair,
    u -> v iff u is an ancestor of v, v -> u iff the reverse, and u <-> v iff neither.
    Intervening on BOTH endpoints therefore fixes the mark outright, with no residual
    ambiguity, and intervening on every node determines the window completely. The
    exemption was a GRADING CHOICE dressed as a necessity, and it shrank precisely the
    part of the problem coordination can win -- shared nodes are the contended surface.

    `require_all_types=False` restores the old grading, for reproducing pre-2026-08-26
    numbers only. Any comparison must hold it fixed across arms.
    """
    return _tally(belief, true_mag, private_positions, bar,
                  require_all_types=require_all_types,
                  confounding_claims=confounding_claims)


def _tally(belief, true_mag: np.ndarray, private_positions: Sequence[int] = (),
           bar: float = 0.7, require_all_types: bool = True,
           confounding_claims: bool = True) -> ClaimScore:
    """`score_window`'s counts, without materialising the claims.

    WHY THIS EXISTS, AND WHY IT IS NOT A SECOND DEFINITION. The environment scores every
    agent's window on every step -- four `score_window` calls per step, and in a profiled
    training run that single call was 13% of the entire wall clock, spent building
    C(k,2) + |E| frozen dataclasses only to tally them and drop them. This is the same
    arithmetic done over arrays.

    Two definitions of the same thing is exactly how a metric drifts, so
    `tests/test_claims.py` asserts this agrees with `enumerate_claims` EXACTLY -- outcome
    by outcome, over randomised beliefs and MAGs, at every bar and both flag settings.
    `enumerate_claims` remains the readable definition and the one traces read; this is a
    tally of it.
    """
    mag = np.asarray(true_mag)
    k = mag.shape[0]
    if k < 2:
        return ClaimScore(0, 0, 0, 0, 0, 0)
    u, v = np.triu_indices(k, 1)                 # exactly combinations(range(k), 2) order

    adjacency = np.asarray(belief.adjacency, dtype=float)
    directed = np.asarray(belief.directed, dtype=float)
    bidirected = np.asarray(belief.bidirected, dtype=float)

    f_adj = adjacency[u, v]
    truly_adjacent = (mag[u, v] != 0) | (mag[v, u] != 0)
    # Both branches computed rather than one derived from the other: `1 - (1 - x)` is not
    # `x` in floating point, and the comparison is against a bar these can sit exactly on.
    adj_correct = np.where(truly_adjacent, f_adj, 1.0 - f_adj)
    adj_wrong = np.where(truly_adjacent, 1.0 - f_adj, f_adj)

    f_bi = bidirected[u, v]
    d_uv, d_vu = directed[u, v], directed[v, u]
    is_bi = mag[u, v] == MAG_BIDIRECTED
    is_fwd = (~is_bi) & (mag[u, v] == MAG_DIRECTED)
    # The final branch mirrors the enumerator's bare `else`: everything adjacent that is
    # neither bidirected nor forward-directed, whatever the encoding.
    is_back = truly_adjacent & (~is_bi) & (~is_fwd)

    type_present = truly_adjacent & ~(is_bi & (not confounding_claims))
    type_correct = np.select([is_bi, is_fwd, is_back], [f_bi, d_uv, d_vu], default=0.0)
    type_wrong = np.select(
        [is_bi, is_fwd, is_back],
        [np.maximum(d_uv, d_vu), np.maximum(d_vu, f_bi), np.maximum(d_uv, f_bi)],
        default=0.0)

    private = set(int(p) for p in private_positions)
    if require_all_types:
        type_required = np.ones(u.shape, dtype=bool)
    else:
        incident = np.array([a in private or b in private for a, b in zip(u, v)])
        type_required = incident.copy()
    type_required = np.where(is_bi, True, type_required)   # confounding is always required

    def outcomes(correct, wrong):
        right = correct >= bar
        return right, (~right) & (wrong >= bar)

    adj_right, adj_bad = outcomes(adj_correct, adj_wrong)
    typ_right, typ_bad = outcomes(type_correct, type_wrong)
    typ_right &= type_present
    typ_bad &= type_present

    n_claims = int(u.size + type_present.sum())
    n_right = int(adj_right.sum() + typ_right.sum())
    n_wrong = int(adj_bad.sum() + typ_bad.sum())

    req = type_present & type_required
    return ClaimScore(n_right, n_wrong, n_claims - n_right - n_wrong,
                      int(adj_right.sum() + (typ_right & req).sum()),
                      int(u.size + req.sum()),
                      int(adj_bad.sum() + (typ_bad & req).sum()))
