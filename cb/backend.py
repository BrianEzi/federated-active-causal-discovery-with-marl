"""The constraint engine behind the environment's belief protocol.

THE BOUNDARY. `ma/env.py` talks to a window's belief through three calls: `edge_marginals`
(every refresh), and an identification measure (every reward step). This class satisfies
the first with the same signature `crosscheck.belief_dp.WindowBeliefDP` has, so `_refresh`
does not branch at all -- and replaces the second's posterior mass with the per-replicate
credit fraction, which the env reads through one explicit branch.

WHAT `clean` AND `score_rule` MEAN HERE: NOTHING, AND THAT IS THE POINT. The exact engine
mixes clean and dirty score tables with a scalar clean fraction -- the abstraction that
loses node identity and makes it unsound for `widest_hidden > 1` (see the long note in
`ma/env.py.__init__`). This engine conditions every test on the per-row intervention mask
directly, so the clean/dirty distinction never exists. Both arguments are accepted and
ignored so the two engines are call-compatible; ignoring them is the fix, not a shortcut.

IDENTIFICATION IS A FRACTION OF REPLICATES, NOT A POSTERIOR MASS. There is no posterior
here. The analogue adopted (per `docs/CB_IMPLEMENTATION_PLAN.md` and the handover): the
fraction of bootstrap replicates whose recovered graph is CREDITED against the true MAG of
the window (`ma.projection.latent_projection` -- ground truth, never shown to the agent).
A replicate is credited when:

  1. its adjacency equals the MAG's adjacency -- the dependence structure is right;
  2. its bidirected pairs equal the MAG's bidirected pairs EXACTLY -- confounding is
     right, with no false confounders and none missed. Confounding is the thesis, so it
     is the one claim required to be both sound and complete;
  3. every directed edge it asserts is a directed edge of the MAG -- orientations are
     sound. Circles are allowed where the truth is directed: Markov equivalence leaves
     edges unorientable for want of information, and the exact engine's [U14] criterion
     (credit-set mass over the Markov equivalence class) makes the same allowance;
  4. every REQUIRED directed edge is asserted. Which edges are required is the caller's
     policy knob: [U14] pins private-incident edges to the truth (an agent must resolve
     its own private structure -- that is the headline behaviour), so the env passes the
     private positions. `strict=True` requires every directed MAG edge instead, the
     analogue of mass on the exact true DAG for the non-U14 reward path.

The credit fraction lives in [0, 1], moves with evidence, and concentrates as data
accumulates -- the same operational contract the posterior mass had. It is NOT a
probability of the truth, and the write-up must not call it one.

WHAT THIS CRITERION DOES NOT CHECK: the cross-agent union. The exact [U14] path also
requires the union of per-agent answers to be acyclic and Markov-equivalent to the global
truth, built from a representative of each agent's credit set. A replicate PAG has no
single representative DAG -- circles are honest ambiguity -- so the union check does not
port. Divergence accepted 2026-08-24 and DOCUMENTED HERE so the Phase-4 cross-check
(verdict agreement between engines at k=4-5) knows to expect it: the constraint verdict is
per-agent credit only, and is therefore WEAKER on jointly-inconsistent answers.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from cb.bootstrap import BootstrapBelief, bootstrap_belief
from cb.orient import CODE_BIDIRECTED, CODE_DIRECTED
from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED


class ConstraintBackend:
    """Bootstrap constraint engine for one agent's window.

    `can_handle_multi_hidden` is the capability the removed guard became: the env asks the
    backend instead of hard-coding the refusal. This engine never forms the clean/dirty
    mixture, so multiple hidden nodes cost it nothing.
    """

    can_handle_multi_hidden = True

    def __init__(self, k: int, shared_positions: Sequence[int], n_boot: int = 50,
                 alpha: float = 0.01, max_cond: int = 3, base_seed: int = 0,
                 n_jobs: int = 1, skeleton_alpha: Optional[float] = None):
        self.k = int(k)
        self.shared_positions = tuple(int(p) for p in shared_positions)
        self.n_boot = int(n_boot)
        self.alpha = float(alpha)
        # None => same as alpha. See cb/citest.py::FisherZ.__init__ for why the skeleton
        # and the orientation channels want different thresholds.
        self.skeleton_alpha = skeleton_alpha
        self.max_cond = int(max_cond)
        self.base_seed = int(base_seed)
        self.n_jobs = int(n_jobs)
        self._calls = 0
        # Base for the CURRENT episode's resample stream; `set_episode` moves it.
        self._episode_base = int(self.base_seed)
        self.last: Optional[BootstrapBelief] = None
        # Set by the env at reset when oracle_obs_structure is on: (adjacency, sepsets)
        # from ma.projection.observational_skeleton -- the true observational limit for
        # THIS episode's graph. None means estimate the skeleton from data as usual.
        self.oracle_skeleton = None

    # -- the WindowBeliefDP-shaped surface ------------------------------------------------

    def set_episode(self, episode_seed: int) -> None:
        """Start a fresh resample stream for one episode.

        Called by the environment at every reset. Without it the stream carried process
        history, which made baselines vary across runs of identical configs.
        """
        self._calls = 0
        self._episode_base = int(self.base_seed) + 9973 * (int(episode_seed) % 100_003)

    def edge_marginals(self, data: np.ndarray, known: np.ndarray, clean=None,
                       score_rule=None, blocks=None) -> np.ndarray:
        """`[k, k]` directed-edge frequencies. Signature-compatible with the exact engine.

        `known` is the per-row intervention mask the env already maintains -- exactly what
        `FisherZ` consumes. `score_rule` is ignored (module docstring).

        `clean` is NOT ignored, and its meaning here differs from the exact engine's. The
        env passes the per-row fraction of HIDDEN nodes clamped, zeroed unless
        `disclose_regime` -- the "I clamped something you cannot see" bit. The exact
        engine used it to weight a clean/dirty score mixture; here any nonzero value marks
        the row as a FOREIGN REGIME, excluded from the two-sample ancestry contrasts so
        another agent's private clamp is not attributed to whatever variable is under
        test (see `FisherZ.__init__`). The disclosure discipline is unchanged: the no-bit
        arm passes zeros and differs in exactly what the agent is told.

        SEEDING IS PER EPISODE, not per process. It used to be `base_seed + calls`, with
        `calls` counting every refresh since the backend was CONSTRUCTED -- so the resample
        stream depended on how much had happened earlier in the process. Two consequences,
        both measured on 2026-08-26: a baseline scored 0.145 in one run and 0.100 in
        another on an identical config and seed, because training length in refreshes
        differed; and arms evaluated in sequence each saw a different stream, so "paired"
        comparisons were not paired. `set_episode` resets the stream, so a given
        (base_seed, episode) reproduces regardless of what ran before it.
        """
        self._calls += 1
        foreign = None if clean is None else np.asarray(clean, dtype=float) > 0
        self.last = bootstrap_belief(
            np.asarray(data, dtype=float), np.asarray(known),
            n_boot=self.n_boot, alpha=self.alpha, max_cond=self.max_cond,
            seed=self._episode_base + self._calls, foreign=foreign, blocks=blocks,
            n_jobs=self.n_jobs, oracle_skeleton=self.oracle_skeleton,
            skeleton_alpha=self.skeleton_alpha)
        return self.last.edge_marginals()

    @property
    def bidirected(self) -> np.ndarray:
        """`[k, k]` bidirected frequencies from the last refresh -- the separate channel
        the federation design is about. Zeros before the first refresh."""
        if self.last is None:
            return np.zeros((self.k, self.k), dtype=float)
        return self.last.bidirected

    # -- identification -------------------------------------------------------------------

    def credit_fraction(self, true_mag: np.ndarray,
                        required_positions: Sequence[int] = (),
                        strict: bool = False) -> float:
        """Fraction of replicates credited against `true_mag` (see module docstring).

        `true_mag` is `ma.projection.latent_projection` output in WINDOW positions.
        `required_positions`: directed MAG edges incident to these positions must be
        asserted (the [U14] private-pinning analogue). `strict=True` requires every
        directed MAG edge instead and ignores `required_positions`.
        """
        if self.last is None or self.last.replicates is None:
            return 0.0
        mag = np.asarray(true_mag)
        k = self.k
        required = set(int(p) for p in required_positions)

        mag_adj = (mag != 0) | (mag != 0).T
        mag_bidirected = {(u, v) for u, v in combinations(range(k), 2)
                          if mag[u, v] == MAG_BIDIRECTED}
        mag_directed = [(u, v) for u in range(k) for v in range(k)
                        if u != v and mag[u, v] == MAG_DIRECTED]
        needed = [(u, v) for u, v in mag_directed
                  if strict or u in required or v in required]

        credited = 0
        for codes in self.last.replicates:
            rep_adj = (codes != 0) | (codes != 0).T
            if not np.array_equal(rep_adj, mag_adj):
                continue
            rep_bidirected = {(u, v) for u, v in combinations(range(k), 2)
                              if codes[u, v] == CODE_BIDIRECTED}
            if rep_bidirected != mag_bidirected:
                continue
            # Sound: every directed claim is a MAG directed edge.
            claimed = [(u, v) for u in range(k) for v in range(k)
                       if u != v and codes[u, v] == CODE_DIRECTED]
            if any(mag[u, v] != MAG_DIRECTED for u, v in claimed):
                continue
            # Complete where required.
            if any(codes[u, v] != CODE_DIRECTED for u, v in needed):
                continue
            credited += 1
        return credited / len(self.last.replicates)
