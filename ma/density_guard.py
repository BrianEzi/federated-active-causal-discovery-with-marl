"""Reject draws whose windows are too dense for the attributed backend to enumerate.

WHY A GUARD IS NEEDED AT ALL. Measured 2026-08-27: at four agents the attributed episode
cost is heavy-TAILED rather than high. The median reset is under a second, one draw in five
took 48 s, and one did not finish in twenty minutes. cProfile over the first two episodes
reports a comfortable 1.76 s/episode -- which is exactly the number that would have
justified launching sixteen 4000-episode runs into a wall.

WHY IT REJECTS ON EDGES AND NOT ON CANDIDATES. `AttributedVersionSpaceBackend` already caps
its candidate count at `max_candidates`, but that truncation runs AFTER
`cb.versionspace.equivalence_class`, and the enumeration is the expensive step -- it
searches 3^(edges in the window's MAG). A cap read off the candidate count therefore cannot
bound the cost that produced it. Window EDGE COUNT is knowable before any enumeration, from
the latent projection alone, and it is what the cost is exponential in.

WHY IT IS A NEW FILE AND A SUBCLASS. `ma/env.py` is being edited by another session against
the transfer axis (docs/HANDOVER_2026_08_27.md section 4). Overriding one method from
outside keeps the two work streams from touching the same lines.

WHAT IT COSTS, AND WHY THE 3-AGENT CONTROL IS NOT OPTIONAL. Rejecting dense draws changes
the episode distribution, and the precedent is bad: the 2026-08-26 density guard at k=7
rejected 94% of worlds, sampled an unrepresentative sparse tail, and the row was discarded
rather than reported. The difference here is that three agents is cheap BOTH ways, so the
same configuration can be run guarded and unguarded and the distortion MEASURED instead of
assumed. No guarded result above three agents should be reported without that control
beside it.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ma.env import TwoAgentEnv
from ma.projection import bidirected_pairs, latent_projection


def window_edges(adjacency: np.ndarray, nodes: Tuple[int, ...]) -> int:
    """Adjacencies in the window's MAG -- the exponent the enumeration cost runs on."""
    mag = latent_projection(adjacency, tuple(nodes))
    k = len(nodes)
    return int(sum(1 for i in range(k) for j in range(i + 1, k) if mag[i, j] or mag[j, i]))


class DensityGuardedEnv(TwoAgentEnv):
    """`TwoAgentEnv` that also refuses draws whose densest window is past `max_edges`.

    `max_edges=None` disables the guard entirely and the class behaves exactly like its
    parent -- which is what the unguarded half of the three-agent control runs on, so that
    both halves of that comparison go through the same code path.

    `density_rejections` counts draws thrown out for density alone, cumulatively over the
    environment's life. It is the number that has to be reported alongside any guarded
    result; a guard whose rejection rate is not quoted is a silent change to the task.
    """

    def __init__(self, *args, max_edges: int = None, **kwargs):
        self.max_edges = None if max_edges is None else int(max_edges)
        self.density_rejections = 0
        self.density_draws = 0
        super().__init__(*args, **kwargs)

    def _too_dense(self, candidate: np.ndarray) -> bool:
        if self.max_edges is None:
            return False
        return any(window_edges(candidate, tuple(w.nodes)) > self.max_edges
                   for w in self.windows.values())

    def _sample_mixed_dag(self, cfg) -> Tuple[np.ndarray, int]:
        """The parent's mix condition, AND the density condition, in one rejection loop.

        Deliberately not `super()` plus a retry: rejecting after the parent returns would
        re-draw the mix condition from a different point in the RNG stream, and the two
        conditions would then interact in a way neither is written to expect.
        """
        if self.max_edges is None:
            return super()._sample_mixed_dag(cfg)
        if cfg.episode_mix == "any":
            for draw in range(1, 201):
                self.density_draws += 1
                candidate = self._draw(cfg)
                if not self._too_dense(candidate):
                    return candidate, draw
                self.density_rejections += 1
            raise RuntimeError(
                f"max_edges={self.max_edges}: no sparse enough graph in 200 draws on "
                f"topology {self.topology.name!r}")
        for draw in range(1, 201):
            self.density_draws += 1
            candidate = self._draw(cfg)
            confounded = any(bidirected_pairs(candidate, tuple(w.nodes))
                             for w in self.windows.values())
            if confounded != (cfg.episode_mix == "confounded"):
                continue
            if self._too_dense(candidate):
                self.density_rejections += 1
                continue
            return candidate, draw
        raise RuntimeError(
            f"episode_mix={cfg.episode_mix!r} with max_edges={self.max_edges}: no "
            f"qualifying graph in 200 draws on topology {self.topology.name!r} at "
            f"prior_p={cfg.prior_p:.3f}")
