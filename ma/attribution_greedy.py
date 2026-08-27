"""A greedy baseline scored on the SAME task the learner is scored on.

WHY IT HAS TO EXIST. `UncertaintyGreedyAgent` counts unsure STRUCTURE claims and knows
nothing about attribution, while the attributed environment grades the learner on structure
AND attribution together. Comparing them is the same unfair comparison found on 2026-08-26
when the learner was the blindfolded one, with the handicap now on the other side.
`ProbeThenWorkAgent` addresses this, but it is a FIXED schedule that ignores the belief --
so between them the project has an adaptive baseline playing the wrong game and a
right-game baseline that cannot adapt. This is the missing cell: adaptive, and playing the
right game.

WHAT MADE IT POSSIBLE, measured 2026-08-27. The obvious objection is that attribution
evidence arrives only when a PARTNER probes privately, so an agent's own actions cannot
move its own attribution and the extra term would be constant in the action. That is false.
Per round, the drop in an agent's own attribution uncertainty:

    own private move   -> own attribution   1.265   (42% of rounds)
    own shared move    -> own attribution   0.473   (22%)
    partner private    -> my attribution    0.425   (23%)
    partner shared     -> my attribution    0.456   (20%)

An agent's own private probe advances its OWN attribution about three times more than a
partner's does. The belief is factored into (structures x attributions) per bidirected-pair
bucket, so pruning structures deletes whole buckets and the attributions hanging off them.
Attribution is therefore reachable by structural evidence, and a selfish rule has a real
argmax over it.

That measurement also bears on the cooperation claim: if an own private probe is primarily
SELF-serving for attribution, then budget share separates the motives even less well than
already recorded. It is not evidence against cooperation, but it is evidence that private
probes cannot be read as altruism.

STILL TRUTH-FREE. It reads `group_frequency` from its own belief -- the same object the
observation already exposes -- and never the true groups.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ma.baselines import UncertaintyGreedyAgent
from ma.env import VARY, TwoAgentEnv


class AttributionGreedyAgent(UncertaintyGreedyAgent):
    """Myopic targeting over unsure STRUCTURE claims and unsure ATTRIBUTION groups.

    `attribution_weight` scales the attribution term against the structure term. At 0.0 this
    is exactly `UncertaintyGreedyAgent`, which makes the weight a clean ablation rather than
    a second implementation to keep in step.
    """

    def __init__(self, agent: int, seed: int = 0, bar: float = 1.0,
                 attribution_weight: float = 1.0):
        super().__init__(agent, seed, bar=bar)
        self.attribution_weight = float(attribution_weight)

    def _unsure_groups(self, belief, k: int) -> np.ndarray:
        """Per window node, how many unsure latent groups name it as a child.

        A group is a candidate hidden cause and its `children` are window positions. An
        intervention on a child is what separates "one latent explains all three" from
        "three separate latents", so children are the nodes worth spending a round on.
        """
        counts = np.zeros(k)
        freqs = getattr(belief, "group_frequency", None)
        if not freqs:
            return counts
        for group, f in freqs.items():
            f = float(f)
            if max(f, 1.0 - f) >= self.bar:
                continue                      # settled either way; nothing to buy
            for child in group.children:
                if 0 <= int(child) < k:
                    counts[int(child)] += 1.0
        return counts

    def __call__(self, env: TwoAgentEnv, result) -> int:
        window = env.windows[self.agent]
        belief = window.belief.last
        if belief is None:
            return int(self.rng.integers(0, window.n_actions - 1))
        counts = self._unsure_touching(belief, window.k)
        if self.attribution_weight:
            counts = counts + self.attribution_weight * self._unsure_groups(belief, window.k)
        scores = {node: counts[window.pos[node]] for node in window.authority}
        best = max(scores.values())
        if best <= 0:
            return window.pass_index
        candidates = [n for n, s in scores.items() if s == best]
        node = int(self.rng.choice(candidates))
        return window.action_index(node, prefer=VARY)
