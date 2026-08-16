"""Per-agent baseline policies for the two-agent environment.

Each policy sees ONLY its own agent's belief and acts within its own authority. No policy
here has access to the other agent's belief, window, or action -- that would be CTDE, which
the supervisor's constraint rules out.

The greedy policy is the myopic expected-information-gain oracle, the same opponent the
single-agent work used, restricted to one agent's window. It is the thing a learned policy
has to beat, and it is beatable in principle because optimal sequential design is not
greedy design chained together.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ma.env import CLAMP, PASS_ACTION, VARY, TwoAgentEnv
from sa.graphs import build_graph_space
from sa.oracle import _partition_entropy


class RandomAgentPolicy:
    """Uniform over the agent's own authority. Never passes -- passing is a decision the
    learned policy gets to make, but a random baseline that sometimes stops early would
    confound 'chose badly' with 'chose to stop'."""

    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __call__(self, env: TwoAgentEnv, result) -> int:
        # Uniform over every (target, mode) pair, excluding PASS (the last entry).
        return int(self.rng.integers(env.views[self.name].n_actions - 1))


class PassPolicy:
    """Never acts. The control arm: what does this agent reach with observational data
    plus whatever the OTHER agent happens to do?"""

    def __init__(self, name: str, seed: int = 0):
        self.name = name

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def __call__(self, env: TwoAgentEnv, result) -> int:
        return env.views[self.name].n_actions - 1      # PASS is the last action


class GreedyAgentPolicy:
    """Myopic expected information gain over the agent's OWN hypothesis space.

    Two graphs that give the intervened node the same descendant set within the window are
    indistinguishable by that intervention, so the expected gain is the entropy of the
    partition the intervention induces on the current belief.

    Signatures depend only on the window size, so they are built once and reused.
    """

    def __init__(self, name: str, env: TwoAgentEnv, seed: int = 0):
        self.name = name
        view = env.views[name]
        self.view = view
        self.rng = np.random.default_rng(seed)

        k = view.k
        adjacency = view.dags > 0.5
        reach = adjacency.copy()
        for m in range(k):
            reach |= reach[:, :, m][:, :, None] & reach[:, m, :][:, None, :]
        bit = (1 << np.arange(k)).astype(np.int64)
        codes = reach.astype(np.int64) @ bit          # [n_dags, k]

        self.signatures = np.empty((view.n_dags, k), dtype=np.int32)
        self.n_groups = []
        for node in range(k):
            groups, inverse = np.unique(codes[:, node], return_inverse=True)
            self.signatures[:, node] = inverse.reshape(-1)
            self.n_groups.append(len(groups))

        # Only nodes the agent may actually intervene on are candidates. At (1,1,3) that
        # is the whole window, but it is not in general -- an agent has no authority over
        # the other agent's private nodes, which it cannot see anyway.
        self.candidates = [view.pos[node] for node in view.authority]
        # Greedy always chooses VARY. This is not an oversight -- it is the honest
        # consequence of the oracle's model. Expected information gain is computed over
        # the agent's OWN hypothesis space, in which the other agent's confounding cannot
        # even be represented, so CLAMP has no value the oracle can see. A myopic,
        # self-interested agent will therefore never clamp to help its partner.
        #
        # That is exactly the room a learned policy has to beat it.
        self.candidate_actions = [view.actions.index((node, VARY))
                                  for node in view.authority]

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def scores(self, belief: np.ndarray) -> np.ndarray:
        return np.array([
            _partition_entropy(self.signatures[:, local], belief, self.n_groups[local])
            for local in self.candidates
        ])

    def __call__(self, env: TwoAgentEnv, result) -> int:
        scores = self.scores(result.beliefs[self.name])
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return int(self.candidate_actions[int(self.rng.choice(best))])


def make_agent_baselines(env: TwoAgentEnv, name: str, seed: int = 0) -> dict:
    return {
        "random": RandomAgentPolicy(name, seed=seed),
        "greedy": GreedyAgentPolicy(name, env, seed=seed),
        "pass": PassPolicy(name),
    }
