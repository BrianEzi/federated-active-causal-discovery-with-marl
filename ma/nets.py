"""Permutation-equivariant per-node policy network.

`PerNodeActorCritic`, moved VERBATIM from `sa/policy.py` on 2026-08-23. Not tidied,
not reformatted, not modernised: `tests/test_depth.py` requires that `layers=1`
reproduce the network behind the d=4/5/6 results EXACTLY -- same parameters, same RNG
draw order -- and any edit risks perturbing that silently.

Role features for the multi-agent case are added by the WRAPPER in `ma/policy.py`,
not by editing this class, for the same reason.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PerNodeActorCritic(nn.Module):
    """Scores every node with the SAME small network, then reads the logits off.

    The flat `ActorCritic` maps d(d-1) edge marginals to d+1 logits through a dense layer,
    so it must learn each node's score as a separate function of the whole vector, and
    learn from scratch that the nodes are interchangeable. The oracle's score for node i is
    a function of node i's own descendant structure -- the same function for every i. This
    architecture says so directly: build node i's features from its own row and column of
    the edge-marginal matrix, push them through a shared MLP, and take the output as node
    i's logit.

    That makes the policy **permutation-equivariant**: relabel the nodes and the logits
    permute with them, which is true of the oracle and was not expressible before. It also
    cuts the parameters that must be learned per node to zero -- one scorer serves all d --
    and makes the network's width independent of d, so the same model form carries to d=6
    unchanged.

    Deliberately restricted to the `edge_marginals` observation. The exact posterior has no
    per-node factorisation to exploit, which is the whole reason the scalable
    representation is the interesting one.

    `layers` sets how many rounds of neighbour aggregation run. One round means a node's
    score sees only its immediate edges. The oracle's score depends on each node's
    DESCENDANTS -- reachability, which is inherently multi-hop -- so a single round is a
    plausible explanation for the supervised probe topping out near 0.89 rather than 1.0.
    Extra rounds let a node's embedding carry information from `layers` hops away.

    `layers=1` constructs exactly the network that produced the d=4/5/6 results: the extra
    round modules are created only when `layers > 1`, and only after every existing
    parameter has been initialised, so the RNG draw is untouched and the state dict is
    identical. `tests/test_depth.py` asserts this rather than assuming it.
    """

    def __init__(self, d: int, hidden: int = 128, include_counts: bool = False,
                 allow_pass: bool = True, layers: int = 1):
        super().__init__()
        self.d = d
        self.include_counts = include_counts
        self.allow_pass = allow_pass
        self.layers = int(layers)
        if self.layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")

        # Each NEIGHBOUR of node i is embedded from the pair (i->j, j->i), then those
        # embeddings are pooled. Pooling rather than concatenating in index order is what
        # makes this genuinely equivariant: a fixed-order neighbour vector reorders when
        # the nodes are relabelled, so an earlier version of this class was only equivariant
        # under permutations that happened to preserve neighbour ordering -- which is to
        # say, not equivariant. The test caught it.
        #
        # Mean and max are pooled together: mean carries the typical neighbour, max carries
        # the most extreme one, and a single statistic loses distinctions the score needs
        # (Deep Sets, Zaheer et al. 2017).
        edge_hidden = max(hidden // 4, 8)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, edge_hidden), nn.Tanh(),
            nn.Linear(edge_hidden, edge_hidden), nn.Tanh(),
        )
        per_node_features = 2 * edge_hidden + 1 + (1 if include_counts else 0)
        self.node_encoder = nn.Sequential(
            nn.Linear(per_node_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.node_score = nn.Linear(hidden, 1)
        # Value and the pass logit both depend on the whole state, so they read a pooled
        # summary. Mean-pooling keeps them permutation-INVARIANT, which is correct: how
        # good the state is, and whether to stop, do not depend on node labels.
        self.value_head = nn.Linear(hidden, 1)
        self.pass_head = nn.Linear(hidden, 1) if allow_pass else None

        nn.init.orthogonal_(self.node_score.weight, gain=0.01)
        nn.init.zeros_(self.node_score.bias)
        if self.pass_head is not None:
            nn.init.orthogonal_(self.pass_head.weight, gain=0.01)
            nn.init.zeros_(self.pass_head.bias)

        # Constructed LAST, and only when asked for. Every `nn.Linear` above draws from the
        # torch RNG at construction, so creating these earlier would shift the
        # initialisation of everything after them -- and `layers=1` has to reproduce the
        # network behind the d=4/5/6 results exactly, not merely have the same shape. An
        # empty ModuleList contributes nothing to the state dict.
        self.rounds = nn.ModuleList()
        for _ in range(self.layers - 1):
            self.rounds.append(nn.ModuleDict({
                # A message from neighbour j to node i is built from j's current embedding
                # together with the (i->j, j->i) marginals, so the edge itself keeps
                # influencing what propagates rather than only seeding the first round.
                "message": nn.Sequential(nn.Linear(hidden + 2, hidden), nn.Tanh()),
                # Mean and max pooled, as in the first round: same reason (Zaheer et al.
                # 2017), and it keeps every round permutation-equivariant.
                # Named "combine" rather than "update" because ModuleDict already has an
                # `update` method and registering that key raises.
                "combine": nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh()),
            }))

    def _offdiag_mask(self, device) -> torch.Tensor:
        """[d, d] boolean, True off the diagonal. Depends only on `d` and the device, and
        both forward paths rebuild it on every call otherwise -- pure constant work in the
        inner loop of a rollout that runs millions of forward passes."""
        cached = getattr(self, "_offdiag_cache", None)
        if cached is None or cached.shape[0] != self.d or cached.device != device:
            cached = ~torch.eye(self.d, dtype=torch.bool, device=device)
            self._offdiag_cache = cached
        return cached

    def _neighbour_pairs(self, obs: torch.Tensor) -> torch.Tensor:
        """[batch, d, d-1, 2] -- for each node i and neighbour j, the pair (i->j, j->i).

        The flat layout is d(d-1) off-diagonal marginals in row-major order, then the
        budget, then (optionally) d intervention counts.
        """
        d = self.d
        batch = obs.shape[0]
        mask = self._offdiag_mask(obs.device)

        matrix = torch.zeros(batch, d, d, dtype=obs.dtype, device=obs.device)
        matrix[:, mask] = obs[:, : d * (d - 1)]

        outgoing = matrix[:, mask].view(batch, d, d - 1)
        incoming = matrix.transpose(1, 2)[:, mask].view(batch, d, d - 1)
        return torch.stack([outgoing, incoming], dim=-1)

    def _neighbour_index(self, device) -> torch.Tensor:
        """[d, d-1] -- row i lists every node other than i, in ascending order.

        Used to gather neighbour embeddings for the extra rounds. The ordering is fixed,
        but nothing downstream depends on it: messages are pooled, exactly as in the first
        round, which is what keeps the added depth equivariant.
        """
        cached = getattr(self, "_index_cache", None)
        if cached is None or cached.shape[0] != self.d or cached.device != device:
            d = self.d
            mask = self._offdiag_mask(device)
            cached = torch.arange(d, device=device).repeat(d, 1)[mask].view(d, d - 1)
            self._index_cache = cached
        return cached

    def _node_features(self, obs: torch.Tensor, pairs: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """[batch, d, per_node_features], pooled over neighbours so node order cannot leak.

        `pairs` lets a caller that already built the [b, d, d-1, 2] neighbour tensor hand
        it in rather than have it rebuilt; at `layers > 1` the forward pass needs it twice.
        """
        d = self.d
        batch = obs.shape[0]

        if pairs is None:
            pairs = self._neighbour_pairs(obs)
        embedded = self.edge_encoder(pairs)                        # [b, d, d-1, edge_hidden]
        pooled = torch.cat([embedded.mean(dim=2), torch.amax(embedded, dim=2)], dim=-1)

        budget = obs[:, d * (d - 1)].view(batch, 1, 1).expand(batch, d, 1)
        parts = [pooled, budget]
        if self.include_counts:
            parts.append(obs[:, d * (d - 1) + 1:].view(batch, d, 1))
        return torch.cat(parts, dim=-1)

    def forward(self, obs: torch.Tensor):
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)

        pairs = self._neighbour_pairs(obs) if self.rounds else None
        embeddings = self.node_encoder(self._node_features(obs, pairs))   # [batch, d, hidden]

        # Rounds 2..k. Empty at layers=1, so this loop does not execute and the forward
        # pass is the original one instruction for instruction.
        if self.rounds:
            index = self._neighbour_index(obs.device)              # [d, d-1]
            for block in self.rounds:
                neighbours = embeddings[:, index]                  # [b, d, d-1, hidden]
                messages = block["message"](
                    torch.cat([neighbours, pairs], dim=-1))
                pooled_messages = torch.cat(
                    [messages.mean(dim=2), torch.amax(messages, dim=2)], dim=-1)
                embeddings = block["combine"](
                    torch.cat([embeddings, pooled_messages], dim=-1))

        logits = self.node_score(embeddings).squeeze(-1)           # [batch, d]

        pooled = embeddings.mean(dim=1)                            # [batch, hidden]
        if self.pass_head is not None:
            logits = torch.cat([logits, self.pass_head(pooled)], dim=-1)
        value = self.value_head(pooled).squeeze(-1)

        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value
