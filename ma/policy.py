"""PHASE 5 -- independent PPO for the rebuilt two-agent environment.

No CTDE [supervisor constraint]: each agent has its own actor, its own critic, its own
optimiser, and sees only its own observation. Nothing is shared but the scalar reward,
which is necessary rather than incidental -- a selfish agent has no reason to clamp for its
partner, so a per-agent reward makes the target behaviour strictly dominated [U15].

THE FIRST TASK HERE IS NOT TRAINING, IT IS THE 1-IN-10 SEED COLLAPSE. sd 0.154 on a median
of 0.312, with one seed in ten degenerating into passing immediately. Three ordered
hypotheses, each with a lever in `PPOConfig` that isolates it:

  1. entropy collapse       policy entropy falls before any reward signal arrives.
                            Lever: `entropy_coef`, and `entropy_floor` traces it.
  2. PASS too attractive    with a step cost and a low initial solve rate, passing dominates
                            until the policy is good enough for the +1 to be reachable.
                            Lever: `mask_pass_updates`.
  3. reward never sampled   the +1 is never seen at all on the collapsed seed.
                            Diagnostic: `first_success_episode` in the trace.

`potential_shaping` implements the fix that follows from hypothesis 3 without changing the
task. With potential Phi(s) = -H(belief), the shaping term gamma*Phi(s') - Phi(s) is
POLICY-INVARIANT (Ng, Harada & Russell 1999): the optimal policy is provably unchanged and
only the gradient becomes informative. It also sharpens the headline claim rather than
weakening it -- greedy EIG becomes the myopic optimum of the agent's own reward, so
"beats greedy" becomes "beats the one-step optimum of its own objective".
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ma.env import SIGNALS, TwoAgentEnv
from ma.nets import PerNodeActorCritic

DEVICE = torch.device("cpu")


@dataclass
class PPOConfig:
    hidden: int = 128
    lr: float = 3e-4
    # FEDAVG. 0 keeps the historical pooled path, which CONCATENATES every site's raw
    # trajectories -- data pooling, strictly more centralised than gradient sharing. Any
    # positive value selects FedAvg over the shared network with that many local epochs per
    # site per round, and nothing but weights leaves a site. E=1 is the correctness check
    # (it should reproduce pooled closely); E>1 is the experiment, because cutting
    # communication by a factor of E is what FedAvg is for. See `_fedavg_update`.
    local_epochs: int = 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    # PORTED FROM sa/policy.py, 2026-08-22 -- NOT yet the default, deliberately. sa/
    # measured that at gain=default (PyTorch's), the actor head can start near-deterministic
    # and never explore ("Without this the agent can start almost deterministic and never
    # explore"), and separately measured entropy_coef=0.01 causing entropy to plateau at
    # 1.09 against a 1.386 maximum (near-uniform but stuck, argmax arbitrary) -- moved to
    # 0.003 to fix it. ma's own "1-in-10 seed collapse" investigation names entropy_coef as
    # hypothesis #1's lever and was fixed by the turn-budget mechanism instead (hypothesis
    # 3), so it is NOT confirmed this bites ma the same way -- hence a flag and a measured
    # comparison, not a silent default change. See docs/logs/MA_BUILD_LOG.md, 2026-08-22.
    orthogonal_init: bool = False
    # TURN-AWARE CREDIT (2026-08-27). Under a turn-taking protocol `env.step` forces every
    # agent except the active one to pass and DISCARDS its submitted move, but `collect`
    # stored a transition for every agent every round -- pairing an action that never
    # executed with a reward a partner caused. At n agents that is (n-1)/n of the batch.
    #
    # The measured fingerprint, on the other session's agent ladder (window and budget per
    # agent held fixed, so the SIGNAL per agent is constant at 3 real transitions while the
    # noise grows linearly): final entropy 0.467 / 0.699 / 1.645 / 1.814 at 2 / 3 / 6 / 8
    # agents -- monotonic in (n-1)/n. Four episodes' worth of extra training, a lower
    # entropy coefficient and a shared trunk all failed to move the 8-agent case, which is
    # what a structural noise floor looks like rather than a tuning problem.
    #
    # ON: one transition per agent per TURN, with rewards arriving while it is inactive
    # accrued FORWARD onto its most recent action -- a partner's probe really does sharpen
    # this agent's belief and that credit belongs to the action that preceded it.
    # OFF (default): byte-identical to the old behaviour, so no existing result moves.
    turn_aware_credit: bool = False
    value_coef: float = 0.5
    epochs: int = 4
    episodes_per_update: int = 16
    total_episodes: int = 4000
    seed: int = 0
    # Hypothesis 2's lever: PASS is unavailable for this many updates, so the policy cannot
    # settle into "do nothing" before it has ever seen the terminal reward.
    mask_pass_updates: int = 0
    # Hypothesis 3's fix. Potential-based, hence policy-invariant.
    potential_shaping: float = 0.0
    # Aggregation rounds for the GNN (policy_arch="gnn" on MAConfig). 2 rather than 1
    # because a node's usefulness depends on its DESCENDANTS -- multi-hop by nature; the
    # supervised probe topping out at 0.89 with layers=1 is the standing evidence.
    gnn_layers: int = 2
    # RETURN NORMALISATION (2026-08-28). The advantages are already standardised before the
    # policy loss, but the CRITIC's target is not: `F.mse_loss(values, ret)` runs on raw
    # returns, so its magnitude grows as the SQUARE of the return while the policy term
    # stays O(1). Per-agent return grows 1.66 to 11.86 from two agents to eight, which puts
    # roughly a 50x weight ratio between the two ends of the agent ladder -- and a hand-
    # picked `MAConfig.reward_scale` of 0.214 was measured to take eight agents from 0.110
    # to 0.653. This is that fix without the magic constant: divide rewards by a RUNNING
    # estimate of the discounted-return scale, so the critic sees an O(1) target at every
    # agent count and nothing is tuned per rung.
    #
    # Scaling only, never centring. Subtracting a mean from the reward changes the
    # optimal policy under a finite horizon with a terminal bonus -- it would pay an agent
    # for surviving steps. Dividing by a positive scalar leaves the argmax untouched.
    #
    # OFF by default: flag off is byte-identical, the same discipline turn_aware_credit was
    # held to, so no existing result moves.
    normalise_returns: bool = False



def _upgrade_checkpoint_keys(blob: dict) -> dict:
    """Translate pre-2026-08-22 checkpoints, which keyed agents by name, to integer keys.

    FORMAT VERSIONING, not a semantic shim. The n-agent refactor switched agents from the
    strings "A"/"B" to integers 0..n-1, and silently broke `load` for every checkpoint
    written before it -- including every policy behind the current headline numbers. The
    failure was a bare KeyError deep in the shape check, with nothing pointing at the cause.

    This is safe in a way `a_private`/`b_private` accessors were not: it reads a serialised
    artefact whose format is known and fixed, and the mapping ("A", "B") -> (0, 1) is exact
    because names never went past two agents. Nothing about live semantics changes.

    "Save the policies" was a lesson paid for once already, when ten trained pairs were
    evaluated and discarded. Saving them is not enough if they cannot be read back.
    """
    names = ("A", "B")
    for field in ("nets", "obs_size"):
        table = blob.get(field)
        if isinstance(table, dict) and any(k in table for k in names):
            blob[field] = {names.index(k) if k in names else k: v
                           for k, v in table.items()}
    return blob


class ActorCritic(nn.Module):
    """Deliberately small and feedforward. The observation is a belief summary, so the
    problem is close to a proper MDP and recurrence has nothing obvious to add; adding it
    would also reintroduce the saturating running state previously diagnosed as the cause
    of a greedy collapse."""

    def __init__(self, obs_size: int, n_actions: int, hidden: int,
                 orthogonal_init: bool = False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        # Ported from sa/policy.py: small-gain orthogonal init on the ACTION head only
        # (the value head is left at PyTorch's default there too) so the initial policy
        # starts close to uniform rather than risking a near-deterministic start that
        # never explores. Off by default -- see PPOConfig.orthogonal_init.
        if orthogonal_init:
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.zeros_(self.actor.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.body(obs)
        return self.actor(latent), self.critic(latent).squeeze(-1)


class RolePerNodeActorCritic(PerNodeActorCritic):
    """The GNN, made multi-agent: `PerNodeActorCritic` plus ROLE FEATURES and the routing
    of the extra observation. The wrapper lives here, not in `ma/nets.py` -- that file is
    verbatim-frozen by `tests/test_depth.py`, and this subclass leaves every parent module
    and its RNG draw order untouched except `node_encoder`, which it REPLACES with a wider
    one (a new architecture may draw fresh parameters; only the SA reproduction may not).

    WHY ROLES ARE FEATURES AND NOT LEARNED. The parent is permutation-equivariant over all
    nodes, but the multi-agent task is not: "clamp my own private node" -- the headline
    behaviour -- is not expressible by a network that cannot tell a private node from a
    shared one. `is_shared` and `has_authority` per node break the symmetry exactly where
    the task does, so equivariance holds WITHIN roles and not across them.

    ROUTING, per the plan: budget and the partner signals are GLOBAL (broadcast to every
    node); the disclosed shared-target vector is PER-NODE (a shared node carries how many
    partners just targeted it, a private node carries 0); the regime bit is global.

    ACTIONS: the parent scores every window node; the action space is the AUTHORITY subset
    plus PASS, in the environment's own action order. Selecting authority positions is the
    action mask -- a non-authority node simply has no logit to pick. Single intervention
    mode only (the arms train clamp-only or vary-only); two modes would need a mode head
    the parent does not have, and silently averaging them is exactly the kind of decision
    that has cost this project four bugs in a day.
    """

    def __init__(self, window, n_others: int, hidden: int = 128, layers: int = 2):
        if len(window.modes) != 1:
            raise NotImplementedError(
                "RolePerNodeActorCritic supports a single intervention mode; "
                f"got {window.modes}. A mode head is a design decision, not a default.")
        super().__init__(d=window.k, hidden=hidden, include_counts=False,
                         allow_pass=True, layers=layers)
        k = window.k
        self.n_others = int(n_others)
        self.n_shared = len(window.shared)
        self.shared_positions = [window.pos[n] for n in window.shared]
        self.authority_positions = [window.pos[n] for n in window.authority]

        role = np.zeros((k, 2), dtype=np.float32)
        for node in window.shared:
            role[window.pos[node], 0] = 1.0
        for node in window.authority:
            role[window.pos[node], 1] = 1.0
        self.register_buffer("role", torch.as_tensor(role))

        # role(2) + disclosed-count(1) + regime(1) + partner signals (global broadcast)
        # + own-intervention count (1, per node -- the "what have I already done" input
        # without which "touch each node once" is unlearnable; added 2026-08-25).
        # +2 for the per-node confounding/adjacency aggregates, ONLY when the window's
        # environment supplies them. The width has to follow the observation: making it
        # unconditional silently broke every checkpoint trained before the channels
        # existed (caught 2026-08-26 when the baselines would not load back).
        self._channels = bool(getattr(window, "_observe_channels", False))
        # +3 for the cumulative partner counts, when the environment supplies them: two
        # PER-NODE (how many partner interventions this shared node has absorbed, and how
        # many DISTINCT partners have touched it -- "twice by one agent" and "once by each
        # of two" are different coordination states and a single sum cannot tell them
        # apart) and one GLOBAL (how much private work partners have done in total, which
        # is not attached to any node this agent can see). Same conditional-width rule as
        # the belief channels: making it unconditional voids every earlier checkpoint.
        self._partner_counts = bool(getattr(window, "_observe_partner_counts", False))
        self._extra = (2 + 1 + 1 + self.n_others * len(SIGNALS) + 1
                       + (2 if self._channels else 0)
                       + (3 if self._partner_counts else 0))
        edge_hidden = max(hidden // 4, 8)
        self.node_encoder = nn.Sequential(
            nn.Linear(2 * edge_hidden + 1 + self._extra, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

    def _split(self, obs: torch.Tensor):
        """Slice the MA observation layout (see `TwoAgentEnv.observation`)."""
        k = self.d
        m = k * (k - 1)
        i = m + 1
        core = obs[:, :i]                                   # marginals + budget
        disclosed = obs[:, i:i + self.n_others * self.n_shared]
        i += self.n_others * self.n_shared
        regime = obs[:, i:i + 1]
        i += 1
        signals = obs[:, i:i + self.n_others * len(SIGNALS)]
        i += self.n_others * len(SIGNALS)
        counts = obs[:, i:i + k]
        i += k
        # Bidirected + adjacency upper triangles, when the environment supplies them
        # (`observe_belief_channels`). Empty otherwise, so the old layout still parses.
        width = k * (k - 1) if self._channels else 0
        channels = obs[:, i:i + width]
        i += width
        # Cumulative partner counts, [n_others, n_shared + 1] flattened. Empty otherwise.
        partner = obs[:, i:]
        return core, disclosed, regime, signals, counts, channels, partner

    def forward(self, obs: torch.Tensor):
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        batch, k = obs.shape[0], self.d

        core, disclosed, regime, signals, counts, channels, partner = self._split(obs)
        # Built once and reused by the extra rounds below; at layers=1 it is not needed
        # again, so it is not built.
        pairs = self._neighbour_pairs(core) if self.rounds else None
        base = self._node_features(core, pairs)                    # parent path, verbatim modules

        per_node_disclosed = torch.zeros(batch, k, 1, dtype=obs.dtype, device=obs.device)
        if self.n_shared and self.n_others:
            # NOT named `counts`: that name is the OWN-intervention block from _split,
            # and shadowing it here fed a [b, n_shared] tensor into the extras stack --
            # caught immediately by the shape mismatch, worth a comment so it stays caught.
            partner_hits = disclosed.view(batch, self.n_others, self.n_shared).sum(dim=1)
            for s_index, pos in enumerate(self.shared_positions):
                per_node_disclosed[:, pos, 0] = partner_hits[:, s_index]

        # Pair-shaped beliefs become PER-NODE features: how much confounding, and how much
        # adjacency, this node's pairs carry. Aggregating rather than flattening keeps the
        # encoder permutation-equivariant within roles, which is the whole reason the GNN
        # is here -- a flat 12-vector would tie parameters to node indices.
        per_node_channels = torch.zeros(batch, k, 2 if self._channels else 0,
                                        dtype=obs.dtype, device=obs.device)
        if self._channels and channels.shape[-1] == k * (k - 1):
            half = k * (k - 1) // 2
            rows, cols = torch.triu_indices(k, k, offset=1)
            for offset, block in enumerate((channels[:, :half], channels[:, half:])):
                dense = torch.zeros(batch, k, k, dtype=obs.dtype, device=obs.device)
                dense[:, rows, cols] = block
                dense[:, cols, rows] = block
                per_node_channels[:, :, offset] = dense.sum(dim=2) / max(k - 1, 1)

        # Cumulative partner history. The per-node pair goes to the shared nodes it is
        # about; the private total is global, because the agent is told THAT partners
        # worked privately and never WHERE, so there is no node to attach it to.
        per_node_partner = torch.zeros(batch, k, 2 if self._partner_counts else 0,
                                       dtype=obs.dtype, device=obs.device)
        partner_private = torch.zeros(batch, 1 if self._partner_counts else 0,
                                      dtype=obs.dtype, device=obs.device)
        if self._partner_counts and self.n_others and partner.shape[-1]:
            table = partner.view(batch, self.n_others, self.n_shared + 1)
            shared_table = table[:, :, :self.n_shared]
            total = shared_table.sum(dim=1)                       # [b, n_shared]
            distinct = (shared_table > 0).to(obs.dtype).sum(dim=1)
            for s_index, pos in enumerate(self.shared_positions):
                per_node_partner[:, pos, 0] = total[:, s_index]
                per_node_partner[:, pos, 1] = distinct[:, s_index] / max(self.n_others, 1)
            partner_private[:, 0] = table[:, :, -1].sum(dim=1)

        extras = torch.cat([
            self.role.unsqueeze(0).expand(batch, k, 2),
            per_node_disclosed,
            regime.unsqueeze(1).expand(batch, k, 1),
            signals.unsqueeze(1).expand(batch, k, signals.shape[-1]),
            counts.unsqueeze(-1),
            per_node_channels,
            per_node_partner,
            partner_private.unsqueeze(1).expand(batch, k, partner_private.shape[-1]),
        ], dim=-1)

        embeddings = self.node_encoder(torch.cat([base, extras], dim=-1))

        if self.rounds:                                     # parent's extra rounds, on core
            index = self._neighbour_index(obs.device)
            for block in self.rounds:
                neighbours = embeddings[:, index]
                messages = block["message"](torch.cat([neighbours, pairs], dim=-1))
                pooled_messages = torch.cat(
                    [messages.mean(dim=2), torch.amax(messages, dim=2)], dim=-1)
                embeddings = block["combine"](
                    torch.cat([embeddings, pooled_messages], dim=-1))

        node_logits = self.node_score(embeddings).squeeze(-1)          # [batch, k]
        pooled = embeddings.mean(dim=1)
        logits = torch.cat([node_logits[:, self.authority_positions],
                            self.pass_head(pooled)], dim=-1)
        value = self.value_head(pooled).squeeze(-1)

        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


class PortableRoleActorCritic(RolePerNodeActorCritic):
    """The same network, freed of the window it was built for.

    WHAT ACTUALLY TIED A CHECKPOINT TO ONE WINDOW. Nothing in the parent's learned
    parameters depends on `k`: the edge encoder maps a single (i->j, j->i) pair, the node
    encoder maps one node's feature vector, and the score, value and pass heads read one
    node embedding or one pooled summary. Every width is per-NODE or per-PAIR, so the
    parameter count is already independent of the window size. Two things tied it down
    anyway, and both are fixed here:

      1. `role`, a [k, 2] BUFFER, which lands in the state dict and so pins its shape.
         Registered non-persistently here and rebuilt from whatever window the net is
         currently bound to.
      2. The partner blocks -- disclosed targets, signals, and cumulative counts -- were
         CONCATENATED in agent order, so the encoder width grew with the number of agents
         and the layout encoded partner identity by position. Pooled here (mean and max
         together, Deep Sets) so the width is fixed and the treatment is permutation-
         INVARIANT over partners. That is the right symmetry: a priori partners are
         interchangeable, and a policy that behaves differently towards "partner 0" and
         "partner 1" because of their index is fitting the labelling, not the task.

    The cost of (2) is real and worth stating: pooling discards WHICH partner did what, so
    per-partner attribution -- the clique-attribution question -- is not expressible by
    this variant. It keeps HOW MANY and HOW MUCH. `RolePerNodeActorCritic` remains the
    fixed-topology architecture, and remains the one to use when partner identity matters.

    `rebind(window, n_others)` points a trained net at a different window. Only the
    non-learned bookkeeping changes.
    """

    def __init__(self, window, n_others: int, hidden: int = 128, layers: int = 2):
        super().__init__(window, n_others, hidden=hidden, layers=layers)
        # Pooled partner features: 2 (mean, max) x [disclosed, signal one-hot, shared
        # counts, distinct-partner share, private counts]. Per-node where the feature is
        # about a node, global where it is not -- see `forward`.
        self._extra = (2 + 2 + 1 + 2 * len(SIGNALS) + 1
                       + (2 if self._channels else 0)
                       + (4 if self._partner_counts else 0))
        edge_hidden = max(hidden // 4, 8)
        self.node_encoder = nn.Sequential(
            nn.Linear(2 * edge_hidden + 1 + self._extra, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self._rebind(window, n_others)

    def _rebind(self, window, n_others: int) -> None:
        self.d = window.k
        self.n_others = int(n_others)
        self.n_shared = len(window.shared)
        self.shared_positions = [window.pos[n] for n in window.shared]
        self.authority_positions = [window.pos[n] for n in window.authority]
        role = np.zeros((window.k, 2), dtype=np.float32)
        for node in window.shared:
            role[window.pos[node], 0] = 1.0
        for node in window.authority:
            role[window.pos[node], 1] = 1.0
        # NON-PERSISTENT: the state dict must not carry a [k, 2] tensor, or loading into a
        # different window size fails on a shape mismatch -- the very thing this class is
        # for. It is derived from the window, so there is nothing to save.
        self.register_buffer("role", torch.as_tensor(role), persistent=False)

    def rebind(self, window, n_others: int) -> "PortableRoleActorCritic":
        if bool(getattr(window, "_observe_channels", False)) != self._channels:
            raise ValueError("cannot rebind across the belief-channel flag: the observation "
                             "layout differs, so the encoder would slice the wrong columns")
        if bool(getattr(window, "_observe_partner_counts", False)) != self._partner_counts:
            raise ValueError("cannot rebind across the partner-count flag: the observation "
                             "layout differs, so the encoder would slice the wrong columns")
        self._rebind(window, n_others)
        return self

    @staticmethod
    def _pool(blocks: torch.Tensor) -> torch.Tensor:
        """[b, n_others, width] -> [b, 2 * width], mean and max together.

        Both, not one: mean carries the typical partner and max carries the most extreme,
        and a single statistic loses distinctions the score needs (Zaheer et al. 2017 --
        the same argument the parent's neighbour pooling rests on).
        """
        if blocks.shape[1] == 0:
            return torch.zeros(blocks.shape[0], 2 * blocks.shape[2],
                               dtype=blocks.dtype, device=blocks.device)
        return torch.cat([blocks.mean(dim=1), torch.amax(blocks, dim=1)], dim=-1)

    def forward(self, obs: torch.Tensor):
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        batch, k = obs.shape[0], self.d

        core, disclosed, regime, signals, counts, channels, partner = self._split(obs)
        # Built once and reused by the extra rounds below; at layers=1 it is not needed
        # again, so it is not built.
        pairs = self._neighbour_pairs(core) if self.rounds else None
        base = self._node_features(core, pairs)

        # Disclosed shared targets: pooled over partners, then routed to the shared node
        # each column is about. Two per node (mean, max across partners).
        per_node_disclosed = torch.zeros(batch, k, 2, dtype=obs.dtype, device=obs.device)
        if self.n_shared and self.n_others:
            table = disclosed.view(batch, self.n_others, self.n_shared)
            mean_block = table.mean(dim=1)                 # [b, n_shared]
            max_block = torch.amax(table, dim=1)
            for s_index, pos in enumerate(self.shared_positions):
                per_node_disclosed[:, pos, 0] = mean_block[:, s_index]
                per_node_disclosed[:, pos, 1] = max_block[:, s_index]

        pooled_signals = self._pool(signals.view(batch, self.n_others, len(SIGNALS))
                                    if self.n_others else
                                    signals.view(batch, 0, len(SIGNALS)))

        per_node_channels = torch.zeros(batch, k, 2 if self._channels else 0,
                                        dtype=obs.dtype, device=obs.device)
        if self._channels and channels.shape[-1] == k * (k - 1):
            half = k * (k - 1) // 2
            rows, cols = torch.triu_indices(k, k, offset=1)
            for offset, block in enumerate((channels[:, :half], channels[:, half:])):
                dense = torch.zeros(batch, k, k, dtype=obs.dtype, device=obs.device)
                dense[:, rows, cols] = block
                dense[:, cols, rows] = block
                per_node_channels[:, :, offset] = dense.sum(dim=2) / max(k - 1, 1)

        per_node_partner = torch.zeros(batch, k, 2 if self._partner_counts else 0,
                                       dtype=obs.dtype, device=obs.device)
        partner_global = torch.zeros(batch, 2 if self._partner_counts else 0,
                                     dtype=obs.dtype, device=obs.device)
        if self._partner_counts and self.n_others and partner.shape[-1]:
            table = partner.view(batch, self.n_others, self.n_shared + 1)
            shared_table = table[:, :, :self.n_shared]
            for s_index, pos in enumerate(self.shared_positions):
                column = shared_table[:, :, s_index]
                per_node_partner[:, pos, 0] = column.sum(dim=1)
                per_node_partner[:, pos, 1] = (column > 0).to(obs.dtype).mean(dim=1)
            private = table[:, :, -1]
            partner_global[:, 0] = private.mean(dim=1)
            partner_global[:, 1] = torch.amax(private, dim=1)

        globals_ = torch.cat([regime, pooled_signals, partner_global], dim=-1)
        extras = torch.cat([
            self.role.unsqueeze(0).expand(batch, k, 2),
            per_node_disclosed,
            globals_.unsqueeze(1).expand(batch, k, globals_.shape[-1]),
            counts.unsqueeze(-1),
            per_node_channels,
            per_node_partner,
        ], dim=-1)

        embeddings = self.node_encoder(torch.cat([base, extras], dim=-1))

        if self.rounds:
            index = self._neighbour_index(obs.device)
            for block in self.rounds:
                neighbours = embeddings[:, index]
                messages = block["message"](torch.cat([neighbours, pairs], dim=-1))
                pooled_messages = torch.cat(
                    [messages.mean(dim=2), torch.amax(messages, dim=2)], dim=-1)
                embeddings = block["combine"](
                    torch.cat([embeddings, pooled_messages], dim=-1))

        node_logits = self.node_score(embeddings).squeeze(-1)
        pooled = embeddings.mean(dim=1)
        logits = torch.cat([node_logits[:, self.authority_positions],
                            self.pass_head(pooled)], dim=-1)
        value = self.value_head(pooled).squeeze(-1)

        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


def belief_entropy(marginals: np.ndarray) -> float:
    """H of the edge-marginal field, in nats -- the shaping potential's magnitude.

    Not the exact posterior entropy: that would need the full joint, which is the thing the
    DP exists to avoid materialising. Potential-based shaping stays policy-invariant for ANY
    potential function, so an approximate one costs nothing in correctness -- only in how
    informative the gradient is.
    """
    off = ~np.eye(marginals.shape[0], dtype=bool)
    p = np.clip(marginals[off], 1e-9, 1 - 1e-9)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).sum())


class IndependentPPO:
    """One PPO learner per agent. No shared parameters, gradients, or observations."""

    def __init__(self, env: TwoAgentEnv, config: PPOConfig, seed_torch: bool = True):
        self.env = env
        self.config = config
        # SEEDING THE GLOBAL TORCH STREAM IS A TRAINING CONCERN, NOT A LOADING ONE.
        # This ran unconditionally, so every `load()` reseeded torch globally and every
        # subsequent evaluation of that policy replayed ONE fixed sample path -- which is
        # why no confidence interval this project has reported included policy
        # stochasticity. `load` now passes seed_torch=False: the checkpoint overwrites
        # every weight anyway, so construction randomness is irrelevant there, and the
        # caller's sample path survives. Training still seeds, so runs stay reproducible.
        if seed_torch:
            torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        # Running discounted-return statistics for `normalise_returns`. Pooled over every
        # batch, shared across agents: the scale is a property of the TASK, and a per-agent
        # divisor would rescale the agents relative to each other, which is a reward change
        # rather than a normalisation.
        self._return_mean, self._return_var, self._return_count = 0.0, 0.0, 0
        # Weight-averaging rounds under FedAvg. The axis a communication-cost result is
        # plotted against -- E local epochs cuts this by a factor of E.
        self.communication_rounds = 0
        arch = getattr(env.config, "policy_arch", "mlp")
        n_others = env.topology.n_agents - 1
        if arch == "gnn_hybrid":
            # SHARED TRUNK, PER-AGENT HEADS. The middle ground between one network for
            # everyone and one per agent, and it separates two things the other two arms
            # confound: how to READ a causal belief (universal, and worth pooling everyone's
            # experience to learn) from what to DO about it (this agent's own role and
            # preferences). Measured motivation: `gnn_solo` did WORSE than full sharing at 8
            # agents (0.040 against 0.080), which points at sample efficiency rather than
            # representation -- solo gives each network an eighth of the data.
            self.nets = {
                agent: PortableRoleActorCritic(env.windows[agent], n_others,
                                               hidden=config.hidden,
                                               layers=config.gnn_layers)
                for agent in env.topology.agents}
            first = self.nets[env.topology.agents[0]]
            for agent in env.topology.agents[1:]:
                # Tie the trunk by sharing the MODULE OBJECTS, so gradients from every agent
                # land on one set of trunk parameters while each head stays private.
                self.nets[agent].edge_encoder = first.edge_encoder
                self.nets[agent].node_encoder = first.node_encoder
                self.nets[agent].rounds = first.rounds
            self.shared_net = None
            self.hybrid = True
            unique = {id(param): param
                      for net in self.nets.values() for param in net.parameters()}
            optimiser = torch.optim.Adam(list(unique.values()), lr=config.lr)
            self.opts = {agent: optimiser for agent in env.topology.agents}
        elif arch == "gnn_solo":
            # FULLY DECENTRALISED, and the point of it: portability comes from the
            # ARCHITECTURE, not from sharing weights. `PortableRoleActorCritic` scores one
            # variable at a time from features that never reference how many variables
            # exist or which specific partner acted (partner blocks are pooled), so the
            # same weights are meaningful at any window size or agent count. Giving each
            # agent its OWN complete copy -- own body, own scorer, own critic, trained only
            # on its own experience with no gradients crossing -- keeps every bit of that
            # and removes the one thing that was not decentralised.
            #
            # Weight sharing decides WHOSE EXPERIENCE trains the weights. The architecture
            # decides whether the weights MEAN anything at a different size. Two separate
            # choices; `gnn_portable` conflates them and this arm separates them.
            #
            # Testing at MORE agents than were trained needs more networks than exist. That
            # is a deployment question, not a training one: a new participant joins by
            # adopting a peer's published policy, which is a real federated scenario and
            # involves no shared training.
            self.nets = {
                agent: PortableRoleActorCritic(env.windows[agent], n_others,
                                               hidden=config.hidden,
                                               layers=config.gnn_layers)
                for agent in env.topology.agents}
            self.shared_net = None
            self.opts = {agent: torch.optim.Adam(net.parameters(), lr=config.lr)
                         for agent, net in self.nets.items()}
        elif arch == "gnn_portable":
            # ONE network, shared by every agent, because with the number of agents varying
            # across the training mixture there is no stable agent identity to give a net
            # to. This is PARAMETER SHARING with decentralised execution: each agent still
            # acts on its own observation only, and nothing centralises the critic or pools
            # observations, so the CTDE constraint is untouched. It is nonetheless a
            # departure from "one learner per agent" and must be reported as one -- the
            # arm is named separately for exactly that reason.
            shared = PortableRoleActorCritic(env.windows[env.topology.agents[0]], n_others,
                                             hidden=config.hidden, layers=config.gnn_layers)
            self.nets = {agent: shared for agent in env.topology.agents}
            self.shared_net: Optional[nn.Module] = shared
            optimiser = torch.optim.Adam(shared.parameters(), lr=config.lr)
            self.opts = {agent: optimiser for agent in env.topology.agents}
        elif arch == "gnn":
            self.nets = {
                agent: RolePerNodeActorCritic(env.windows[agent], n_others,
                                              hidden=config.hidden,
                                              layers=config.gnn_layers)
                for agent in env.topology.agents}
            self.shared_net = None
            self.opts = {agent: torch.optim.Adam(net.parameters(), lr=config.lr)
                         for agent, net in self.nets.items()}
        else:
            self.nets = {
                agent: ActorCritic(env.obs_size(agent), env.n_actions(agent), config.hidden,
                                   orthogonal_init=config.orthogonal_init)
                for agent in env.topology.agents}
            self.shared_net = None
            self.opts = {agent: torch.optim.Adam(net.parameters(), lr=config.lr)
                         for agent, net in self.nets.items()}
        self.hybrid = getattr(self, "hybrid", False)
        self.history: List[dict] = []
        self.first_success_episode: Optional[int] = None

    # -- portability --------------------------------------------------------------------

    def bind(self, env: TwoAgentEnv) -> "IndependentPPO":
        """Point a SHARED portable policy at a different environment, in place.

        Only legal for `gnn_portable`, and that is the point of the restriction: any other
        architecture has per-agent parameters whose shapes are tied to one window, so
        "binding" them to another environment would be a silent reinterpretation of learned
        weights rather than a rebind.
        """
        n_others = env.topology.n_agents - 1
        if self.shared_net is not None:
            self.shared_net.rebind(env.windows[env.topology.agents[0]], n_others)
            self.env = env
            self.nets = {agent: self.shared_net for agent in env.topology.agents}
            optimiser = next(iter(self.opts.values()))
            self.opts = {agent: optimiser for agent in env.topology.agents}
            return self
        if not all(isinstance(net, PortableRoleActorCritic) for net in self.nets.values()):
            raise ValueError("bind requires a portable architecture ('gnn_portable' or "
                             "'gnn_solo') -- others have window-shaped parameters")
        # ONE NETWORK PER AGENT, each rebound to that agent's own window. Where the new
        # environment has MORE agents than were trained, the extra slots adopt an existing
        # agent's policy -- onboarding, not training-time sharing: no gradient ever crossed.
        trained = list(self.nets.values())
        self.nets = {agent: trained[index % len(trained)].rebind(
                         env.windows[agent], n_others)
                     for index, agent in enumerate(env.topology.agents)}
        opts = list(self.opts.values())
        self.opts = {agent: opts[index % len(opts)]
                     for index, agent in enumerate(env.topology.agents)}
        self.env = env
        return self

    # -- rollout ------------------------------------------------------------------------

    def _act(self, agent: int, obs: np.ndarray, mask_pass: bool):
        # Rollout only -- gradients come from the recomputed forward pass in `update`, so
        # nothing here needs the graph. Without no_grad these floats drag a live autograd
        # graph into the buffers.
        with torch.no_grad():
            logits, value = self.nets[agent](torch.as_tensor(obs, dtype=torch.float32))
            if mask_pass:
                logits = logits.clone()
                logits[self.env.windows[agent].pass_index] = -1e9
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return (int(action), float(dist.log_prob(action)), float(value),
                    float(dist.entropy()))

    def collect(self, episodes: int, episode_offset: int, mask_pass: bool) -> Dict[str, dict]:
        cfg = self.config
        buffers = {agent: {k: [] for k in
                           ("obs", "action", "logp", "value", "reward", "done")}
                   for agent in self.env.topology.agents}
        entropies: List[float] = []
        solved = 0
        # Per-WINDOW identification alongside the joint rate. The joint number falls
        # exponentially in the number of agents whatever the policy does, so it is a poor
        # training signal to watch; this is the quantity the headroom was measured in.
        windows_identified = 0.0

        for episode in range(episodes):
            result = self.env.reset(seed=int(self.rng.integers(1 << 30)))
            potential = {a: -belief_entropy(result.beliefs[a]) for a in self.env.topology.agents}
            # Index of each agent's most recent OPEN transition, per episode. Rewards from
            # rounds it did not act in accrue here rather than becoming their own rows.
            open_at = {a: None for a in self.env.topology.agents}
            while not result.done:
                obs = {a: self.env.observation(a) for a in self.env.topology.agents}
                picks = {a: self._act(a, obs[a], mask_pass) for a in self.env.topology.agents}
                result = self.env.step({a: picks[a][0] for a in self.env.topology.agents})
                new_potential = {a: -belief_entropy(result.beliefs[a]) for a in self.env.topology.agents}
                per_agent = result.info.get("agent_rewards")
                # Whose move the protocol actually applied. None under simultaneous play,
                # where every agent acts and the old and new paths coincide exactly.
                active = result.info.get("active") if cfg.turn_aware_credit else None
                for agent in self.env.topology.agents:
                    # Per-agent pay when the environment provides it, otherwise the shared
                    # scalar. Nothing else in the loop changes.
                    shaped = (result.reward if per_agent is None
                              else float(per_agent[agent]))
                    if cfg.potential_shaping:
                        shaped += cfg.potential_shaping * (
                            cfg.gamma * new_potential[agent] - potential[agent])
                    action, logp, value, entropy = picks[agent]
                    buf = buffers[agent]
                    acted = (not cfg.turn_aware_credit) or active is None or agent == active
                    if acted:
                        buf["obs"].append(obs[agent])
                        buf["action"].append(action)
                        buf["logp"].append(logp)
                        buf["value"].append(value)
                        # Under turn-aware credit the row opens EMPTY and this round's
                        # reward is accrued just below, together with every later round
                        # until the agent acts again.
                        buf["reward"].append(0.0 if cfg.turn_aware_credit else shaped)
                        buf["done"].append(float(result.done))
                        open_at[agent] = len(buf["reward"]) - 1
                        entropies.append(entropy)
                    if cfg.turn_aware_credit and open_at[agent] is not None:
                        buf["reward"][open_at[agent]] += shaped
                        # `done` belongs to the agent's LAST row, whenever the episode ends.
                        buf["done"][open_at[agent]] = float(result.done)
                potential = new_potential
            windows_identified += result.info.get("identified_fraction", 0.0)
            if result.info["both_identified"]:
                solved += 1
                if self.first_success_episode is None:
                    self.first_success_episode = episode_offset + episode

        for agent in self.env.topology.agents:
            for key in buffers[agent]:
                buffers[agent][key] = np.asarray(buffers[agent][key], dtype=np.float32)
        return {"buffers": buffers, "entropy": float(np.mean(entropies)),
                "solve_rate": solved / episodes,
                "window_rate": windows_identified / episodes}

    # -- learning -----------------------------------------------------------------------

    def _return_scale(self, rewards: np.ndarray, dones: np.ndarray) -> float:
        """A running estimate of the discounted return's standard deviation.

        Estimated from the REWARDS alone, never from the critic's own predictions: a scale
        read off `buf["value"]` would be a function of the thing it is about to rescale, and
        the two would chase each other. The accumulator is the same backward discounted sum
        the returns use, reset at episode boundaries.

        RUNNING, not per batch. A per-batch divisor changes the effective learning rate
        every update and, early in training when almost every episode scores zero, divides
        by the noise in a handful of lucky episodes. The running variance is pooled across
        every batch seen so far, so it settles quickly and then moves slowly.
        """
        cfg = self.config
        accumulator, discounted = 0.0, np.empty_like(rewards)
        for t in range(len(rewards) - 1, -1, -1):
            accumulator = rewards[t] + cfg.gamma * (0.0 if dones[t] else accumulator)
            discounted[t] = accumulator
        count = len(discounted)
        if count == 0:
            return max(float(np.sqrt(self._return_var)), 1e-6)
        mean, var = float(discounted.mean()), float(discounted.var())
        total = self._return_count + count
        delta = mean - self._return_mean
        # Chan's parallel variance update: the batch is pooled with everything seen so far
        # rather than blended with a decay constant, which would be one more free parameter.
        new_mean = self._return_mean + delta * count / total
        m2 = (self._return_var * self._return_count + var * count
              + delta * delta * self._return_count * count / total)
        self._return_mean, self._return_var, self._return_count = new_mean, m2 / total, total
        # Floored, not clamped to 1.0: before any episode has scored, the returns really are
        # all zero and the honest scale is "unknown". Dividing by the floor leaves the
        # all-zero batch all-zero, which is the same no-op the raw path would produce.
        return max(float(np.sqrt(self._return_var)), 1e-6)

    def _advantages(self, buf: dict) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        rewards, values, dones = buf["reward"], buf["value"], buf["done"]
        if cfg.normalise_returns:
            # Scale BEFORE the GAE loop so rewards, values, returns and advantages are all
            # in one unit system. Scaling the returns afterwards instead would leave the
            # bootstrap `values[t + 1]` -- which the critic produced in scaled units -- being
            # added to an unscaled reward, and the two would silently disagree.
            rewards = rewards / self._return_scale(rewards, dones)
        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            next_value = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
            delta = rewards[t] + cfg.gamma * next_value - values[t]
            running = delta + cfg.gamma * cfg.gae_lambda * (0.0 if dones[t] else running)
            advantages[t] = running
        returns = advantages + values
        return advantages, returns

    def _fedavg_update(self, buffers: Dict[int, dict]) -> None:
        """FedAvg over one shared network: local epochs per site, then average the weights.

        WHAT IT REPLACES, AND WHY THAT MATTERS. The pooled path concatenates every site's
        RAW TRAJECTORIES into one buffer -- data pooling, which is strictly more centralised
        than gradient sharing and is precisely the thing a federated setting forbids. Here
        nothing but weights leaves a site.

        THE ONE KNOB IS `local_epochs` (E), AND IT IS THE WHOLE EXPERIMENT. E>1 is what
        FedAvg is FOR: it cuts communication rounds by a factor of E. The cost is client
        drift, and our clients are non-IID by construction -- each holds a different private
        block and plays a different role -- which is the regime where FedAvg degrades most.
        So the degradation is the result, and the axis to plot it against is communication
        rounds, not updates.

        E=1 IS NOT EQUIVALENT TO THE POOLED PATH, and it would be convenient but wrong to
        claim it is. Two differences survive. The pooled path takes `cfg.epochs` steps on
        the MERGED batch, so matching it needs E = cfg.epochs, not E = 1. And advantage
        normalisation differs by construction: pooled normalises over every site's
        experience together, FedAvg normalises within a site, because a site that had to
        share its advantage statistics would be leaking exactly what FedAvg exists to keep
        local. The clean equivalence check is therefore the SINGLE-SITE case, where there is
        nothing to average and nothing to normalise differently -- pinned in
        `tests/ma/test_fedavg.py`.

        WHY AVERAGING IS MEANINGFUL HERE AT ALL. Only because the architecture is portable:
        `PortableRoleActorCritic` scores one variable at a time from features that never
        reference how many variables exist, so every site's parameters carry the same
        semantics and a coordinate-wise mean is a policy rather than a mixture of
        incompatible representations. Averaging non-portable nets across sites with
        different window sizes would be meaningless, which is why this path requires
        `shared_net`.

        The optimiser is re-created per site per round on purpose: carrying one server-side
        optimiser across rounds would let its moment estimates accumulate across weight
        averages they were never computed for, which is a different algorithm.
        """
        import copy

        cfg = self.config
        agents = list(self.env.topology.agents)
        server = {key: value.detach().clone()
                  for key, value in self.shared_net.state_dict().items()}
        # Weight each site by how much experience it contributed, as FedAvg specifies.
        # Under round-robin every site holds the same share, but under random turn order it
        # does not -- and an unweighted mean would then quietly over-count a site that
        # happened to draw few turns.
        sizes = {a: max(1, len(buffers[a]["obs"])) for a in agents}
        total = float(sum(sizes.values()))

        accumulated = {key: torch.zeros_like(value) for key, value in server.items()}
        for agent in agents:
            self.shared_net.load_state_dict(server)          # every site starts from the
            local = torch.optim.Adam(self.shared_net.parameters(), lr=cfg.lr)  # same weights
            buf = buffers[agent]
            advantages, returns = self._advantages(buf)
            std = advantages.std()
            normed = (advantages - advantages.mean()) / (std + 1e-8)
            obs = torch.as_tensor(buf["obs"], dtype=torch.float32)
            actions = torch.as_tensor(buf["action"], dtype=torch.long)
            old_logp = torch.as_tensor(buf["logp"], dtype=torch.float32)
            adv = torch.as_tensor(normed, dtype=torch.float32)
            ret = torch.as_tensor(returns, dtype=torch.float32)

            for _ in range(cfg.local_epochs):
                logits, values = self.shared_net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions)
                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                loss = (-torch.min(ratio * adv, clipped * adv).mean()
                        + cfg.value_coef * F.mse_loss(values, ret)
                        - cfg.entropy_coef * dist.entropy().mean())
                local.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.shared_net.parameters(), 0.5)
                local.step()

            share = sizes[agent] / total
            for key, value in self.shared_net.state_dict().items():
                accumulated[key] += value.detach() * share

        self.shared_net.load_state_dict(accumulated)
        self.communication_rounds += 1

    def update(self, buffers: Dict[int, dict]) -> None:
        cfg = self.config
        if self.hybrid:
            # ONE optimiser step over everyone. Each agent's experience flows through its
            # OWN head and the shared trunk, so gradients must be accumulated across agents
            # and stepped once -- stepping per agent would apply the trunk update n times
            # and make the effective learning rate scale with the agent count.
            optimiser = next(iter(self.opts.values()))
            for _ in range(cfg.epochs):
                optimiser.zero_grad()
                for agent in self.env.topology.agents:
                    buf = buffers[agent]
                    advantages, returns = self._advantages(buf)
                    std = advantages.std()
                    normed = (advantages - advantages.mean()) / (std + 1e-8)
                    obs = torch.as_tensor(buf["obs"], dtype=torch.float32)
                    actions = torch.as_tensor(buf["action"], dtype=torch.long)
                    old_logp = torch.as_tensor(buf["logp"], dtype=torch.float32)
                    adv = torch.as_tensor(normed, dtype=torch.float32)
                    ret = torch.as_tensor(returns, dtype=torch.float32)
                    logits, values = self.nets[agent](obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(actions)
                    ratio = torch.exp(logp - old_logp)
                    clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                    loss = (-torch.min(ratio * adv, clipped * adv).mean()
                            + cfg.value_coef * F.mse_loss(values, ret)
                            - cfg.entropy_coef * dist.entropy().mean())
                    (loss / len(self.env.topology.agents)).backward()
                for net in self.nets.values():
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimiser.step()
            return
        if self.shared_net is not None and cfg.local_epochs > 0:
            return self._fedavg_update(buffers)
        if self.shared_net is not None:
            # One network, so ONE update over every agent's experience pooled. Stepping the
            # shared optimiser once per agent instead would make the effective learning rate
            # scale with the number of agents, and would make the update order matter --
            # both of which would show up as the mixture arm behaving differently at 4 and
            # at 8 agents for reasons that have nothing to do with the task.
            #
            # NOTE WHAT THIS IS: raw trajectories from every site are CONCATENATED. That is
            # data pooling, which is strictly more centralised than gradient sharing, and it
            # is not what "federated" means. `local_epochs > 0` selects FedAvg instead.
            merged = {key: np.concatenate([buffers[a][key]
                                           for a in self.env.topology.agents])
                      for key in buffers[self.env.topology.agents[0]]}
            buffers = {self.env.topology.agents[0]: merged}
        for agent in buffers:
            buf = buffers[agent]
            advantages, returns = self._advantages(buf)
            # Normalise over the whole batch, AFTER computing returns. Normalising before
            # would corrupt the critic's target -- the exact advantage-normalisation bug
            # that once put a floor of ~400 under the critic loss.
            std = advantages.std()
            normed = (advantages - advantages.mean()) / (std + 1e-8)

            obs = torch.as_tensor(buf["obs"], dtype=torch.float32)
            actions = torch.as_tensor(buf["action"], dtype=torch.long)
            old_logp = torch.as_tensor(buf["logp"], dtype=torch.float32)
            adv = torch.as_tensor(normed, dtype=torch.float32)
            ret = torch.as_tensor(returns, dtype=torch.float32)

            for _ in range(cfg.epochs):
                logits, values = self.nets[agent](obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions)
                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = F.mse_loss(values, ret)
                entropy = dist.entropy().mean()
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy)
                self.opts[agent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.nets[agent].parameters(), 0.5)
                self.opts[agent].step()

    def load_resume(self, path) -> int:
        """Restore a run mid-flight and return the update it should CONTINUE FROM.

        Weights alone do not restart a run. The optimiser moments are part of the
        trajectory, and so are both random streams -- the numpy generator that draws each
        episode's seed and torch's global stream that the policy samples from. Restoring
        three of the four gives a run that looks resumed and is not.

        `CheckpointWriter` writes this state AFTER update `u` has completed and after its
        own MI evaluation has consumed whatever randomness it consumes, so the correct
        continuation point is `u + 1` -- which is what this returns.
        """
        import torch

        state = torch.load(pathlib.Path(path), weights_only=False)
        for agent, weights in state["nets"].items():
            self.nets[int(agent)].load_state_dict(weights)
        for agent, moments in state["opts"].items():
            self.opts[int(agent)].load_state_dict(moments)
        torch.set_rng_state(state["torch_rng"])
        self.rng.bit_generator.state = state["numpy_rng"]
        self.history = list(state.get("history", []))
        for key, value in state.get("trainer", {}).items():
            setattr(self, key, value)
        return int(state["update"]) + 1

    # The trajectory state that is NEITHER weights, NOR optimiser moments, NOR a random
    # stream. It was missed on the first attempt at resume and the omission was silent:
    # `_return_mean/_var/_count` are RUNNING statistics accumulated over every update, and
    # `--normalise_returns` divides the rewards by them. A resumed run restarted them at
    # zero, so its first update scaled its rewards differently, and from there the two runs
    # were different runs. It cost nothing visible -- the rollout immediately after the
    # resume point still reproduced exactly, because the weights were right; the update
    # that followed it did not. Every sweep run passes --normalise_returns.
    #
    # Kept as a NAMED LIST rather than "everything on __dict__" so that adding a field to
    # the trainer is a decision about whether it belongs to the trajectory, not an
    # accident either way.
    RESUME_FIELDS = ("_return_mean", "_return_var", "_return_count",
                     "communication_rounds", "first_success_episode")

    def resume_state(self) -> dict:
        return {field: getattr(self, field) for field in self.RESUME_FIELDS
                if hasattr(self, field)}

    def train(self, verbose: bool = False, on_update=None,
              start_update: int = 0) -> List[dict]:
        """`on_update(record)` is called after every update, if given -- the hook the
        live telemetry hangs off. It is deliberately a plain callback: nothing in `ma/`
        imports a tracking library, so a broken or absent logger cannot take a run down.

        `start_update` continues an interrupted run; see `load_resume`. The episode offset
        is derived from the update index, so a resumed run draws the same episode seeds it
        would have drawn had it never stopped.
        """
        cfg = self.config
        n_updates = max(1, cfg.total_episodes // cfg.episodes_per_update)
        for update in range(int(start_update), n_updates):
            mask_pass = update < cfg.mask_pass_updates
            batch = self.collect(cfg.episodes_per_update,
                                 update * cfg.episodes_per_update, mask_pass)
            self.update(batch["buffers"])
            record = {"update": update, "entropy": batch["entropy"],
                      "solve_rate": batch["solve_rate"],
                      "window_rate": batch.get("window_rate"), "mask_pass": mask_pass}
            self.history.append(record)
            if on_update is not None:
                on_update(record)
            if verbose and update % 10 == 0:
                print(f"  update {update:4d}  entropy {record['entropy']:.3f}  "
                      f"solve {record['solve_rate']:.3f}  "
                      f"window {record['window_rate']:.3f}", flush=True)
        return self.history

    # -- use ----------------------------------------------------------------------------

    def policy(self, agent: int, deterministic: bool = False):
        def act(env: TwoAgentEnv, result) -> int:
            obs = torch.as_tensor(env.observation(agent), dtype=torch.float32)
            with torch.no_grad():
                logits, _ = self.nets[agent](obs)
            if deterministic:
                return int(torch.argmax(logits))
            return int(torch.distributions.Categorical(logits=logits).sample())

        act.reset = lambda seed=None: None
        return act

    def policies(self, deterministic: bool = False) -> Dict[int, object]:
        return {agent: self.policy(agent, deterministic) for agent in self.env.topology.agents}

    # -- persistence --------------------------------------------------------------------

    def save(self, path) -> None:
        """Write all agents' weights, with the shapes needed to rebuild them.

        Added after the fact, and the omission had a cost worth recording: ten trained
        two-agent policies were evaluated, reported, and then discarded, because nothing
        wrote them to disk. Reproducing any qualitative claim about what an agent LEARNED
        -- which variable it targets, when it clamps, what graph it ends up believing --
        meant retraining from scratch.

        The observation and action sizes are stored alongside the weights because they are
        derived from the topology, and a checkpoint that cannot say what environment it
        belongs to is a checkpoint you cannot trust.
        """
        import torch as _torch

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _torch.save({
            "nets": {agent: net.state_dict() for agent, net in self.nets.items()},
            "hidden": self.config.hidden,
            "seed": self.config.seed,
            "obs_size": {agent: self.env.obs_size(agent) for agent in self.env.topology.agents},
            "n_actions": {agent: self.env.n_actions(agent) for agent in self.env.topology.agents},
            "topology": self.env.topology.name,
            "score_rule": self.env.config.score_rule,
            "disclose_regime": self.env.config.disclose_regime,
            "policy_arch": getattr(self.env.config, "policy_arch", "mlp"),
            "belief_backend": getattr(self.env.config, "belief_backend", "exact"),
            # Provenance only -- `obs_size` already refuses a mismatched environment. These
            # are here so a checkpoint can SAY what it was trained with rather than leaving
            # it to be inferred from a width.
            "observe_belief_channels": getattr(
                self.env.config, "observe_belief_channels", False),
            "observe_partner_counts": getattr(
                self.env.config, "observe_partner_counts", False),
            "mode_by_role": getattr(self.env.config, "mode_by_role", False),
            # A portable checkpoint carries no window-shaped tensor, so `load` may skip the
            # obs_size check that pins an ordinary one to its environment.
            "portable": self.shared_net is not None,
            "claims_require_all_types": getattr(
                self.env.config, "claims_require_all_types", True),
        }, path)

    @classmethod
    def load(cls, path, env: TwoAgentEnv, config: Optional[PPOConfig] = None,
             allow_backend_transfer: bool = False):
        """Rebuild a trained set against `env`, refusing a mismatched environment.

        `allow_backend_transfer` permits loading a policy trained on ONE belief backend into
        an environment using another. That is normally void -- performance belongs to the
        (policy, backend) pair -- but it is exactly the transfer experiment: a policy
        trained in the deterministic idealisation, evaluated on noisy data, testing whether
        good experiment SELECTION is a structural skill that survives the noise. It must be
        asked for explicitly, and what was trained on what is reported alongside the number.
        """
        import torch as _torch

        blob = _torch.load(path, map_location=DEVICE, weights_only=False)
        blob = _upgrade_checkpoint_keys(blob)
        # A PORTABLE checkpoint is deliberately exempt from the obs_size check. Every
        # learned width in it is per-node or per-pair and the partner blocks are pooled, so
        # it has no tensor whose shape encodes the window size or the agent count -- which
        # is the whole reason it exists. The observation LAYOUT flags still have to match,
        # because the encoder slices the observation positionally, and `rebind` enforces
        # that; a mismatched layout raises there rather than silently mis-slicing.
        if not blob.get("portable", False):
            for agent in env.topology.agents:
                if blob["obs_size"][agent] != env.obs_size(agent):
                    raise ValueError(
                        "checkpoint is for a different environment: agent %s has obs_size "
                        "%d, this env has %d"
                        % (agent, blob["obs_size"][agent], env.obs_size(agent)))
        if blob.get("score_rule") != env.config.score_rule:
            # Cross-rule numbers are void -- a joint_conf-trained policy scored under
            # `subset` collapses below random. Performance belongs to the (policy, rule)
            # pair, so loading across rules is refused rather than warned about.
            raise ValueError("checkpoint was trained under rule %r, env uses %r"
                             % (blob.get("score_rule"), env.config.score_rule))
        # Same argument, two new axes (2026-08-24): performance belongs to the
        # (policy, backend, architecture) triple. A checkpoint predating the axes
        # (no keys in the blob) loads only into the historical defaults.
        blob_arch = blob.get("policy_arch", "mlp")
        env_arch = getattr(env.config, "policy_arch", "mlp")
        if blob_arch != env_arch:
            raise ValueError("checkpoint was trained with policy_arch %r, env uses %r"
                             % (blob_arch, env_arch))
        blob_backend = blob.get("belief_backend", "exact")
        env_backend = getattr(env.config, "belief_backend", "exact")
        if blob_backend != env_backend and not allow_backend_transfer:
            raise ValueError("checkpoint was trained on belief_backend %r, env uses %r "
                             "(pass allow_backend_transfer=True to run this deliberately)"
                             % (blob_backend, env_backend))
        learner = cls(env, config or PPOConfig(hidden=blob["hidden"],
                                                  seed=blob["seed"]),
                      seed_torch=False)          # see __init__: do not hijack evaluation
        if learner.shared_net is not None:
            # Every agent key holds the same weights; load one and every agent has it.
            first = next(iter(blob["nets"].values()))
            # `strict=False` tolerates a `role` buffer left over from a checkpoint written
            # before it became non-persistent. Nothing learned can be missing silently:
            # every learned module is constructed here, so an absent weight would surface
            # as an untrained head rather than a shape error.
            learner.shared_net.load_state_dict(first, strict=False)
            return learner
        for agent in env.topology.agents:
            learner.nets[agent].load_state_dict(blob["nets"][agent])
        return learner

