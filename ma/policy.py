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
        self._extra = 2 + 1 + 1 + self.n_others * len(SIGNALS) + 1
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
        return core, disclosed, regime, signals, counts

    def forward(self, obs: torch.Tensor):
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        batch, k = obs.shape[0], self.d

        core, disclosed, regime, signals, counts = self._split(obs)
        base = self._node_features(core)                    # parent path, verbatim modules

        per_node_disclosed = torch.zeros(batch, k, 1, dtype=obs.dtype, device=obs.device)
        if self.n_shared and self.n_others:
            # NOT named `counts`: that name is the OWN-intervention block from _split,
            # and shadowing it here fed a [b, n_shared] tensor into the extras stack --
            # caught immediately by the shape mismatch, worth a comment so it stays caught.
            partner_hits = disclosed.view(batch, self.n_others, self.n_shared).sum(dim=1)
            for s_index, pos in enumerate(self.shared_positions):
                per_node_disclosed[:, pos, 0] = partner_hits[:, s_index]

        extras = torch.cat([
            self.role.unsqueeze(0).expand(batch, k, 2),
            per_node_disclosed,
            regime.unsqueeze(1).expand(batch, k, 1),
            signals.unsqueeze(1).expand(batch, k, signals.shape[-1]),
            counts.unsqueeze(-1),
        ], dim=-1)

        embeddings = self.node_encoder(torch.cat([base, extras], dim=-1))

        if self.rounds:                                     # parent's extra rounds, on core
            pairs = self._neighbour_pairs(core)
            index = self._neighbour_index(obs.device)
            for block in self.rounds:
                neighbours = embeddings[:, index]
                messages = block["message"](torch.cat([neighbours, pairs], dim=-1))
                pooled_messages = torch.cat(
                    [messages.mean(dim=2), messages.max(dim=2).values], dim=-1)
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

    def __init__(self, env: TwoAgentEnv, config: PPOConfig):
        self.env = env
        self.config = config
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        arch = getattr(env.config, "policy_arch", "mlp")
        if arch == "gnn":
            n_others = env.topology.n_agents - 1
            self.nets = {
                agent: RolePerNodeActorCritic(env.windows[agent], n_others,
                                              hidden=config.hidden,
                                              layers=config.gnn_layers)
                for agent in env.topology.agents}
        else:
            self.nets = {
                agent: ActorCritic(env.obs_size(agent), env.n_actions(agent), config.hidden,
                                   orthogonal_init=config.orthogonal_init)
                for agent in env.topology.agents}
        self.opts = {agent: torch.optim.Adam(net.parameters(), lr=config.lr)
                     for agent, net in self.nets.items()}
        self.history: List[dict] = []
        self.first_success_episode: Optional[int] = None

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
            while not result.done:
                obs = {a: self.env.observation(a) for a in self.env.topology.agents}
                picks = {a: self._act(a, obs[a], mask_pass) for a in self.env.topology.agents}
                result = self.env.step({a: picks[a][0] for a in self.env.topology.agents})
                new_potential = {a: -belief_entropy(result.beliefs[a]) for a in self.env.topology.agents}
                per_agent = result.info.get("agent_rewards")
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
                    buf["obs"].append(obs[agent])
                    buf["action"].append(action)
                    buf["logp"].append(logp)
                    buf["value"].append(value)
                    buf["reward"].append(shaped)
                    buf["done"].append(float(result.done))
                    entropies.append(entropy)
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

    def _advantages(self, buf: dict) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        rewards, values, dones = buf["reward"], buf["value"], buf["done"]
        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            next_value = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
            delta = rewards[t] + cfg.gamma * next_value - values[t]
            running = delta + cfg.gamma * cfg.gae_lambda * (0.0 if dones[t] else running)
            advantages[t] = running
        returns = advantages + values
        return advantages, returns

    def update(self, buffers: Dict[int, dict]) -> None:
        cfg = self.config
        for agent in self.env.topology.agents:
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

    def train(self, verbose: bool = False, on_update=None) -> List[dict]:
        """`on_update(record)` is called after every update, if given -- the hook the
        live telemetry hangs off. It is deliberately a plain callback: nothing in `ma/`
        imports a tracking library, so a broken or absent logger cannot take a run down."""
        cfg = self.config
        n_updates = max(1, cfg.total_episodes // cfg.episodes_per_update)
        for update in range(n_updates):
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
        for agent in env.topology.agents:
            if blob["obs_size"][agent] != env.obs_size(agent):
                raise ValueError(
                    "checkpoint is for a different environment: agent %s has obs_size %d, "
                    "this env has %d" % (agent, blob["obs_size"][agent], env.obs_size(agent)))
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
                                                  seed=blob["seed"]))
        for agent in env.topology.agents:
            learner.nets[agent].load_state_dict(blob["nets"][agent])
        return learner

