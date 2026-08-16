"""PPO agent for single-agent active causal discovery.

Deliberately plain. A feedforward actor-critic, a clipped surrogate objective, GAE, and
nothing else -- no recurrence, no exploration bonus, no reward shaping. Each of those was
a lever in the previous codebase and none of them earned its place.

**No recurrence, on purpose.** Under observation condition A the agent sees the exact
posterior, which is a sufficient statistic for everything observed so far: nothing in the
history adds information beyond it. That makes the problem a proper MDP, so a feedforward
network is enough. It also structurally removes the saturating running-average state that
was diagnosed as the cause of the previous project's greedy-policy collapse -- a posterior
sharpens with evidence, it does not saturate. Under condition B (edge marginals) the
representation is lossy and the problem is formally partially observed, so recurrence could
help there; that is left as a measured question rather than assumed.

**Reward: +1 on identification, minus a small cost per step.** A shortest-path objective.
Nothing is added to it, which means no potential-based-shaping analysis is needed to know
what the optimal policy is (see docs/THEORY_NOTES.md #8 for why that mattered before).

**Action space is `d + 1`**: one action per node, plus pass. Passing is a real decision --
it ends the episode -- so "should I act at all" is something the agent must learn rather
than something the action space decides for it. It is also what makes under-acting
measurable, which is a hard-fail criterion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig


@dataclass
class PPOConfig:
    observation: str = "posterior"     # or "edge_marginals"
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    # Lowered from 0.01 after the first training run: entropy plateaued at 1.09 against a
    # 1.386 maximum, so the policy stayed near-uniform and argmax was arbitrary -- the
    # deterministic policy always picked the same node regardless of belief. That is the
    # same greedy-collapse signature as the previous project, reproduced here from an
    # exploration bonus that never decayed.
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs_per_update: int = 4
    minibatch: int = 256
    episodes_per_update: int = 32
    total_episodes: int = 6000
    step_cost: float = 0.05
    seed: int = 0

    # -- two diagnostic levers, both added to test one measured failure ----------------
    #
    # The agent learns NOT TO PASS within ~1500 episodes and then never learns WHERE to
    # intervene, settling at exactly random-policy cost. The suspected reason is that
    # pass-versus-act is a large, consistent contrast in return while which-node is a small
    # one, and both share a single batch-wide advantage normalisation -- so the small
    # contrast is a small share of the normalised advantage. These are the two ways to
    # attack that directly.

    # Drop `pass` from the action space entirely. If the mechanism above is right, removing
    # the large contrast should let the small one dominate. NOTE: an agent without `pass`
    # satisfies the under-acting criterion by construction, so that check becomes VACUOUS
    # for it and must not be read as evidence -- exactly the trap that produced a retracted
    # oracle-agreement figure earlier in this project.
    allow_pass: bool = True

    # Potential-based shaping on the posterior's entropy: F = gamma*phi(s') - phi(s) with
    # phi = -H(posterior)/log(n_dags), i.e. higher when belief is sharper. This gives a
    # dense per-step signal about whether THIS intervention was informative, which is
    # precisely the credit assignment the sparse terminal reward fails to deliver.
    #
    # Policy-invariant by construction (Ng, Harada & Russell 1999): any potential-based
    # term leaves the optimal policy unchanged, whatever its scale. That is why this is
    # admissible here when arbitrary reward shaping is not -- it cannot invent a better
    # policy that the unshaped objective would not also prefer. phi is forced to 0 at
    # terminal states, which the guarantee requires for episodic tasks.
    shaping_coef: float = 0.0

    # "flat" is the dense MLP; "pernode" is the permutation-equivariant scorer that matches
    # the oracle's own structure (see PerNodeActorCritic). Added after a supervised probe
    # showed the flat network reaches only 0.42 accuracy predicting the oracle's choice
    # even with full supervision -- evidence that the architecture, not the reward or the
    # exploration, is what cannot express the mapping.
    arch: str = "flat"
    # Rounds of neighbour aggregation in the per-node scorer. 1 is the network behind the
    # d=4/5/6 results; higher values let a node's embedding reach further, which is the
    # candidate explanation for the probe's ~0.89 ceiling. Ignored by arch="flat".
    layers: int = 1


class ActorCritic(nn.Module):
    """Two-headed MLP. Shared trunk, separate policy and value heads."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        # Small final-layer policy weights so the initial policy is close to uniform.
        # Without this the agent can start almost deterministic and never explore.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


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

    def _neighbour_pairs(self, obs: torch.Tensor) -> torch.Tensor:
        """[batch, d, d-1, 2] -- for each node i and neighbour j, the pair (i->j, j->i).

        The flat layout is d(d-1) off-diagonal marginals in row-major order, then the
        budget, then (optionally) d intervention counts.
        """
        d = self.d
        batch = obs.shape[0]
        mask = ~torch.eye(d, dtype=torch.bool, device=obs.device)

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
        d = self.d
        mask = ~torch.eye(d, dtype=torch.bool, device=device)
        return torch.arange(d, device=device).repeat(d, 1)[mask].view(d, d - 1)

    def _node_features(self, obs: torch.Tensor) -> torch.Tensor:
        """[batch, d, per_node_features], pooled over neighbours so node order cannot leak."""
        d = self.d
        batch = obs.shape[0]

        embedded = self.edge_encoder(self._neighbour_pairs(obs))   # [b, d, d-1, edge_hidden]
        pooled = torch.cat([embedded.mean(dim=2), embedded.max(dim=2).values], dim=-1)

        budget = obs[:, d * (d - 1)].view(batch, 1, 1).expand(batch, d, 1)
        parts = [pooled, budget]
        if self.include_counts:
            parts.append(obs[:, d * (d - 1) + 1:].view(batch, d, 1))
        return torch.cat(parts, dim=-1)

    def forward(self, obs: torch.Tensor):
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)

        embeddings = self.node_encoder(self._node_features(obs))   # [batch, d, hidden]

        # Rounds 2..k. Empty at layers=1, so this loop does not execute and the forward
        # pass is the original one instruction for instruction.
        if self.rounds:
            pairs = self._neighbour_pairs(obs)                     # [b, d, d-1, 2]
            index = self._neighbour_index(obs.device)              # [d, d-1]
            for block in self.rounds:
                neighbours = embeddings[:, index]                  # [b, d, d-1, hidden]
                messages = block["message"](
                    torch.cat([neighbours, pairs], dim=-1))
                pooled_messages = torch.cat(
                    [messages.mean(dim=2), messages.max(dim=2).values], dim=-1)
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


def action_to_env(action_index: int, d: int) -> int:
    """Index `d` is pass; indices `0..d-1` are node interventions.

    With `allow_pass=False` the action space is only `d` wide, so index `d` never occurs
    and every action maps straight through to a node.
    """
    return PASS_ACTION if action_index == d else action_index


class PPOAgent:
    """Trains and acts. `as_policy()` yields a callable usable by `sa.evaluate`."""

    def __init__(self, env_config: EnvConfig, ppo_config: PPOConfig, space=None,
                 backend=None):
        self.env_config = env_config
        self.cfg = ppo_config
        self.backend = backend
        torch.manual_seed(ppo_config.seed)
        self.env = (backend.make_env() if backend is not None
                    else CausalDiscoveryEnv(env_config, space=space))

        self.d = env_config.d
        self.n_actions = self.d + 1 if ppo_config.allow_pass else self.d
        # Normaliser for the shaping potential: entropy of a uniform posterior, so phi
        # lands in [-1, 0] regardless of d and shaping_coef means the same thing at every
        # problem size.
        # None on the DP path: the potential is the entropy of a distribution over
        # graphs, and that array does not exist at d=7. Shaping measured dead in the
        # Phase 2 sweep, so this refuses rather than quietly optimising something else.
        self._max_entropy = (backend.max_entropy if backend is not None
                             else float(np.log(self.env.space.n_dags)))
        if ppo_config.shaping_coef and self._max_entropy is None:
            raise ValueError(
                "shaping_coef requires the enumerated posterior, which does not exist on "
                "the subset-DP path. Measured 'dead' in the Phase 2 sweep; set it to 0.")
        self.obs_dim = self.env.observation_dim[ppo_config.observation]
        if ppo_config.arch == "pernode":
            if ppo_config.observation != "edge_marginals":
                raise ValueError(
                    "arch='pernode' requires observation='edge_marginals'; the exact "
                    "posterior has no per-node factorisation to exploit"
                )
            self.net = PerNodeActorCritic(
                self.d, ppo_config.hidden, env_config.include_counts,
                ppo_config.allow_pass, ppo_config.layers,
            )
        elif ppo_config.arch == "flat":
            self.net = ActorCritic(self.obs_dim, self.n_actions, ppo_config.hidden)
        else:
            raise ValueError(f"unknown arch {ppo_config.arch!r}")
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=ppo_config.lr)
        self.rng = np.random.default_rng(ppo_config.seed)
        self.history: List[dict] = []

    # -- acting --------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        return self.env.observation(self.cfg.observation)

    def _potential(self, result) -> float:
        """phi(s) = -H(posterior) / log(n_dags), in [-1, 0]; higher means sharper belief.

        Normalised by the uniform-posterior entropy so that `shaping_coef` carries the same
        meaning at every d, rather than silently scaling with the size of the graph space.
        """
        posterior = np.asarray(result.posterior, dtype=np.float64)
        nonzero = posterior[posterior > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
        return -entropy / self._max_entropy

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        logits, value = self.net(torch.as_tensor(obs, dtype=torch.float32))
        if deterministic:
            action = int(torch.argmax(logits).item())
            return action, torch.tensor(0.0), value
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), value

    # -- training ------------------------------------------------------------------

    def _collect(self, n_episodes: int) -> dict:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        solved, lengths = [], []

        for _ in range(n_episodes):
            seed = int(self.rng.integers(1 << 31))
            result = self.env.reset(seed=seed)
            # A graph already identified from observational data is a valid episode with
            # nothing to do; skip it rather than training on a zero-length trajectory.
            if result.done:
                solved.append(float(result.identified))
                lengths.append(0)
                continue

            while not result.done:
                obs = self._observe()
                # Only when it will be used. Computing it unconditionally costs an entropy
                # over the whole DAG space every step, and on the DP path there is no such
                # array at all -- so the unused value was a crash rather than mere waste.
                potential = self._potential(result) if self.cfg.shaping_coef else 0.0
                action, logp, value = self.act(obs)
                result = self.env.step(action_to_env(action, self.d))

                reward = -self.cfg.step_cost
                if result.identified:
                    reward += 1.0
                if self.cfg.shaping_coef:
                    # F = gamma * phi(s') - phi(s), with phi forced to 0 at terminal
                    # states as the policy-invariance guarantee requires.
                    next_potential = 0.0 if result.done else self._potential(result)
                    reward += self.cfg.shaping_coef * (
                        self.cfg.gamma * next_potential - potential
                    )

                obs_buf.append(obs)
                act_buf.append(action)
                logp_buf.append(float(logp))
                val_buf.append(float(value))
                rew_buf.append(reward)
                done_buf.append(float(result.done))

            solved.append(float(result.identified))
            lengths.append(result.n_interventions)

        return {
            "obs": np.array(obs_buf, dtype=np.float32),
            "actions": np.array(act_buf, dtype=np.int64),
            "logp": np.array(logp_buf, dtype=np.float32),
            "values": np.array(val_buf, dtype=np.float32),
            "rewards": np.array(rew_buf, dtype=np.float32),
            "dones": np.array(done_buf, dtype=np.float32),
            "solve_rate": float(np.mean(solved)) if solved else float("nan"),
            "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        }

    def _advantages(self, rewards, values, dones):
        """GAE. Bootstrapping is cut at episode boundaries via `dones`.

        Normalised over the whole batch AFTER computing returns, so returns stay on the
        reward's scale -- normalising before would corrupt the value targets, which is a
        bug this project has hit before.
        """
        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
            delta = rewards[t] + self.cfg.gamma * next_value * (1.0 - dones[t]) - values[t]
            running = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * running
            advantages[t] = running
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _update(self, batch: dict) -> dict:
        advantages, returns = self._advantages(batch["rewards"], batch["values"], batch["dones"])
        obs = torch.as_tensor(batch["obs"])
        actions = torch.as_tensor(batch["actions"])
        old_logp = torch.as_tensor(batch["logp"])
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        n = len(actions)
        losses = []
        for _ in range(self.cfg.epochs_per_update):
            order = torch.randperm(n)
            for start in range(0, n, self.cfg.minibatch):
                idx = order[start:start + self.cfg.minibatch]
                logits, values = self.net(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])

                ratio = torch.exp(logp - old_logp[idx])
                unclipped = ratio * advantages_t[idx]
                clipped = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * advantages_t[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((values - returns_t[idx]) ** 2).mean()
                entropy = dist.entropy().mean()

                loss = (policy_loss
                        + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy)

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimiser.step()
                losses.append((policy_loss.item(), value_loss.item(), entropy.item()))

        arr = np.array(losses)
        return {"policy_loss": float(arr[:, 0].mean()),
                "value_loss": float(arr[:, 1].mean()),
                "entropy": float(arr[:, 2].mean())}

    def train(self, verbose: bool = False) -> List[dict]:
        episodes_done = 0
        while episodes_done < self.cfg.total_episodes:
            batch = self._collect(self.cfg.episodes_per_update)
            episodes_done += self.cfg.episodes_per_update
            if len(batch["actions"]) == 0:
                continue  # every episode was pre-solved; nothing to learn from
            stats = self._update(batch)
            record = {"episodes": episodes_done,
                      "solve_rate": batch["solve_rate"],
                      "mean_length": batch["mean_length"], **stats}
            self.history.append(record)
            if verbose:
                print(f"ep {episodes_done:>6} | solve {record['solve_rate']:.2f} "
                      f"| len {record['mean_length']:.2f} | entropy {record['entropy']:.3f}")
        return self.history

    # -- deployment ------------------------------------------------------------------

    def as_policy(self, deterministic: bool = True):
        """A callable `(env, result) -> action` for `sa.evaluate`.

        `deterministic=True` is the deployment condition and the one every pass/fail
        number comes from. The sampled variant exists only to detect a collapse gap
        between the two -- the specific failure that ended the previous round.
        """
        def policy(env: CausalDiscoveryEnv, result) -> int:
            obs = env.observation(self.cfg.observation)
            action, _, _ = self.act(obs, deterministic=deterministic)
            return action_to_env(action, env.config.d)
        return policy
