"""PHASE 5 -- independent PPO for the rebuilt two-agent environment.

No CTDE [supervisor constraint]: each agent has its own actor, its own critic, its own
optimiser, and sees only its own observation. Nothing is shared but the scalar reward,
which is necessary rather than incidental -- a selfish agent has no reason to clamp for its
partner, so a per-agent reward makes the target behaviour strictly dominated [U15].

THE FIRST TASK HERE IS NOT TRAINING, IT IS THE 1-IN-10 SEED COLLAPSE. sd 0.154 on a median
of 0.312, with one seed in ten degenerating into passing immediately. Three ordered
hypotheses, each with a lever in `MA2PPOConfig` that isolates it:

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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ma.env2 import AGENTS, TwoAgentEnv2

DEVICE = torch.device("cpu")


@dataclass
class MA2PPOConfig:
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
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


class ActorCritic(nn.Module):
    """Deliberately small and feedforward. The observation is a belief summary, so the
    problem is close to a proper MDP and recurrence has nothing obvious to add; adding it
    would also reintroduce the saturating running state previously diagnosed as the cause
    of a greedy collapse."""

    def __init__(self, obs_size: int, n_actions: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.body(obs)
        return self.actor(latent), self.critic(latent).squeeze(-1)


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


class IndependentPPO2:
    """One PPO learner per agent. No shared parameters, gradients, or observations."""

    def __init__(self, env: TwoAgentEnv2, config: MA2PPOConfig):
        self.env = env
        self.config = config
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.nets = {
            name: ActorCritic(env.obs_size(name), env.n_actions(name), config.hidden)
            for name in AGENTS}
        self.opts = {name: torch.optim.Adam(net.parameters(), lr=config.lr)
                     for name, net in self.nets.items()}
        self.history: List[dict] = []
        self.first_success_episode: Optional[int] = None

    # -- rollout ------------------------------------------------------------------------

    def _act(self, name: str, obs: np.ndarray, mask_pass: bool):
        # Rollout only -- gradients come from the recomputed forward pass in `update`, so
        # nothing here needs the graph. Without no_grad these floats drag a live autograd
        # graph into the buffers.
        with torch.no_grad():
            logits, value = self.nets[name](torch.as_tensor(obs, dtype=torch.float32))
            if mask_pass:
                logits = logits.clone()
                logits[self.env.windows[name].pass_index] = -1e9
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return (int(action), float(dist.log_prob(action)), float(value),
                    float(dist.entropy()))

    def collect(self, episodes: int, episode_offset: int, mask_pass: bool) -> Dict[str, dict]:
        cfg = self.config
        buffers = {name: {k: [] for k in
                          ("obs", "action", "logp", "value", "reward", "done")}
                   for name in AGENTS}
        entropies: List[float] = []
        solved = 0

        for episode in range(episodes):
            result = self.env.reset(seed=int(self.rng.integers(1 << 30)))
            potential = {n: -belief_entropy(result.beliefs[n]) for n in AGENTS}
            while not result.done:
                obs = {n: self.env.observation(n) for n in AGENTS}
                picks = {n: self._act(n, obs[n], mask_pass) for n in AGENTS}
                result = self.env.step(picks["A"][0], picks["B"][0])
                new_potential = {n: -belief_entropy(result.beliefs[n]) for n in AGENTS}
                for name in AGENTS:
                    shaped = result.reward
                    if cfg.potential_shaping:
                        shaped += cfg.potential_shaping * (
                            cfg.gamma * new_potential[name] - potential[name])
                    action, logp, value, entropy = picks[name]
                    buf = buffers[name]
                    buf["obs"].append(obs[name])
                    buf["action"].append(action)
                    buf["logp"].append(logp)
                    buf["value"].append(value)
                    buf["reward"].append(shaped)
                    buf["done"].append(float(result.done))
                    entropies.append(entropy)
                potential = new_potential
            if result.info["both_identified"]:
                solved += 1
                if self.first_success_episode is None:
                    self.first_success_episode = episode_offset + episode

        for name in AGENTS:
            for key in buffers[name]:
                buffers[name][key] = np.asarray(buffers[name][key], dtype=np.float32)
        return {"buffers": buffers, "entropy": float(np.mean(entropies)),
                "solve_rate": solved / episodes}

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

    def update(self, buffers: Dict[str, dict]) -> None:
        cfg = self.config
        for name in AGENTS:
            buf = buffers[name]
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
                logits, values = self.nets[name](obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions)
                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = F.mse_loss(values, ret)
                entropy = dist.entropy().mean()
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy)
                self.opts[name].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.nets[name].parameters(), 0.5)
                self.opts[name].step()

    def train(self, verbose: bool = False) -> List[dict]:
        cfg = self.config
        n_updates = max(1, cfg.total_episodes // cfg.episodes_per_update)
        for update in range(n_updates):
            mask_pass = update < cfg.mask_pass_updates
            batch = self.collect(cfg.episodes_per_update,
                                 update * cfg.episodes_per_update, mask_pass)
            self.update(batch["buffers"])
            record = {"update": update, "entropy": batch["entropy"],
                      "solve_rate": batch["solve_rate"], "mask_pass": mask_pass}
            self.history.append(record)
            if verbose and update % 10 == 0:
                print(f"  update {update:4d}  entropy {record['entropy']:.3f}  "
                      f"solve {record['solve_rate']:.3f}", flush=True)
        return self.history

    # -- use ----------------------------------------------------------------------------

    def policy(self, name: str, deterministic: bool = False):
        def act(env: TwoAgentEnv2, result) -> int:
            obs = torch.as_tensor(env.observation(name), dtype=torch.float32)
            with torch.no_grad():
                logits, _ = self.nets[name](obs)
            if deterministic:
                return int(torch.argmax(logits))
            return int(torch.distributions.Categorical(logits=logits).sample())

        act.reset = lambda seed=None: None
        return act

    def policies(self, deterministic: bool = False) -> Dict[str, object]:
        return {name: self.policy(name, deterministic) for name in AGENTS}
