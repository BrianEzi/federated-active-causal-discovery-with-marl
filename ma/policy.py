"""Independent PPO for the two-agent case. One network per agent, no CTDE.

Each agent has its own actor-critic, sees only its own observation (edge marginals over its
own window plus its remaining budget), and emits an index into its own (target, mode)
action list. Nothing in the training loop lets one agent see the other's belief,
observation, action, or gradient. That is the supervisor's constraint, and it is enforced
structurally rather than by convention -- `_observe` takes an agent name and can only reach
that agent's view.

REWARD IS SHARED, and this is the one decision that genuinely changes the problem.
Recorded here rather than buried:

    A purely self-interested B has NO reason to clamp its private node for A's benefit.
    Clamping costs B a turn and teaches B nothing it wants -- the entire benefit lands in
    A's window, which B cannot even see. Under per-agent reward the coordination behaviour
    this whole design exists to study is strictly dominated, and no amount of training
    would produce it. It would not be a hard exploration problem; it would be a
    correctly-solved different problem.

    So both agents receive the same terminal reward, paid when BOTH have identified their
    own induced DAG. Each still pays for its own interventions, so budgets stay separate
    and the agents are not merged into one decision-maker.

    This is a cooperative team objective, not centralised training: the shared quantity is
    a scalar reward, not observations, parameters, or gradients. Two labs jointly mapping
    one system share the goal without sharing the data, which is the setting.

    The alternative -- per-agent reward plus an explicit "helping" bonus -- was rejected as
    circular: it would hand-code the answer the experiment is supposed to measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ma.env import MAConfig, TwoAgentEnv


@dataclass
class MAPPOConfig:
    lr: float = 1e-3
    hidden: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    episodes_per_update: int = 16
    total_episodes: int = 4000
    step_cost: float = 0.05
    seed: int = 0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        features = self.trunk(obs)
        return self.actor(features), self.critic(features).squeeze(-1)


class IndependentPPO:
    """Two independent PPO learners on one shared environment."""

    def __init__(self, env_config: MAConfig, ppo_config: MAPPOConfig):
        self.cfg = ppo_config
        self.env = TwoAgentEnv(env_config, seed=ppo_config.seed)
        self.env_config = env_config
        torch.manual_seed(ppo_config.seed)
        self.rng = np.random.default_rng(ppo_config.seed)

        self.names = ("A", "B")
        self.nets: Dict[str, ActorCritic] = {}
        self.opts: Dict[str, torch.optim.Optimizer] = {}
        for name in self.names:
            net = ActorCritic(self.env.observation_dim(name),
                              self.env.n_actions(name), ppo_config.hidden)
            self.nets[name] = net
            self.opts[name] = torch.optim.Adam(net.parameters(), lr=ppo_config.lr)

    # -- rollout -------------------------------------------------------------------

    def _act(self, name: str, obs: np.ndarray, deterministic: bool = False):
        tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.nets[name](tensor)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
            return action, 0.0, float(value.item())
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def collect(self, n_episodes: int, deterministic: bool = False) -> dict:
        cfg = self.cfg
        buf = {name: {"obs": [], "act": [], "logp": [], "val": [],
                      "rew": [], "done": []} for name in self.names}
        stats = {"solved": [], "length": [], "clamp_fraction": [],
                 "solved_A": [], "solved_B": []}

        for _ in range(n_episodes):
            result = self.env.reset(seed=int(self.rng.integers(1 << 30)))
            steps = 0
            clamps = 0
            actions_taken = 0

            while not result.done and steps < self.env_config.budget:
                chosen, logps, values, observations = {}, {}, {}, {}
                for name in self.names:
                    observations[name] = self.env.observation(name)
                    chosen[name], logps[name], values[name] = self._act(
                        name, observations[name], deterministic)

                for name in self.names:
                    target, mode = self.env.views[name].actions[chosen[name]]
                    if target != -1:
                        actions_taken += 1
                        if mode == "clamp":
                            clamps += 1

                result = self.env.step(chosen["A"], chosen["B"])
                steps += 1
                done = result.done or steps >= self.env_config.budget

                # SHARED terminal reward, per-agent step cost. See the module docstring.
                team = 1.0 if result.info["both_identified"] else 0.0
                for name in self.names:
                    target, _ = self.env.views[name].actions[chosen[name]]
                    cost = cfg.step_cost if target != -1 else 0.0
                    buf[name]["obs"].append(observations[name])
                    buf[name]["act"].append(chosen[name])
                    buf[name]["logp"].append(logps[name])
                    buf[name]["val"].append(values[name])
                    buf[name]["rew"].append((team if done else 0.0) - cost)
                    buf[name]["done"].append(done)

            stats["solved"].append(float(result.info["both_identified"]))
            stats["solved_A"].append(float(result.identified["A"]))
            stats["solved_B"].append(float(result.identified["B"]))
            stats["length"].append(steps)
            stats["clamp_fraction"].append(clamps / max(actions_taken, 1))

        return {"buf": buf, "stats": stats}

    # -- learning ------------------------------------------------------------------

    def _advantages(self, rewards, values, dones):
        cfg = self.cfg
        adv = np.zeros(len(rewards), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
            delta = rewards[t] + cfg.gamma * next_value - values[t]
            last = delta + cfg.gamma * cfg.lam * (0.0 if dones[t] else last)
            adv[t] = last
        returns = adv + np.asarray(values, dtype=np.float32)
        # Normalised over the whole batch, AFTER the recursion -- normalising inside the
        # loop was the single-agent `compute_gae` bug (memory: advantage normalisation).
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def _update(self, name: str, data: dict) -> dict:
        cfg = self.cfg
        obs = torch.as_tensor(np.asarray(data["obs"]), dtype=torch.float32)
        act = torch.as_tensor(np.asarray(data["act"]), dtype=torch.long)
        old_logp = torch.as_tensor(np.asarray(data["logp"]), dtype=torch.float32)
        adv_np, ret_np = self._advantages(data["rew"], data["val"], data["done"])
        adv = torch.as_tensor(adv_np, dtype=torch.float32)
        ret = torch.as_tensor(ret_np, dtype=torch.float32)

        net, opt = self.nets[name], self.opts[name]
        losses = {}
        for _ in range(cfg.epochs):
            logits, values = net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
            policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
            value_loss = F.mse_loss(values, ret)
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            losses = {"policy_loss": float(policy_loss.item()),
                      "value_loss": float(value_loss.item()),
                      "entropy": float(entropy.item())}
        return losses

    def save(self, path) -> None:
        """Persist both agents' networks and the config needed to rebuild them.

        Without this a trained pair is unrecoverable once the process exits, and the
        cross-rule evaluation -- score a policy trained under one belief rule against
        another -- would mean retraining every arm from scratch. The rule the policy was
        TRAINED under is stored alongside, because evaluating a policy under a different
        rule is the whole point and mixing the two up silently would be easy.
        """
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "nets": {name: self.nets[name].state_dict() for name in self.names},
            "obs_dims": {name: self.env.observation_dim(name) for name in self.names},
            "n_actions": {name: self.env.n_actions(name) for name in self.names},
            "hidden": self.cfg.hidden,
            "seed": self.cfg.seed,
            "trained_under_rule": self.env_config.score_rule,
            "topology": self.env_config.topology.name,
        }, path)

    def load(self, path) -> dict:
        """Restore both agents' networks. Returns the stored metadata so a caller can check
        which rule the policy was trained under before scoring it."""
        payload = torch.load(path, weights_only=False)
        for name in self.names:
            self.nets[name].load_state_dict(payload["nets"][name])
            self.nets[name].eval()
        return {k: v for k, v in payload.items() if k != "nets"}

    def train(self, verbose: bool = True) -> List[dict]:
        cfg = self.cfg
        history = []
        done_episodes = 0
        while done_episodes < cfg.total_episodes:
            batch = self.collect(cfg.episodes_per_update)
            done_episodes += cfg.episodes_per_update
            record = {"episodes": done_episodes,
                      "solved": float(np.mean(batch["stats"]["solved"])),
                      "solved_A": float(np.mean(batch["stats"]["solved_A"])),
                      "solved_B": float(np.mean(batch["stats"]["solved_B"])),
                      "length": float(np.mean(batch["stats"]["length"])),
                      "clamp_fraction": float(np.mean(batch["stats"]["clamp_fraction"]))}
            for name in self.names:
                for key, value in self._update(name, batch["buf"][name]).items():
                    record[f"{name}_{key}"] = value
            history.append(record)
            if verbose and (done_episodes % (cfg.episodes_per_update * 20) == 0):
                print(f"  ep {done_episodes:>6}  solved {record['solved']:.3f}"
                      f"  (A {record['solved_A']:.3f} B {record['solved_B']:.3f})"
                      f"  len {record['length']:.2f}"
                      f"  clamp {record['clamp_fraction']:.3f}"
                      f"  H_A {record['A_entropy']:.3f}", flush=True)
        return history
