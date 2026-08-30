"""Checkpointing must survive its own failures and never take a run down.

A checkpointing bug that raises costs the whole run; one that silently writes nothing costs
the whole run too, but quietly. Both are pinned here.
"""
import pathlib

import numpy as np
import pytest

from ma.checkpoints import CheckpointWriter, DEFAULT_SCHEDULE, default_schedule, mi_ratio
from ma.env import MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import federated_topology


def _env():
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(3, 2, 3), n_obs=40, n_int=10, budget=9,
        turn_order="round_robin", belief_backend="factored", action_modes=("vary",),
        claim_bar=1.0, reward_criterion="claims", policy_arch="gnn_portable",
        graph_model="sf", sf_m=2, episode_mix="confounded", vs_evidence="oracle"))


def _writer(tmp_path, ppo, env, **kw):
    return CheckpointWriter(ppo, env, tmp_path / "run.json", n_updates=250,
                            mi_episodes=2, **kw)


# -- the schedule -----------------------------------------------------------------------

def test_schedule_is_dense_early_because_that_is_where_diagnostics_resolve():
    """Entropy collapse and did-it-train-at-all resolve inside ~70 updates. A uniform
    every-100 schedule would have two points over a 250-update run and miss all of it."""
    points = default_schedule(250)
    assert sum(1 for p in points if p <= 70) >= 5
    assert sum(1 for p in points if p > 70) <= 4


def test_schedule_always_includes_the_final_update():
    for n in (30, 250, 1000):
        assert max(default_schedule(n)) == n - 1


def test_schedule_is_clipped_to_short_runs_and_never_exceeds_them():
    points = default_schedule(12)
    assert points and max(points) == 11
    assert all(p < 12 for p in points)


# -- the three kinds --------------------------------------------------------------------

def test_eval_checkpoints_are_written_only_on_schedule(tmp_path):
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0))
    writer = _writer(tmp_path, ppo, env, schedule=[2, 4], resume_every=0)
    for update in range(6):
        writer({"update": update, "entropy": 1.0, "solve_rate": 0.0})
    written = sorted(p.name for p in tmp_path.glob("run_u*.pt"))
    assert written == ["run_u0002.pt", "run_u0004.pt"]


def test_resume_state_rotates_and_self_prunes(tmp_path):
    """Otherwise a long run fills the disk with restart states nobody will use."""
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0))
    writer = _writer(tmp_path, ppo, env, schedule=[], resume_every=2, keep_resume=2)
    for update in range(1, 11):
        writer({"update": update})
    kept = sorted(p.name for p in tmp_path.glob("run_resume_u*.pt"))
    assert len(kept) == 2 and kept == ["run_resume_u0008.pt", "run_resume_u0010.pt"]


def test_resume_state_carries_what_weights_alone_cannot(tmp_path):
    """Optimiser moments and RNG positions are part of the trajectory; dropping them
    silently changes the run you resume into."""
    import torch
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0))
    writer = _writer(tmp_path, ppo, env, schedule=[], resume_every=1)
    writer({"update": 1})
    blob = torch.load(next(tmp_path.glob("run_resume_u*.pt")), weights_only=False)
    assert {"update", "nets", "opts", "torch_rng", "numpy_rng"} <= set(blob)


def test_best_is_tracked_by_mi_not_by_the_final_update(tmp_path, monkeypatch):
    """The final policy is not reliably the best -- entropy rises again late in measured
    runs. Selecting on reward would select on the thing being measured, so the criterion is
    the training-health gate instead."""
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0))
    scores = iter([0.1, 0.9, 0.2])
    monkeypatch.setattr("ma.checkpoints.mi_ratio", lambda *a, **k: next(scores))
    writer = _writer(tmp_path, ppo, env, schedule=[0, 1, 2], resume_every=0)
    for update in range(3):
        writer({"update": update})
    assert writer.best_update == 1                     # not the last
    assert (tmp_path / "run_best.pt").exists()


# -- it must never take a run down -------------------------------------------------------

def test_a_failing_checkpoint_write_is_swallowed_and_reported(tmp_path, monkeypatch):
    """A run that has spent hours of compute must not die because a disk write failed."""
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0))
    messages = []
    writer = _writer(tmp_path, ppo, env, schedule=[1], resume_every=0,
                     log=messages.append)
    monkeypatch.setattr(ppo, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
    writer({"update": 1})                              # must not raise
    assert any("FAILED" in m for m in messages)
    assert writer.written == []


def test_manifest_records_where_everything_went(tmp_path):
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0))
    writer = _writer(tmp_path, ppo, env, schedule=[1], resume_every=0)
    writer({"update": 1, "entropy": 1.0})
    manifest = writer.manifest()
    assert manifest["best_update"] == 1
    assert manifest["checkpoints"][0]["path"].endswith("run_u0001.pt")
    assert "mi_ratio" in manifest["checkpoints"][0]


def test_mi_ratio_is_a_ratio(tmp_path):
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0))
    value = mi_ratio(ppo, env, episodes=2, seed=0)
    assert 0.0 <= value <= 1.0
