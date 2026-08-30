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


# =======================================================================================
# RESUME. Written since the checkpointing work, read since 30 Aug 2026 -- until then the
# `*_resume_uNNNN.pt` files were dead weight on exactly the long runs they exist for.
#
# The property that makes resume worth having is not "it starts again". It is that the
# resumed run is the SAME run: a 60-run sweep whose interrupted members follow a different
# trajectory from its uninterrupted ones is a sweep with an uncontrolled variable in it.
# So this asserts identity of the history, not merely that training continues.
# =======================================================================================
import numpy as np
import torch

from ma.checkpoints import CheckpointWriter
from ma.env import MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import federated_topology


def _env():
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(2, 2, 2), n_obs=40, n_int=12, budget=8,
        turn_order="round_robin", belief_backend="factored", action_modes=("vary",),
        claim_bar=1.0, reward_criterion="claims", policy_arch="gnn_portable",
        graph_model="sf", sf_m=2, episode_mix="confounded", vs_evidence="oracle"))


def _ppo(env, seed):
    return IndependentPPO(env, PPOConfig(total_episodes=48, episodes_per_update=8,
                                         epochs=2, hidden=16, seed=seed))


def _weights(ppo):
    return torch.cat([p.detach().flatten()
                      for agent in sorted(ppo.nets) for p in ppo.nets[agent].parameters()])


def test_a_resumed_run_reproduces_the_uninterrupted_one_exactly(tmp_path):
    torch.manual_seed(0)
    straight = _ppo(_env(), seed=0)
    straight.train(verbose=False)

    # The same run, stopped after update 2 and restarted from the state written there.
    torch.manual_seed(0)
    interrupted = _ppo(_env(), seed=0)
    writer = CheckpointWriter(interrupted, interrupted.env, tmp_path / "run.json",
                              n_updates=6, schedule=[], resume_every=2, keep_resume=4,
                              mi_episodes=2, seed=0)
    interrupted.train(verbose=False, on_update=writer, start_update=0)

    resume_state = tmp_path / "run_resume_u0002.pt"
    assert resume_state.exists(), "no resume state was written at update 2"

    torch.manual_seed(999)                     # deliberately WRONG: the state must win
    revived = _ppo(_env(), seed=123)           # deliberately a different seed, likewise
    start = revived.load_resume(resume_state)
    assert start == 3, f"resume should continue at the update AFTER the one saved, got {start}"
    revived.train(verbose=False, start_update=start)

    assert [r["update"] for r in revived.history] == [r["update"] for r in straight.history]
    for got, want in zip(revived.history, straight.history):
        assert got["entropy"] == want["entropy"], (
            f"update {want['update']}: resumed entropy {got['entropy']!r} != "
            f"uninterrupted {want['entropy']!r} -- the resumed run is a DIFFERENT run")
        assert got["solve_rate"] == want["solve_rate"]
    assert torch.equal(_weights(revived), _weights(straight)), (
        "final weights differ, so something outside nets/optimiser/RNG is part of the "
        "trajectory and is not being restored")


def test_resume_restores_the_manifest_so_a_resumed_result_file_is_complete(tmp_path):
    """The checkpoints written before the interruption are still on disk and `best_path`
    may still point at one of them. A manifest that starts at the resume point would
    silently drop them."""
    torch.manual_seed(0)
    ppo = _ppo(_env(), seed=0)
    writer = CheckpointWriter(ppo, ppo.env, tmp_path / "run.json", n_updates=6,
                              schedule=[1], resume_every=2, keep_resume=4,
                              mi_episodes=2, seed=0)
    ppo.train(verbose=False, on_update=writer)
    assert writer.written, "no scheduled checkpoint was written, so the test proves nothing"

    state = torch.load(tmp_path / "run_resume_u0002.pt", weights_only=False)
    revived_writer = CheckpointWriter(ppo, ppo.env, tmp_path / "run.json", n_updates=6,
                                      schedule=[1], resume_every=2, mi_episodes=2, seed=0,
                                      resumed=state)
    assert revived_writer.written == writer.written[:len(revived_writer.written)]
    assert revived_writer.best_update == state["best_update"]
    assert revived_writer.manifest()["checkpoints"], "manifest lost the earlier checkpoints"
