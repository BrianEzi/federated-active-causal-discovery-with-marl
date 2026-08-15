"""Tracking must be impossible to die from.

Every test here breaks WandB in a different way and asserts the caller carries on. The
standard is deliberate: this module observes runs that cost hours, so a tracking failure
at hour three must cost the log, never the run.
"""
import sys
import types

import pytest

from sa.tracking import NullTracker, start_run


def test_disabled_by_default():
    """No project means logging was not requested -- not a failure, and silent."""
    tracker = start_run(project=None, name="x")
    assert isinstance(tracker, NullTracker)
    assert tracker.enabled is False


def test_null_tracker_accepts_every_call():
    tracker = start_run(project=None, name="x")
    tracker.log({"entropy": 1.2}, step=3)
    tracker.summarise({"gap_closed": 1.2})
    tracker.finish()
    assert tracker.directory is None


def test_import_failure_degrades_to_noop(monkeypatch, capsys):
    """The venv on a compute node may not have wandb at all."""
    # A None entry in sys.modules makes `import wandb` raise ImportError, which is the
    # same thing the caller sees when the package is genuinely absent.
    monkeypatch.setitem(sys.modules, "wandb", None)
    tracker = start_run(project="p", name="x")
    assert isinstance(tracker, NullTracker)
    assert "import failed" in capsys.readouterr().out


def _fake_wandb(monkeypatch, init):
    module = types.ModuleType("wandb")
    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_init_failure_degrades_to_noop(monkeypatch, capsys):
    """A full disk or a permissions problem in the run directory."""
    def init(**kwargs):
        raise OSError("No space left on device")

    _fake_wandb(monkeypatch, init)
    tracker = start_run(project="p", name="x")
    assert isinstance(tracker, NullTracker)
    assert "init failed" in capsys.readouterr().out


def test_init_returning_none_degrades_to_noop(monkeypatch, capsys):
    _fake_wandb(monkeypatch, lambda **kwargs: None)
    tracker = start_run(project="p", name="x")
    assert isinstance(tracker, NullTracker)
    assert "init returned no run" in capsys.readouterr().out


class _Run:
    """A run whose every method fails, which is the case that matters most.

    A tracker that survives `init` and then throws on the first `log` would take down a
    run mid-training -- later, and so more expensively, than a clean failure at startup.
    """

    def __init__(self, fail=True):
        self.fail = fail
        self.logged = []
        self.summary = {}
        self.dir = "/tmp/run"
        self.finished = False

    def log(self, metrics, step=None):
        if self.fail:
            raise RuntimeError("network unreachable")
        self.logged.append((metrics, step))

    def finish(self):
        if self.fail:
            raise RuntimeError("broken pipe")
        self.finished = True


def test_log_failure_mid_run_does_not_propagate(monkeypatch, capsys):
    run = _Run(fail=True)
    _fake_wandb(monkeypatch, lambda **kwargs: run)
    tracker = start_run(project="p", name="x")
    assert tracker.enabled

    tracker.log({"entropy": 1.2}, step=1)   # must not raise
    tracker.finish()                        # must not raise
    out = capsys.readouterr().out
    assert "log failed" in out and "finish failed" in out


def test_summary_failure_does_not_propagate(monkeypatch, capsys):
    run = _Run(fail=False)

    class Exploding(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("summary is read-only")

    run.summary = Exploding()
    _fake_wandb(monkeypatch, lambda **kwargs: run)
    tracker = start_run(project="p", name="x")
    tracker.summarise({"gap_closed": 1.2})
    assert "summary failed" in capsys.readouterr().out


def test_happy_path_passes_metrics_through(monkeypatch):
    run = _Run(fail=False)
    _fake_wandb(monkeypatch, lambda **kwargs: run)
    tracker = start_run(project="p", name="x")

    tracker.log({"entropy": 1.2}, step=1)
    tracker.summarise({"gap_closed": 1.23})
    tracker.finish()

    assert run.logged == [({"entropy": 1.2}, 1)]
    assert run.summary["gap_closed"] == 1.23
    assert run.finished


def test_offline_mode_is_the_default(monkeypatch):
    """Online mode on a compute node hangs rather than failing, which is worse."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return _Run(fail=False)

    _fake_wandb(monkeypatch, init)
    start_run(project="p", name="x")
    import os
    assert os.environ["WANDB_MODE"] == "offline"


def test_explicit_mode_is_respected(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    _fake_wandb(monkeypatch, lambda **kwargs: _Run(fail=False))
    start_run(project="p", name="x", mode="online")
    import os
    assert os.environ["WANDB_MODE"] == "online"


def test_grouping_is_passed_through(monkeypatch):
    """Sweeps are only navigable if seeds collapse by config and arch is filterable."""
    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return _Run(fail=False)

    _fake_wandb(monkeypatch, init)
    start_run(project="p", name="lr1e-3_s0", group="lr1e-3", job_type="pernode",
              config={"lr": 1e-3}, tags=["d5", "n5000"])

    assert captured["group"] == "lr1e-3"
    assert captured["job_type"] == "pernode"
    assert captured["name"] == "lr1e-3_s0"
    assert captured["tags"] == ["d5", "n5000"]
    assert captured["config"] == {"lr": 1e-3}
