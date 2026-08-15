"""Optional WandB logging. A second view of a run, never the record.

Three properties are load-bearing, and each is a response to something specific:

**Off unless asked.** Logging activates only when `--wandb_project` is passed. The result
JSON files remain the single source of truth for every number that gets reported, and
nothing in `sa/` reads anything back out of WandB. If this module were deleted the
experiments would produce identical results.

**Offline by default.** Myriad compute nodes have no outbound internet -- confirmed by
submitting a curl probe to one, which returned no HTTP status. An online `wandb.init()`
there does not fail fast, it hangs, which would burn the whole walltime allocation of a
job that was otherwise going to succeed. So runs write to a local directory and
`scripts/sync_wandb.py` uploads them afterwards from the login node.

**Never fatal.** Every call is wrapped. A missing package, a full disk, a permissions
problem, a version mismatch, 34 array tasks writing concurrently -- all degrade to a
printed warning and a no-op tracker. Instrumentation that can abort a run it is merely
observing is a worse trade than no instrumentation: the point of this phase is to make
experiments interpretable, not to add a new way for them to die at hour three.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


class NullTracker:
    """What every failure path returns, so callers never branch on availability."""

    enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def summarise(self, metrics: Dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass

    @property
    def directory(self) -> Optional[str]:
        return None


class WandbTracker:
    """A thin wrapper over a live `wandb` run. Constructed only by `start_run`."""

    enabled = True

    def __init__(self, run, module):
        self._run = run
        self._wandb = module

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as exc:  # noqa: BLE001
            _warn(f"log failed: {type(exc).__name__}: {exc}")

    def summarise(self, metrics: Dict[str, Any]) -> None:
        """Final values, which is what the run table in the UI sorts and filters on."""
        try:
            for key, value in metrics.items():
                self._run.summary[key] = value
        except Exception as exc:  # noqa: BLE001
            _warn(f"summary failed: {type(exc).__name__}: {exc}")

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001
            _warn(f"finish failed: {type(exc).__name__}: {exc}")

    @property
    def directory(self) -> Optional[str]:
        try:
            return str(self._run.dir)
        except Exception:  # noqa: BLE001
            return None


def _warn(message: str) -> None:
    print(f"  [wandb] {message} -- continuing without tracking", flush=True)


def start_run(project: Optional[str], name: str, group: Optional[str] = None,
              job_type: Optional[str] = None, config: Optional[dict] = None,
              tags: Optional[list] = None, mode: Optional[str] = None,
              directory: Optional[str] = None):
    """Begin a tracked run, or return a `NullTracker` if that is not possible.

    `project=None` means logging was not requested, which is the default and is not a
    failure -- no warning is printed for it.

    Grouping is set up so a 34-configuration x 3-seed sweep stays navigable: `group` is
    the configuration tag (so seeds of one config collapse into a single line with a
    band), `job_type` is the architecture (so the per-node and flat sweeps of E1 and E2
    can be split with one filter), and `tags` carry d and n_obs.
    """
    if not project:
        return NullTracker()

    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        _warn(f"import failed: {type(exc).__name__}: {exc}")
        return NullTracker()

    # Set before init, not passed to it: `wandb` reads the environment in several places
    # and honouring it here keeps any child process consistent too.
    os.environ.setdefault("WANDB_MODE", mode or "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    # Job arrays start dozens of processes at once; without this they contend over a
    # shared cache directory in the home filesystem.
    os.environ.setdefault("WANDB_CONSOLE", "off")

    try:
        run = wandb.init(project=project, name=name, group=group, job_type=job_type,
                         config=config or {}, tags=tags or [], dir=directory,
                         reinit=True)
        if run is None:
            _warn("init returned no run")
            return NullTracker()
        return WandbTracker(run, wandb)
    except Exception as exc:  # noqa: BLE001
        _warn(f"init failed: {type(exc).__name__}: {exc}")
        return NullTracker()
