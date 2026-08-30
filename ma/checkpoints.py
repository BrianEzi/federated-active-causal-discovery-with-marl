"""Checkpointing: what to keep from a long run, and why each kind exists.

THREE KINDS, and they answer different questions. Keeping only the last one -- which is what
this project did until 2026-08-30 -- answers none of them well.

  EVAL checkpoints, log-spaced and dense early. Weights only, ~384 KB apiece because the
    portable architecture's parameter count does not depend on k. They exist so the FULL
    metric suite can be recomputed at a point, which the history log cannot give: it records
    entropy and solve rate every update but not the MI gate, the pooled global graph,
    duplicate coverage or effort evenness. Dense early because every diagnostic we care
    about resolves in the first ~70 updates -- a06diff reached entropy 1.275 by update 70
    while its twin sat at 1.940, and w20iso was still at 3.042 at 70. A uniform
    every-100 schedule would have two points and miss all of it.

  RESUME state, rotated. Weights plus optimiser moments plus the RNG positions plus the
    update index. Weights alone cannot restart a run: the optimiser's second-moment
    estimates and the random streams are part of the trajectory. This is the difference
    between a cluster job dying at hour nine costing ten minutes and costing nine hours.

  BEST-BY-MI, tracked as it goes. The final policy is NOT reliably the best one: entropy
    rises again late in several measured runs, and a08long peaked mid-run. Selecting on
    reward would select on the thing we are trying to measure, so the criterion is the MI
    gate -- did the policy condition on its observation -- which is a training-health
    question and independent of the score being reported.

WHY MI AND NOT REWARD. `scripts/mi_gate.py` is the gate every learned number must pass;
using the same quantity to choose a checkpoint keeps one definition of "this policy
trained". The estimator here mirrors that file exactly -- same conditional-entropy formula,
same construction of the distribution the policy actually acts on -- so a checkpoint chosen
here and a checkpoint audited there cannot disagree.
"""
from __future__ import annotations

import pathlib
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Dense early, sparse late. See the module docstring for why this shape and not uniform.
DEFAULT_SCHEDULE: Sequence[int] = (5, 10, 20, 40, 70, 100, 150, 200, 250)
LONG_SUFFIX: Sequence[int] = (350, 500, 700, 1000)


def default_schedule(n_updates: int) -> List[int]:
    """The log-spaced schedule, clipped to the run length and always including the last."""
    points = [u for u in list(DEFAULT_SCHEDULE) + list(LONG_SUFFIX) if u < n_updates]
    if n_updates - 1 not in points:
        points.append(n_updates - 1)
    return sorted(set(points))


def _entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    return float(-(p * np.log(p)).sum())


def mi_ratio(ppo, env, episodes: int = 8, seed: int = 0) -> float:
    """I(S;A)/H(A), averaged over agents -- the same estimator as `scripts/mi_gate.py`.

    Deliberately cheap (8 episodes by default): this runs at every eval checkpoint and is
    used to RANK checkpoints, not to certify one. The certifying measurement is the gate
    script at its own episode count, run afterwards on whichever checkpoint this selects.
    """
    import torch

    conditionals: Dict[int, List[np.ndarray]] = {a: [] for a in env.topology.agents}
    policies = ppo.policies(deterministic=False)
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            for agent in env.topology.agents:
                with torch.no_grad():
                    logits, _ = ppo.nets[agent](
                        torch.as_tensor(env.observation(agent), dtype=torch.float32))
                conditionals[agent].append(torch.softmax(logits, dim=-1).numpy())
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})

    ratios = []
    for rows in conditionals.values():
        if not rows:
            continue
        stacked = np.asarray(rows)
        h_marginal = _entropy(stacked.mean(axis=0))
        h_conditional = float(np.mean([_entropy(row) for row in stacked]))
        # CLAMPED AT ZERO. Mutual information is non-negative by definition, but when a
        # policy is near-uniform h_conditional and h_marginal agree to within float error
        # and the difference can come out at -2e-08. A negative "information" would be
        # nonsense in a manifest and would sort below a genuine zero.
        ratios.append(0.0 if h_marginal <= 0
                      else max(0.0, (h_marginal - h_conditional) / h_marginal))
    return float(np.mean(ratios)) if ratios else 0.0


class CheckpointWriter:
    """Drop-in `on_update` hook for `IndependentPPO.train`.

    Writes eval checkpoints on a schedule, rotates resume state, and tracks the best policy
    by MI. Every failure mode is swallowed and reported rather than raised: a checkpointing
    problem must never take down a run that has already spent hours of compute.
    """

    def __init__(self, ppo, env, out: pathlib.Path, n_updates: int,
                 schedule: Optional[Sequence[int]] = None, resume_every: int = 50,
                 keep_resume: int = 2, mi_episodes: int = 8, seed: int = 0,
                 log: Optional[Callable[[str], None]] = None,
                 resumed: Optional[dict] = None):
        self.ppo, self.env = ppo, env
        self.out = pathlib.Path(out).with_suffix("")
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.schedule = set(schedule if schedule is not None else default_schedule(n_updates))
        self.resume_every = int(resume_every)
        self.keep_resume = int(keep_resume)
        self.mi_episodes, self.seed = int(mi_episodes), int(seed)
        self.log = log or (lambda msg: None)
        self.best_mi, self.best_update = -1.0, None
        self.written: List[dict] = []
        self._resume_paths: List[pathlib.Path] = []
        # A resumed run must not report a manifest that starts at the resume point: the
        # checkpoints written before the interruption are still on disk and still the ones
        # `best_path` might be pointing at. Restored from the resume payload so the result
        # file of a resumed run is indistinguishable from that of an uninterrupted one.
        if resumed:
            self.written = list(resumed.get("written", []))
            self.best_mi = float(resumed.get("best_mi", -1.0))
            self.best_update = resumed.get("best_update")

    # -- the hook ------------------------------------------------------------------------

    def __call__(self, record: dict) -> None:
        update = int(record.get("update", -1))
        if update in self.schedule:
            self._write_eval(update, record)
        if self.resume_every > 0 and update > 0 and update % self.resume_every == 0:
            self._write_resume(update)

    # -- the three kinds -----------------------------------------------------------------

    def _write_eval(self, update: int, record: dict) -> None:
        path = self.out.parent / f"{self.out.name}_u{update:04d}.pt"
        try:
            self.ppo.save(path)
            score = mi_ratio(self.ppo, self.env, self.mi_episodes, self.seed)
        except Exception as error:                       # never take a run down
            self.log(f"  checkpoint u{update} FAILED: {error!r}")
            return
        entry = {"update": update, "path": str(path), "mi_ratio": score,
                 "entropy": record.get("entropy"), "solve_rate": record.get("solve_rate")}
        self.written.append(entry)
        if score > self.best_mi:
            self.best_mi, self.best_update = score, update
            try:
                self.ppo.save(self.out.parent / f"{self.out.name}_best.pt")
            except Exception as error:
                self.log(f"  best-checkpoint write FAILED: {error!r}")
        self.log(f"  checkpoint u{update:4d}  mi {score:.3f}"
                 f"{'  <- best so far' if self.best_update == update else ''}")

    def _write_resume(self, update: int) -> None:
        """Weights are not enough to restart: optimiser moments and RNG positions are part
        of the trajectory, and dropping them silently changes the run you resume into."""
        import torch

        path = self.out.parent / f"{self.out.name}_resume_u{update:04d}.pt"
        try:
            torch.save({
                "update": update,
                "nets": {a: net.state_dict() for a, net in self.ppo.nets.items()},
                "opts": {a: opt.state_dict() for a, opt in self.ppo.opts.items()},
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": self.ppo.rng.bit_generator.state,
                "history": list(self.ppo.history),
                # This writer's own state, so a resumed run's manifest is complete.
                "written": list(self.written),
                "best_mi": self.best_mi,
                "best_update": self.best_update,
            }, path)
        except Exception as error:
            self.log(f"  resume u{update} FAILED: {error!r}")
            return
        self._resume_paths.append(path)
        while len(self._resume_paths) > self.keep_resume:      # self-pruning
            stale = self._resume_paths.pop(0)
            stale.unlink(missing_ok=True)

    # -- reporting -----------------------------------------------------------------------

    def manifest(self) -> dict:
        """Goes into the result file, so a reader can find and rank every checkpoint
        without listing the directory or re-deriving the schedule."""
        return {"schedule": sorted(self.schedule), "resume_every": self.resume_every,
                "best_update": self.best_update, "best_mi_ratio": self.best_mi,
                "best_path": (str(self.out.parent / f"{self.out.name}_best.pt")
                              if self.best_update is not None else None),
                "checkpoints": self.written}
