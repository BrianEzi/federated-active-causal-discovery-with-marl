# legacy/scripts/ — one-offs and superseded pipelines

Moved 2026-08-21 under the consolidation rule:

> **A script stays in the main tree only if it reproduces a number we cite, or is imported
> by something that does.**

43 of 60 scripts failed that test. **Nothing was deleted** — they are the record of how the
project got here, and several document dead ends worth remembering.

Roughly: the Kaggle notebook generators (`generate_kaggle_*`), the staged sweeps
(`sweep_stage2`..`6`, `sweep_phase2`), the pre-DP two-agent pipeline (`ma_train`,
`ma_coordination_gate`, `ma_regime_*`, `ma_role_analysis`), and one-shot diagnostics
(`probe_*`, `step0_diagnostic`, `oracle_residual`, `sampler_comparison`).

Imports inside these files were repointed to `legacy.scripts.*` so they still run. Two test
modules — `tests/test_analyse_phase2.py` and `tests/test_sweep_phase2.py` — were repointed
rather than moved, so this tree stays protected against accidental breakage.

`legacy/scripts/ma_train.py` is the V1 trainer. The name is reused by the CURRENT trainer at
`scripts/ma_train.py`, which was `ma_train2.py` before the rename.
