# legacy/ — the superseded generation

Kept, not deleted. Nothing here is on the path of any current result.

## `ma_v1/` — the first two-agent implementation

Replaced by the `ma/` package on 2026-08-21:

| legacy | replaced by | why |
|---|---|---|
| `ma_v1/env.py` | `ma/env.py` | belief moved from 543-DAG enumeration to the subset DP; enumeration died at k=6 |
| `ma_v1/policy.py` | `ma/policy.py` | checkpointing, and the [U14] reward |
| `ma_v1/baselines.py` | `ma/baselines.py` | tie-break variants, forced-clamp arm |
| `ma_v1/gates.py` | `scripts/ma_gates2.py` | gates rebuilt on the DP belief |

## `ma_v1/env.py` MUST NOT BE DELETED

It generated `tests/fixtures/ma_reference_posteriors.npz`, and that fixture is the
INDEPENDENT ground truth the subset DP was validated against to 1e-10. Two live tests still
import it for exactly that reason:

    tests/ma/test_belief_dp.py
    tests/test_score_regimes.py

Regenerating the fixture from the current code would make the validation circular and
destroy its entire value. The file looks like dead code. It is not.

## Naming

`legacy.ma_v1.env` and `ma.env` both define `MAConfig` and `TwoAgentEnv`. They are different
classes with different semantics, kept deliberately: the canonical package gets the clean
name, and the legacy path is explicit at every import site.
