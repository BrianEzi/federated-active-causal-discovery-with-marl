# Retired tests — the v1 generation

These nineteen files test `legacy/src/` and `legacy/scripts/`, the generation retired on
21 August 2026. They were moved out of `tests/` on 22 August and are **excluded from
`testpaths`**, so `pytest` does not collect them.

**Moved, not deleted, deliberately.** The thesis write-up still refers to v1 behaviour in
places, and a moved test is recoverable evidence of what v1 actually did; a deleted one is
not. They passed at the moment they were moved (111 tests, 68 s).

To run them anyway:

    PYTHONPATH=. python -m pytest legacy/tests -q

**Not moved, and must not be:** `tests/ma/test_belief_dp.py` and `tests/test_score_regimes.py`
also import from `legacy/`, but there v1 is the **independent reference oracle** for current
code, not the subject under test. See the note at the top of each.
