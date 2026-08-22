# Test suite — what was cut, and what is left

**Implemented 22 August 2026.** Drafted as a scope, executed the same day; the measured
outcome replaces the estimates.

The suite had reached 876 s over 570 tests, which is long enough that it stopped being run
casually — and a suite you skip protects nothing.

---

## Result

| | tests | wall clock |
|---|---:|---:|
| before | 570 | **876 s** |
| after, full suite | 462 | **484 s** |
| after, `-m "not slow"` | 446 | **124 s** |

The default local loop is now **2 minutes**, and nothing was deleted to get there.

---

## 1. Nineteen dead-v1 files moved out

They tested `legacy/src/` and `legacy/scripts/`, the generation retired on 21 August: green
gates on code no result depends on. Moved to `legacy/tests/` and dropped from `testpaths`.

**Moved, not deleted.** The write-up still refers to v1 behaviour, and a moved test is
recoverable evidence of what v1 actually did. Run them with `pytest legacy/tests`.

Saving: 111 tests, 68 s.

## 2. Two files that look identical to a grep, and must NOT go

`tests/ma/test_belief_dp.py` and `tests/test_score_regimes.py` also import from `legacy/`.
There v1 is the **independent reference oracle** for current code — the check is worth
something precisely because the reference shares no code with the thing under test, so a
shared bug cannot hide in both. `test_belief_dp.py` is the evidence behind the 1e-10 claim in
`STATE_OF_TRUTH.md`. Both now carry a do-not-move banner. If `legacy/ma_v1/` is ever removed,
convert them to frozen fixtures FIRST.

## 3. The real win: memoising `build_graph_space`

`build_graph_space(d)` enumerates every DAG on `d` nodes and groups them into equivalence
classes. It is a **pure function of `(d, fast)`** and a `d=6` build costs ~33 s. The suite
called it from ~55 sites across 15 files, rebuilding the same handful of spaces over and over.
That was the single largest cost in the 876 s run — and it was pure waste, not coverage.

Cached session-wide in `tests/conftest.py`. Measured effect:

| test | before | after |
|---|---:|---:|
| `test_sampled_singleton_fraction_is_unbiased[6-0.5]` | 41.6 s | **5.4 s** |
| `test_confinement_also_holds_with_two_private_nodes_each` | 40.2 s | **6.0 s** |
| `test_masked_indices_agree_with_the_generator` | 34.3 s | out of the top 15 |

Two deliberate choices worth defending:

- **In `conftest.py`, not in `sa/graphs.py`.** Production jobs build large spaces, and an
  unbounded process-lifetime cache there is a memory leak waiting to happen at `d >= 8`. The
  test session builds a few small ones and exits. The cost belongs where the benefit is.
- **At conftest import time, not in a fixture.** Test modules do
  `from sa.graphs import build_graph_space`, binding the function object at their own import
  time. conftest is imported first, so patching the module attribute is picked up. A fixture
  would run too late.

Safe because `GraphSpace` is a frozen dataclass and no test assigns into its arrays. The
arrays are ordinary mutable numpy arrays, so this is a **convention, not a guarantee**: a test
that mutated a returned space would now corrupt every later test asking for the same `d`.

## 4. Markers, not deletions, for what is genuinely expensive

After the cache, seven tests held 333 s of the remaining 484 s. Every one earns its cost —
the confinement proof, metric earnability, DP-against-enumeration, the `d=6` known counts.
**None was cut.** They carry `@pytest.mark.slow`, so `-m "not slow"` gives a 124 s loop while
the full suite still gates anything touching `ma/` or `sa/`.

`test_onepass_is_faster_than_constrained_runs` carries `@pytest.mark.perf`: it asserts on wall
clock, and it failed once on 22 August purely because another job was running. It measures
something real but cannot be trusted under contention.

## 5. Stale names fixed

The consolidation renamed `env2.py` to `env.py` but not its tests: `tests/ma/test_env2.py`,
`tests/ma/test_evaluate2.py`, `tests/test_env2_turns.py`, `tests/test_env2_turn_budget.py`.

That collided two basenames (`tests/ma/test_env.py` against `tests/sa/test_env.py`), so
`tests/` is now a package — without `__init__.py` pytest imports by bare basename and the
two fail collection with "import file mismatch".

(`test_analyse_phase2` and `test_sweep_phase2` keep their `2`: a phase number, not a
generation suffix, and both moved to `legacy/tests/` anyway.)

---

## Both open points, now closed

**`test_metric_reachability.py`: 179 s -> 25.7 s.** The three tests each called
`split_by_confounding(topology, episodes=70)` with *identical arguments*, so the same 70
episodes were simulated three times. Now a module-scoped fixture. No coverage lost: the
function is deterministic in its arguments, and the rows are shared read-only.

**`test_env_turns.py` vs `test_env_turn_budget.py`: no overlap worth removing.** Exactly one
pair duplicates -- `test_clean_rounds_are_reachable` against
`test_clean_rounds_are_still_reachable` -- and the second is item **12.8 of the turn-budget
spec**, whose point is that the older guard survives the new shared-budget semantics. Same
assertion, different claim. Both are sub-second. Kept, with a note on each so the
duplication reads as deliberate rather than as an oversight.

Everything else in the two files is complementary: `turns.py` covers turn-order mechanics,
`turn_budget.py` covers budget, forfeits and signalling.

## Still open

Nothing. The next thing that would move the number is `tests/test_dp.py` and
`tests/test_projection.py`, and both are exact-enumeration checks whose cost IS the check.
