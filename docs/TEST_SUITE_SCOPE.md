# Test suite — scope for slimming

**Drafted 22 August 2026. NOT IMPLEMENTED — this is a scope, for sign-off.**

The suite is slow enough to have stopped being run casually, which is the real cost: a suite
you avoid running is a suite that is not protecting anything. This document says what is in
it, what should go, what must not go, and in what order.

---

## 1. Inventory, measured

570 tests collected across 48 files, collection alone taking 9.7 s.

| tier | files | tests | what it exercises |
|---|---:|---:|---|
| **A — dead v1** | 19 | 111 | `legacy/src/`, `legacy/scripts/` — the retired generation |
| **B — v1 as reference oracle** | 2 | 11 | `legacy/ma_v1/`, used to cross-check *current* code |
| **C — active** | 27 | ~448 | `ma/`, `sa/`, current `scripts/` |

Tier A runs in **68 s**. That is only a slice of the total, so **deleting it is the easy win,
not the main one.** The main one is in tier C and needs the duration data below before
anything is cut.

---

## 2. Tier A — delete

Nineteen files whose subject is code we retired on 21 August. They import `legacy.src.*` or
`legacy.scripts.*`, and they currently pass, which is precisely the problem: they are green
gates on code no result depends on any more, and they are the reason `legacy/` needed its
internal imports rewritten during consolidation at all.

```
test_analyse_phase2   test_avici_buffer      test_baselines
test_bayes_optimal_estimator                 test_curriculum
test_episode_metrics  test_evaluate_checkpoint_config
test_evaluator_env    test_graph_estimator   test_inductive_head
test_jit_acceleration test_metrics           test_ppo_agent
test_ppo_trainer      test_rewards           test_stitching
test_sweep_phase2     test_topologies        test_two_stage_loop
```

**Move to `legacy/tests/`, do not delete outright**, and drop that directory from `testpaths`
in `pytest.ini`. The thesis write-up still cites v1 behaviour in places, and a moved test is
recoverable evidence of what v1 actually did; a deleted one is not. This costs nothing —
excluded directories are not collected.

**Saving: 111 tests, ~68 s, 19 files.**

## 3. Tier B — must NOT go

Two files import `legacy.ma_v1` and look like tier A on a `grep`. They are the opposite:

- `tests/ma/test_belief_dp.py` — checks the subset DP against
  `tests/fixtures/ma_reference_posteriors.npz`, **generated independently by
  `legacy/ma_v1/env.py`**. This is the evidence behind the 1e-10 agreement claim in
  `STATE_OF_TRUTH.md`.
- `tests/test_score_regimes.py` — same pattern for the regime rules.

Here v1 is the **independent oracle**, and its independence is the whole value: a
cross-check against a reimplementation that shares no code with the thing under test. If
`legacy/ma_v1/` is ever removed these tests must be converted to frozen fixtures **first**,
never dropped.

A comment saying so belongs at the top of both files, because the next person to grep for
`legacy` in `tests/` will make exactly this mistake.

## 4. Tier C — needs measurement before it is touched

A duration run is what decides this, and **nothing in tier C should be cut on structural
grounds alone.** The known shape of the problem:

- only two files import torch (`tests/sa/test_policy.py`, `tests/test_depth.py`), so the cost
  is unlikely to be model construction
- the environment tests run real episodes with `n_obs = 1000` and exact posteriors, which is
  where the time almost certainly is
- the honest lever there is **fixture reuse and smaller `d`**, not deletion — a test that
  stops running a real episode stops testing the thing that has broken most often

The candidate actions, in the order they should be considered:

1. **Session-scoped fixtures** for anything that builds a `GraphSpace` or a `DPPosterior`.
   These are deterministic functions of `d` and are currently rebuilt per test.
2. **A `slow` marker** with `-m "not slow"` as the default local run, full suite in CI and
   before any commit that touches `ma/` or `sa/`. This keeps coverage while making the suite
   runnable casually again.
3. **Drop `d` where the test does not depend on it.** Several environment tests would prove
   exactly as much at `d=4` as at `d=6`.

## 5. Stale names, worth fixing in the same pass

The consolidation renamed `env2.py` → `env.py` but not the tests that cover it:

| now | should be |
|---|---|
| `tests/ma/test_env2.py` | `tests/ma/test_env.py` |
| `tests/ma/test_evaluate2.py` | `tests/ma/test_evaluate.py` |
| `tests/test_env2_turns.py` | `tests/test_env_turns.py` |
| `tests/test_env2_turn_budget.py` | `tests/test_env_turn_budget.py` |

(`test_analyse_phase2` and `test_sweep_phase2` keep their `2` — it is a phase number, not a
generation suffix, and both are tier A anyway.)

There is also a possible overlap between `test_env2_turns.py` (9 tests) and
`test_env2_turn_budget.py` (18 tests) — the first predates the turn-budget spec. **Check
before merging them**: the turn-budget spec's nine acceptance tests live in the second file
and are the gate on the collapse fix, so they must survive intact whatever happens.

---

## 6. Order of work

1. Move tier A to `legacy/tests/`, exclude from `testpaths`. Contained, reversible, ~68 s.
2. Add the "this is an oracle, do not delete" note to the two tier B files.
3. Rename the four stale files.
4. **Then** read the duration data and decide tier C — fixtures first, marker second.

Steps 1–3 are mechanical and safe. Step 4 is the one that needs judgement, and it should not
start until the numbers are in.
