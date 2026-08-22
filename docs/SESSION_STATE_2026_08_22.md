# Session state — 22 August 2026, morning

Written before a compact. Read `docs/STATE_OF_TRUTH.md` first for what is true; this file is
only *where things stand right now* and *what to do next*.

**Freeze: 31 August. Dissertation due 8 September.**

---

## Where the code is

Branch `feat/single-agent-clean`, HEAD `0b02c60`, pushed. Consolidation is complete: 570
tests green, `ma/env2.py` → `ma/env.py` (and the `2` suffixes gone from classes too), v1 in
`legacy/`, 18 scripts, 12 authoritative docs.

Revert point if anything goes wrong: tag `pre-consolidation-2026-08-21`.

## What happened overnight

**Confinement HOLDS for n agents** — 16 configurations, 230,529 projections, zero violations,
2–4 agents, disjoint and partial-overlap. The largest blocker to scaling is retired. It also
corrected our edge rule: see below.

**No training ran.** Four arms were launched locally at 23:40; the machine stalled around
01:00 and nothing completed. Zero new seeds.

**The miss that matters:** the instruction was to use Myriad as much as possible. It was
treated as optional and never set up. A cluster job would have been immune to the laptop
sleeping. This is the fix in progress (below).

## IN PROGRESS — Myriad submission

1. `~/ma_tb` on Myriad is a clone of this branch at `0b02c60`. **The first clone was killed
   by a 120 s timeout mid-checkout and only `ma/` landed** — a `git checkout -f HEAD` was
   running when this file was written. **Verify `sa/`, `scripts/` and `tests/` exist before
   submitting.**
2. The venv is **`~/envs/sa_env`** (torch 2.6.0+cpu). **`~/envs/marl_env` has no torch** — it
   is the old environment and will fail.
3. `cluster/submit_turnbudget_arms.sh` is written and committed: an SGE array, `-t 1-40`,
   one task per (arm, seed), skipping any output that already exists.

**To submit:**

    ssh myriad "cd ~/ma_tb && git pull && qsub cluster/submit_turnbudget_arms.sh"

**To collect:**

    scp myriad:~/ma_tb/results/ma_fixed/'*.json' results/ma_fixed/
    PYTHONPATH=. python scripts/ma_night_summary.py

## The four arms, and why each matters

| arm | seeds | question |
|---|---|---|
| `nobit_clamp` | 0–9 | **the ablation.** Every number we have has the regime bit ON. Without this we cannot claim the federation channel earns its place. Highest value |
| `randturn_clamp` | 0–9 | random turn order vs round-robin — the supervisor raised it, never swept |
| `tb_clamp` | 10–19 | resolve the +1.8pp clamp-only lean |
| `tb_both` | 10–19 | its pair — the comparison is paired, so both need the same seeds |

## Two lessons to apply before the next unattended run

1. **Submit to the cluster, not the laptop.** A local overnight run has no protection against
   the machine sleeping, and a stalled background job notifies nobody.
2. **Verify throughput early instead of explaining it.** At 00:50 the arm was running ~3×
   slower than the previous batch, and a plausible reason was found (the no-bit arm never
   terminates early). The reasoning was sound and wrong, and it stopped further checking.

## Decisions waiting on the student

- **Sign off `docs/N_AGENT_REFACTOR_SPEC.md`** before any code. Flags that per-block
  confounding subsets are a **blocker for n ≥ 3**, not a later nicety.
- **`prior_p = 2 ln(d)/d`** — measured, not yet applied. ER-2 gives **1% connected graphs at
  d=30**.
- **Clamp-only as the default** — currently a *trade* (≤4pp, ahead 6/10 tied 2 behind 2), not
  a proven equivalence.
- **Merge to `main`** — Phase 4 of the consolidation, deliberately left undone.
- **Supervisor**: is the three-category signalling channel admissible?

## Immediate next steps, in order

1. Finish the Myriad checkout, verify, `qsub`.
2. While it runs: `ma/topology.py` refactor only — integer agents and the
   **jointly-visible edge rule**, every existing test still passing.
3. **Rung 0** — two agents on refactored code reproducing today's numbers — before any third
   agent exists anywhere.

## The edge-rule correction (carry this forward)

`ma/topology.py` currently forbids an edge between two nodes *each private to a different
agent*. That is the two-agent special case. The rule that generalises:

> **An edge may exist only if some agent observes both of its endpoints.**

Identical under a disjoint partition; required the moment visibility overlaps. Under the old
rule the overlap family violates confinement in all four configurations tested.

## Artifacts

- Overnight report: https://claude.ai/code/artifact/3c6cf39a-bb6d-4306-a06c-e21012e49d6d
- Supervisor briefing (superseded): https://claude.ai/code/artifact/f0ad745e-0a27-44ef-bb53-eba0a4f0db14
- Graph examples (superseded protocol): https://claude.ai/code/artifact/459c8931-7305-4367-a2c9-8a19f5c8cb28
