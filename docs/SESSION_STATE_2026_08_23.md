# Session state — 23 August 2026

Written before a compact. `docs/STATE_OF_TRUTH.md` is what is TRUE; this is *where things
stand* and *what to do next*.

**Freeze: 31 August. Dissertation due 8 September. Eight days.**

---

## Where the code is

Branch **`feat/n-agent-topology`**, all pushed. `main` is at `e490387` and **untouched by
this session's work** — the branch has not been merged and needs sign-off.

475 tests green. Full suite ~220 s on a quiet machine; `-m "not slow"` ~124 s.

Worktree: `C:/Workspace/MSc Project/.claude/worktrees/single-agent-clean`.

**Branch contents beyond `main`:** the n-agent topology refactor (rung 0 PASSED), Gemini's
`ma/env.py` generalisation, the S_r retraction + restored guard, structural ceiling, IV
decomposition, and the disclosure design doc.

## The four results from the overnight run

Report: https://claude.ai/code/artifact/ee44cf12-968c-4cae-b809-ad8585c917ae

1. **Structural ceiling ~2%** — only 2.3% (CI [1.3, 4.0], 12/516 pooled) of confounded
   windows are observationally detectable at INFINITE data. `results/structural_ceiling.json`.
2. **Confounding doubles at 3 agents** — 16.9% vs 8.8%, non-overlapping CIs.
3. **`sa/`'s policy settings hurt `ma/`** — −0.033 paired, additive across both changes,
   entropy falls monotonically. `results/sa_derived/`, `results/sa_split/`. Adopted neither.
4. **IV hypothesis refuted** — advantage is +0.030 and identical with/without instrument
   structure; DiD +0.005, CI [−0.030, +0.040]. `results/iv_decomposition.json`.

## Decisions waiting on the student

- **Merge `feat/n-agent-topology` → `main`.** Rung 0 passed (paired CI includes zero,
  private-clamp share holds). **CORRECTED 2026-08-23: this said "fast-forward is not possible
  — `main` has moved — so this is a real merge". That was wrong.** `origin/main` was an
  ancestor of the branch (0 commits behind, 22 ahead), and the merge was a clean fast-forward.
  Done on 2026-08-23; `main` is now at `c58543e`.
- **Mirco, and it is ONE question:** is an existential confounding claim about SHARED
  variables admissible, and is the clique-structure leak acceptable? `docs/DISCLOSURE_DESIGN.md`
  §7 has the framing. If refused outright, the ceiling says cross-boundary discovery is
  impossible here — write that up as a finding.
- **Latent projection vs S_r.** Recommend latent projection; `SR_MATH.md` stays as the
  fallback and its §14 modularity argument holds either way.
- **Port `PerNodeActorCritic` to `ma/`?** Evidence-backed (flat MLP measured 0.42 accuracy
  against the oracle in `sa/`) but a new axis with eight days left.

## Open work, in dependency order

1. **Implement the disclosure design** — blocked on Mirco. §6 of `DISCLOSURE_DESIGN.md`
   lists the three things to resolve first; double-counting is resolved in principle.
2. **Rung 1 (3 agents)** — blocked on (1) or on S_r. Note the env still REFUSES that shape;
   the guard in `ma/env.py.__init__` is correct and deliberate.
3. **Greedy baselines (sequential / joint, SGA / Corah)** — INDEPENDENT of all the above,
   thesis-facing, and the weakest point in the results chapter. Our greedy conditions on
   nothing. This is the best candidate if the disclosure question stalls.
4. **`docs/RESULTS.md`** — not started, sequenced last, but "last" starts about now.

## Traps that have already bitten, do not re-learn them

- **`ma/env.py` refuses topologies where any agent has >1 hidden node.** That guard is
  correct. It covers both multi-private AND n>=3 (three agents at one private node each
  still hides two nodes from each agent — the ORIGINAL guard missed this case).
- **Pre-refactor checkpoints** needed `_upgrade_checkpoint_keys` to load at all; the n-agent
  switch silently broke every one of them. Regression test in `tests/ma/test_env.py`.
- **Do not run the full suite alongside a CPU-bound job.** Done on 22/23 August: the suite
  took 2:25 instead of 3:40 total, and starved the IV job for hours.
- **Myriad's fetch refspec** was pinned to a single branch and had to be widened; new
  branches were invisible to it before that.
- **`ma_train` refuses to overwrite a result whose config differs** (`--force` overrides).
  Added after an overnight local run silently replaced five committed Myriad results that
  differed only in `step_cost`.

## Cluster

`~/ma_tb` on Myriad, branch `feat/n-agent-topology`. Venv is **`~/envs/sa_env`** (torch
2.6.0+cpu); `~/envs/marl_env` has NO torch. Jobs this session: 188953 (rung 0), 191136 +
193040 (policy A/B), all complete and pulled.
