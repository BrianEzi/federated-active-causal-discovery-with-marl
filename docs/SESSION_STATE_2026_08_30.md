# Session state — 30 August 2026

**Resume point.** Everything on the pre-run build list is DONE, tested and committed. The
next action is to launch runs. Freeze was 31 Aug morning; experiments to 2 Sep; write-up to
7 Sep 3pm.

Companion documents, in reading order:
[`THESIS_QUESTIONS.md`](THESIS_QUESTIONS.md) (the question hierarchy and where each is
answered) · [`ROADMAP_RUNGS_2026_08_29.md`](ROADMAP_RUNGS_2026_08_29.md) ·
[`METRICS.md`](METRICS.md) · [`ENGINES_AND_FLOW.md`](ENGINES_AND_FLOW.md) ·
[`FINDINGS_SHD_2026_08_29.md`](FINDINGS_SHD_2026_08_29.md) ·
[`FINDINGS_TURN_ORDER_2026_08_29.md`](FINDINGS_TURN_ORDER_2026_08_29.md)

---

## 1. THE COSTING — the sweep fits on the laptop

Measured, not estimated. Fitted from the ladder's own `train_seconds`:
`t ~ 8.38 * k^1.81 * n^0.56` seconds at 250 updates. And the sampled/oracle ratio was
**re-measured after the `estimated_reveal_all` speedup: 3.82x, down from 8.4x.**

| | core-hours | wall on 8 cores |
|---|---|---|
| oracle, 3 seeds | 35.8 | **4.5 h** |
| sampled, 3 seeds | 136.9 | **17.1 h** |
| **both, 3 seeds** | **173** | **21.6 h** |

20 cells x 3 seeds x 2 regimes. **The cluster is now optional rather than blocking.**

## 2. WHAT WAS BUILT TONIGHT — all committed, 243 tests green

| commit | what |
|---|---|
| `69016fa` | **Tier 1 correctness** — three defects that would have corrupted every number |
| `5843bf2` | **Checkpointing** — log-spaced eval, resume state, best-by-MI |
| `83c3019` | **Sweep** — (k, sigma, n, beta) parameterisation, 20 cells |
| `095b947` | **A5** — oracle-cover planner, the optimal arm at any k |
| `f8ba0a3` | **FedAvg** and the **owner channel** |

### The three correctness defects, because they change what everything means

1. **Every confidence interval excluded policy stochasticity.** `IndependentPPO.__init__`
   seeded the global torch stream and `load()` went through it, so every evaluation replayed
   ONE fixed sample path. Verified fixed: consecutive evals now give 0.4250 / 0.4500 where
   they were identical, and `run_arm_paths` reports **0.470 +/- 0.029**, path SD 0.033 — an
   interval previously reported as zero.
2. **Greedy was built at bar 0.7** while the task grades at 1.0. Worth +0.233 to greedy;
   inverted a headline once. Now read off the env's own config.
3. **329 of 436 result files record no `vs_evidence`.** `_config_record` now sweeps every
   `MAConfig` field automatically — 52 fields, plus `k`, `sigma_contended`, `n_agents`.

### Things I asserted and then had to correct

- **FedAvg E=1 is NOT equivalent to pooled.** Pooled takes `cfg.epochs` steps on the merged
  batch, and advantage normalisation differs by construction. Even single-site a 0.006 gap
  survives from the deliberate optimiser reset. The test now pins what is true: averaging
  drift dominates the optimiser reset.
- **"2 hours" for the pooled-graph metric was ~20 minutes**, and **"a day+" for FedAvg was an
  evening.** I have over-estimated three times; discount my estimates accordingly.

## 3. THE FINDING THAT MAY EXPLAIN THREE ANOMALIES

**The window ladder never varied sigma = shared/k.** It is 0.50 at w08/w12/w20/w30 and
**0.75 at w04**. So w04 is not on the same line as the others, and every w04 anomaly — argmax
reversing, the learner winning SHD only there, the de-dup gap going non-significant only
there — is a live candidate for a **sigma effect read as a k effect**. Pinned as a test in
`tests/test_sweep.py`. The sigma sweep is cheap and settles it.

## 4. DECISIONS TAKEN

- **Attribution is deprioritised.** Allocation leads; attribution is a bounded chapter at
  k=5 on `arch=gnn` if R5 lands, otherwise future work. Decided on progress.
- **Cut entirely:** C2 as a results axis, D5 (GRPO).
- **Extension list:** B3/B4/B5, C3, D4/D6/D7 — but **C3 is promoted**, because the
  decentralisation gap is **-0.017 at two agents**, the 1/n data signature, so C3 decides
  whether decentralisation costs data or capability. That is the direct answer to Mirco.
- **Round-robin stays.** It beats random turns (0.620/0.687 vs 0.373/0.393) and the JCI
  objection to it was retracted on three independent grounds.
- **"Homogeneous interventions" means equal COST, not one type.** Clamp is available. Note
  `mode_at_scale` measured clamp-only 0.233 vs vary-only 0.589 before it was cut.

## 5. WHAT TO DO NEXT, IN ORDER

1. **`scripts/preflight_metrics.py`** — gates the launch, currently green.
2. **Oracle sweep**, 20 cells x 3 seeds: `scripts/sweep.py --emit sh --seeds 3`. ~4.5 h.
3. **Re-score everything** — every SHD number predates the argmax + de-duplication defaults,
   and the w08 flip (greedy winning -> learner winning) shows this changes verdicts.
4. **Sampled sweep**, same cells. ~17 h.
5. Then, as time allows: FedAvg E in {1,4}; solo + `normalise_returns`; the owner-channel
   attribution re-run at k=5.

## 6. STILL BLOCKED ON A HUMAN

**The push.** 21 commits ahead of origin, keychain-blocked. The cluster agent can see none of
this. Less urgent now the sweep fits locally, but the work is unbacked-up.

## 7. THE RULE

Every number is quoted with three things or not at all: **the MI gate, the evidence mode, and
the evaluation policy (argmax or sampled).** Every wrong claim on this project has come from
one of the three being left implicit.
