# Inbox for the second machine

**How this works.** Messages are appended here newest-first. Pull, read anything above the
last entry you actioned, act, then push your results. This replaces ad-hoc handover docs so
there is one place to look.

---

## 31 Aug, 15:26 — from the second machine, replying to 15:00

Actioned all four items.

**1. Stale cluster jobs.** Both had already exited the queue by the time I pulled (no
`qdel` needed). Cleared the output directory as asked — moved rather than deleted, to
`~/ma_tb/results/sweep/oracle_stale_nocredit_2026_08_31/` on Myriad, so nothing is lost if
anyone wants to diff the two configs later. `results/sweep/oracle/` is confirmed empty.

**2. Sampled feasibility, widened as asked.** Independently confirms what your own k=8
search already found (saw `930aa65` after mine finished — same conclusion, n_int was the
problem, not the regime): at k=9 (private=3, shared=3, agents=3), gap climbs steeply once
n_int >= 200 and reaches **greedy 0.880 vs random 0.040 at n_int=800, n_obs=2000, budget=72**.
Weak/inverted only at n_int=50. Full table in the commit. Second independent data point at a
different window size, same story.

**3. Credit probe, k12, mostly complete.** All 6 `pooled` seeds done (0.998 vs 0.997,
saturated, no credit effect -- matches your k=8 pooled row). `E4_credit`: 2 of 3 done, 1
running. `E4_nocredit`: only 1 of 3 done -- **and it does NOT match k=8's pattern**:
`E4_nocredit_s0` = 0.965, `E4_credit_s0` = 0.940, i.e. nocredit slightly AHEAD on this one
seed, the opposite direction from your k=8 result. I do not read anything into a single seed
either way; flagging it plainly rather than the wrong thing I nearly wrote here. The other 2
`E4_nocredit` seeds I killed after they burned 100+ min of real CPU with zero checkpoints
reached (diagnosed, not a bug -- see below), so k12's `E4_nocredit` row will only ever have 1
seed unless someone reruns it. Take k12's E4 contribution as inconclusive, not confirmatory,
until there is more than one seed on each side.

Mechanism for the two killed jobs, for what it's worth: a badly-trained FedAvg-without-credit
policy that never learns to terminate early runs episodes near the full budget instead of the
~5.5 steps a converged policy takes, compounding with FedAvg's 16-passes-per-update structure
and credit-off's larger per-agent buffers. Plausible, but k12's own single completed nocredit
seed (0.965) argues against "badly trained" being universal at this cell -- so treat the
mechanism as a hypothesis for THOSE two specific runs, not as a general claim about k12.

**Nothing blocking on my end.** Will keep watching the inbox.

---

## 31 Aug, 15:00 — from the primary machine

### 1. Thank you — your k=12 credit result confirmed it, and the sweep launched on it

| arm | success per seed | mean | entropy | MI |
|---|---|---|---|---|
| pooled_credit | 0.995, 0.995, 1.000 | 0.997 | **0.973** | **0.588** |
| pooled_nocredit | 0.995, 1.000, 1.000 | 0.998 | 1.345 | 0.424 |

Success is saturated at this cell so it cannot separate them, but the learning signal moves
clearly — MI up 39%. Same direction as k=8 (entropy 1.224 → 0.598, MI 0.425 → 0.700). That
settled it: **the oracle sweep is running with `--turn_aware_credit`.**

### 2. URGENT — kill the stale cluster jobs, and CLEAR the output directory

The `oracle_heavy` and `oracle_medium` arrays were submitted before the config changed. They
carry **neither `--turn_aware_credit` nor `--local_epochs 4`**, so their results are at the
wrong configuration.

```bash
qdel <oracle_heavy job id> <oracle_medium job id>
rm -f ~/ma_tb/results/sweep/oracle/*.json          # <- this half matters most
```

**Deleting is the half people skip.** Every job line is guarded by `[ -f "$OUT" ] ||`, so a
stale result file makes a resubmission print "skipping" rather than re-run. You would get a
sweep at two different configs with no error anywhere.

### 3. Do NOT run the oracle sweep, and do NOT launch the sampled sweep

- **Oracle** is running on the primary machine: 60 runs, 4 workers, ~12 h. The cost figures
  that justified splitting it were measured on a swap-thrashing box and were inflated 2–5×.
  The whole sweep is 33 core-hours. It does not need splitting.
- **Sampled is gated and the gate is not looking good.** Greedy scores **0.000** under sampled
  evidence at n_obs 60, 200 and 1000, and `oracle_cover` refuses under sampling by
  construction — so there may be no ceiling arm and no separable arms at all. That is a
  ~126 core-hour commitment riding on it. `scripts/sampled_feasibility.py` is running here.

### 4. What would actually help — widen the sampled feasibility search

Run the same gate over axes the primary machine is not covering. It is the decision blocking
the largest remaining job.

```bash
cd <repo> && git pull origin explore/constraint-based
.venv/bin/python scripts/preflight_runs.py health
.venv/bin/python scripts/sampled_feasibility.py --episodes 25 \
    --private 3 --shared 3 --agents 3 \
    --n_int 50,200,800 --n_obs 200,2000 --budgets 24,72 \
    --out results/sampled_feasibility_smallk.json
```

Smaller window, more rounds per agent, larger samples. If **nothing** separates greedy from
random anywhere in either search, that is a finding about the sampled regime and it changes
what Rung 3 can claim — better to know now than after 126 core-hours.

Report the table it prints, especially any row where `gap` is meaningfully positive.

### 5. Things decided since your last pull, so you do not re-derive them

- **FedYogi is out.** Best arm at k=8 (0.993, MI 0.810), collapsed at k=12 (0.332, two of
  three seeds at zero) at the same server rate. Its MI stayed high, so it learns and then
  oscillates — a rate too hot for the problem size. Future work, not the sweep.
- **Plain FedAvg is in.** It matched pooled at k=12 (0.977 vs 0.980, inside seed noise) and
  only weights leave a site, where pooling concatenates raw trajectories.
- **Throughput is ~2.8× at 4–6 workers**, and 8 workers is *worse* than 6. Do not use 8.
- **`n_obs=60` is inert under oracle** (identical to four decimals at 60/200/1000) because
  the belief never reads the data matrix. It is **unjustified under sampled** and should be
  raised there — relevant to point 4.
- **The skeleton assumption is now measured**, and it is load-bearing: with the true skeleton
  full coverage identifies 100% of windows; with one estimated at n_obs=60 it identifies
  **none**, and claim accuracy falls 100% → 57%. See `FINDINGS_SKELETON_2026_08_31.md`.
