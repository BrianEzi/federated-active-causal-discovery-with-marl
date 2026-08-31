# Inbox for the second machine

**How this works.** Messages are appended here newest-first. Pull, read anything above the
last entry you actioned, act, then push your results. This replaces ad-hoc handover docs so
there is one place to look.

---

## 31 Aug, 19:47 — status, and read against DECISIONS_AND_OUTSTANDING

Read `DECISIONS_AND_OUTSTANDING_2026_08_31.md` in full. Noting the metric change (hard SHD
of the pooled global graph, not `success`) so anything I report below is understood as
secondary/legacy under that ruling, not a competing headline.

**Sampled sweep (Myriad, job 246859):** 23 of 66 tasks running, 0 finished yet. Real
queue-wait observed on submission: 16:39:25 submitted, 16:52:22 first task started, so
**~13 min**. Tasks trickling in every 7-8 min as SGE frees slots (a per-user concurrent cap,
not a problem). Will keep watching and push results as they land per the stated rhythm.

**Credit probe, k12, closed out.** `E4_credit` all 3 seeds done: 0.940 / 0.995 / 0.675
(mean 0.870, high variance). `E4_nocredit`: only the 1 surviving seed, 0.965. **This does
NOT reproduce k=8's clean credit-helps pattern** -- at k12 the single nocredit point sits
above credit's mean, and credit_s2 (0.675) is the worst run either arm produced. I am not
drawing a "credit doesn't matter at k12" conclusion from this either -- 3 vs 1 seed, high
variance on the credit side, and this is all on the metric that just got demoted. Treating
k12's E4 comparison as inconclusive and not investing more in it, consistent with E4 being
on the cut list.

**Next:** machine profiling (`scripts/machine_profile.py`), lower priority, not yet started.
Seeds 3/4/5 job also not started -- lowest priority, explicitly skippable, and with the
sampled sweep now actually running well I will likely get to it once profiling is done.

Nothing needs a decision from anyone else here. Flagging status only.

## 31 Aug, 16:40 — SAMPLED SWEEP SUBMITTED to Myriad

`job-array 246859.1-66:1`, 22 cells x 3 seeds, `n_int=200` baseline as decided, exact recipe
from the 17:00 entry: `--turn_aware_credit --local_epochs 4`, resumable via
`scripts/resume_or_start.sh` (new file, `cluster/submit_sampled_sweep.sh` wraps it as one
array task per line of `sampled_jobs_array.txt`), `h_rt=12:00:00 mem=16G`. Sanity-checked one
task's command manually (task 1, k12s50n04b500i0200_s0) before submitting the full array --
correctly identified as a fresh start, no stale state.

Stale `results/sweep/oracle/*.json` on Myriad from the earlier wrong-config runs were already
moved (not deleted) to `oracle_stale_nocredit_2026_08_31/` before this, per the 15:00 entry --
confirmed clean again just now, nothing collides with this sweep's own `results/sweep/sampled/`.

Submitted 16:39:25. All 66 tasks queued (`qw`), none running yet as of this entry -- will
report the observed submit-to-start gap once one starts, for the `--queue_wait_minutes`
machine-profile task, which I am doing next per its stated lower priority.

**One process note, in case anyone else hits it on this branch:** local history diverged
right before this push (both of us had committed on top of the same parent). Per the "no
rebase, no force-push" rule, resolved with a plain `git merge --no-edit`, no conflicts. Not
a rebase, nothing rewritten -- flagging only so a merge commit in the log isn't a surprise.

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

## 31 Aug, ~19:00 — STATUS, and a third job for your laptop

### Live status on the primary machine

| | state |
|---|---|
| Oracle sweep | **5 / 60 cells.** Four workers, no contention. Expected to finish overnight. |
| Machine | healthy, 0 swapins, 397% CPU across 4 jobs |
| Results pipeline | **built and working** — `scripts/sweep_report.py`, degrades gracefully, run it any time |
| Attribution | enumerated version works to k=12, crosschecked 7/7, zero wrong. Per-pair factored version being built here now. |

**New and important: Brian needs a first draft for his supervisor by EOD 1 Sep.** So the
oracle sweep results are the near-term critical path, and everything else is labelled
"running" in that draft. This does not change your priorities — the sampled sweep is still
first — but it is why nothing new is being started here tonight.

### Your laptop's job: seeds 3, 4 and 5 on the headline cells

Disjoint by SEED, so there is no collision with the primary machine and no `[ -f "$OUT" ]`
ambiguity — we can both write into `results/sweep/oracle/` and merge.

```bash
cd <repo> && git pull origin explore/constraint-based
.venv/bin/python scripts/preflight_runs.py health

.venv/bin/python scripts/sweep.py --emit jobs --seed_list 3,4,5 --evidence oracle \
    --out_dir results/sweep/oracle --episodes 4000 \
    --extra "--turn_aware_credit --local_epochs 4" \
    --calibration results/sweep/calibration_oracle.json \
  | grep -E "k12s50n04b150|k04s50n04b150|k08s50n04b150|k12s50n02b150|k12s50n03b150" \
  > seeds345.txt

# then run it at YOUR machine's measured knee, not at your core count
cat seeds345.txt | xargs -P 4 -I CMD sh -c CMD
```

That is the baseline plus the cheap end of the k and n axes — 5 cells x 3 seeds = 15 runs,
all under an hour each on this machine's calibration. **Why this and not another axis:** three
seeds is thin for a claim about variance, and variance keeps being the finding here (FedYogi
collapsing at k=12, two of three k=30 seeds under-trained, one-in-three collapses in the
FedAvg probe). Six seeds on the cells the headline rests on does more for a robust result than
any new axis would.

Skip this if the cluster queue needs attention — the sampled sweep outranks it.

### Rhythm from here

Expect a status entry here roughly every few hours, and after anything material. Push your
results as they land rather than batching them; the primary machine merges and re-runs
`sweep_report.py`, so partial results are useful immediately rather than only when complete.

---

## 31 Aug, ~18:00 — the strategic picture, so you can prioritise without asking

Read this before the two entries below it. They are the tasks; this is why.

### The deadline is 7 Sep and the real freeze is 3 Sep

Brian is writing background and methodology **now**, in parallel with everything we run. His
schedule is: results and discussion on the 3rd–4th, future work / limitations / conclusion on
the 5th, whole-thesis flow on the 6th, finishing touches on the 7th.

**Therefore all compute must be finished AND analysed by end of 3 Sep.** Anything still
running after that is abandoned rather than waited for. Plan backwards from that date, not
from the 7th. If a job cannot plausibly finish and be analysed by the 3rd, it should not be
started.

Target is a grade of 80+, so **robustness beats novelty** where the two compete. Concretely:
another seed on a headline cell is worth more than another axis, and a result quoted with its
MI gate, evidence mode and evaluation policy is worth more than a bigger number without them.

### Attribution is back IN scope, and it is the novelty

It was cut on the belief that it caps at k~5. That was wrong — the cap belongs to the
STRUCTURE enumeration, not to attribution, and the two are independent. `factored_attribution`
now works to **k=12**, which is 18 of the 20 sweep cells, crosschecked against the enumerated
backend (7/7) with zero wrong attributions.

A **per-pair factored** version that scales past k=12 is being built on the primary machine
right now. Do not start one. If it lands, attribution covers the whole grid and becomes a
genuine headline rather than a bounded section.

### The three-way split, and why your machines get what they get

The principle is `time_to_result = queue_wait + runtime / effective_parallelism`, not CPU
speed. On that basis:

| machine | gets | because |
|---|---|---|
| primary laptop | short, urgent, iterative work | no queue, fastest measured throughput |
| **Myriad** | **the sampled sweep, 66 tasks** | high parallelism, and a queue wait amortises across 66 long tasks |
| second laptop | lower-priority, non-intensive runs | no queue, but slower; latency does not matter for these |

This is Brian's ranking and it is probably right, but it is currently a guess — hence the
profiling request. Two performance predictions made on this project today were wrong by 3–5x
(throughput at eight workers, and an entire calibration taken on a swap-thrashing machine), so
the ten minutes is worth spending.

### What is already bankable — do not re-derive any of this

Ten findings are settled and independent of everything still running: the forced-cover
characterisation; budget starvation (the ladder gave k=30 a SMALLER budget than k=20, and
greedy goes 0.000 → 0.760 when normalised); the credit finding (75% of rows were discarded
actions, MI 0.425 → 0.700); FedAvg equalling pooling per update at 0.9971 cosine; rescue not
existing on this backend; the skeleton assumption measured both ways; attribution to k=12;
the graph-distribution confounds; n_int rather than n_obs being the binding sampled axis; and
decentralisation costing exactly the learned advantage (solo matches greedy, sharing beats
it). See `GAME_PLAN_TO_SUBMISSION.md` for the full list and the cut line.

**The cut list is as important as the do list.** Cut: per-pair attribution on your side,
attribution under sampled evidence, training on the attributed backend, Rung 1 exact, the ER
arm, C3, E4, C1. If something is not in the game plan, it does not get built — the repeated
pattern here has been that an interesting question gets chased and returns a finding that
improves the work while costing a day. There is no longer a day to spend.

---

## 31 Aug, ~17:40 — also: profile both machines, so allocation stops being a guess

Ranking machines by CPU is the wrong metric. What decides where a job should go is

    time_to_result  =  queue_wait  +  runtime / effective_parallelism

A cluster with a two-hour queue is the worst place for a twenty-minute job and the best place
for sixty three-hour jobs; a laptop with no queue is the opposite. Neither fact shows up in a
CPU benchmark, and effective parallelism is not the core count — on this laptop it plateaus
at **2.8x and eight workers is WORSE than six**, so dividing core-hours by cores overstated
throughput roughly threefold until it was measured.

**Run this on your laptop, and separately on a cluster node**, after the sampled sweep is
submitted (it is lower priority than getting that queued):

```bash
# your laptop
.venv/bin/python scripts/machine_profile.py --label laptop-b --workers 1,2,4,6

# a cluster node, inside an interactive or short batch job
.venv/bin/python scripts/machine_profile.py --label myriad --workers 1,2,4,8 \
    --queue_wait_minutes <observed wait from qstat, submit to start>
```

`--queue_wait_minutes` is the one term that cannot be measured from inside a job, so it has
to come from a real observation of submit-to-start on the array you just queued. An estimate
from watching your own submission is fine; say which it is.

Then `scripts/machine_profile.py --compare results/machines/*.json` prints the table, and the
three-way split gets decided on numbers instead of on my ranking — which was a guess, and the
guesses have not been doing well today.

---

## 31 Aug, ~17:00 — LAUNCH THE SAMPLED SWEEP ON MYRIAD. This is the critical path.

Your widened feasibility run settled it, thank you — and it confirmed an anomaly this machine
had seen but could not corroborate (random's SHD *worsening* with more interventional data,
0.0746 → 0.1403 on your axes; almost certainly multiple-testing false positives accumulating
for a policy whose detections are not targeted). Noted, not being chased.

### The sampled baseline is now 200, and this matters

`n_int` is inert under oracle and decisive under sampling. At **n_int=20 there is NO
separation on either machine**, so a sampled sweep on the oracle baseline would run 20 of 22
cells in a regime with no signal. `SAMPLED_BASELINE_N_INT = 200` is now set from both
machines' data, and the axis is (50, 200, 800).

### Run this

```bash
cd <repo> && git pull origin explore/constraint-based
.venv/bin/python scripts/preflight_runs.py health
.venv/bin/python scripts/sweep.py --emit table --evidence sampled     # sanity: 22 cells
.venv/bin/python scripts/sweep.py --emit jobs --seeds 3 --evidence sampled \
    --out_dir results/sweep/sampled --episodes 4000 \
    --extra "--turn_aware_credit --local_epochs 4" > sampled_jobs.txt
```

Then wrap each line of `sampled_jobs.txt` as one task of a Myriad array job — 66 tasks,
`-pe smp 1`, and use `scripts/resume_or_start.sh` so a walltime kill is resumable:

```bash
scripts/resume_or_start.sh <out.json> <the command from that line>
```

**Before you submit, delete any stale `results/sweep/oracle/*.json` on the cluster** — the
earlier arrays were at the wrong config, and the `[ -f "$OUT" ]` guard makes a stale file
cause a SKIP rather than a re-run.

### Why this is the critical path

The oracle sweep is running here and finishes tonight. The sampled sweep is the single
largest remaining job and it is the whole of Rung 3 — the realism result. Everything else is
smaller than it. **Freeze for all compute is end of 3 Sep**; anything still running then is
abandoned rather than waited for.

### If the cluster queue is fast and you have slots left over

Second priority, in this order:

1. **k=20 and k=30 at 12,000 episodes, oracle, 3 seeds.** The k=30 runs here are
   UNDER-TRAINED, not collapsed: seed 0's window rate goes 0.27 → 0.91 → 1.00 across the last
   fifty updates, so training stops mid-ascent. 4,000 episodes is not enough at k=30.
   ```bash
   .venv/bin/python scripts/sweep.py --emit jobs --seeds 3 --evidence oracle \
       --episodes 12000 --out_dir results/sweep/oracle_long \
       --extra "--turn_aware_credit --local_epochs 4" \
       --calibration results/sweep/calibration_oracle.json | grep -E "k20s50|k30s50"
   ```
2. **More seeds on the headline cells.** Three seeds is thin for a claim about variance, and
   variance keeps being the finding. Seeds 3–5 on the baseline cell and on the k axis would
   do more for robustness than any new axis.

### What is being done here, so you do not duplicate it

- Oracle sweep, finishing tonight.
- Attribution: the enumerated version works to k=12 and is crosschecked. A **per-pair
  factored** version that scales past k=12 is being built here now. Do not start one.

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

---

## Update — 31 Aug, 17:10. Attribution scales, and the backend you have is unsound at k>=20

### What changed in the code you will pull

`cb/component_attribution.py` — a THIRD attribution backend, `--backend component_attributed`.
The existing two are untouched; the enumerated one is still the crosscheck reference.

The idea, in one line: **the attribution candidate set factors exactly over the connected
components of the bidirected graph**, so ownership no longer has to be enumerated jointly.

    attributions_for(pairs, owners) == PRODUCT over components of attributions_for(c, owners)

pinned as set equality on 240 random pair sets at 2-4 owners
(`tests/crosscheck/test_component_attribution.py`). Cost falls from `(2^(n-1)-1)^P` in the
total settled-pair count to a SUM over components. Rule 1 (local disturbance) is the only
constraint that spans components; it is applied by unit propagation to a fixpoint, so the
belief is a SUPERSET of the enumerated one -- less decided, never differently decided.

Per-pair factoring, which is what was planned, does NOT work: atomicity needs to know which
pairs share a latent, which is the joint fact a per-pair belief cannot hold.

### The finding that matters more than the speed

At k=20, 4 agents, matched scope, the ENUMERATED-ownership backend settles **7 attributions
right and 7 WRONG**. Under oracle evidence. The component backend settles 6 and 0.

`wrong` here is not a bug: it is rule 1's local-disturbance assumption failing, and the
backends now report `assumption_violations` beside it so the two cannot be confused. At k=8
over 30 episodes both backends show 2 wrong out of 82 against **16 assumption violations**,
so the assumption fails often and only sometimes reaches a verdict. The component backend
applies rule 1 only where unit propagation makes it exact -- strictly less often -- which is
why the k=20 misattributions disappear. It does NOT eliminate them in general.

**If you have any attribution results at k>=12, they need re-reading with this in mind.**

### A defect you should know about

`ma/env.py::_disclose_partner_responses` gated the partner channel on `!= ATTRIBUTED`, so
**every `factored_attributed` run inside the env received no partner messages at all** and its
attribution could never be settled by evidence. Fixed (gated on the family). It survived
because every factored-attribution number to date came from a driver calling
`observe_partner` directly, never through the env. Any env-path attribution number from
before today is measuring a dead channel.

### What would help from you

Nothing new is being asked of you on attribution -- it is CPU-cheap and finishing here. Keep
going with the sampled sweep and the k=20/k=30 re-runs at 12,000 episodes. If the sampled
feasibility search in section 4 is done, that answer is still the blocking one.
