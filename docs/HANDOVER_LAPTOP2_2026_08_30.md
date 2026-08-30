# Handover — second machine, 30 August 2026

**You are running the MEDIUM TIER of the oracle sweep.** Everything you need is committed.
Read section 1, run section 2, and report section 5. Nothing here needs a decision from you.

Context: MSc thesis, *Federated Active Causal Discovery via Multi-Agent RL*, branch
`explore/constraint-based`. Freeze 31 Aug morning, experiments to 2 Sep, write-up to 7 Sep.
Work is split across three machines by MEASURED cost — the primary laptop is running the
cheap tier, you have the medium tier, and the cluster has the heavy cell and the sampled
sweep.

---

## 1. What you are running, and why it is safe to just start it

Six cells x three seeds, **54.9 core-hours**, longest single run 3.9 h. On six workers that
is roughly 10–11 hours wall, so it is an overnight job.

| cell | h/run | x3 seeds |
|---|---|---|
| `k20s50n04b150` | 3.94 | 11.8 |
| `k12s75n08b150` | 3.50 | 10.5 |
| `k12s25n08b150` | 3.36 | 10.1 |
| `k12s50n10b150` | 3.13 | 9.4 |
| `k12s50n08b150` | 2.29 | 6.9 |
| `k12s50n04b500` | 2.06 | 6.2 |

**It is restart-safe and there is no state to keep in sync.** Every job line is guarded by
`[ -f "$out" ] ||`, so re-running the launcher picks up exactly the runs with no result file
yet. The result files ARE the resume state. If the machine reboots, a job dies, or you Ctrl-C
it, just run the same command again.

Jobs are ordered longest-first from a real calibration (`results/sweep/calibration_oracle.json`),
so the long cells start immediately and the workers do not idle at the tail.

---

## 2. Run it

```bash
cd <repo>
git pull origin explore/constraint-based

# sanity: this project has NO bare `python` -- always .venv/bin/python
.venv/bin/python -c "import torch, numpy; print('env ok')"

bash scripts/launch.sh oracle 6 medium
```

That is the whole job. The launcher gates itself first (~1 minute: metric preflight, then a
feasibility check on just these six cells), then launches under `caffeinate` on macOS.

**Tune the worker count to the machine.** `6` assumes at least 8 cores. Runs are
single-threaded (`OMP_NUM_THREADS=1`), so throughput comes from running many at once; leave
2–4 cores free or the machine becomes unusable and the timings distort.

### Optional, only if you have spare cores

`k30s50n04b150` is the single heavy cell at 12.3 h/run, assigned to the cluster. Seed 0 is
the fallback in case the cluster queue does not deliver, and it is the cell carrying the
SCALING claim — the headline result. If you can spare one core for 12 hours:

```bash
SEED_LIST=0 bash scripts/launch.sh oracle 1 heavy
```

Run it in a separate terminal, after the medium tier is underway.

---

## 3. What NOT to do

- **Do not start the sampled sweep** (`--vs_evidence sampled` / `scripts/launch.sh sampled ...`).
  The oracle→sampled cost multiplier is stale — it was measured at 3.82x before an
  optimisation pass that made evaluation ~2x faster and training 1.18x faster, so the ratio
  has moved and nobody has re-measured it. Committing a machine to ~500 core-hours on a stale
  number is the thing to avoid. The primary machine is re-measuring it.
- **Do not change any training flag.** Every cell must differ only in `(k, sigma, n, beta)`
  and seed. Twice on this project a comparison was built from a neighbouring run's flags and
  had to be thrown away. `scripts/sweep.py` emits the commands precisely so nobody hand-types
  one.
- **Do not re-run the cheap tier.** The primary machine has it.
- **Do not `git push --force`,** and do not rebase the branch. Three machines share it.

---

## 4. If something goes wrong

| symptom | what it means | do |
|---|---|---|
| a job prints `FAILED` | one run died; the others are unaffected | re-run the same launch command, it retries only that one |
| `FEASIBILITY GATE FAILED` | a cell's budget cannot admit the optimal arm | **stop and report** — this means the experiment design is wrong, not the machine |
| `METRIC PREFLIGHT FAILED` | a metric is computed but dropped before reaching the file | **stop and report** — every number would be corrupted |
| machine slept, runs stopped | normal on a laptop | re-run the launch command |
| a run seems stuck for hours | `k20` legitimately takes 3.9 h | check `results/sweep/oracle/logs/<cell>_s<seed>.log` is still growing |

Never delete a `*.json` in `results/sweep/oracle/` to "start clean" — that is a finished run,
and deleting it costs hours to reproduce.

---

## 5. Report back

Push results as they finish — result JSONs are tracked deliberately (`.gitignore` has
`!results/**/*.json`; they are DATA, not build artefacts). The `.pt` checkpoints are NOT
tracked; leave them local.

```bash
git add results/sweep/oracle/*.json
git commit -m "results: medium tier, oracle sweep, <n> runs from laptop 2"
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true git push origin explore/constraint-based
```

**The `GIT_ASKPASS` part matters** — a plain `git push` on this repo hangs on an interactive
credential prompt and looks like a permissions failure. It is not; the credential is there.

When you report, state three things for any number you quote, or do not quote it:
**the MI gate, the evidence mode, and the evaluation policy (argmax or sampled).** Every
wrong claim on this project has come from one of those three being left implicit.

Useful one-liner for progress:

```bash
ls results/sweep/oracle/*.json 2>/dev/null | wc -l    # of 18 expected
```

---

## 6. Background, if you want it

- [`RUN_PLAN_2026_08_30.md`](RUN_PLAN_2026_08_30.md) — the plan, the costing, and the gates.
- [`FINDINGS_GRAPH_DISTRIBUTION_2026_08_30.md`](FINDINGS_GRAPH_DISTRIBUTION_2026_08_30.md) —
  what the training graphs are, and two confounds in them.
- [`FINDINGS_CLAMP_2026_08_30.md`](FINDINGS_CLAMP_2026_08_30.md) — why interventions are
  vary-only, and why "rescue" does not exist on this backend.
- [`METRICS.md`](METRICS.md) — what every reported field means.

Two things found today that explain why the gates exist: `scripts/ma_train.py` kept a private
copy of the baseline registry, so the sweep would have run with **no ceiling arm and no
coordinated control**; and resume state was being written and never read, so any run killed
by a walltime limit would have silently restarted its reward normaliser. Both are fixed. The
gates are cheap and they have already earned their keep — do not skip them.
