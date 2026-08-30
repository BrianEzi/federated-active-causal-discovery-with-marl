# Second machine — next task, 30 August 2026 (evening)

Supersedes the "run the medium tier" instruction in
[`HANDOVER_LAPTOP2_2026_08_30.md`](HANDOVER_LAPTOP2_2026_08_30.md). **Do not run the medium
tier locally** — you have already submitted it to Myriad, and the local cost figures that
justified splitting it across machines were wrong (see section 3).

---

## 1. Run this

```bash
cd <repo>
git pull origin explore/constraint-based

.venv/bin/python scripts/preflight_runs.py health
.venv/bin/python scripts/credit_probe.py --cell k12s50n04b150 --seeds 3 --workers 4
```

About 2.8 core-hours; roughly 40 minutes on four workers. **This is the decision blocking
both machines and the cluster**, so it comes before anything else.

Run the health gate first and believe it. This machine spent most of today swap-thrashing
without anyone noticing, and every timing taken during it was inflated 2–5x. The metric is
**swapins per second, not swap used** — macOS leaves pages in swap indefinitely at no cost,
and it is reading them back that destroys throughput. 0–10 is fine, 1000+ means stop.

## 2. What it decides, and why it matters more than the FedAvg question it came from

Under round-robin only the ACTIVE agent's action is applied, but `turn_aware_credit=False`
— the default, and what every run so far has used — makes **every** agent store a
transition **every** round. Measured at four agents:

| | count | share |
|---|---|---|
| transitions stored | 1600 | |
| action actually applied | 400 | **25%** |
| action discarded | 1200 | **75%** |

The reward on a discarded row (+0.188, sd 0.387) is statistically indistinguishable from a
real one (+0.197, sd 0.391), because it is the consequence of *another* agent's move
credited to this agent's thrown-away action. And the observation carries 173 features, none
of which encode whose turn it is — so the policy cannot separate the two even in principle.
At eight agents it would be 87.5% phantom.

The probe crosses `turn_aware_credit` on/off with `pooled`/`E4`, because either factor alone
is uninterpretable. **If turning credit on closes the FedAvg gap, the gap was credit rather
than federation — and every run in the sweep needs `--turn_aware_credit` regardless of what
we decide about FedAvg.**

The primary machine is running the same probe at `k08s50n04b150`. Yours is at
`k12s50n04b150` deliberately: k=8 at beta=1.5 is nearly saturated (ceiling 0.995, greedy
0.945) so it has little room to show a difference, while k=12 has real headroom (greedy
0.92, learner 1.000). **Yours is the cell that actually settles it.**

Report: the four-row table the script prints, plus the per-seed successes.

## 3. Two things about the cluster jobs you submitted

**They may need resubmitting, and a stale file will silently prevent that.** All 21 tasks
carry the current config, with no `--turn_aware_credit`. If the probe says credit matters,
those results are at the wrong configuration. `_config_record` now sweeps every `MAConfig`
field, so the JSONs will record `turn_aware_credit: false` and are identifiable — but the
`[ -f "$OUT" ]` guard means the stale files would make a resubmission print "skipping"
rather than re-run. **If we change the config, that output directory must be cleared, not
merely relaunched.** Worth knowing before you see 21 tasks skip.

**The local cost figures that sent the medium tier to the cluster were wrong.** This machine
was swap-thrashing when they were measured. Re-measured clean:

| cell | measured while thrashing | measured clean |
|---|---|---|
| baseline k12 | 34.1 m | **14.0 m** |
| k20 | 236.5 m | **46.0 m** |
| k30 | 739.1 m | **207.5 m** |
| whole oracle sweep | 108 core-h | **33.2 core-h** |

The sweep is an evening on one machine, not a three-machine job. Leaving the medium tier on
Myriad is still fine — it costs queue time rather than ours — but the split is no longer
necessary, and the cluster is no longer load-bearing for the oracle sweep.

**Your k=30 OOM finding stands and is valuable.** A single ~2.8 GB tensor in the GNN's
neighbour-gather at k=30 with budget 98 is real arithmetic:
`1568 rows x 30 nodes x 29 neighbours x 128 hidden`. It is almost certainly what tipped the
primary machine into swap while its calibration was running, so you diagnosed from one
direction what was being chased from the other. Keep the 16G-per-slot request.

## 4. Do not do these yet

- **The sampled sweep.** The oracle→sampled multiplier is stale — 3.82x, measured before an
  optimisation pass that made evaluation ~2x faster. Being re-measured on the primary
  machine.
- **Rung 1 exact (k=5, attributed).** Also blocked on the credit decision, since
  `turn_aware_credit` affects any multi-agent run.
- **Re-scoring old results.** Needs the `.pt` checkpoints, which are not tracked in git and
  live only on the primary machine.

## 5. After the probe reports

Stop and report before starting anything else. The next task depends on the answer, and
starting the wrong thing costs more than the wait: the sampled sweep is the only genuinely
large job left, and running it at the wrong config would waste days rather than hours.
