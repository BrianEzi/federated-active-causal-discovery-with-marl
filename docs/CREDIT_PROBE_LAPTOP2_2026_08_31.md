# Credit probe — laptop 2's k12 contribution, read alongside k08

Written 31 Aug 2026. This machine ran `k12s50n04b150` per
[`HANDOVER_LAPTOP2_NEXT.md`](HANDOVER_LAPTOP2_NEXT.md); the primary machine's
`k08s50n04b150` result (`results/credit/k08s50n04b150_credit_probe.json`) already answers
the decisive question, pulled in by this session's own `git pull` before writing this note.

## The k08 result, read directly off the primary machine's own file

| arm | mean | per-seed |
|---|---|---|
| pooled, credit off | 0.977 | 0.975, 0.985, 0.970 |
| pooled, credit on | 0.958 | 0.940, 0.955, 0.980 |
| E4 (FedAvg), credit off | **0.510** | 0.425, 0.845, 0.260 |
| E4 (FedAvg), credit on | **0.922** | 0.915, 0.985, 0.865 |

**Turning credit on takes E4 from 0.510 to 0.922 — closing nearly all of the gap to
pooled, and collapsing the seed variance at the same time** (spread 0.26-0.845 down to
0.865-0.985). Pooled barely moves either way (0.977 vs 0.958, both near ceiling with
greedy at 0.94-0.95). This is exactly the signature the probe was designed to detect: the
FedAvg gap was credit, not federation.

## This machine's contribution: k12, chosen because it is NOT saturated

k12s50n04b150 has real headroom (greedy ~0.92, learner up to 1.000) where k08 is nearly
saturated (ceiling 0.995, greedy 0.94-0.95), so it is the cell that would show a credit
effect if pooled sharing were masking one. All 6 `pooled` seeds are done:

| arm | mean | per-seed |
|---|---|---|
| pooled, credit off | 0.998 | 0.995, 1.000, 1.000 |
| pooled, credit on | 0.997 | 0.995, 0.995, 1.000 |

**No credit effect under pooled sharing at k12 either** -- both saturate near ceiling,
consistent with k08's pooled row. This is the SAME direction as k08, at a cell explicitly
chosen because it should have shown a difference if one existed at the pooled arm. It
did not, which is itself informative: pooled sharing's variance reduction is strong enough
to swamp the phantom-transition noise even where the task has headroom to show a gap.

**The E4 arm (the one that actually decides anything new) is still running** as of this
commit -- 6 more jobs, `E4_credit_s{0,1,2}` and `E4_nocredit_s{0,1,2}`. Will push as a
follow-up commit once complete. Expect it to look like a milder version of k08's E4 rows,
since k12's pooled arm already shows less headroom-for-a-gap than k08's did.

## A tooling fix worth carrying forward

`scripts/credit_probe.py` had two portability bugs, fixed in this commit:

1. `ENV = {"PATH": "/usr/bin:/bin", ...}` REPLACED the whole environment rather than adding
   to it -- fatal on Windows (deletes `SystemRoot`, breaks DLL loading before the child can
   even print an error). Now copies `os.environ` and overrides only the three variables the
   probe cares about.
2. `command()` hardcodes `argv[0] = ".venv/bin/python"`, which only resolves via a shell
   that understands shebangs (`launch.sh`'s xargs+sh). `subprocess.run()` without a shell
   calls the OS process API directly and cannot execute it. Now overwritten with
   `sys.executable` -- the interpreter the probe is already running under, portable
   everywhere.

**A red herring worth naming so nobody re-chases it**: on this machine, every python.exe
launched this way shows up as TWO OS processes -- one at ~4.5MB / near-zero CPU (a launcher
stub) and one at 650MB-2GB / real CPU (the actual worker). `Get-Process`/Task Manager will
show 2x the expected process count for `--workers N`; this is normal on this Python
distribution, not duplicate work. Confirmed by checking CPU/memory per PID rather than
count alone -- do that before killing anything that looks doubled.

**Operational note, so the timing on this machine's numbers isn't over-read**: this run
was interrupted for about 12 hours overnight by the machine sleeping (an operator error
correcting a power setting that had been reset earlier, now fixed for good). This affected
wall-clock only -- no job failed, no output was corrupted, `git log` for `logs/*` (untracked,
not pushed) has the full timeline if anyone wants to audit it.
