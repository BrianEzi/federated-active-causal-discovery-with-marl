# Myriad rescue, 20 August 2026

Everything pulled off `myriad.rc.ucl.ac.uk` so that nothing depends on a cluster checkout
staying alive. Fetched read-only; nothing on the cluster was popped, reset or deleted.

## What is here

    myriad_rescue_20260820.tar.gz    ~/marl_sa_fast/results  (363 JSON files)
                                     ~/marl_sa_fast/logs     (job stdout/stderr)
                                     three git stash patches (see below)
    stash0_sa_fast.patch             re-fetched separately -- see the warning

## The two jobs that were never synced

- **180127 `sa_d6`** -- COMPLETE. 20 runs in `results/d6_exact/`: `d=6`, `n_obs` in
  {100, 1000}, budget in {2, 3}, seeds 0-4. Finished 20 Aug 16:07.
- **180124 `ma_cost0`** -- ran; logs in `logs/ma_cost0.*`.

**Flagged, not interpreted:** `sa_d6` reports `[FAIL] G5 gate 1 recorded: observational-only
rate 0.0000 against a singleton fraction of 0.0810`. The sample printed was an `n_obs=100`
configuration, and `n_obs=100` is already documented as too low for the posterior to reach
the identification threshold at all, so that is the first thing to rule out before treating
it as a `d=6` finding. The `n_obs=1000` runs are in the same directory and have not been
read yet.

## The stash patches

Three stashes existed and each held work that was not on any branch:

    stash0_sa_fast.patch        ~/marl_sa_fast   "cluster results before stepcost pull"
    stash1_sa_fast.patch        ~/marl_sa_fast   "untracked d7nobs results before gnn_budget pull"
    stash0_marl_causal.patch    ~/marl_causal    "pre-diagnostic-experiment-sync"

**Warning worth keeping.** `git stash show -p` without `--include-untracked` reports an
EMPTY patch when the stash is entirely untracked files. `stash0_sa_fast` first came back at
0 bytes and looked like an empty stash; with the flag it is 3.8 MB. If you ever export a
stash, pass `--include-untracked` and check the byte count.

To inspect without applying: `git apply --stat <patch>`. To restore: `git apply <patch>`.

The stashes are still on the cluster as well. This is a copy, not a move.
