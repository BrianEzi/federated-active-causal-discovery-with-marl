# Raw experiment data

Everything reported in the write-up derives from these files. Nothing is transcribed by
hand — `scripts/analyse_sweep.py` reads `raw/` and emits `all_runs.csv`, and the charts and
prose are built from that CSV. The previous round of this project assembled figures from
several places by hand and had to retract them; a single derivation path is the cheapest
guard against repeating that.

## Layout

    raw/<tag>.json     one file per configuration, every stage in one directory
    all_runs.csv       one row per (configuration, seed) — the analysis table
    all_summary.json   per-configuration seed spread (min / median / max)

## What each result file contains

    args               every flag the run was invoked with
    provenance         git commit, python / numpy / torch versions, host, UTC finish time
    space              n_dags, n_mecs, singleton_fraction for that d
    references         the four baseline policies' solve rate and mean cost
    per_seed           full metric set per seed, deterministic AND sampled
    training_history   entropy, solve rate, mean length, losses over training
    summary            the pass/fail verdict against the pinned criteria

`training_history` is the bulk of the size and is kept deliberately: entropy and solve-rate
trajectories are how a collapse is diagnosed after the fact, and re-running a night of jobs
to recover a curve is not an acceptable cost.

## Reproducing a run

Every result file carries its own command line. To repeat one:

    python -m scripts.run_experiment $(python - <<'PY'
    import json; a = json.load(open('results/raw/<tag>.json'))['args']
    print(' '.join(f'--{k} {v}' for k, v in a.items() if v not in (None, False)))
    PY
    )

Note that `torch` differs between the cluster (2.6.0+cpu) and the laptop (2.10.0+cpu);
numpy and scipy are pinned to match. The version used is recorded per run in `provenance`.

## A caveat on reading the table

`gap_closed` is `(random − agent) / (random − greedy)` on episode cost, where unsolved
episodes are charged at the full budget. That last clause means **`budget` is not a neutral
lever**: raising it multiplies the penalty for the same underlying failure, so `budget_10`
and `budget_40` are not comparable as "the same agent under a different budget".
