# Reproducing a reported number

One worked example, then the general pattern.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest -m "not slow and not perf"

The fast subset is 489 tests and takes a few minutes. It includes a cross-validation of the
belief engine against the independent implementation in `crosscheck/` and `legacy/`, so a
clean run is evidence about correctness rather than merely about imports.

## Worked example: the principal cell

The principal cell is twelve variables per window, four agents, half the interface shared,
at the default intervention budget. To re-score its policies:

    .venv/bin/python scripts/global_shd_paired.py \
        submission/sweep12k/data/k12s50n04b150_s0.json \
        submission/sweep12k/data/k12s50n04b150_s1.json \
        submission/sweep12k/data/k12s50n04b150_s2.json \
        --episodes 200 --checkpoint best --sample \
        --out /tmp/principal.json

This rebuilds the environment from each run's own recorded config, loads the selected
checkpoint, replays 200 seeded episodes for the learned, myopic and random arms, and writes
per-episode vectors alongside the means. Because the episodes are seeded, the result should
match `submission/checkpoint/data/shd__k12s50n04b150.json` to the digits reported.

Add `--checkpoint final` for the other convention. The dissertation reports both, and they
disagree.

## The general pattern

1. Find the claim in `thesis_results/CLAIMS.md`. Read its boundary and its MUST NOT lines
   first; several record readings the data does not support.
2. Find the group in `submission/MANIFEST.md` that supports it.
3. Re-score from `submission/<group>/data/`, as above. Checkpoints are resolved from the
   sibling `checkpoints/` directory automatically.

## Rebuilding the reported tables and figures

    .venv/bin/python scripts/collect_thesis_results.py   # gather measurements
    .venv/bin/python scripts/build_claims.py             # regenerate CLAIMS.md
    .venv/bin/python scripts/figures.py                  # every figure
    .venv/bin/python scripts/build_tables12k.py          # the axis tables
    .venv/bin/python scripts/build_robustness.py         # the robustness table
    .venv/bin/python scripts/build_appendix.py           # the appendix

These read `results/` and write into the dissertation source tree, which is versioned
separately and is not in this repository. They will report the numbers either way.

## Verifying the shipped data

    .venv/bin/python scripts/build_submission.py --check

Re-hashes every file in `submission/` against its source in `results/` and reports drift or
absence. Both counts should be zero.

## Checks that guard the reporting

    .venv/bin/python scripts/check_reporting.py --captions   # thin support, caption completeness
    .venv/bin/python scripts/mark_provenance.py --check      # registered data patterns resolve

`check_reporting.py` counts how many episodes actually contribute a non-zero error to each
reported mean and flags any resting on five or fewer. It exists because a figure once drew
a confident line between two points, one of which was a mean over 600 episodes with exactly
one non-zero.

## Re-running training

Training is expensive: a single 12,000-episode cell is hours, and the reported sweep is
sixty runs. To retrain one cell with the settings a stored run used:

    .venv/bin/python scripts/train_from_config.py <a run json> --seed 0 --out <new json>

Without `--run` it prints the reconstructed command and verifies, field by field, that the
config it would produce matches the source. Run that check before spending the compute; it
exists because a reconstructed command silently dropped a flag once and inverted an
ablation.
