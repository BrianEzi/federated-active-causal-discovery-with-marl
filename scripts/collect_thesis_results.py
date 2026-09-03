"""Assemble `thesis_results/` -- the files the dissertation actually quotes.

WHY THIS EXISTS. The numbers in the results chapter were scattered over `results/`, a 1.5 GB
working directory holding sixty scratch runs, superseded builds and abandoned experiments,
where `attr_scale.json`, `attr_scale_final.json`, `attr_scale_postfix.json` and
`attr_scale_component.json` all contain a k=30 row and only one of them is the one quoted.
That is how a figure ends up citing the wrong run.

So: one registry, below, mapping each CLAIM to the files that support it, the chapter section
that makes it, and the command that regenerates it. Copying is the small part; the registry is
the point, and it is the thing to update when a result changes.

    python scripts/collect_thesis_results.py            # build thesis_results/
    python scripts/collect_thesis_results.py --check    # verify nothing has drifted

`--check` recomputes every hash against the source. A source that has been re-run since the
copy was made is reported as DRIFTED, which is the signal to re-read the number in the thesis
rather than to blindly re-copy.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "thesis_results"

# (folder, claim, thesis section, [source globs], regenerating command)
REGISTRY = [
    ("sweep",
     "Learned against myopic and random on all four swept axes: window size, agent count, "
     "contended fraction, budget. 20 cells x 3 seeds. Final-policy evaluation.",
     "4.2 (RQ1), figure sweep_grid",
     ["results/sweep/oracle/k*s*n*b*_s?.json"],
     "scripts/ma_train.py per results/sweep/oracle/jobs/*.sh"),

    ("sweep12k",
     "THE PRIMARY SWEEP. Every cell retrained to 12,000 episodes, the budget at which the "
     "policies converge, and measured with scripts/global_shd_paired.py at the selected "
     "checkpoint, the final update, and update 500 (8,000 episodes). The 4,000-episode "
     "`sweep` folder is retained because Chapter 4 reports what that budget did to three "
     "structural claims, not because it is the headline. Joint recovery has the learned arm "
     "ahead of the myopic rule in 2 of 18 cells at 4,000 episodes and 16 of 18 at 12,000.",
     "RQ1, tables tab:12k_*, figure crossover_budget",
     ["results/sweep12k/k*s*n*b*_s?.json",
      "results/sweep12k/shd/*.json", "results/sweep12k/shd_final/*.json",
      "results/sweep12k/shd_u0500/*.json",
      "results/longcheck/shd_n05_12k.json", "results/longcheck/shd_n08_12k.json",
      "results/longcheck/shd_n10_12k.json", "results/longcheck/shd_s75_12k.json"],
     "scripts/build_sweep12k.py to generate jobs, then scripts/measure_sweep12k.py "
     "--conventions best,final,u0500"),

    ("checkpoint",
     "Early-stopped against final policy on the window-size axis, 200 paired episodes per "
     "seed. Establishes that the checkpoint choice is inert below the crossover and worth "
     "2.3x at k=20 and 16x at k=30 above it.",
     "4.1.1 and 4.2 (RQ1), figure checkpoint",
     # results/rerows, not results/ckpt: the ckpt/ set was measured before the evaluation
     # RNG was seeded (2 Sep 21:15) and does not reproduce. The rerows set is the same
     # measurement under the fixed path, and its myopic arm matches ckpt/ to five decimals
     # at every k, which is what confirms only the learned arm was ever affected.
     ["results/rerows/k??_best.json", "results/rerows/k??_final.json"],
     "scripts/global_shd_paired.py --episodes 200 --sample --checkpoint {best,final}"),

    ("attribution",
     "RQ4. attr_ceiling: recovery by group size and peer count. attr_ceiling_matched_budget: "
     "the control holding rounds-per-agent fixed, which is what rules out budget starvation. "
     "attr_ceiling_budget: the coverage step function (21/1056 at budget 30; 349/1056 at "
     "60, 120 and 240 -- IDENTICAL counts, not merely equal rates). attr_scale_final and "
     "attr_reach: k=30/40/50 at 30 episodes each, zero misattributions. attr_train: training "
     "under the ATTRIBUTION BELIEF BACKEND, scored on the structural criterion -- NOT an "
     "attribution reward, which no run in this project uses. attr/transfer_*: the "
     "self-interested attribution baseline.",
     "RQ4, figure attribution_law",
     ["results/attr_ceiling.json", "results/attr_ceiling_matched_budget.json",
      "results/attr_ceiling_budget.json", "results/attr_scale_final.json",
      "results/attr_reach.json", "results/attr_train/*.json",
      "results/attr/transfer_k12s50n04b200_s?.json"],
     "scripts/attr_ceiling.py, scripts/attr_model.py"),

    ("federation",
     "RQ3, plus the sweep's training-budget limitation. v2_k12_* and v2_k20_*: arm A is "
     "the federated baseline, arm E removes the "
     "information partition (partners' beliefs and counts observed) and the optimiser "
     "partition (trajectories pooled instead of FedAvg). Action rights stay partitioned in "
     "both. shd_k20_*: the same arms on the primary metric, where the recovery rate has "
     "saturated and cannot separate them. longcheck/*_long_s2: all seven competence-floor "
     "exclusions retrained at 12,000 episodes; all seven pass and all seven beat the myopic "
     "rule. lrcheck/*: the same runs at lr 1e-4, which makes them worse and rules out an "
     "unstable step size.",
     "4.4 (RQ3), figure federation",
     # central12k, not central: the k=12 ladder at 4,000 episodes is superseded, and its one
     # significant seed was an unconverged centralised run. k=20 stays in central/ because
     # those six runs trained at 12,000 and exist nowhere else.
     ["results/central12k/v2_k12_?_s?.json", "results/central/v2_k20_?_s?.json",
      "results/rerows/ladder12k_?_best.json", "results/rerows/ladder12k_?_final.json",
      "results/rerows/shd_A.json", "results/rerows/shd_E.json",
      "results/rerows/shd_A_s345.json", "results/rerows/shd_E_s345.json",
      "results/rerows/shd_k20_A.json", "results/rerows/shd_k20_E.json",
      "results/longcheck/*_long_s?.json", "results/longcheck/*_conv_s?.json",
      "results/lrcheck/*.json"],
     "results/central/jobs/*.sh and jobs2/*.sh, then global_shd_paired.py"),

    ("power",
     "RQ2, the answer-rate arm. Policies trained under a PARTIAL ORACLE -- ancestry answers "
     "withheld with probability 1 - rho, ~0.085 s/episode -- evaluated under genuine sampled "
     "evidence at 6-9 s/episode, which they never saw in training. The grid is 7 rates x 3 "
     "seeds x 200 paired episodes, taken from `deterministic/`: the same 21 cells were first "
     "built on an evaluation path that did not seed the torch RNG, so a learned arm scored "
     "with --sample was not reproducible, and `deterministic/` is the rebuild after that fix. "
     "Do not mix the two directories in one table. transfer_p{10,07,05} with p{10,07,05} is "
     "the isolation pair, differing in exactly one config field. repeat/ carries the coverage "
     "and repeat-rate mechanism probes; argmax/ the action-selection control.",
     "4.2 (RQ2), figure rho_curve",
     ["results/power/rho/deterministic/xfer_rho*_s?.json",
      "results/power/rho/rho[01].[0-9]*_s?.json",
      "results/power/rho/CURVE.json",
      "results/power/transfer_p[0-9][0-9].json", "results/power/p[0-9][0-9].json",
      "results/power/rho/repeat/*.json", "results/power/rho/argmax_det/*.json"],
     "scripts/run_rho_fleet.sh, then scripts/rebuild_grid_deterministic.sh, then "
     "scripts/rho_curve_report.py --dir results/power/rho/deterministic"),
]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def resolve(patterns):
    out = []
    for pattern in patterns:
        found = sorted(glob.glob(str(ROOT / pattern)))
        if not found:
            print(f"  !! no match: {pattern}")
        out += [pathlib.Path(f) for f in found]
    return out


def build(check_only: bool) -> int:
    lines = ["# Thesis results — the files the dissertation quotes", "",
             "Built by `scripts/collect_thesis_results.py`. **Do not edit by hand.** To add a",
             "result, add it to that script's `REGISTRY` so the claim it supports and the",
             "command that regenerates it travel with the data.", "",
             "`python scripts/collect_thesis_results.py --check` verifies that no source has",
             "been re-run since these copies were taken.", ""]
    drift = missing = copied = 0
    index = {}
    for folder, claim, section, patterns, command in REGISTRY:
        files = resolve(patterns)
        target = DEST / folder
        if not check_only:
            target.mkdir(parents=True, exist_ok=True)
        lines += [f"## `{folder}/` — {section}", "", claim, "",
                  f"Regenerate with: `{command}`", "",
                  "| file | sha256 (16) | source |", "|---|---|---|"]
        # Two sources can share a basename. results/sweep12k/shd/, shd_final/ and shd_u0500/
        # all hold `<cell>.json`, so a flat copy silently overwrote two of the three
        # conventions with the third -- 108 files collapsing to 76, with no error anywhere.
        # Disambiguate by the source's parent directory whenever a name repeats in this folder.
        seen_names = {}
        for src in files:
            seen_names.setdefault(src.name, []).append(src)
        clashing = {n for n, v in seen_names.items() if len(v) > 1}
        if clashing:
            print(f"  {folder}: {len(clashing)} basenames appear in more than one source "
                  f"directory; prefixing those with their parent")
        for src in files:
            digest = sha(src)
            rel = src.relative_to(ROOT)
            name = f"{src.parent.name}__{src.name}" if src.name in clashing else src.name
            dst = target / name
            if check_only:
                if not dst.exists():
                    print(f"  MISSING  {folder}/{name}"); missing += 1
                elif sha(dst) != digest:
                    print(f"  DRIFTED  {folder}/{name}  (source re-run since copy)"); drift += 1
            else:
                shutil.copy2(src, dst); copied += 1
            lines.append(f"| `{name}` | `{digest}` | `{rel}` |")
            index.setdefault(folder, []).append({"file": name, "sha256_16": digest,
                                                 "source": str(rel)})
        lines += ["", f"{len(files)} files.", ""]

    if check_only:
        print(f"\n{drift} drifted, {missing} missing")
        return 1 if (drift or missing) else 0
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "MANIFEST.md").write_text("\n".join(lines))
    (DEST / "manifest.json").write_text(json.dumps(index, indent=1))
    print(f"\ncopied {copied} files into {DEST.relative_to(ROOT)}/")
    print(f"wrote {(DEST / 'MANIFEST.md').relative_to(ROOT)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the copies still match their sources; do not write")
    return build(ap.parse_args(argv).check)


if __name__ == "__main__":
    raise SystemExit(main())
