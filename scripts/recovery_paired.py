"""Joint recovery rate of an EXISTING checkpoint, on the run_arm convention.

WHY THIS EXISTS. scripts/global_shd_paired.py records hard/soft/resolved SHD and nothing
else, so the 4,000-episode policies at k=20 and k=30 -- which survive only as u0249
checkpoints, their run JSONs having been overwritten by the 12,000-episode copies on
1 Sep -- have measured SHD but no measured joint recovery. Every other recovery number in
the chapter comes from the run's own end-of-training eval pass, which is
ma/evaluate.py::run_arm at the FINAL checkpoint. This script calls that same run_arm on a
named checkpoint tag, so a number produced here sits on the same convention as the run
files: same success criterion, same episode worlds (seed * 100_000 + episode), same
episode count.

THE BUILT-IN CHECK. The greedy arm never reads the trained policy, so for a fixed
(cell, seed, episodes, evidence) its success rate here must EQUAL the run file's
greedy_uncertainty success exactly. If it does not, the environments differ and the
learned number is wrong too.

The torch RNG is seeded per arm (the 2 Sep fix); run_arm itself does not seed it. The
learned arm SAMPLES, matching both the run files' eval pass and the night's headline
convention.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                            # noqa: E402

from ma.baselines import make_baselines                                 # noqa: E402
from ma.evaluate import run_arm                                         # noqa: E402
from ma.policy import IndependentPPO                                    # noqa: E402
from scripts.rescore_from_config import env_from_config                 # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="run JSONs whose checkpoints to score")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--checkpoint", default="best",
                    help="best | final | an update tag such as u0249")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    payload = []
    for path in args.results:
        path = pathlib.Path(path)
        report = json.loads(path.read_text())
        config = report["config"]
        seed = report.get("seed", 0)
        env = env_from_config(config, seed=seed)

        if args.checkpoint == "best":
            checkpoint = path.with_name(path.stem + "_best.pt")
        elif args.checkpoint == "final":
            checkpoint = path.with_suffix(".pt")
        else:
            checkpoint = path.with_name(f"{path.stem}_{args.checkpoint}.pt")
        if not checkpoint.exists():
            print(f"!! {path.stem}: no {checkpoint.name}, skipped", flush=True)
            continue

        ppo = IndependentPPO.load(str(checkpoint), env)
        arms = {}
        torch.manual_seed(seed)
        arms["learned"] = run_arm(env, ppo.policies(deterministic=False),
                                  args.episodes, seed=seed)
        torch.manual_seed(seed)
        # The SAME registry ma_train.py uses, so the bar and construction match the run
        # file's arm exactly -- a hand-built agent here is how the bar drifted once before.
        builders = {a: make_baselines(env, a, seed) for a in env.topology.agents}
        arms["greedy_uncertainty"] = run_arm(
            env, {a: builders[a]["greedy_uncertainty"] for a in env.topology.agents},
            args.episodes, seed=seed)

        run_greedy = (report.get("arms", {})
                            .get("greedy_uncertainty", {}).get("success"))
        print(f"=== {path.stem} ({args.episodes} episodes, {args.checkpoint}, sampled) ===",
              flush=True)
        print(f"  learned success {arms['learned']['success']:.4f}   "
              f"greedy success {arms['greedy_uncertainty']['success']:.4f}   "
              f"run-file greedy {run_greedy}", flush=True)

        payload.append({
            "source": str(path), "seed": seed, "episodes": args.episodes,
            "checkpoint": args.checkpoint, "sampled": True,
            "eval_evidence": config.get("vs_evidence"),
            "greedy_matches_run_file": (run_greedy is not None and
                                        abs(arms["greedy_uncertainty"]["success"]
                                            - run_greedy) < 1e-12),
            "arms": arms,
        })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out} ({len(payload)} entries)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
