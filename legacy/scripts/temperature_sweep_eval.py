"""Runs evaluate_checkpoint at a sweep of temperatures against one saved checkpoint,
saving one evaluation_trace.json per temperature and printing a quick collapse-metric
summary for each -- see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md's greedy-policy-
collapse discussion (a fully deterministic policy doesn't make much sense to treat as
*the* evaluation target for an active-experiment-design task, so it's worth checking a
range rather than only temperature=0.0).

Usage: python -m legacy.scripts.temperature_sweep_eval --checkpoint_path diag_runs/RUN/checkpoints/best_ippo_params.pkl --output_dir diag_runs/RUN
"""
import argparse
import json
import os

from src.evaluate import evaluate_checkpoint


def summarize(trace: dict) -> dict:
    static_count = never_intervenes_count = reached0_count = diverse_count = 0
    total = 0
    for g in range(8):
        steps = trace[f"graph_{g}"]["steps"]
        first_shd, final_shd = steps[0]["shd"], steps[-1]["shd"]
        targets0, targets1 = set(), set()
        n_interv = 0
        for s in steps:
            a0, a1 = s["actions"]["agent_0"], s["actions"]["agent_1"]
            if a0["cat"] == 0:
                targets0.add(a0["target"]); n_interv += 1
            if a1["cat"] == 0:
                targets1.add(a1["target"]); n_interv += 1
        total += 1
        static_count += int(first_shd == final_shd)
        never_intervenes_count += int(n_interv == 0)
        reached0_count += int(any(s["shd"] == 0.0 for s in steps))
        diverse_count += int(len(targets0 | targets1) >= 2)
    return {
        "static_rate": static_count / total,
        "never_intervenes_rate": never_intervenes_count / total,
        "reached0_rate": reached0_count / total,
        "diverse_rate": diverse_count / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperatures", type=str, default="0.0,0.2,0.5,1.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temperatures = [float(t) for t in args.temperatures.split(",")]

    print(f"{'temperature':>12} | {'static':>8} | {'never_int':>10} | {'reached0':>9} | {'diverse':>8}")
    for t in temperatures:
        trace = evaluate_checkpoint(ckpt_path=args.checkpoint_path, temperature=t, seed=args.seed)
        out_path = os.path.join(args.output_dir, f"eval_trace_temp{t}.json")
        with open(out_path, "w") as f:
            json.dump(trace, f, indent=2)
        s = summarize(trace)
        print(f"{t:>12.2f} | {s['static_rate']:>8.1%} | {s['never_intervenes_rate']:>10.1%} | "
              f"{s['reached0_rate']:>9.1%} | {s['diverse_rate']:>8.1%}")


if __name__ == "__main__":
    main()
