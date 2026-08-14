"""Runs evaluate_checkpoint at a sweep of temperatures against one saved checkpoint,
saving one evaluation_trace.json per temperature and printing a quick collapse-metric
summary for each -- see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md's greedy-policy-
collapse discussion (a fully deterministic policy doesn't make much sense to treat as
*the* evaluation target for an active-experiment-design task, so it's worth checking a
range rather than only temperature=0.0).

Usage: python -m scripts.temperature_sweep_eval --checkpoint_path diag_runs/RUN/checkpoints/best_ippo_params.pkl --output_dir diag_runs/RUN
"""
import argparse
import json
import os

from src.evaluate import evaluate_checkpoint


def summarize(trace: dict) -> dict:
    static_count = never_intervenes_count = reached0_count = diverse_count = 0
    ended_at_zero_count = 0
    total = 0
    oracle_informative = oracle_optimal = oracle_scored = 0
    step0_informative = step0_total = 0
    for g in range(8):
        steps = trace[f"graph_{g}"]["steps"]
        first_shd, final_shd = steps[0]["shd"], steps[-1]["shd"]
        targets0, targets1 = set(), set()
        n_interv = 0
        for i, s in enumerate(steps):
            a0, a1 = s["actions"]["agent_0"], s["actions"]["agent_1"]
            if a0["cat"] == 0:
                targets0.add(a0["target"]); n_interv += 1
            if a1["cat"] == 0:
                targets1.add(a1["target"]); n_interv += 1
            for a in (a0, a1):
                o = a.get("oracle")
                if not o:
                    continue
                oracle_scored += 1
                informative = o["best_score"] > 1e-9
                oracle_informative += int(informative)
                oracle_optimal += int(informative and o["is_optimal"] > 0.5)
                if i == 0:
                    step0_total += 1
                    step0_informative += int(informative)
        total += 1
        static_count += int(first_shd == final_shd)
        never_intervenes_count += int(n_interv == 0)
        reached0_count += int(any(s["shd"] == 0.0 for s in steps))
        # The metric that actually matters for deployment: did the episode FINISH correct,
        # not merely pass through SHD=0 transiently (often at step 0, from the initial
        # observational guess) and then drift away. See the investigation doc's Track B
        # section for why the naive reached0 figure was misleading.
        ended_at_zero_count += int(final_shd == 0.0)
        diverse_count += int(len(targets0 | targets1) >= 2)
    return {
        "static_rate": static_count / total,
        "never_intervenes_rate": never_intervenes_count / total,
        "reached0_rate": reached0_count / total,
        "ended_at_zero_rate": ended_at_zero_count / total,
        "diverse_rate": diverse_count / total,
        # Oracle agreement over INFORMATIVE steps only -- steps where the oracle actually
        # had a preference. Counting steps where every legal target ties at zero is what
        # produced the retracted 99.4-100% figure; see src/evaluate.py's oracle_summary.
        "oracle_optimal_rate": (oracle_optimal / oracle_informative) if oracle_informative else None,
        "oracle_informative_rate": (oracle_informative / oracle_scored) if oracle_scored else None,
        # Regression canary: step 0 must stay informative. If this drops toward zero the
        # observational shortcut has returned and the oracle metric is vacuous again.
        "step0_informative_rate": (step0_informative / step0_total) if step0_total else None,
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

    def pct(v):
        return "     n/a" if v is None else f"{v:>7.1%}"

    header = (f"{'temp':>5} | {'static':>7} | {'never_int':>9} | {'reached0':>8} | "
              f"{'ENDED@0':>7} | {'diverse':>7} | {'orc_opt':>7} | {'orc_inf':>7} | {'step0_inf':>9}")
    print(header)
    print("-" * len(header))
    for t in temperatures:
        trace = evaluate_checkpoint(ckpt_path=args.checkpoint_path, temperature=t, seed=args.seed)
        out_path = os.path.join(args.output_dir, f"eval_trace_temp{t}.json")
        with open(out_path, "w") as f:
            json.dump(trace, f, indent=2)
        s = summarize(trace)
        print(f"{t:>5.2f} | {s['static_rate']:>7.1%} | {s['never_intervenes_rate']:>9.1%} | "
              f"{s['reached0_rate']:>8.1%} | {s['ended_at_zero_rate']:>7.1%} | "
              f"{s['diverse_rate']:>7.1%} | {pct(s['oracle_optimal_rate'])} | "
              f"{pct(s['oracle_informative_rate'])} | {pct(s['step0_informative_rate']):>9}")


if __name__ == "__main__":
    main()
