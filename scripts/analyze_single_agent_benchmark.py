"""Aggregates and formats Single-Agent Optimal Experiment Designer benchmark results."""
import argparse
import glob
import json
import os
import numpy as np


def analyze_single_agent_matrix(runs_dir: str = "diag_runs"):
    eval_files = glob.glob(os.path.join(runs_dir, "single_agent_*", "eval_trace_temp0.0.json"))
    if not eval_files:
        print(f"No single-agent eval traces found in {runs_dir}/single_agent_*/eval_trace_temp0.0.json")
        return

    print("==========================================================================================")
    print("                SINGLE-AGENT OPTIMAL EXPERIMENT DESIGNER: EVALUATION RESULTS              ")
    print("==========================================================================================")
    print(f"{'Experiment':<35} | {'Mean SHD':<9} | {'SHD=0 Rate':<11} | {'Oracle Agree':<13} | {'Budget Left':<11}")
    print("------------------------------------------------------------------------------------------")

    for fpath in sorted(eval_files):
        exp_name = os.path.basename(os.path.dirname(fpath))
        with open(fpath, "r") as f:
            data = json.load(f)

        shds = []
        reached0 = []
        oracle_agrees = []
        budgets_left = []

        for g in range(8):
            g_data = data.get(f"graph_{g}")
            if not g_data:
                continue
            steps = g_data.get("steps", [])
            if steps:
                final_shd = steps[-1].get("shd", 0.0)
                shds.append(final_shd)
                reached0.append(float(final_shd == 0.0))
                budgets_left.append(float(steps[-1].get("budgets", [0.0])[0]))
            
            oracle_summary = g_data.get("oracle_summary")
            if oracle_summary and oracle_summary.get("optimal_rate") is not None:
                oracle_agrees.append(oracle_summary["optimal_rate"])

        mean_shd = np.mean(shds) if shds else float("nan")
        shd0_rate = np.mean(reached0) if reached0 else float("nan")
        mean_agree = np.mean(oracle_agrees) if oracle_agrees else float("nan")
        mean_budget = np.mean(budgets_left) if budgets_left else float("nan")

        print(f"{exp_name:<35} | {mean_shd:<9.2f} | {shd0_rate:<11.1%} | {mean_agree:<13.1%} | {mean_budget:<11.2f}")

    print("==========================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="diag_runs")
    args = parser.parse_args()
    analyze_single_agent_matrix(args.runs_dir)
