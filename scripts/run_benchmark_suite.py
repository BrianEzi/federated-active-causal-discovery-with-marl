import argparse
import datetime
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPERIMENT_DEFINITIONS = {
    "EXP-1": {
        "name": "EXP-1 (Single Topo G0)",
        "description": "Sanity check on fixed forward chain Graph 0",
        "flags": ["--fixed_graph", "0", "--initial_budget", "20.0", "--no_inductive_graph_head"]
    },
    "EXP-2": {
        "name": "EXP-2 (Multi-Topo Standard)",
        "description": "Standard multi-agent benchmark across all 8 topologies with Inductive Graph Head",
        "flags": ["--initial_budget", "20.0", "--use_inductive_graph_head"]
    },
    # NOTE: EXP-3A/EXP-3B were designed to compare the Skew-Symmetric Tournament graph head
    # against a baseline unconstrained MLP graph head. That graph head was removed from the
    # actor networks in the ActionCategory INTERVENE/NOOP collapse refactor, so
    # --use_inductive_graph_head / --no_inductive_graph_head now select architecturally
    # identical actor classes -- running this pair currently only measures training-run
    # seed variance, not an architectural difference. Left in place for reference; do not
    # spend compute expecting a real distinction until the graph head is restored.
    "EXP-3A": {
        "name": "EXP-3A (Inductive Head Probe)",
        "description": "[Currently a no-op vs EXP-3B -- graph head removed] Architectural Inductive Bias Probe",
        "flags": ["--initial_budget", "20.0", "--use_inductive_graph_head"]
    },
    "EXP-3B": {
        "name": "EXP-3B (Baseline MLP Probe)",
        "description": "[Currently a no-op vs EXP-3A -- graph head removed] Architectural Inductive Bias Probe",
        "flags": ["--initial_budget", "20.0", "--no_inductive_graph_head"]
    },
    "EXP-4A": {
        "name": "EXP-4A (Curiosity Beta=0.0)",
        "description": "Curiosity exploration sweep beta=0.0 (Pure reward)",
        "flags": ["--intrinsic_coef", "0.0", "--use_inductive_graph_head"]
    },
    "EXP-4B": {
        "name": "EXP-4B (Curiosity Beta=0.05)",
        "description": "Curiosity exploration sweep beta=0.05 (Standard)",
        "flags": ["--intrinsic_coef", "0.05", "--use_inductive_graph_head"]
    },
    "EXP-4C": {
        "name": "EXP-4C (Curiosity Beta=0.10)",
        "description": "Curiosity exploration sweep beta=0.10 (High curiosity)",
        "flags": ["--intrinsic_coef", "0.10", "--use_inductive_graph_head"]
    },
    "EXP-5A": {
        "name": "EXP-5A (Curriculum Stage 1-3)",
        "description": "Topology Curriculum learning schedule",
        "flags": ["--curriculum", "--use_inductive_graph_head"]
    },
    "EXP-5B": {
        "name": "EXP-5B (No Curriculum Uniform)",
        "description": "Uniform random topology sampling without curriculum",
        "flags": ["--use_inductive_graph_head"]
    },
    "EXP-6A": {
        "name": "EXP-6A (Budget Scarcity B=5)",
        "description": "Strict budget scarcity (B=5.0)",
        "flags": ["--initial_budget", "5.0", "--action_cost", "1.0", "--use_inductive_graph_head"]
    },
    "EXP-6B": {
        "name": "EXP-6B (Budget Abundance B=20)",
        "description": "Abundant budget allocation (B=20.0)",
        "flags": ["--initial_budget", "20.0", "--action_cost", "1.0", "--use_inductive_graph_head"]
    },
    "EXP-7": {
        "name": "EXP-7 (Nonlinear ANM Mechanism)",
        "description": "Mechanism complexity probe with Additive Noise Models",
        "flags": ["--mechanism_type", "NONLINEAR_ANM", "--use_inductive_graph_head"]
    },
    "EXP-8A": {
        "name": "EXP-8A (Low Noise Scale 0.05)",
        "description": "Exogenous noise sensitivity sigma=0.05",
        "flags": ["--noise_scale", "0.05", "--use_inductive_graph_head"]
    },
    "EXP-8B": {
        "name": "EXP-8B (High Noise Scale 0.50)",
        "description": "Exogenous noise sensitivity sigma=0.50",
        "flags": ["--noise_scale", "0.50", "--use_inductive_graph_head"]
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Causal MARL Benchmark Suite Runner")
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma-separated list of experiment IDs (e.g. 'EXP-1,EXP-2,EXP-3A') or 'all'")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated list of random seeds (default: '42,43,44')")
    parser.add_argument("--num_episodes", type=int, default=2000,
                        help="Number of training episodes per run (default: 2000)")
    parser.add_argument("--eval_freq", type=int, default=50,
                        help="Evaluation logging frequency (default: 50)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log benchmark runs to Weights & Biases under project 'federated-causal-benchmarks'")
    return parser.parse_args()

def run_single_experiment(exp_id: str, seed: int, num_episodes: int, eval_freq: int,
                           base_out_dir: str, use_wandb: bool) -> dict:
    exp_def = EXPERIMENT_DEFINITIONS[exp_id]
    run_dir = os.path.join(base_out_dir, f"{exp_id}_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "src.train",
        "--seed", str(seed),
        "--num_episodes", str(num_episodes),
        "--eval_freq", str(eval_freq),
        "--output_dir", run_dir,
        "--checkpoint_dir", run_dir,
        "--save_file"
    ] + exp_def["flags"]
    
    if use_wandb:
        cmd += ["--use_wandb", "--wandb_project", "federated-causal-benchmarks", "--run_name", f"{exp_id}_s{seed}"]
        
    print(f"\n[RUNNING] {exp_def['name']} | Seed {seed} | Episodes {num_episodes}")
    print(f"Command: {' '.join(cmd)}")
    
    res = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    
    if res.returncode != 0:
        print(f"❌ Run failed with exit code {res.returncode}")
        print(f"STDERR:\n{res.stderr}")
        return {
            "exp_id": exp_id, "seed": seed, "success": False,
            "error": res.stderr
        }
        
    # Read metrics CSV and trace JSON
    metrics_path = os.path.join(run_dir, "training_metrics.csv")
    trace_path = os.path.join(run_dir, "evaluation_trace.json")
    
    final_reward, final_shd, final_f1 = np.nan, np.nan, np.nan
    if os.path.exists(metrics_path):
        df_m = pd.read_csv(metrics_path)
        if "train/episode_reward" in df_m.columns:
            final_reward = float(df_m["train/episode_reward"].dropna().iloc[-1])
        if "eval/shd" in df_m.columns:
            final_shd = float(df_m["eval/shd"].dropna().iloc[-1])
        if "eval/f1" in df_m.columns:
            final_f1 = float(df_m["eval/f1"].dropna().iloc[-1])
            
    trace_shd_mean = np.nan
    trace_f1_mean = np.nan
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            trace = json.load(f)
        shds, f1s = [], []
        for g_k, g_v in trace.items():
            if g_k.startswith("graph_"):
                last_step = g_v["steps"][-1]
                shds.append(last_step.get("shd", np.nan))
        if shds:
            trace_shd_mean = float(np.nanmean(shds))
            
    print(f"✅ Finished: Final Reward={final_reward:.2f}, Final SHD={final_shd:.2f}, Final F1={final_f1:.3f}, Trace Avg SHD={trace_shd_mean:.2f}")
    
    return {
        "exp_id": exp_id,
        "exp_name": exp_def["name"],
        "seed": seed,
        "success": True,
        "final_reward": final_reward,
        "final_shd": final_shd,
        "final_f1": final_f1,
        "trace_avg_shd": trace_shd_mean,
        "run_dir": run_dir
    }

def main():
    args = parse_args()
    
    if args.experiments.lower() == "all":
        target_exps = list(EXPERIMENT_DEFINITIONS.keys())
    else:
        target_exps = [e.strip() for e in args.experiments.split(",") if e.strip() in EXPERIMENT_DEFINITIONS]
        
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = os.path.join("benchmarks", f"suite_{timestamp}")
    os.makedirs(base_out_dir, exist_ok=True)
    
    print(f"================================================================================")
    print(f"🚀 STARTING BENCHMARK SUITE: {len(target_exps)} Experiments x {len(seeds)} Seeds")
    print(f"Output Directory: {base_out_dir}")
    print(f"================================================================================")
    
    results = []
    for exp_id in target_exps:
        for seed in seeds:
            res = run_single_experiment(exp_id, seed, args.num_episodes, args.eval_freq, base_out_dir, args.use_wandb)
            results.append(res)
            
    # Aggregate results across seeds
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(base_out_dir, "raw_benchmark_results.csv"), index=False)
    
    successful = df_res[df_res["success"] == True]
    if not successful.empty:
        agg = successful.groupby("exp_name").agg({
            "final_reward": ["mean", "std"],
            "final_shd": ["mean", "std"],
            "final_f1": ["mean", "std"],
            "trace_avg_shd": ["mean", "std"]
        })
        agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
        agg.to_csv(os.path.join(base_out_dir, "benchmark_summary.csv"))
        
        # Build Markdown summary report
        md_lines = [
            f"# 📊 Benchmark Suite Summary Report",
            f"**Timestamp**: `{timestamp}` | **Episodes per Run**: `{args.num_episodes}` | **Seeds**: `{seeds}`",
            "",
            "## 📋 Consolidated Performance Metrics (Mean ± Std)",
            "",
            "| Experiment Name | Final Reward | Final SHD | Final F1 | Trace Avg SHD |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]
        
        for exp_name, row in agg.iterrows():
            r_m, r_s = row["final_reward_mean"], row["final_reward_std"]
            shd_m, shd_s = row["final_shd_mean"], row["final_shd_std"]
            f1_m, f1_s = row["final_f1_mean"], row["final_f1_std"]
            t_shd_m, t_shd_s = row["trace_avg_shd_mean"], row["trace_avg_shd_std"]
            
            md_lines.append(
                f"| **{exp_name}** | {r_m:.2f} ± {r_s:.2f} | {shd_m:.2f} ± {shd_s:.2f} | {f1_m:.3f} ± {f1_s:.3f} | {t_shd_m:.2f} ± {t_shd_s:.2f} |"
            )
            
        md_content = "\n".join(md_lines)
        with open(os.path.join(base_out_dir, "benchmark_summary.md"), "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print("\n" + md_content)
        print(f"\nSaved benchmark summary to {os.path.join(base_out_dir, 'benchmark_summary.md')}")

if __name__ == "__main__":
    main()
