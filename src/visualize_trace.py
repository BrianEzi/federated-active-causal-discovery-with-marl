import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional

def parse_and_visualize_trace(
    trace_path: str = "evaluation_trace.json",
    output_dir: str = "plots",
    temperature: Optional[float] = None
):
    """
    Parses an evaluation_trace.json file and creates visual plots of the SHD 
    progression for each evaluated graph, as well as an aggregate summary plot.
    Supports user-specified or auto-detected sampling temperature annotations.
    """
    if not os.path.exists(trace_path):
        print(f"Error: {trace_path} not found.")
        return

    with open(trace_path, "r") as f:
        trace = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    
    # Extract temperature from metadata if available
    metadata = trace.get("metadata", {})
    if temperature is None and "temperature" in metadata:
        temperature = metadata["temperature"]

    temp_str = f" (Temp: {temperature})" if temperature is not None else ""
    print(f"=== Loaded Trace: {trace_path}{temp_str} ===")

    all_shd_histories = {}
    
    for graph_key, episode_data in trace.items():
        if not graph_key.startswith("graph_"):
            continue
            
        graph_idx = episode_data["graph_idx"]
        print(f"\n--- Topology {graph_idx} ---")
        
        steps = episode_data["steps"]
        shd_history = []
        
        for step_data in steps:
            step_idx = step_data["step"]
            shd = step_data["shd"]
            shd_history.append(shd)
            
            # Print human-readable actions
            action_strs = []
            for agent_id, acts in step_data["actions"].items():
                cat = acts["cat"]
                target = acts["target"]
                
                cat_str = "Observe" if cat == 0 else ("Intervene" if cat == 1 else "Peer Request")
                action_strs.append(f"{agent_id}: {cat_str} on Node {target}")
            
            print(f"Step {step_idx} | SHD: {shd:.1f} | " + " | ".join(action_strs))

        all_shd_histories[graph_idx] = shd_history

        # Plot individual SHD progression
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(shd_history)), shd_history, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        plt.title(f"SHD Progression - Topology {graph_idx}{temp_str}", fontsize=12, fontweight='bold')
        plt.xlabel("Rollout Step", fontsize=10)
        plt.ylabel("Structural Hamming Distance (SHD)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, max(max(shd_history) + 2, 8))
        
        plot_path = os.path.join(output_dir, f"shd_progression_graph_{graph_idx}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved SHD progression plot to {plot_path}")

    # Generate 8-Topology Combined Grid Summary
    if all_shd_histories:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for idx in range(8):
            ax = axes[idx]
            if idx in all_shd_histories:
                hist = all_shd_histories[idx]
                ax.plot(range(len(hist)), hist, marker='o', color='#2ca02c', linewidth=2, label=f"Top {idx}")
                ax.set_title(f"Topology {idx} (Final SHD: {hist[-1]:.0f})", fontsize=11, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(0, 8)
            if idx >= 4:
                ax.set_xlabel("Step", fontsize=9)
            if idx % 4 == 0:
                ax.set_ylabel("SHD", fontsize=9)
                
        fig.suptitle(f"Multi-Topology Causal Discovery Progression{temp_str}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "all_topologies_summary.png")
        plt.savefig(summary_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved 8-topology grid summary to {summary_path}")

def compare_temperatures_and_visualize(
    ckpt_path: str = "checkpoints/best_ippo_params.pkl",
    temperatures: List[float] = [0.0, 0.2, 0.5, 1.0],
    output_dir: str = "plots/temperature_comparison",
    seed: int = 42
):
    """
    Evaluates a checkpoint across a range of temperature scales, plots comparative
    SHD trajectories for each topology, and generates an aggregate mean-SHD comparison.
    """
    from src.evaluate import evaluate_checkpoint
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Running Multi-Temperature Evaluation across T = {temperatures} ===")
    
    temp_results = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for t_idx, temp in enumerate(temperatures):
        print(f"\nEvaluating with Temperature = {temp:.2f}...")
        trace = evaluate_checkpoint(ckpt_path=ckpt_path, temperature=temp, seed=seed)
        
        shds_per_topo = {}
        for graph_key, ep in trace.items():
            if graph_key.startswith("graph_"):
                g_idx = ep["graph_idx"]
                shds_per_topo[g_idx] = [s["shd"] for s in ep["steps"]]
        temp_results[temp] = shds_per_topo

    # 1. Overlay plots per topology
    for g_idx in range(8):
        plt.figure(figsize=(7, 4.5))
        for t_idx, temp in enumerate(temperatures):
            color = colors[t_idx % len(colors)]
            if g_idx in temp_results[temp]:
                hist = temp_results[temp][g_idx]
                plt.plot(range(len(hist)), hist, marker='o', label=f"Temp = {temp:.2f} (Final: {hist[-1]:.0f})", color=color, linewidth=2)
                
        plt.title(f"Temperature Comparison - Topology {g_idx}", fontsize=12, fontweight='bold')
        plt.xlabel("Step", fontsize=10)
        plt.ylabel("SHD", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper right", framealpha=0.9)
        plt.ylim(0, 8)
        
        plot_path = os.path.join(output_dir, f"temp_comparison_topo_{g_idx}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()

    # 2. Aggregate Mean-SHD curve across all topologies for each temperature
    plt.figure(figsize=(8, 5))
    for t_idx, temp in enumerate(temperatures):
        color = colors[t_idx % len(colors)]
        all_hist = [temp_results[temp][g] for g in range(8) if g in temp_results[temp]]
        if all_hist:
            mean_curve = np.mean(all_hist, axis=0)
            plt.plot(range(len(mean_curve)), mean_curve, marker='s', label=f"Temp {temp:.2f} (Final Mean SHD: {mean_curve[-1]:.2f})", color=color, linewidth=2.5)
            
    plt.title("Mean SHD Across All 8 Topologies vs. Temperature Scale", fontsize=13, fontweight='bold')
    plt.xlabel("Rollout Step", fontsize=11)
    plt.ylabel("Mean Structural Hamming Distance (SHD)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    
    summary_path = os.path.join(output_dir, "temperature_scale_mean_comparison.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved temperature comparison plots to {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Evaluation Trace with Temperature Control")
    parser.add_argument("--trace_path", type=str, default="evaluation_trace.json", help="Path to evaluation_trace.json")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory where output PNG plots are saved")
    parser.add_argument("--temperature", "-t", type=float, default=None, help="Sampling temperature scale annotation")
    parser.add_argument("--compare_temperatures", nargs="+", type=float, default=None, help="List of temperatures to compare (e.g. --compare_temperatures 0.0 0.2 0.5 1.0)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_ippo_params.pkl", help="Checkpoint path for multi-temperature sweep")
    args = parser.parse_args()
    
    if args.compare_temperatures is not None:
        compare_temperatures_and_visualize(
            ckpt_path=args.checkpoint_path,
            temperatures=args.compare_temperatures,
            output_dir=args.output_dir
        )
    else:
        parse_and_visualize_trace(args.trace_path, args.output_dir, temperature=args.temperature)
