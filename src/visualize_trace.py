import json
import matplotlib.pyplot as plt
import os
from typing import Dict, Any

def parse_and_visualize_trace(trace_path: str = "evaluation_trace.json", output_dir: str = "plots"):
    """
    Parses the evaluation_trace.json file and creates visual plots of the SHD 
    progression for each evaluated graph, as well as printing a human-readable log.
    """
    if not os.path.exists(trace_path):
        print(f"Error: {trace_path} not found.")
        return

    with open(trace_path, "r") as f:
        trace = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Loaded Trace: {trace_path} ===")
    
    for graph_key, episode_data in trace.items():
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

        # Plot SHD progression
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(shd_history)), shd_history, marker='o', linestyle='-', color='b')
        plt.title(f"SHD Progression for Graph Topology {graph_idx}")
        plt.xlabel("Step")
        plt.ylabel("SHD")
        plt.grid(True)
        plt.ylim(0, max(max(shd_history) + 2, 10))
        
        plot_path = os.path.join(output_dir, f"shd_progression_graph_{graph_idx}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved SHD progression plot to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Evaluation Trace")
    parser.add_argument("--trace_path", type=str, default="evaluation_trace.json")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()
    
    parse_and_visualize_trace(args.trace_path, args.output_dir)
