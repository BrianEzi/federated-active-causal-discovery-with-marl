import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure output dir exists
os.makedirs("docs/figures", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# Define runs to analyze
runs = {
    "Oracle (Ceiling)": "diag_runs/single_agent_bayes_optimal_s42/eval_trace_temp0.0.json",
    "PPO + AVICI": "diag_runs/single_agent_avici_s42/eval_trace_temp0.0.json",
    "PPO + Bayes-Optimal": "diag_runs/single_agent_bayes_optimal_s42/eval_trace_temp0.0.json",
    "PPO + Learned Net": "diag_runs/single_agent_learned_s42/eval_trace_temp0.0.json",
    "Round-Robin (Baseline)": "diag_runs/single_agent_round_robin_avici/eval_trace_temp0.0.json",
    "Random (Baseline)": "diag_runs/single_agent_random_avici/eval_trace_temp0.0.json"
}

loaded_traces = {}
for name, path in runs.items():
    if os.path.exists(path):
        with open(path, "r") as f:
            loaded_traces[name] = json.load(f)

# Topologies to visualize:
# Graph 0: Non-Collider Chain (0 -> 1 -> 2 -> 3) - Ambiguous observationally, requires boundary intervention
# Graph 2: X1-Collider (0 -> 1 <- 2 -> 3) - Unique v-structure at 1, 0-1 and 1-2 identified without intervention
# Graph 4: Reversed Chain (3 -> 2 -> 1 -> 0) - Ambiguous observationally, symmetrical to Graph 0
# Graph 6: X1-Collider with reversed right wing (0 -> 1 <- 2 <- 3)
selected_graphs = [0, 2, 4, 6]
graph_names = {
    0: "Graph 0: Non-Collider Chain (0 -> 1 -> 2 -> 3)",
    2: "Graph 2: X1-Collider (0 -> 1 <- 2 -> 3)",
    4: "Graph 4: Reversed Chain (3 -> 2 -> 1 -> 0)",
    6: "Graph 6: Symmetrical Collider (0 -> 1 <- 2 <- 3)"
}

# Plot 1: Step-by-Step SHD Trajectory Comparison across Methods
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

colors = {
    "Oracle (Ceiling)": "#2ca02c", # green
    "PPO + AVICI": "#1f77b4", # blue
    "PPO + Bayes-Optimal": "#9467bd", # purple
    "PPO + Learned Net": "#17becf", # cyan
    "Round-Robin (Baseline)": "#ff7f0e", # orange
    "Random (Baseline)": "#d62728" # red
}
styles = {
    "Oracle (Ceiling)": "--",
    "PPO + AVICI": "-",
    "PPO + Bayes-Optimal": "-",
    "PPO + Learned Net": "-.",
    "Round-Robin (Baseline)": ":",
    "Random (Baseline)": ":"
}

for ax_idx, g in enumerate(selected_graphs):
    ax = axes[ax_idx]
    ax.set_title(graph_names[g], fontsize=12, fontweight="bold")
    ax.set_xlabel("Step Index", fontsize=10)
    ax.set_ylabel("SHD to True DAG", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    for name, trace in loaded_traces.items():
        g_data = trace.get(f"graph_{g}")
        if not g_data:
            continue
        steps = g_data["steps"]
        shds = [s["shd"] for s in steps]
        steps_x = list(range(len(shds)))
        ax.plot(steps_x, shds, label=name, color=colors.get(name, "black"),
                linestyle=styles.get(name, "-"), linewidth=2.2, marker="o" if "PPO" in name or "Oracle" in name else "x", markersize=5)
        
    ax.set_ylim(-0.2, 5.2)
    ax.set_xlim(-0.2, 8.2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=11, frameon=True)
plt.suptitle("Step-by-Step SHD Progression by Method & Estimator", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig("docs/figures/step_by_step_shd_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Plot 2: Step-by-Step Action & Budget Trajectory
fig, axes = plt.subplots(len(selected_graphs), len(loaded_traces), figsize=(18, 11), sharex=True, sharey="row")

for row_idx, g in enumerate(selected_graphs):
    for col_idx, (name, trace) in enumerate(loaded_traces.items()):
        ax = axes[row_idx, col_idx]
        g_data = trace.get(f"graph_{g}")
        if not g_data:
            continue
        steps = g_data["steps"][:6] # First 6 steps
        
        budgets = [s["budgets"][0] if isinstance(s["budgets"], list) else s["budgets"] for s in steps]
        actions = []
        action_colors = []
        for s in steps:
            act = s["actions"]["agent_0"]
            if act.get("cat") == 0:
                actions.append(f"Int(X{act.get('target')})")
                action_colors.append("#d95f02") # orange/red for intervene
            else:
                actions.append("NOOP")
                action_colors.append("#1b9e77") # green for NOOP
                
        steps_x = list(range(len(steps)))
        ax.plot(steps_x, budgets, color="#386cb0", linewidth=2, marker="s", label="Budget")
        
        # Annotate actions
        for i, (x, y, act_str, ac_col) in enumerate(zip(steps_x, budgets, actions, action_colors)):
            ax.annotate(act_str, (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, fontweight="bold", color=ac_col)
            
        if row_idx == 0:
            ax.set_title(name, fontsize=11, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(f"Graph {g}\nBudget ($B$)", fontsize=10, fontweight="bold")
        if row_idx == len(selected_graphs) - 1:
            ax.set_xlabel("Step", fontsize=10)
            
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylim(-0.5, 6.0)

plt.suptitle("Step-by-Step Action Selection & Budget Depletion Across Topologies", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
fig.savefig("docs/figures/step_by_step_action_budget.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generated docs/figures/step_by_step_shd_comparison.png and docs/figures/step_by_step_action_budget.png successfully!")
