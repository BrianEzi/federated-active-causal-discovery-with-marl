# 🔬 Deep Analytical Agent Guidelines: Federated Causal MARL

This document establishes the mandatory protocol, analytical framework, and execution standards for any AI agent tasked with analyzing training runs, evaluation traces, model architectures, and agent behaviors in the `federated-causal-marl` repository.

---

## 🎯 1. Core Philosophy & Mandate

1. **Empirical Grounding**: NEVER deliver qualitative speculation or hand-waving summaries. Every single analytical assertion MUST be backed by raw data extracted directly from WandB run logs, CSV metrics, or `evaluation_trace.json` files.
2. **Zero Tolerance for NULL/NA Data**: Prior to rendering tables or generating plots, verify raw JSON/DataFrame keys. Displaying empty or `N/A` metrics due to key mismatches (e.g. `train/episode_reward` vs `mean_reward`) is an explicit failure.
3. **Execution & Notebook Integrity**: Analysis MUST be output as a fully executable, self-contained Jupyter Notebook (`notebooks/run_analysis_<run_id>.ipynb`) containing active Python code (`matplotlib`/`pandas`) that renders real plots and tables.

---

## 📐 2. The 5-Layer Analytical Framework

When analyzing a single run or a suite of WandB runs, you MUST systematically evaluate five distinct layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Problem Statement & High-Level Goal Alignment                   │
│ - SCM Identifiability, Markov Equivalence Class (MEC) breakdown         │
│ - Multi-Agent Jurisdiction Boundaries & Budget Constraints              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 2: RL & Optimization Dynamics ("RL Quirks")                       │
│ - Return scale & horizon-induced variance (1/T_max normalization)      │
│ - Value loss scaling, GAE targets, Critic divergence, Entropy decay     │
│ - Intrinsic Information Gain Curiosity Bonus (Frobenius covariance shift)│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Agent Behavioral & Exploration Dynamics                        │
│ - Category action modes (LOCAL_INTERVENTION vs PEER_REQUEST vs NOOP)    │
│ - Target selection distribution & budget depletion curves               │
│ - Passive Observation Collapse vs Coordinated Active Probing            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 4: Neural Architecture & Inductive Bias Diagnosis                 │
│ - Graph Head Edge Scorer: Static Prior Memorization vs Dynamic Inference│
│ - Unconditioned Covariance input vs Interventional Contrast Tensor (ΔΣ) │
│ - Skew-Symmetric Tournament Decomposition & 2-Cycle Algebraic Bounds     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 5: Per-Topology Performance Breakdown (MEC Topologies 0–7)        │
│ - Graph 0 (Forward Chain), Graph 1 (Reverse Chain), Colliders, Forks    │
│ - SHD, F1 Score, Precision, Recall, False Positives vs False Negatives  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Layer 1: Problem Statement & High-Level Goal Alignment
- **Domain Context**: Active causal discovery seeks to learn the directed acyclic graph (DAG) $G^* = (V, E)$ of an SCM by performing targeted interventions $do(X_i)$. Observational data alone can only identify the graph up to its Markov Equivalence Class (MEC). Interventions break observational symmetry.
- **Federated Constraints**: $K$ sovereign agents partition $V$. Agent $k$ observes only its localized sub-vector and pays an action cost $c_k$ per intervention from budget $B_k$.
- **Analysis Mandate**: Assess whether the agent policy learned to prioritize interventions that resolve MEC ambiguities within its allowed budget.

---

### Layer 2: RL & Optimization Mechanics ("RL Quirks")
- **Reward Horizon Scale**: Verify if rewards are normalized by $T_{\max} = 20.0$. Unnormalized penalties cause variance explosion ($\sigma > 100$) and panic policy collapse.
- **Critic Value Loss**: Check for critic overestimation or divergence. High value loss indicates unnormalized return targets or non-stationary transition dynamics.
- **Entropy Decay**: Trace actor policy entropy ($\mathcal{H}(\pi_k)$). Premature entropy decay to $0.0$ signifies premature convergence into sub-optimal deterministic policies (e.g. 100% NOOP).
- **Curiosity Bonus ($\beta \cdot I_k$)**: Evaluate whether the Frobenius covariance shift reward $I_k = \frac{1}{|O_k|} \| (\Sigma_t - \Sigma_{t-1}) \odot M_k \|_F$ successfully prevents passive idling.

---

### Layer 3: Agent Behavioral & Exploration Dynamics
- **Action Mode Counts**: Quantify step-by-step action choices across `ActionCategory`:
  - `0`: Local Intervention ($do(X_{\text{local}})$)
  - `1`: Peer Interventional Request ($do(X_{\text{boundary}})$)
  - `2`: NOOP / Observe (Passive sample drawing)
- **Target Node Distribution**: Plot which specific causal nodes $X_i$ agents choose to intervene on. Check if agents target boundary nodes ($X_1, X_2$) that yield maximal information gain across jurisdiction boundaries.

---

### Layer 4: Neural Architecture & Inductive Bias Diagnosis
- **Static Prior Memorization**: Inspect whether the Graph Head outputs the exact same edge prediction regardless of SCM data.
  - *Diagnosis*: An unconstrained MLP edge scorer trained on single-topology curriculum stages minimizes BCE loss by memorizing constant output bias weights.
- **Interventional Contrast ($ \Delta \Sigma $)**: Verify if the input representation includes $\Delta \Sigma = \Sigma_{\text{running}} - \Sigma_{\text{obs}}$.
- **Anti-Symmetric Tournament Head**: Verify if the architecture enforces skew-symmetric decomposition:
  $$\text{Logit}(i \to j) = S_{\theta}(e_i, e_j) + \frac{1}{2}\Big(\mathcal{O}_{\phi}(i, j) - \mathcal{O}_{\phi}(j, i)\Big) + \gamma \mathbf{A}_{ij}$$
  Check if 2-cycle conflicts ($X_1 \rightleftarrows X_2$) are zero.

---

### Layer 5: Per-Topology Performance Breakdown (MEC Topologies 0–7)
Inspect deterministic evaluation traces across all 8 standard 4-node topologies:
- **Graph 0**: Forward Chain ($Z_1 \to X_1 \to X_2 \to Z_2$)
- **Graph 1**: Reverse Chain ($Z_1 \leftarrow X_1 \leftarrow X_2 \leftarrow Z_2$)
- **Graph 2**: Collider on $X_2$ ($Z_1 \to X_1 \to X_2 \leftarrow Z_2$)
- **Graph 3**: Fork on $X_1$ ($Z_1 \leftarrow X_1 \to X_2 \to Z_2$)
- **Graph 4**: Dual Collider ($X_1 \to Z_1$, $X_2 \to Z_2$, $X_1 \to X_2$)
- **Graph 5–7**: Mixed Chain / Fork / Collider Topologies

For each topology, extract:
- Final Structural Hamming Distance (SHD)
- Precision, Recall, F1 Score
- False Positives (FP) & False Negatives (FN)
- Agent Action Modes

---

## 📦 3. Mandatory Output Notebook Structure

Every deep run analysis MUST produce a clean, self-contained notebook containing:
1. **Title & Executive Summary Markdown Cell**
2. **Setup & Data Loading Code Cell** (relies on local data copies, pure `matplotlib`/`pandas`).
3. **Comparative Configuration Table** (Episodes, Budgets, Rewards, Normalization, Curiosity, Curriculum, Final SHD, Final F1).
4. **Statistical Metric Progression Table** (Mean, Std, Min, Max for Returns, SHD, F1).
5. **Multi-Panel Trajectory Plot** (4 subplots: Returns, SHD, F1, Loss/Entropy).
6. **Per-Topology Heatmap Matrix** (SHD and F1 heatmaps across Graph 0–7).
7. **Grounded Analytical Synthesis Markdown Cell** (What We Are Doing Right, What We Are Doing Wrong, High-Level Causal MARL Synthesis, Actionable Next Steps).
