# 📊 Standardized Benchmark & Evaluation Suite Specification

This document defines the official, reproducible **8-Experiment Benchmark Suite** for evaluating model architectures, MARL algorithms, exploration mechanics, and environment stress conditions in the `federated-causal-marl` project.

---

## 🎯 Purpose & Methodology

To ensure objective empirical comparison across all model iterations and architectural commits:
1. **Multi-Seed Averaging**: Every benchmark experiment is executed across 3 fixed random seeds (`42`, `43`, `44`). Reported metrics are mean $\pm$ standard deviation.
2. **Automated Launcher**: Experiments are managed and launched by `scripts/run_benchmark_suite.py`.
3. **Reproducible Artifacts**: Results, summary CSV tables, and markdown reports are stored under `benchmarks/<run_timestamp>/` and logged to Weights & Biases under project `federated-causal-benchmarks`.

---

## 🧪 The 8 Benchmark Experiments

### 1. EXP-1: `single_topo_g0` (Sanity & Baseline Convergence)
- **Goal**: Verify if the policy can learn a single static 4-node forward chain ($Z_1 \to X_1 \to X_2 \to Z_2$).
- **Configuration**: `--fixed_graph 0 --num_episodes 1000 --initial_budget 20.0 --action_cost 1.0`
- **Target Metrics**: Final SHD $\le 0.5$, F1 $\ge 0.90$.

---

### 2. EXP-2: `multi_topo_standard` (Standard Multi-Topology Generalization)
- **Goal**: Standard multi-agent benchmark across all 8 MEC 4-node topologies.
- **Configuration**: `--num_episodes 2000 --initial_budget 20.0 --action_cost 1.0 --use_inductive_graph_head`
- **Target Metrics**: Mean SHD $\le 1.0$, F1 $\ge 0.85$.

---

### 3. EXP-3: `ablation_inductive_vs_mlp` (Architectural Inductive Bias Probe)
- **Goal**: Direct head ablation comparing Skew-Symmetric Tournament Head vs Baseline Unconstrained MLP.
- **Configurations**:
  - **EXP-3A**: `--use_inductive_graph_head`
  - **EXP-3B**: `--no_inductive_graph_head`
- **Target Metrics**: 2-Cycle conflict rate ($X_1 \rightleftarrows X_2$), Graph 1 (Reverse Chain) SHD.

---

### 4. EXP-4: `curiosity_sweep` (Exploration & Observation Collapse Probe)
- **Goal**: Evaluates policy behavior when intrinsic information gain scaling $\beta$ is varied.
- **Configurations**: `--intrinsic_coef {0.0, 0.01, 0.05, 0.10}`
- **Target Metrics**: Action category distributions (Observe vs Intervene), budget depletion rate.

---

### 5. EXP-5: `curriculum_ablation` (Topology Curriculum Schedule Probe)
- **Goal**: Evaluates 3-stage curriculum learning vs uniform random topology sampling.
- **Configurations**:
  - **EXP-5A**: `--curriculum --curriculum_stage1_ratio 0.20 --curriculum_stage2_ratio 0.30`
  - **EXP-5B**: `--no-curriculum`
- **Target Metrics**: Stage-by-stage SHD progression, learning stability.

---

### 6. EXP-6: `budget_scarcity` (Resource Constraint Stress Test)
- **Goal**: Tests decision-making under strict intervention budget scarcity.
- **Configurations**: `--initial_budget {5.0, 10.0, 20.0} --action_cost 1.0`
- **Target Metrics**: Interventional efficiency, action-cost ROI, final SHD under budget depletion.

---

### 7. EXP-7: `nonlinear_anm` (Mechanism Complexity Probe)
- **Goal**: Tests covariance & invariance performance under nonlinear Additive Noise Models ($X_j = f_j(\text{PA}_j) + \epsilon_j$).
- **Configurations**: `--mechanism_type NONLINEAR_ANM --num_episodes 2000`
- **Target Metrics**: SHD and F1 score under non-linear functional mechanisms.

---

### 8. EXP-8: `noise_robustness` (Exogenous Noise Sensitivity Sweep)
- **Goal**: Sweeps exogenous Gaussian noise standard deviation $\sigma_{\text{noise}} \in \{0.05, 0.10, 0.50\}$.
- **Configurations**: `--noise_scale {0.05, 0.10, 0.50}`
- **Target Metrics**: Covariance estimation stability, false-positive edge rate.

---

## 🏃 Running the Suite

To run the complete benchmark suite across all experiments and seeds:
```bash
python scripts/run_benchmark_suite.py
```

To run a specific subset of experiments:
```bash
python scripts/run_benchmark_suite.py --experiments EXP-1,EXP-3,EXP-6 --num_episodes 500 --seeds 42,43
```
