# Federated Active Causal Discovery Framework (JAX)

This repository contains the foundational simulation backend and Deep Multi-Agent Reinforcement Learning (MARL) framework for **Federated Active Causal Discovery**, formulated as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) in JAX, Flax, and PyTorch/NumPy.

The primary goal of this framework is to allow multiple decentralized RL agents to interact with, observe, and intervene on a hidden global causal graph to collaboratively discover its true structural equations and Partial Ancestral Graph (PAG).

---

## 📑 System Documentation (`docs/`)

Detailed technical documentation and algorithmic specifications are available in the [`docs/`](file:///c:/Workspace/MSc%20Project/docs/README.md) directory:

- [**Documentation Index (`docs/README.md`)**](file:///c:/Workspace/MSc%20Project/docs/README.md): Overview and table of contents.
- [**System Architecture (`docs/ARCHITECTURE.md`)**](file:///c:/Workspace/MSc%20Project/docs/ARCHITECTURE.md): JAX GPU simulation vs. NumPy CPU graph interpretation, Dec-POMDP formulation, and statistical alignment.
- [**MARL Agent Architectures (`docs/AGENTS_AND_MODELS.md`)**](file:///c:/Workspace/MSc%20Project/docs/AGENTS_AND_MODELS.md): `MLPAgent`, `RNNAgent` (GRU), `CausalTransformerAgent`, QMIX monotonic mixing network, and Optax trainer.
- [**Causal Evaluator & PAG Engine (`docs/CAUSAL_EVALUATOR.md`)**](file:///c:/Workspace/MSc%20Project/docs/CAUSAL_EVALUATOR.md): Partial Ancestral Graphs (PAGs), 120x BLAS matrix Meek rules, interventional mean shift testing, and structural metrics (SHD, Precision, Recall, F1).
- [**Project Changelog (`docs/CHANGELOG.md`)**](file:///c:/Workspace/MSc%20Project/docs/CHANGELOG.md): History of optimizations, bug fixes, and feature additions.

---

## 🏗️ Core Architecture Overview

### 1. The JAX Simulation Backend (`src/scm.py`, `src/environment.py`)
- **Topological `jax.lax.scan`**: Simulates linear and non-linear Additive Noise Models (ANM) following topological ordering without dynamic Python loops.
- **Batched Interventions (`InterventionSpec`)**: Supports `HARD` ($do(X_i = c)$), `SOFT_SHIFT`, and `SOFT_SCALE` interventions seamlessly compiled in JAX.
- **Statistical Alignment (`src/alignment.py`)**: Fuses disparate agent p-values via Stouffer's method and stitches overlapping local covariances into a global matrix.

### 2. The Hybrid Evaluator Engine (`src/pag.py`, `src/evaluator_env.py`)
- **Vectorized FCI Meek Rules**: Uses NumPy BLAS matrix operations (`np.dot`) to propagate Meek rules R1 & R2, yielding a **121x speedup** over dynamic Python loops.
- **Interventional Mean Shift Testing**: Measures $E[X_j \mid \text{do}(X_i = 5.0)]$ to orient directed edges ($X_i \to X_j$) without wiping out unobserved variable pairs.

### 3. QMIX Control Layer (`src/marl/`)
- **Decentralized Agents**: Supports `MLPAgent`, `RNNAgent` (GRU cell for memory in Dec-POMDPs), and `CausalTransformerAgent` (Self-Attention over trajectory history).
- **Monotonic Mixing Network**: Enforces $\frac{\partial Q_{\text{tot}}}{\partial Q_k} \ge 0$ via absolute-value hypernetwork weights (`jnp.abs`).
- **Action Masking**: Masks out-of-jurisdiction or over-budget actions with $-1\text{e}9$.

---

## 🚀 Quick Start & CLI Usage

### 1. Running Automated Verification Suite
Verify the architecture and test all 15 unit and integration tests using `pytest`:
```bash
python -m pytest tests/ -v
```

### 2. Launching Training Runs
Run QMIX training with your choice of agent model (`mlp`, `rnn`, or `transformer`):

#### A. Recurrent GRU Agent (Recommended for Dec-POMDPs)
```bash
python -m src.train \
    --agent_type rnn \
    --num_variables 5 \
    --num_agents 2 \
    --graph_type ER \
    --num_episodes 150 \
    --batch_size 16 \
    --action_cost 0.05 \
    --circle_reward 10.0 \
    --noop_penalty 0.5 \
    --epsilon_decay_frac 0.8 \
    --use_wandb \
    --wandb_project "federated-causal-marl"
```

#### B. Causal Transformer Agent (Self-Attention)
```bash
python -m src.train \
    --agent_type transformer \
    --num_variables 5 \
    --num_agents 2 \
    --num_episodes 150 \
    --use_wandb
```

---

## 📁 Repository Structure
```text
├── docs/                     # Architectural, mathematical, and model documentation
│   ├── README.md             # Documentation index
│   ├── ARCHITECTURE.md       # JAX/NumPy Hybrid system architecture
│   ├── AGENTS_AND_MODELS.md  # MLP, RNN (GRU), and CausalTransformer specs
│   ├── CAUSAL_EVALUATOR.md   # PAG tracker, Meek rules, and metric equations
│   └── CHANGELOG.md          # Chronological update record
├── src/
│   ├── types.py              # JAX Dataclass structures (SCMConfig, EnvState)
│   ├── functional.py         # SCM mathematical primitives (Linear, ANM)
│   ├── scm.py                # JIT-compiled topological sampling
│   ├── environment.py        # JAX Dec-POMDP environment step & local masking
│   ├── generators.py         # Erdős-Rényi and Barabási-Albert DAG generators
│   ├── alignment.py          # Stouffer Z-score fusion & covariance stitching
│   ├── pag.py                # CPU-side PAG tracker & 120x BLAS Meek rules
│   ├── rewards.py            # Global reward shaping with NO-OP penalty
│   ├── metrics.py            # SHD, Precision, Recall, and F1 calculations
│   ├── evaluator_env.py      # PettingZoo/Gym wrapper bridging JAX & CPU PAG
│   └── marl/
│       ├── agent.py          # MLPAgent, RNNAgent (GRU), CausalTransformerAgent
│       ├── mixer.py          # QMIX monotonic mixing hypernetwork
│       ├── buffer.py         # Trajectory replay buffer
│       └── trainer.py        # Optax-powered double Q-learning trainer
├── tests/
│   ├── test_agents.py        # Verification for MLP, GRU, and Transformer agents
│   ├── test_evaluator.py     # Verification for PAG orientations & rewards
│   ├── test_jax_pipeline.py  # Verification for Phase 2 Interventions & JIT
│   ├── test_marl.py          # Verification for QMIX Monotonicity
│   └── test_metrics.py       # Verification for SHD, Precision, Recall, F1
├── notebooks/
│   └── kaggle_training.ipynb # Production Kaggle GPU training notebook
├── README.md
└── requirements.txt
```
