# Federated Active Causal Discovery Framework (JAX)

This repository contains the foundational simulation backend and Deep Multi-Agent Reinforcement Learning (MARL) framework for **Federated Active Causal Discovery**, formulated as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) in JAX and Haiku/Optax.

The primary goal of this framework is to allow multiple decentralized RL agents to interact with, observe, and intervene on their local jurisdiction of a hidden global causal graph, collaborating to discover the true Directed Acyclic Graph (DAG) without sharing private node data.

---

## 📑 System Documentation (`docs/`)

Detailed technical documentation and algorithmic specifications are available in the [`docs/`](file:///c:/Workspace/MSc%20Project/docs/README.md) directory:

- [**Documentation Index (`docs/README.md`)**](file:///c:/Workspace/MSc%20Project/docs/README.md): Overview and table of contents.
- [**System Architecture (`docs/ARCHITECTURE.md`)**](file:///c:/Workspace/MSc%20Project/docs/ARCHITECTURE.md): Strict privacy boundaries, meta-learning topologies, algorithmic covariance aggregation, and JAX GPU simulation details.
- [**MARL Agent Architectures (`docs/AGENTS_AND_MODELS.md`)**](file:///c:/Workspace/MSc%20Project/docs/AGENTS_AND_MODELS.md): Dual-Head `IPPOActor` (Node Embeddings, Multi-Discrete Actions, Graph Edge Scorer) and `IPPOCritic`.
- [**Causal Evaluator Engine (`docs/CAUSAL_EVALUATOR.md`)**](file:///c:/Workspace/MSc%20Project/docs/CAUSAL_EVALUATOR.md): Deterministic continuous graph stitching, DFS cycle detection, and Dense Structural Hamming Distance (SHD) mixed-cooperative rewards.
- [**Project Changelog (`docs/CHANGELOG.md`)**](file:///c:/Workspace/MSc%20Project/docs/CHANGELOG.md): History of optimizations, architectural pivots (e.g. QMIX to IPPO), and feature additions.

---

## 🏗️ Core Architecture Overview

### 1. The JAX Simulation Backend (`src/scm.py`, `src/environment.py`)
- **Topological `jax.lax.scan`**: Simulates linear and non-linear Additive Noise Models (ANM) following topological ordering natively in JAX.
- **Meta-Learning Topologies**: Dynamically generates Chains, Colliders, and Forks at the start of every episode to force structural generalization.
- **Algorithmic State Aggregation**: Tracks a running covariance matrix to maintain the Markov property without relying on unstable RNNs.

### 2. Dual-Head IPPO MARL Architecture (`src/marl/ppo_agent.py`)
- **Independent PPO (IPPO)**: Fully decentralized training execution, ensuring agents learn to act on localized partial observability.
- **Hierarchical Action Space**: Stage 1 selects intervention category (`Local`, `Peer Request`, `NO-OP`); Stage 2 targets specific boundary/local nodes via embedded vector projection.
- **Continuous Graph Refinement**: The agent's neural network contains a shared Edge Scorer MLP that outputs dense predictions of the local DAG structure at every time step.

### 3. Stitching & Rewards (`src/stitching.py`, `src/rewards.py`)
- **Deterministic Server-Side Stitching**: Averages continuous boundary predictions across agents to resolve overlapping edge conflicts.
- **Dense SHD Penalty**: Replaces terminal rewards with a continuous step-by-step SHD penalty that penalizes individual local errors but shares penalties for boundary mistakes and topological cycles.

---

## 🚀 Quick Start & CLI Usage

### 1. Running Automated Verification Suite
Verify the architecture and test all 18 unit and integration tests using `pytest` inside the virtual environment:
```bash
python -m pytest tests/ -v
```

### 2. Launching Training Runs
Run IPPO training or evaluate a non-learning baseline (Random/Round-Robin):

#### A. IPPO Agent
```bash
python -m src.train \
    --agent_type ippo \
    --num_variables 4 \
    --num_agents 2 \
    --num_episodes 150 \
    --batch_size 16 \
    --action_cost 0.5 \
    --initial_budget 10.0 \
    --learning_rate 3e-4 \
    --use_wandb \
    --wandb_project "federated-causal-marl"
```

#### B. Random Baseline (Sanity Check)
```bash
python -m src.train \
    --agent_type random \
    --num_episodes 10
```

---

## 📁 Repository Structure
```text
├── docs/                     # Architectural, mathematical, and model documentation
├── src/
│   ├── types.py              # JAX Dataclass structures (SCMConfig, EnvState)
│   ├── functional.py         # SCM mathematical primitives (Linear, ANM)
│   ├── scm.py                # JIT-compiled topological sampling
│   ├── environment.py        # Algorithmic state aggregation and covariance updates
│   ├── generators.py         # Meta-Learning Topologies (Chain, Collider, Fork)
│   ├── stitching.py          # Deterministic DAG stitching and DFS cycle detection
│   ├── rewards.py            # Dense SHD and mixed cooperative/competitive shaping
│   ├── metrics.py            # SHD, Precision, Recall, and F1 calculations
│   ├── evaluator_env.py      # PettingZoo/Gym wrapper bridging SCM and IPPO logic
│   ├── baselines.py          # Random and Round-Robin benchmark agents
│   └── marl/
│       ├── ppo_agent.py      # Haiku Dual-Head IPPOActor and IPPOCritic
│       └── ppo_trainer.py    # Optax IPPO rollout buffer and update loop
├── tests/
│   ├── test_evaluator_env.py # Verification for env initialization and steps
│   ├── test_metrics.py       # Verification for SHD evaluation
│   ├── test_ppo_agent.py     # Verification for Haiku network shapes & masking
│   ├── test_rewards.py       # Verification for local/boundary penalties
│   └── test_stitching.py     # Verification for overlap merging & cycles
├── shelved/                  # Deprecated QMIX & PAG engines (saved for future phases)
├── .agents/                  # Autonomous agent configuration and rules
├── notebooks/                # Production Kaggle GPU training notebooks
├── README.md
└── requirements.txt
```
