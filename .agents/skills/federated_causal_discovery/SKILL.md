---
name: Federated Causal Discovery Framework Context
description: Immediate architectural context for the JAX SCM environment and IPPO DAG training pipeline.
---

# 🧠 Federated Causal Discovery Architecture Context

You are working on a multi-agent reinforcement learning (MARL) causal discovery engine. Here is the dense contextual map of the system following the Decentralized IPPO pivot:

## The JAX Backend (Environment & Topologies)
The environment simulates structural causal models (SCMs) natively in JAX using `src/scm.py`. 
- **Meta-Learning Topologies**: Graphs are dynamically sampled at $t=0$ using `src/generators.py` (Chains, Colliders, Forks) to force policy generalization.
- **Algorithmic State Aggregation**: Instead of relying on recurrent memory (RNNs), the environment natively tracks and updates a **running covariance matrix** across all steps.
- **Jurisdictions**: Agents observe only a localized slice of the true graph, governed strictly by `agent_masks` and budget constraints. 

## The Hybrid Evaluator (Stitching & Rewards)
Because complex graph algorithms choke the JAX static compiler, we utilize a Hybrid CPU-GPU Architecture.
- `src/stitching.py` runs on the CPU. It deterministically merges the agents' localized continuous DAG predictions. Boundary edge conflicts are resolved by averaging probabilities.
- **DFS Cycle Detection**: If boundary predictions cause a cycle in the stitched graph, a Depth-First Search triggers a massive joint penalty.
- `src/rewards.py` evaluates the stitched DAG against $G^*$ at every step to calculate a **Dense Structural Hamming Distance (SHD)** penalty, applying exclusive penalties for local errors and shared penalties for boundary/cycle errors.

## The IPPO Control Layer (Dual-Head Architecture)
The distributed agents are controlled by Independent PPO (`src/marl/`).
- **`IPPOActor`**: A Haiku network containing:
  - **Node Embeddings**: Projects covariance rows into dense vectors.
  - **Action Head**: Multi-discrete (Category & Target) with strict $-1\text{e}9$ masking on unobservable targets.
  - **Graph Head**: A shared MLP Edge Scorer that predicts the local DAG matrix.
- **`IPPOTrainer`**: Executes PPO steps (Clipping, GAE, Value Loss, Entropy Bonus, plus supervised Graph BCE Loss).

**When debugging or extending the framework, ensure you strictly respect the JAX/NumPy hybrid boundaries and NEVER introduce in-place updates into the JAX `step_env` pipeline.**

## 🛠️ Software Engineering & Performance Protocol
Every AI agent modifying or optimizing this codebase must execute the following protocol:
1. **Performance Profiling**: Before claiming an algorithm is slow or optimized, create an empirical benchmark script measuring execution time with `time.perf_counter()`. 
2. **Matrix Vectorization**: Convert any dynamic nested Python `for` loops in CPU code into vectorized NumPy matrix operations.
3. **JIT Type Safety**: Never pass `chex.dataclass` objects to `static_argnums` in `@jax.jit`. Extract primitive integers (`int(config.d)`) outside compilation boundaries.
4. **Empirical Verification**: Always run the relevant scripts (e.g. `python src/train.py` or `pytest`) to verify functionality before pushing or declaring a task complete.
