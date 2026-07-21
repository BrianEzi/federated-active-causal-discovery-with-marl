# 🧠 MARL Agent Architectures & QMIX Control Layer

This document details the Multi-Agent Reinforcement Learning (MARL) control layer implemented in `src/marl/`, built on **Flax** and **Optax** following the Centralized Training with Decentralized Execution (CTDE) paradigm.

---

## 1. Decentralized Agent Architectures (`src/marl/agent.py`)

We provide three distinct agent models selectable via `--agent_type [mlp|rnn|transformer]`:

### A. `MLPAgent` (`--agent_type mlp`)
- **Type**: Feedforward Multi-Layer Perceptron.
- **Formulation**: $Q_k(o_{k,t}) = \text{Dense}(\text{ReLU}(\text{Dense}(o_{k,t})))$.
- **Use Case**: Simple baseline for 1-step observation mapping.
- **Limitation**: Stateless. Has no memory of past interventions within an episode.

### B. `RNNAgent` (`--agent_type rnn`)
- **Type**: Gated Recurrent Unit (GRU) Recurrent Neural Network.
- **Formulation**: 
  $$h_{k,t}, Q_k(o_{k,t}, h_{k,t-1}) = \text{GRUCell}(h_{k,t-1}, \text{Dense}(o_{k,t}))$$
- **Use Case**: Active causal discovery under partial observability (Dec-POMDP).
- **Advantage**: Maintains a hidden memory state $h_{k,t}$ across episode steps, allowing agents to remember previous intervention outcomes and learn adaptive multi-step intervention strategies.

### C. `CausalTransformerAgent` (`--agent_type transformer`)
- **Type**: Self-Attention Trajectory Transformer.
- **Formulation**: Applies Multi-Head Self-Attention over the sequence of past step observations:
  $$Q_k(o_{k, 1:t}) = \text{LayerNorm}(X + \text{MultiHeadAttention}(X, X))$$
- **Use Case**: Long-context reasoning across episode steps.
- **Advantage**: Directly attends to past intervention steps to correlate downstream changes and resolve ambiguous edges.

---

## 2. QMIX Monotonic Mixing Network (`src/marl/mixer.py`)

QMIX factorizes the joint action-value function $Q_{\text{tot}}(s, \mathbf{a})$ into individual agent Q-values $Q_1, \dots, Q_K$:

$$Q_{\text{tot}}(s, \mathbf{a}) = \text{Mixer}(Q_1(o_1, a_1), \dots, Q_K(o_K, a_K), s)$$

### Monotonicity Guarantee
To ensure decentralized greedy choices align with global joint value maximization:
$$\frac{\partial Q_{\text{tot}}}{\partial Q_k} \ge 0 \quad \forall k \in \{1 \dots K\}$$

This constraint is enforced by generating all hypernetwork weights via absolute value operations (`jnp.abs`):
$$W_1 = |W_{\text{raw}, 1}(s)|, \quad W_2 = |W_{\text{raw}, 2}(s)|$$

---

## 3. Training Protocol (`src/marl/trainer.py`)

- **Double Q-Learning**: Target action selection is decoupled from target evaluation:
  $$a^*_k = \arg\max_{a'} Q_k(o'_k, a'; \theta), \quad Y_t = R_t + \gamma Q_{\text{tot}}(s', \mathbf{a}^*; \theta^-)$$
- **Epsilon Schedule**: Linear decay from `--epsilon_start` to `--epsilon_min` over `--epsilon_decay_frac` fraction of total training episodes.
