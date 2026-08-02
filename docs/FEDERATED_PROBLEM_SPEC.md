# Federated Active Causal Discovery with Decentralized MARL: Problem Specification & Architectural Constraints

## 1. Executive Overview

This document provides the formal mathematical problem formulation, privacy boundaries, and architectural constraints for **Federated Active Causal Discovery with Multi-Agent Reinforcement Learning (MARL)**.

The objective is to discover the global causal Directed Acyclic Graph (DAG) $G^*$ governing a decentralized system partitioned across $K$ sovereign, non-trusting agents, while adhering to strict privacy, budget, and informational boundaries.

---

## 2. Global Ground Truth & Structural Causal Models (SCMs)

### 2.1 The Global System
Let $\mathbf{V} = \{v_1, \dots, v_d\}$ be a set of $d$ random variables whose data-generating process is defined by an underlying Structural Causal Model (SCM) $\mathcal{M}^* = \langle G^*, \mathbf{F}, P(\boldsymbol{\epsilon}) \rangle$:
\[
v_i := f_i(\mathbf{Pa}_{G^*}(v_i), \epsilon_i), \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)
\]
where $G^* = (\mathbf{V}, \mathbf{E})$ is a ground truth DAG, $\mathbf{Pa}_{G^*}(v_i)$ denotes the direct causal parents of $v_i$, and $\epsilon_i$ are mutually independent exogenous noise terms.

### 2.2 Observational Identifiability vs. Active Interventions
- **Observational Limit (Markov Equivalence Class):** Observational data generated from $\mathcal{M}^*$ identifies $G^*$ only up to its Markov Equivalence Class (MEC), sharing the same skeleton and v-structures ($X \to Y \leftarrow Z$).
- **Active Interventions ($do(v_i = c)$):** Fully orienting all causal edges in $G^*$ requires active hard interventions that sever incoming causal edges to $v_i$, setting its value to a constant $c \in \mathbb{R}$.

---

## 3. Decentralized Jurisdictions & Variable Taxonomy

The variable set $\mathbf{V}$ is partitioned across $K$ decentralized jurisdictions $\mathcal{A}_1, \dots, \mathcal{A}_K$.

### 3.1 Standard Variable Taxonomy ($Z$ vs. $X$)
We strictly standardize the variable notation across the codebase:
- **$Z$ denotes Private Local Variables:** Variables that reside entirely within a single agent's private domain and have no direct external visibility.
  - $Z_1$ (index 0): Private local variable owned exclusively by **Agent 0**.
  - $Z_2$ (index 3): Private local variable owned exclusively by **Agent 1**.
- **$X$ denotes Boundary / Exposed Variables:** Variables that interface across jurisdictions.
  - $X_1$ (index 1): Boundary variable residing in **Agent 0's** jurisdiction.
  - $X_2$ (index 2): Boundary variable residing in **Agent 1's** jurisdiction.
- **Global Variable Ordering:** $\mathbf{V} = [Z_1, X_1, X_2, Z_2]$.

```
┌────────────────────────────────────────────────────────┐
│                        GLOBAL DAG                      │
│                                                        │
│   [Agent 0 Jurisdiction]      [Agent 1 Jurisdiction]   │
│   ┌─────────────────────┐    ┌─────────────────────┐   │
│   │  Z₁ (Private Local) │    │  Z₂ (Private Local) │   │
│   │         │           │    │         ▲           │   │
│   │         ▼           │    │         │           │   │
│   │  X₁ (Boundary)      │    │  X₂ (Boundary)      │   │
│   └─────────┬───────────┘    └─────────▲───────────┘   │
│             │                          │               │
│             └────────► [X₁ ──► X₂] ────┘               │
│                      (Shared Boundary)                 │
└────────────────────────────────────────────────────────┘
```

---

## 4. Privacy Boundaries & Federated Communication

### 4.1 Strict Privacy Guarantees
1. **No Raw Data Sharing:** Raw sample matrices $\mathbf{S} \in \mathbb{R}^{N \times d}$ are never pooled or shared.
2. **Private Local Variable Isolation:** Agent 0 never observes raw samples, sample moments, or local edge predictions involving $Z_2$. Symmetrically, Agent 1 never observes $Z_1$.
3. **No Central Parameter Sharing:** Each agent $\mathcal{A}_k$ maintains completely disjoint, private neural network parameters $(\theta_k, \phi_k)$. Sharing network weights, gradients, or latent representations across agents is strictly prohibited.

### 4.2 Permitted Federated Communication
Agents coordinate strictly through two non-private channels:
1. **Boundary Summary Statistics:** Agents are permitted to compute and exchange sample cross-moments over shared boundary variables $\{X_1, X_2\}$:
   \[
   \widehat{\text{Cov}}(X_1, X_2) = \frac{1}{N-1} \sum_{n=1}^N (x_{1}^{(n)} - \bar{x}_1)(x_{2}^{(n)} - \bar{x}_2)
   \]
   This statistical moment is aggregated into each agent's running covariance matrix without revealing private variables.
2. **Peer Intervention Requests:** An agent may send a discrete message requesting a peer agent to perturb a boundary node residing in the peer's jurisdiction.

---

## 5. Hierarchical Action Space & Budgeting

Each agent $\mathcal{A}_k$ operates under a finite intervention budget $B_k \in \mathbb{R}^+$. At each decision round $t$, the agent executes a **2-Level Hierarchical Action** $a_k = (c_k, t_k)$ in a single forward pass:

```
                  ┌────────────────────────────────────────┐
                  │          Observation Vector o_k        │
                  │   [Flattened Covariance (d²), Budget]  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Private MLP Backbone    │
                        └─────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │ Macro Head: cat_logits│           │Micro Head: tgt_logits │
        │     Shape: [batch, 3] │           │     Shape: [batch, d] │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │                                   │
                    │ Sample c_k                        │ Conditioned Masking
                    │                                   ▼
                    │                   ┌───────────────────────────────┐
                    └──────────────────►│  mask_invalid_targets(c_k)    │
                                        │  Invalid targets set to -1e9  │
                                        └───────────────┬───────────────┘
                                                        │
                                                        ▼ Sample t_k
                                        ┌───────────────────────────────┐
                                        │   Micro Action: t_k ∈ {0..d}  │
                                        └───────────────────────────────┘
```

### 5.1 Level 1: Macro-Action (Category)
- `0: LOCAL_INTERVENTION` — Perturb a variable in the agent's owned domain ($do(V = 5.0)$).
- `1: PEER_REQUEST` — Request the peer to perturb a boundary node in its domain.
- `2: NOOP` — Conserve budget; perform no intervention (cost: $0.0$).

### 5.2 Level 2: Micro-Action (Target Selection & Masking)
Invalid targets are masked with $-1\text{e}9$ prior to action selection:
- **Agent 0 (`LOCAL_INTERVENTION`):** Allowed targets: $\{Z_1, X_1\}$.
- **Agent 1 (`LOCAL_INTERVENTION`):** Allowed targets: $\{X_2, Z_2\}$.
- **Both Agents (`PEER_REQUEST`):** Allowed targets: $\{X_1, X_2\}$.

---

## 6. Disjoint IPPO Control Layer

### 6.1 Disjoint Parameter Optimization
Each agent $\mathcal{A}_k$ maintains its own private actor-critic pair:
\[
\text{Actor: } \pi_{\theta_k}(a_k, \hat{G}_k \mid o_k), \quad \text{Critic: } V_{\phi_k}(o_k)
\]
- **Local Rollout Collection:** Agent $k$ collects experience exclusively in `buffers[k]`.
- **Local Policy Gradient:**
  \[
  L_k^{\text{CLIP}}(\theta_k) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta_k) \hat{A}_{k, t}, \, \text{clip}(r_t(\theta_k), 1 \pm \epsilon) \hat{A}_{k, t} \right) \right]
  \]
- **Supervised Graph BCE Loss:** In addition to policy gradients, each agent's graph prediction head is supervised on its visible sub-adjacency:
  \[
  \mathcal{L}_k^{\text{graph}}(\theta_k) = \text{BCE}(\hat{G}_k \odot \mathbf{M}_k^{\text{obs}}, \, G^* \odot \mathbf{M}_k^{\text{obs}})
  \]

---

## 7. Deterministic Hybrid Evaluator & Graph Stitching

Because static compilation of cycle detection stalls JAX, graph evaluation executes on the CPU via pure NumPy:

1. **Deterministic Stitching (`src/stitching.py`):**
   - Merges local DAG predictions $\{\hat{G}_k\}_{k=1}^K$ into a global DAG $\hat{G}_{\text{stitched}} \in \{0, 1\}^{d \times d}$.
   - Resolves boundary edge conflicts by averaging prediction confidences: $\hat{G}_{X_1 \to X_2} = \frac{1}{2}(\hat{G}_{0, X_1 \to X_2} + \hat{G}_{1, X_1 \to X_2}) \ge 0.5$.
2. **Acyclicity Verification:**
   - Evaluates DFS cycle detection on $\hat{G}_{\text{stitched}}$. If a cycle exists, a severe structural penalty is incurred.
3. **Metrics:**
   - **Structural Hamming Distance (SHD):** Sum of missing, extra, and reversed edges relative to $G^*$.
   - **F1 Score:** Harmonic mean of edge precision and recall.
