# 🏛️ System Architecture: JAX/NumPy Hybrid Pipeline

The **Federated Active Causal Discovery Engine** is formulated as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**. It enables distributed agents to collaboratively discover a hidden global causal graph through active interventions and localized continuous structural predictions.

---

## 1. Hybrid Architecture Boundaries

```text
+-----------------------------------------------------------------------+
|                         ACCELERATOR (GPU/TPU)                         |
|                                                                       |
|  +---------------------+      +------------------------------------+  |
|  |   SCM Generation    | ---> |      Covariance Calculation        |  |
|  | (sample_scm in JAX) |      | (Algorithmic State Aggregation)    |  |
|  +---------------------+      +------------------------------------+  |
|                                                  |                    |
+--------------------------------------------------|--------------------+
                                                   |
                                         Host-Device Bridge
                                                   |
                                                   v
+-----------------------------------------------------------------------+
|                               CPU (NUMPY)                             |
|                                                                       |
|  +---------------------+      +------------------------------------+  |
|  |     IPPO Network    | ---> |        DAG Edge Prediction         |  |
|  | (Haiku Forward Pass)|      |   (Local Adjacency Matrix)         |  |
|  +---------------------+      +------------------------------------+  |
|                                                  |                    |
|                                                  v                    |
|                               +------------------------------------+  |
|                               |    Deterministic Graph Stitching   |  |
|                               |     (Stouffer/Average Merging)     |  |
|                               +------------------------------------+  |
|                                                  |                    |
|                                                  v                    |
|                               +------------------------------------+  |
|                               |     Reward & Metric Calculation    |  |
|                               |       (Dense SHD, DFS Cycles)      |  |
|                               +------------------------------------+  |
+-----------------------------------------------------------------------+
```

### Why a Hybrid Architecture?
- **Accelerated Simulation (JAX)**: Structural Causal Model (SCM) data generation produces thousands of observational and interventional samples per step. Executing this on a GPU/TPU using `jax.lax.scan` and `jax.vmap` yields a massive speedup over standard Python loops.
- **CPU Graph Interpretation & Network Execution (NumPy / JAX-CPU)**: Complex dynamic graph algorithms (such as Depth-First Search for cycle detection during stitching) involve data-dependent recursion. Compiling such dynamic recursions into static JAX arrays causes compilation stalls. These run safely on the CPU alongside the agent's inference pass.

---

## 2. Dec-POMDP Mathematical Formulation

> **⚠️ Architectural note (see `docs/CHANGELOG.md`, "Collapsed ActionCategory to INTERVENE/NOOP"):** the action space and graph-prediction head below describe a prior design. The action space is now `[INTERVENE, NOOP]` -- INTERVENE natively targets any node in the agent's local domain or the shared boundary via a unified mask, rather than distinguishing a "Local Intervention" category from a "Peer Request" category. The per-agent graph-prediction head has also been removed from the actor networks; predicted DAG structure now comes entirely from the fixed analytic invariance scorer, not a learned network.

- **State Space $\mathcal{S}$**: Global ground-truth Directed Acyclic Graph $G^*$, current agent budget array $\mathbf{b} \in \mathbb{R}^K$, and the running global covariance matrix $\Sigma_{\text{global}}$.
- **Observation Space $\Omega_k$**: Agent $k$'s private running covariance matrix $\Sigma_k$ (containing only its local nodes and exposed boundary nodes), its remaining budget $b_k$, its local jurisdictional mask $M_k \in \{0, 1\}^d$, and its previous predicted local DAG matrix.
- **Action Space $\mathcal{A}_k$**: Hierarchical Multi-Discrete Actions.
  - **Stage 1 (Category)**: `[Local Intervention, Peer Request, NO-OP]`.
  - **Stage 2 (Target)**: Node index $t \in \{0, 1, \dots, d-1\}$ generated via node embeddings.
  - *Constraints*: Masking forces agents to only target nodes they are permitted to see and act upon (e.g. peer requests can only target boundary nodes).

---

## 3. Data Flow & Meta-Learning Topologies

1. **Meta-Learning Topologies**: At $t=0$, the environment randomly samples a base topology (Chain, Collider, Fork, Fork+Collider) rather than a fixed Erdős-Rényi graph. The agents must adapt and generalize.
2. **Algorithmic State Aggregation**: Rather than relying on Recurrent Neural Networks (RNNs) which are unstable in MARL causal tasks, the environment continuously tracks and updates a running covariance matrix of all samples observed during the episode. This preserves the Markov property explicitly in the state vector.
3. **Continuous Reward Shaping**: At every step, the agents output a dense predicted local DAG. These are stitched together, checked for cycles, and compared against $G^*$ to compute a Dense Structural Hamming Distance (SHD). Agents are penalized per step for structural errors, forcing them to learn optimal interventions to reduce their error rapidly.
