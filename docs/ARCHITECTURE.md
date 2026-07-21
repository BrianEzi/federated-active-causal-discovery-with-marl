# 🏛️ System Architecture: JAX/NumPy Hybrid Pipeline

The **Federated Active Causal Discovery Engine** is formulated as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**. It enables distributed agents to collaboratively discover a hidden global causal graph through active interventions.

---

## 1. Hybrid Architecture Boundaries

```text
+-----------------------------------------------------------------------+
|                         ACCELERATOR (GPU/TPU)                         |
|                                                                       |
|  +---------------------+      +------------------------------------+  |
|  |   SCM Generation    | ---> |      Covariance Calculation        |  |
|  | (sample_scm in JAX) |      | (compute_local_covariances + vmap) |  |
|  +---------------------+      +------------------------------------+  |
|                                                  |                    |
|                                                  v                    |
|                               +------------------------------------+  |
|                               |    Global Covariance Stitching     |  |
|                               |     (stitch_global_covariance)     |  |
|                               +------------------------------------+  |
+-----------------------------------------------------------------------+
                                                   |
                                         Host-Device Bridge
                                                   |
                                                   v
+-----------------------------------------------------------------------+
|                               CPU (NUMPY)                             |
|                                                                       |
|  +---------------------+      +------------------------------------+  |
|  | Interventional Mean | ---> |         PAG Graph Orientation      |  |
|  |    Shift Testing    |      | (Vectorized Meek Rules & DFS Cycle)|  |
|  +---------------------+      +------------------------------------+  |
|                                                  |                    |
|                                                  v                    |
|                               +------------------------------------+  |
|                               |     Reward & Metric Calculation    |  |
|                               |     (SHD, Precision, Recall, F1)   |  |
|                               +------------------------------------+  |
+-----------------------------------------------------------------------+
```

### Why a Hybrid Architecture?
- **Accelerated Simulation (JAX)**: Structural Causal Model (SCM) data generation produces thousands of observational and interventional samples per step. Executing this on a GPU/TPU using `jax.lax.scan` and `jax.vmap` yields a 100x–1000x speedup over standard Python loops.
- **CPU Graph Interpretation (NumPy)**: Complex dynamic graph algorithms (such as Depth-First Search cycle checks and FCI Meek rule propagation) involve data-dependent recursion. Compiling such dynamic recursions into static JAX arrays causes compilation stalls. Running these on CPU with vectorized NumPy matrix operations maintains high efficiency without compiler overhead.

---

## 2. Dec-POMDP Mathematical Formulation

- **State Space $\mathcal{S}$**: Global stitched covariance matrix $\Sigma_{\text{global}} \in \mathbb{R}^{d \times d}$, ground-truth adjacency $G^*$, and current agent budget array $\mathbf{b} \in \mathbb{R}^K$.
- **Observation Space $\Omega_k$**: Local covariance matrix $\Sigma_k \in \mathbb{R}^{d \times d}$, agent jurisdiction mask $M_k \in \{0, 1\}^d$, remaining budget $b_k$, and global covariance proxy.
- **Action Space $\mathcal{A}_k$**: Discrete choice $a_k \in \{0, 1, \dots, d\}$.
  - $a_k < d$: Execute a Hard Intervention $\text{do}(X_{a_k} = 5.0)$ on node $a_k$.
  - $a_k = d$: Execute `NO-OP` (observe without intervening).

---

## 3. Data Flow & Compilation Safety

1. **Primitive Integers in JIT**: Functions decorated with `@jax.jit` (such as `_sample_scm_jitted`) receive primitive Python integers (`d`, `mechanism_type`, `noise_type`) as static arguments (`static_argnums`).
2. **PyTree State Objects**: Dynamic simulation states (`EnvState`, `InterventionSpec`) remain PyTrees (`chex.dataclass`), allowing seamless gradient tracing without unhashable PyTree compilation errors.
