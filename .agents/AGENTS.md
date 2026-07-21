# Project Rules: Federated Active Causal Discovery

When working on this repository, you must adhere strictly to the following architectural and behavioral constraints:

## 1. Functional Purity & JAX Strictness
- All simulation data generation code must reside in JAX and remain completely functionally pure.
- **No in-place updates.**
- **Topological `lax.scan` execution:** The Structural Causal Model generation (`src/scm.py`) must use `jax.lax.scan` loops over the precomputed topological order. Never revert to dynamic Python `while` loops for graph traversal, as this breaks `jax.jit`.
- All environment state variables must be encapsulated within `chex.dataclass` structures (defined in `src/types.py`) to guarantee PyTree compatibility for vectorization (`jax.vmap`).

## 2. Hybrid Architecture Boundaries
- **GPU/TPU Operations:** Data generation (`sample_scm`), SCM mechanisms, and multi-agent budget transitions are highly batched and must remain in strict JAX.
- **CPU Operations:** Graph interpretation logic, specifically the Partial Ancestral Graph (PAG) tracking (`src/pag.py`) and FCI rule orientation, must remain in pure NumPy on the CPU. Attempting to statically compile DFS cycle detection inside JAX will stall the compiler. The `FederatedCausalEnv` wrapper securely bridges this boundary.

## 3. CTDE MARL Paradigm
- The reinforcement learning layer (`src/marl/`) strictly follows Centralized Training with Decentralized Execution.
- **Monotonicity:** The central mixing network (`QMIXMixer`) MUST enforce absolute value constraints (`jnp.abs`) on all generated mixing weights to guarantee $\frac{\partial Q_{tot}}{\partial Q_k} \ge 0$.
- **Action Masking:** Forbidden actions (interventions on unobservable variables or those exceeding agent budgets) must be masked with `-1e9` prior to greedy action selection.

## 4. Documentation & Style
- Maintain clean, descriptive Docstrings for all functions.
- Ensure all matrices and arrays are annotated with their expected shapes (e.g., `[batch_size, K, d, d]`) in the comments or type hints.
