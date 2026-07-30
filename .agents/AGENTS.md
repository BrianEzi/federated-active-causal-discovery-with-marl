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

## 5. Software Engineering & Performance Discipline
- **Empirical Profiling Over Speculation:** NEVER diagnose a performance bottleneck or claim an optimization works without writing a local profiling/benchmark script (`time.perf_counter()`) to measure actual millisecond execution times.
- **Trace Full Execution Paths:** Trace the complete call stack from top-level training scripts (`train.py`) down to step functions (`evaluator_env.py`), graph interpretation (`pag.py`), and backend kernels (`scm.py`). Identify exact lines causing latency or memory transfer blocks (e.g., `np.array()` host-device syncs).
- **Algorithmic Vectorization:** Avoid $O(d^2)$ or $O(d^3)$ nested Python `for` loops. Always replace dynamic loops with vectorized NumPy matrix operations (`np.dot`, `@`, Boolean indexing) for graph orientations and graph checks.
- **JIT Compilation Safety:** Ensure `@jax.jit` boundaries are respected. Primitive integers (`d`, `mechanism_type`) must be passed statically, while PyTrees (`chex.dataclass`) must remain dynamic to prevent unhashable object compilation errors.

## 6. Testing & Quality Verification
- **Run Unit & Integration Tests First:** Before committing or declaring success, ALWAYS run the full test suite (`pytest tests/ -v`) locally. If `pytest` is unavailable, you **MUST** run the relevant scripts directly (e.g., `python src/train.py`) to confirm that your code executes without crashing and behaves as expected. **Do not assume code works without executing it.**
- **Equivalence Verification:** When refactoring algorithms for performance (e.g., vectorizing loops), verify that the optimized output matches the reference implementation 100% using `np.array_equal` or `np.allclose`.

## 7. Workspace Hygiene & Git Standards
- **Never Commit Scratch Files:** Temporary profiling scripts, benchmark harnesses, or scratch files (e.g., `scratch/`, temporary logs) MUST NEVER be committed to Git.
- **Gitignore Enforcement:** Always ensure temporary folders (e.g., `scratch/`, `shelved/`, `.venv/`) are explicitly listed in `.gitignore`. Clean up or untrack any temporary files before committing work.
- **Commit Messages:** Follow the detailed guidelines outlined in `.agents/COMMIT_CONVENTIONS.md`. Major changes **must** include a descriptive multi-line body explaining *what* changed and *why*.
- **Kaggle Synchronization:** When fixing errors reported from the Kaggle notebook, ALWAYS immediately commit and push the fix to the remote repository so the user can `git pull` the changes instantly.

## 8. Continuous Documentation Maintenance
- **Keep Documentation Updated:** Whenever you add a feature, refactor code, fix a bug, or introduce new hyperparameters or agent architectures, you MUST update the relevant files in `docs/` (`docs/ARCHITECTURE.md`, `docs/AGENTS_AND_MODELS.md`, `docs/CAUSAL_EVALUATOR.md`, `docs/CHANGELOG.md`) and `README.md`.
- **Changelog Tracking:** Always add an entry under `docs/CHANGELOG.md` summarizing what was added, fixed, or refactored.
## 9. Effort Levels & Verification Standards
- **Never Declare Done Prematurely:** Do not announce that a task, refactor, or bug fix is "done" until you have verified it functionally. Passing unit tests is not enough if the entry point (e.g., the training script) is broken.
- **End-to-End Verification:** If you modify core components, you MUST run the primary integration scripts (e.g., `python -m src.train ...` with a few episodes) to guarantee that your changes actually execute in the target environment (Kaggle/CLI) without throwing exceptions.
- **Proactive Debugging:** If you hit an error during your end-to-end verification, you must debug and resolve it iteratively before concluding your turn. Do not pass the burden of testing back to the user.
- **Bare Minimum Standard:** The bare minimum effort requires that all code written must be syntactically valid, type-safe, unit-tested (via `pytest`), AND empirically verified to run via its main execution pipeline without crashing.

