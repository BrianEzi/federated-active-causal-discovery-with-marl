# Project Rules: Federated Active Causal Discovery

When working on this repository, you must adhere strictly to the following architectural and behavioral constraints:

## 1. Functional Purity & JAX Strictness
- All simulation data generation code must reside in JAX and remain completely functionally pure.
- **No in-place updates.**
- **Topological `lax.scan` execution:** The Structural Causal Model generation (`src/scm.py`) must use `jax.lax.scan` loops over the precomputed topological order. Never revert to dynamic Python `while` loops for graph traversal, as this breaks `jax.jit`.
- All environment state variables must be encapsulated within `chex.dataclass` structures (defined in `src/types.py`) to guarantee PyTree compatibility for vectorization (`jax.vmap`).

## 2. Hybrid Architecture Boundaries
- **GPU/TPU Operations:** Data generation (`sample_scm`), SCM mechanisms, and multi-agent budget transitions are highly batched and must remain in strict JAX.
- **CPU Operations:** Graph interpretation logic, specifically the causal DAG stitching and cycle checking (`src/stitching.py`), must remain in pure NumPy on the CPU. Attempting to statically compile DFS cycle detection inside JAX will stall the compiler. The `FederatedCausalEnv` wrapper securely bridges this boundary.

## 3. Independent MARL Paradigm (IPPO)
- The reinforcement learning layer (`src/marl/`) strictly follows a decentralized independent Proximal Policy Optimization (IPPO) paradigm.
- **Shared Parameters & Decentralized Execution:** Each agent acts completely independently during the rollout phase using local observations.
- **Action Masking:** Forbidden actions (interventions on unobservable variables or those exceeding agent budgets) must be masked with `-1e9` prior to greedy action selection.

## 4. Documentation & Style
- Maintain clean, descriptive Docstrings for all functions.
- Ensure all matrices and arrays are annotated with their expected shapes (e.g., `[batch_size, K, d, d]`) in the comments or type hints.

## 5. Software Engineering & Performance Discipline
- **Empirical Profiling Over Speculation:** NEVER diagnose a performance bottleneck or claim an optimization works without writing a local profiling/benchmark script (`time.perf_counter()`) to measure actual millisecond execution times.
- **Absolute Performance Mandate:** You are expected to write highly performant, state-of-the-art code *always*. Do not settle for minimum viable performance. If an algorithmic vectorization or JIT compilation opportunity exists, you must implement it. 
- **Trace Full Execution Paths:** Trace the complete call stack from top-level training scripts (`train.py`) down to step functions (`evaluator_env.py`), graph interpretation (`stitching.py`), and backend kernels (`scm.py`). Identify exact lines causing latency or memory transfer blocks (e.g., `np.array()` host-device syncs).
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
## 10. Deep Run Analysis & Artifact Standards
- **Empirical WandB Extraction**: When requested to analyze a training run or evaluation run, NEVER rely on high-level summaries or hand-waving assertions. You MUST query the WandB API to extract full history logs, run configuration, summary metrics, and JSON traces.
- **Deep Notebook Generation**: Any requested analysis MUST be saved as a fully executable, self-contained Jupyter Notebook under `notebooks/` (e.g. `notebooks/run_analysis_<run_id>.ipynb`).
- **Required Notebook Content**:
  1. Complete Hyperparameter and Configuration Tables.
  2. Tabular Metric Progression across training episodes (Rewards, SHD, F1 Score, Loss components).
  3. Action Category and Target Distribution analysis per agent.
  4. Step-by-step per-topology evaluation traces.
  5. Causal DAG / Theoretical domain explanations (e.g., Markov Equivalence Class breakdowns vs. active interventional requirements).
- **No Placeholder Code**: All analysis notebooks must contain active, functioning Python code (using `matplotlib` / standard libraries) to plot metrics directly from local or extracted data files.

## 11. High-Effort Analysis Standards & Mandatory Output Inspection
- **Zero-Tolerance for NULL/N/A Data**: You MUST inspect raw JSON/DataFrame keys prior to rendering markdown tables or plotting. Delivering tables filled with `N/A`, `NaN`, or `null` due to key mismatches (e.g. `train/episode_reward` vs `mean_reward`) is UNACCEPTABLE.
- **Mandatory Notebook Execution Verification**: Before declaring a notebook complete, you must execute the generation script, inspect the resulting notebook/JSON output, verify that every metric array is populated with valid numbers, and confirm that all plots render real data.
- **Exhaustive Deep Dive**: High-effort analysis requires exhaustive detail: exact step-by-step action sequences, exact loss curve progressions, per-agent action distribution breakdowns, and detailed mathematical explanations of domain failure modes. Never settle for high-level summaries.

## 12. Directness, Precision & Communication Standards
- **Concise & Direct Responses**: Eliminate unnecessary preamble, pleasantries, and conversational fluff. Deliver direct, rigorous, and point-blank technical explanations.
- **Meticulous Technical Accuracy**: Every diagnostic assertion, mathematical derivation, and architectural explanation must be precise, concrete, and empirically backed by codebase inspection or test outputs.
- **Zero Ambiguity**: State exact variable names, mathematical formulas, tensor shapes, and file locations (`file:///path/to/file#L10`) when explaining failures or proposed remedies.

