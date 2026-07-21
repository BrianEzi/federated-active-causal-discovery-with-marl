# 📜 Project Changelog

All notable changes, bug fixes, architectural refactors, and performance optimizations are recorded here.

---

## [Unreleased] - 2026-07-22

### Added
- **`RNNAgent` (GRU)**: Added Gated Recurrent Unit agent model in `src/marl/agent.py` to maintain hidden carry $h_{k,t}$ across episode steps for Dec-POMDP causal discovery.
- **`CausalTransformerAgent`**: Added Self-Attention Trajectory Transformer agent model in `src/marl/agent.py` for long-context trajectory reasoning.
- **Dynamic WandB Run Naming**: Configured automatic WandB run names reflecting agent choice, graph size, agent count, learning rate, action cost, and seed.
- **Complete Argument Parser in `src/train.py`**: Exposed all missing environment, reward, SCM, and QMIX hyperparameters (`--initial_budget`, `--action_cost`, `--sample_count`, `--circle_reward`, `--noop_penalty`, `--violation_penalty`, `--epsilon_decay_frac`, `--agent_type`).
- **Comprehensive Documentation Suite (`docs/`)**: Added `ARCHITECTURE.md`, `AGENTS_AND_MODELS.md`, `CAUSAL_EVALUATOR.md`, `CHANGELOG.md`, and `docs/README.md`.

### Fixed
- **PAG Metric Zeroing Bug (`src/pag.py`, `src/evaluator_env.py`)**: Fixed a bug where `NaN` values (unobserved variable pairs) were converted to 1.0 $p$-value, deleting the entire PAG matrix to 0 at reset. Refactored interventional testing to use **Interventional Mean Shifts** ($\mu_{j, \text{int}}$) rather than raw clamped covariances.
- **NO-OP Penalty Trap (`src/rewards.py`)**: Added `--noop_penalty` (-0.5 default) when all agents NO-OP while unresolved circles remain, preventing policy collapse into passive NO-OP behavior.
- **JIT `chex.dataclass` Unhashable Hashing Crash (`src/scm.py`)**: Resolved `TypeError: unhashable type 'SCMConfig'` by unwrapping primitive static integers (`int(config.d)`) outside the `@jax.jit` boundary.
- **PAG Meek Rule CPU Bottleneck (`src/pag.py`)**: Replaced $O(d^3)$ nested Python loops with vectorized NumPy BLAS matrix operations (`np.dot`), yielding a **121x CPU speedup** for graph orientation.

### Verified
- Full test suite expanded to 15 unit and integration tests (`tests/test_agents.py`, `tests/test_evaluator.py`, `tests/test_jax_pipeline.py`, `tests/test_marl.py`, `tests/test_metrics.py`), passing 100%.
