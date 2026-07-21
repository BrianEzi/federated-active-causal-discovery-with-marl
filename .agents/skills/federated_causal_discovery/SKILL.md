---
name: Federated Causal Discovery Framework Context
description: Immediate architectural context for the JAX SCM environment and QMIX MARL training pipeline.
---

# 🧠 Federated Causal Discovery Architecture Context

You are working on a multi-agent reinforcement learning (MARL) causal discovery engine. Here is the dense contextual map of the system:

## The JAX Backend (Phases 1 & 2)
The environment simulates structural causal models (SCMs) purely in JAX using `src/scm.py`. 
- Graphs are dynamically generated using `src/generators.py` (Erdős-Rényi and Barabási-Albert).
- The `sample_scm` function loops over a topological order using `jax.lax.scan`.
- We support interventions via `InterventionSpec` (Hard, Soft Shift, Soft Scale) injected seamlessly into the mathematical mechanism evaluation inside `src/functional.py`.
- Agents only observe a localized slice of the true graph, masked by `agent_masks`. Local covariance matrices are stitched globally via `src/alignment.py`.

## The Hybrid Evaluator (Phase 3)
Because complex graph traversals (like FCI rule orientations and DFS cycle detection) choke the JAX static compiler, we deployed a Hybrid Architecture.
- `src/pag.py` runs on the CPU in NumPy. It tracks the Partial Ancestral Graph using an explicit integer matrix (`0: NULL, 1: TAIL, 2: ARROW, 3: CIRCLE`).
- It applies Meek rules to orient edges based on proxy p-values derived from interventional covariance tests.
- `src/evaluator_env.py` is the PettingZoo/Gym wrapper bridging JAX and the PAG tracker.

## The QMIX Control Layer (Phase 4)
The distributed agents are controlled by a QMIX architecture (`src/marl/`).
- `MLPAgent` uses Flax to produce masked local Q-values.
- `QMIXMixer` translates the global state into mixing weights, constrained strictly non-negative via absolute value operations (`jnp.abs`).
- `QMIXTrainer` executes the Temporal Difference double Q-learning steps via `optax.adam`, sampling padded sequences from the `TrajectoryBuffer`.

**When debugging or extending the framework, ensure you strictly respect the JAX/NumPy hybrid boundaries and NEVER introduce in-place updates into the JAX `step_env` pipeline.**

## 🛠️ Software Engineering & Performance Protocol
Every AI agent modifying or optimizing this codebase must execute the following protocol:
1. **Performance Profiling**: Before claiming an algorithm is slow or optimized, create an empirical benchmark script measuring execution time with `time.perf_counter()`. Profile step-by-step to isolate CPU vs GPU bottlenecks.
2. **Matrix Vectorization**: Convert any dynamic nested Python `for` loops in CPU code into vectorized NumPy matrix operations (`np.dot`, `@`, Boolean indexing).
3. **JIT Type Safety**: Never pass `chex.dataclass` objects to `static_argnums` in `@jax.jit`. Extract primitive integers (`int(config.d)`) outside compilation boundaries.
4. **Empirical Verification**: Always run `pytest tests/ -v` and verify numerical equivalence (`np.array_equal`) before pushing or completing a task.

