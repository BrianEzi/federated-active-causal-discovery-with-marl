# Repository Versioning & Branching Standards

To maintain codebase stability while aggressively exploring research hypotheses and neural architecture variations, all developers and AI agents MUST adhere to these software engineering standards.

## 1. Git Branching Strategy

The `main` branch represents stable, fully tested, and reproducible baseline code. All non-trivial modifications must be developed on isolated feature or experiment branches:

- **`feat/<feature-name>`**: Major architectural additions, new neural network heads, or environment capabilities (e.g., `feat/anti-symmetric-tournament-head`).
- **`exp/<experiment-name>`**: Experimental curriculum schedules, reward formulation tests, or hyperparameter explorations.
- **`fix/<bug-name>`**: Targeted bug fixes or performance refactors.
- **`main`**: Production reference. Code is merged into `main` ONLY after full automated test suites (`pytest tests/ -v`) pass and training metric behavior is verified.

## 2. Feature Isolation & Ablation Flags

- **Flag-Gated Features**: When introducing a new model architecture or environment mechanic, make it configurable via CLI arguments (e.g., `--use_inductive_graph_head True/False`).
- **Backward Compatibility**: Ensure that turning the feature flag off falls back cleanly to the baseline implementation.

## 3. Workflow Protocol for Architectural Changes

1. **Checkout Feature Branch**:
   ```bash
   git checkout -b feat/<feature-name>
   ```
2. **Implement & Profile**: Keep JAX pure functions functionally pure, adhere to `chex.dataclass` PyTrees, and profile execution performance (`time.perf_counter()`).
3. **Write Unit Tests**: Add unit tests in `tests/test_<feature_name>.py` testing mathematical invariants, JIT compilation, and output tensor shapes.
4. **Run Test Suite**: Run `pytest tests/ -v` to ensure zero regression across existing functionality.
5. **Local Dry-Run Verification**: Execute a short training run (e.g., `python src/train.py --num_episodes 20`) to confirm zero runtime crashes.
6. **Structured Commit**: Commit using conventional commit format ([`.agents/COMMIT_CONVENTIONS.md`](file:///c:/Workspace/MSc%20Project/.agents/COMMIT_CONVENTIONS.md)).
