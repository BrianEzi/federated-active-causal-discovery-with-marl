# 📚 Federated Active Causal Discovery Documentation Index

Welcome to the technical documentation for the **Federated Active Causal Discovery Framework**. This directory contains comprehensive architectural, mathematical, and algorithmic reference guides for the codebase.

---

## 📑 Documentation Structure

- [**System Architecture (`ARCHITECTURE.md`)**](file:///c:/Workspace/MSc%20Project/docs/ARCHITECTURE.md)
  Overview of the JAX/NumPy Hybrid Architecture, data generation pipelines, Dec-POMDP formulation, and statistical alignment layers.

- [**MARL Agent Architectures (`AGENTS_AND_MODELS.md`)**](file:///c:/Workspace/MSc%20Project/docs/AGENTS_AND_MODELS.md)
  Detailed specification of decentralized agent models (`MLPAgent`, `RNNAgent` with GRU, and `CausalTransformerAgent` with Self-Attention), the QMIX monotonic mixing hypernetwork, and the Optax TD trainer.

- [**Causal Evaluator & PAG Tracking (`CAUSAL_EVALUATOR.md`)**](file:///c:/Workspace/MSc%20Project/docs/CAUSAL_EVALUATOR.md)
  Deep dive into Partial Ancestral Graphs (PAGs), vectorized FCI Meek rule orientation, interventional mean shift testing, reward shaping, and structural metrics (SHD, Precision, Recall, F1).

- [**Project Changelog (`CHANGELOG.md`)**](file:///c:/Workspace/MSc%20Project/docs/CHANGELOG.md)
  Chronological record of code updates, performance optimizations (JIT compilation, 120x BLAS matrix speedups), metric bug fixes, and feature additions.

---

## 🚀 Quick Navigation Links
- [Main Project README](file:///c:/Workspace/MSc%20Project/README.md)
- [Project Agents Rules](file:///c:/Workspace/MSc%20Project/.agents/AGENTS.md)
- [Skill Context Guide](file:///c:/Workspace/MSc%20Project/.agents/skills/federated_causal_discovery/SKILL.md)
