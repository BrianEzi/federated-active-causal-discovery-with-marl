# 📚 Federated Active Causal Discovery Documentation Index

Welcome to the technical documentation for the **Federated Active Causal Discovery Framework**. This directory contains comprehensive architectural, mathematical, and algorithmic reference guides for the codebase, specifically reflecting our pivot to Decentralized DAG estimation via IPPO.

---

## 📑 Documentation Structure

- [**System Architecture (`ARCHITECTURE.md`)**](file:///c:/Workspace/MSc%20Project/docs/ARCHITECTURE.md)
  Overview of the JAX Simulation Architecture, data generation pipelines (Meta-Learning Topologies), Algorithmic State Aggregation, and strict privacy boundaries.

- [**MARL Agent Architectures (`AGENTS_AND_MODELS.md`)**](file:///c:/Workspace/MSc%20Project/docs/AGENTS_AND_MODELS.md)
  Detailed specification of the Independent PPO (IPPO) agent architecture, including the Haiku-based `IPPOActor` (Node Embeddings, Hierarchical Action Heads, Shared Edge Scorer) and `IPPOCritic`.

- [**Causal Evaluator Engine (`CAUSAL_EVALUATOR.md`)**](file:///c:/Workspace/MSc%20Project/docs/CAUSAL_EVALUATOR.md)
  Deep dive into deterministic continuous graph stitching, DFS cycle detection, and Dense Structural Hamming Distance (SHD) mixed-cooperative reward shaping. (Also includes a "Future Work" section on the shelved PAG trackers).

- [**Project Changelog (`CHANGELOG.md`)**](file:///c:/Workspace/MSc%20Project/docs/CHANGELOG.md)
  Chronological record of code updates, performance optimizations, architectural pivots (e.g. QMIX to IPPO), and feature additions.

---

## 🚀 Quick Navigation Links
- [Main Project README](file:///c:/Workspace/MSc%20Project/README.md)
- [Project Agents Rules](file:///c:/Workspace/MSc%20Project/.agents/AGENTS.md)
- [Skill Context Guide](file:///c:/Workspace/MSc%20Project/.agents/skills/federated_causal_discovery/SKILL.md)
