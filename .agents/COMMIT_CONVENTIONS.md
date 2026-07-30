# Git Commit Conventions

This repository follows a structured convention for Git commit messages to ensure the project history remains clean, readable, and highly informative—especially during major architectural pivots.

## 1. Commit Message Format
Each commit message consists of a **header**, an optional **body**, and an optional **footer**.

```text
<type>(<optional scope>): <subject>

<body>

<footer>
```

## 2. Commit Types
- **feat**: A new feature (e.g., adding a new agent architecture).
- **fix**: A bug fix.
- **refactor**: Code changes that neither fix a bug nor add a feature (e.g., moving from QMIX to IPPO).
- **test**: Adding missing tests or correcting existing tests.
- **docs**: Documentation only changes (e.g., updating README or Changelog).
- **perf**: Code changes that improve performance (e.g., vectorizing a loop).
- **chore**: Changes to the build process or auxiliary tools and libraries (e.g., updating dependencies).

## 3. The Subject (Header)
- Use the imperative mood ("add feature" not "added feature").
- Keep it under 50-72 characters.
- Do not capitalize the first letter.
- No period (`.`) at the end.

## 4. The Body (Required for Large Changes)
For any significant changes (such as architectural refactors, multi-file feature additions, or environment rewrites), the body **must** be provided.
- Explain **what** and **why** the change was made, not just **how**.
- Provide bullet points for multiple sub-tasks completed in the commit.
- Wrap lines at 72 characters.

### Example of a Good Major Commit Message:
```text
refactor(marl): pivot from centralized QMIX to IPPO

- Replaced `QMIXMixer` with decentralized Haiku Actor-Critic networks to enforce strict local observability.
- Added hierarchical multi-discrete action space (Category + Target) for bounding agent interventions.
- Removed PAG interpretation logic in favor of Dense DAG structure prediction via Edge Scorers.
- Updated `FederatedCausalEnv` to aggregate running covariance states, eliminating the need for RNNs.
- Created `stitching.py` to handle deterministic DAG boundary conflict resolution and DFS cycle penalties.
```

## 5. Footers
Use footers to reference issue tracker IDs or note BREAKING CHANGES.
```text
BREAKING CHANGE: The environment `step()` function signature has changed. `predict_dags` is now required.
Fixes #42
```
