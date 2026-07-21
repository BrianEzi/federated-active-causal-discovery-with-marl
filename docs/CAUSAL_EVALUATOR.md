# 🔬 Causal Evaluator & PAG Tracker Engine

This document details the causal reasoning engine (`src/pag.py`), interventional dependency testing, reward shaping, and structural evaluation metrics (`src/metrics.py`).

---

## 1. Partial Ancestral Graphs (PAGs)

A Partial Ancestral Graph (PAG) represents an equivalence class of causal DAGs under observational and interventional data.

We encode PAG edge marks as an integer matrix $P \in \{0, 1, 2, 3\}^{d \times d}$:
- `0 (NULL)`: No edge exists between $X_i$ and $X_j$.
- `1 (TAIL)`: Tail mark ($-$ facing $X_j$).
- `2 (ARROW)`: Arrowhead mark ($\to$ facing $X_j$).
- `3 (CIRCLE)`: Circle mark ($\circ$ unoriented ambiguity).

---

## 2. Interventional Mean Shift Testing

Under a Hard Intervention $\text{do}(X_i = 5.0)$:
- The variable $X_i$ is clamped to $5.0 + \text{noise}$.
- The sample mean of all variables is computed: $\boldsymbol{\mu}_{\text{int}} = \mathbf{E}[\mathbf{X} \mid \text{do}(X_i = 5.0)]$.
- **Causal Propagation Principle**:
  - If $|\mu_{j, \text{int}}| \ge \text{threshold}$ (e.g. $\ge 0.5$), then $X_i$ causes $X_j$.
  - The edge is oriented: $P[i, j] = \text{TAIL} (1)$ and $P[j, i] = \text{ARROW} (2)$.
  - If $|\mu_{j, \text{int}}| < \text{threshold}$ and both $X_i$ and $X_j$ have been intervened on without mutual dependency, the edge is removed ($P[i, j] = \text{NULL}, P[j, i] = \text{NULL}$).

---

## 3. Vectorized FCI Meek Rule Propagation

Meek orientation rules propagate structural orientation constraints:
- **Rule 1 (R1)**: $a \to b \circ\!\!-\!\!\circ c$ and $a \not\sim c \implies b \to c$.
- **Rule 2 (R2)**: $a \to b \to c$ and $a \circ\!\!-\!\!\circ c \implies a \to c$.

### Vectorized NumPy BLAS Matrix Implementation (120x Speedup)
Instead of nested $O(d^3)$ Python loops, Meek rules are computed using NumPy boolean matrix operations:
```python
directed = (P == TAIL) & (P.T == ARROW)
circles = (P == CIRCLE) & (P.T == CIRCLE)
no_edge = (P == NULL) & (P.T == NULL)

# R1: circles & (directed.T @ no_edge > 0)
r1 = circles & (np.dot(directed.T.astype(int), no_edge.astype(int)) > 0)
P[r1] = TAIL
P.T[r1] = ARROW

# R2: circles & (directed @ directed > 0)
r2 = circles & (np.dot(directed.astype(int), directed.astype(int)) > 0)
P[r2] = TAIL
P.T[r2] = ARROW
```

---

## 4. Global Reward Function (`src/rewards.py`)

To prevent the **NO-OP Penalty Trap**, the global reward is defined as:

$$R_t = c_{\text{circle}} \cdot \Delta \text{Circles} - \sum_{k=1}^K \text{cost}_k + R_{\text{noop}} - c_{\text{viol}} \cdot \text{Violations}$$

- **$\Delta \text{Circles}$**: $\text{Circles}_{t-1} - \text{Circles}_t$. Positive reward for orienting circle marks ($+10.0$ default).
- **$R_{\text{noop}}$**: Penalty ($-0.5$ default) applied if ALL agents choose `NO-OP` while unresolved circles remain.
- **Action Cost**: Cost per intervention ($-0.05$ default).

---

## 5. Structural Evaluation Metrics (`src/metrics.py`)

- **Directed Edge Extraction**: $X_i \to X_j$ iff $P[i, j] = \text{TAIL} (1)$ and $P[j, i] = \text{ARROW} (2)$.
- **Precision**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$.
- **Recall**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$.
- **F1 Score**: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **Structural Hamming Distance (SHD)**: Counts edge additions, deletions, and orientation mismatches against the true DAG $G^*$.
