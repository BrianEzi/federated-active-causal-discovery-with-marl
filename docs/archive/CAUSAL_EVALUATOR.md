# 🔬 Causal Evaluator & DAG Engine

This document details the causal reasoning engine used to combine decentralized predictions (`src/stitching.py`), generate mixed-cooperative rewards (`src/rewards.py`), and evaluate structural performance (`src/metrics.py`).

---

## 1. Deterministic Graph Stitching (`src/stitching.py`)

Under the IPPO framework, agents do not directly orient a global matrix using hard algorithmic rules. Instead, each agent $k$ outputs a continuous probability matrix $\hat{A}_k \in [0, 1]^{d \times d}$ representing its belief of the local DAG structure.

### Resolving Boundary Conflicts
At every timestep, the environment deterministically stitches these local predictions on the server side to form a global adjacency matrix $\hat{A}_{\text{global}}$:
- **Private Nodes**: If an edge only involves nodes within agent $k$'s local jurisdiction, the probability is taken directly from $\hat{A}_k$.
- **Boundary Nodes**: If an edge connects boundary nodes shared between multiple agents, their probabilities are combined using the element-wise maximum (`np.maximum`). This preserves any predicted edge if either agent strongly believes it exists.
- **Thresholding**: The resulting continuous matrix is thresholded at $0.5$ to produce a discrete DAG.

### DFS Cycle Detection
The resulting stitched graph is validated for topological correctness using a Depth-First Search (DFS) algorithm. If conflicts between agent boundary predictions result in a cycle (e.g. $X_1 \to X_2$ from Agent 1, and $X_2 \to X_1$ from Agent 2), the graph is flagged as invalid and heavily penalized.

---

## 2. Mixed Cooperative/Competitive Reward Shaping (`src/rewards.py`)

Rather than waiting until the end of the episode to receive a terminal reward, the environment evaluates the stitched DAG against the true global DAG $G^*$ at *every step*.

### Dense Structural Hamming Distance (SHD) Penalty
Agents receive a continuous step-by-step penalty based on the Structural Hamming Distance (SHD). Because agents begin with an empty or random graph prediction at $t=0$, they incur massive initial penalties, creating an active pressure to rapidly intervene and resolve structural uncertainty.

### Reward Assignment
- **Local Error Penalty**: Agent $k$ is penalized $-1.0$ exclusively if it mispredicts an edge involving its private local nodes.
- **Boundary Error Penalty**: If an edge involving shared boundary nodes is mispredicted, *both* sharing agents receive a $-1.0$ penalty, encouraging cooperation.
- **Cycle Penalty**: If the stitched graph contains a cycle, *all* agents involved in the cycle receive a massive joint penalty (default $-10.0$).

---

## 3. Structural Evaluation Metrics (`src/metrics.py`)

During training and evaluation, the following metrics are tracked by comparing the discrete stitched DAG against $G^*$:
- **Structural Hamming Distance (SHD)**: Counts edge additions, deletions, and orientation mismatches.
- **Precision**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$.
- **Recall**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$.
- **F1 Score**: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.

---

## 4. Future Work: Partial Ancestral Graphs (PAGs)

*(Note: The following logic has been temporarily shelved in `shelved/pag.py` while the system focuses on environments without latent confounders. It will be reintroduced in subsequent phases.)*

In the presence of unobserved latent confounders, DAGs are insufficient. The system will revert to predicting **Partial Ancestral Graphs (PAGs)**:
- `0 (NULL)`: No edge.
- `1 (TAIL)`: Tail mark ($-$).
- `2 (ARROW)`: Arrowhead mark ($\to$).
- `3 (CIRCLE)`: Circle mark ($\circ$ unoriented ambiguity).

**Vectorized FCI Meek Rule Propagation**:
Instead of nested $O(d^3)$ Python loops, PAG Meek rules (R1, R2) are computed using NumPy boolean matrix operations:
```python
directed = (P == TAIL) & (P.T == ARROW)
circles = (P == CIRCLE) & (P.T == CIRCLE)
no_edge = (P == NULL) & (P.T == NULL)

# R1: circles & (directed.T @ no_edge > 0)
r1 = circles & (np.dot(directed.T.astype(int), no_edge.astype(int)) > 0)
P[r1] = TAIL
P.T[r1] = ARROW
```

---

## 5. Post-Training Tracing & Visualization

To understand agent behavior and learning dynamics over time, the system includes an automated evaluation tracing tool.

### WandB Tracing (`src/evaluate.py`)
At the end of training, the main loop loads the parameters of the best-performing model (based on lowest `best_shd` and highest `f1`). It then runs a completely deterministic, greedy evaluation episode across all possible fixed graph topologies. The step-by-step actions and predictions are saved to `evaluation_trace.json` and automatically uploaded to WandB.

### Local Visualization (`src/visualize_trace.py`)
The generated trace can be visually analyzed using the `parse_and_visualize_trace` utility. This parses the JSON trace to:
1. Print a human-readable log of exactly what action each agent took at every environment step.
2. Render line plots representing the SHD progression across the 20 steps of the evaluation episode using `matplotlib`.
