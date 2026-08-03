# Federated Active Causal Discovery with MARL: Problem Statement & Mathematical Formulation

## 1. Executive Summary & Core Objective

The central goal of this repository is to solve **Federated Active Causal Discovery**: discovering the true global causal Directed Acyclic Graph (DAG) $G^* = (\mathbf{V}, \mathbf{E})$ governing a complex continuous system whose variables $\mathbf{V}$ are partitioned across $K$ sovereign, non-trusting federated agents, operating under strict privacy, informational, and finite budget constraints.

Traditional causal discovery algorithms (e.g., PC, GES, NOTEARS) operate under a **centralized paradigm**, assuming full access to a single pooled dataset containing simultaneous observations of all system variables. In real-world multi-institutional networks (e.g., cross-hospital medical research, multi-bank financial risk analysis, multi-tenant cloud telemetry), data pooling is strictly prohibited due to privacy laws (GDPR, HIPAA), proprietary confidentiality, and massive communication overhead.

Furthermore, observational data alone can at best identify $G^*$ up to its **Markov Equivalence Class (MEC)**—a set of DAGs sharing the same skeleton and unshielded v-structures ($X \to Y \leftarrow Z$). Fully resolving edge orientations requires **active interventional probing** ($do(V_i = c)$). In a federated network, performing interventions incurs real physical and computational costs.

We formulate this problem as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** solved via **Disjoint Independent Proximal Policy Optimization (Disjoint IPPO)** with an **Inductive Skew-Symmetric Graph Head**.

---

## 2. Mathematical Problem Formulation

### 2.1 The Global Structural Causal Model (SCM)
Let $\mathbf{V} = \{v_1, v_2, \dots, v_d\}$ be a set of $d$ random variables. The data-generating process is governed by a global Structural Causal Model $\mathcal{M}^* = \langle G^*, \mathbf{F}, P(\boldsymbol{\epsilon}) \rangle$:

$$v_i := f_i(\mathbf{Pa}_{G^*}(v_i), \epsilon_i), \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$$

where:
- $G^* = (\mathbf{V}, \mathbf{E})$ is the unknown true DAG ($d \times d$ binary adjacency matrix).
- $\mathbf{Pa}_{G^*}(v_i)$ denotes the direct causal parents of variable $v_i$ in $G^*$.
- $f_i$ is a linear or non-linear structural mechanism function.
- $\epsilon_i$ are mutually independent exogenous Gaussian noise terms.

### 2.2 Observational Equivalence & Interventional Invariance
Under passive observation, multiple distinct DAGs generate identical observational joint distributions $P(\mathbf{V})$. For example, a forward chain ($Z_1 \to X_1 \to X_2 \to Z_2$) and a reverse chain ($Z_1 \leftarrow X_1 \leftarrow X_2 \leftarrow Z_2$) reside in the same Markov Equivalence Class.

To break MEC ambiguities and orient causal edges, agents must perform active hard interventions $do(v_i = c)$. An intervention severs incoming structural mechanisms to $v_i$:

$$v_i := c, \quad v_j := f_j(\mathbf{Pa}_{G^*}(v_j), \epsilon_j) \, \forall j \neq i$$

According to **Causal Invariance Physics**:
- Intervining on $v_i$ disrupts the marginal distribution of its causal descendants $v_j \in \mathbf{Desc}_{G^*}(v_i)$, causing a variance shift $\text{Var}(v_j \mid do(v_i)) \neq \text{Var}_{\text{obs}}(v_j)$.
- Intervining on $v_i$ leaves non-descendants $\mathbf{NonDesc}_{G^*}(v_i)$ and parent distributions invariant: $\text{Var}(v_k \mid do(v_i)) = \text{Var}_{\text{obs}}(v_k)$.

---

## 3. Decentralized Jurisdictions & Variable Taxonomy

The $d$ system variables $\mathbf{V}$ are partitioned into $K$ disjoint local jurisdictions $\mathbf{V}_1, \dots, \mathbf{V}_K$.

### 3.1 Variable Taxonomy ($Z$ vs. $X$)
To enforce rigorous domain boundaries, variables are classified into two strict categories:
1. **$Z$ (Private Local Variables):** Variables residing entirely within a single agent's private jurisdiction. They have no direct cross-jurisdictional connections.
   - $Z_1$ (index 0): Private local variable owned exclusively by **Agent 0**.
   - $Z_2$ (index 3): Private local variable owned exclusively by **Agent 1**.
2. **$X$ (Boundary / Exposed Variables):** Variables located at the interface between agent jurisdictions.
   - $X_1$ (index 1): Boundary variable residing in **Agent 0's** jurisdiction.
   - $X_2$ (index 2): Boundary variable residing in **Agent 1's** jurisdiction.

Standard 4-variable setup ($d=4, K=2$): $\mathbf{V} = [Z_1, X_1, X_2, Z_2]$.

```
┌────────────────────────────────────────────────────────┐
│                        GLOBAL DAG                      │
│                                                        │
│   [Agent 0 Jurisdiction]      [Agent 1 Jurisdiction]   │
│   ┌─────────────────────┐    ┌─────────────────────┐   │
│   │  Z₁ (Private Local) │    │  Z₂ (Private Local) │   │
│   │         │           │    │         ▲           │   │
│   │         ▼           │    │         │           │   │
│   │  X₁ (Boundary)      │    │  X₂ (Boundary)      │   │
│   └─────────┬───────────┘    └─────────▲───────────┘   │
│             │                          │               │
│             └────────► [X₁ ──► X₂] ────┘               │
│                      (Shared Boundary)                 │
└────────────────────────────────────────────────────────┘
```

---

## 4. Privacy & Informational Constraints

The federated problem formulation enforces three non-negotiable privacy boundaries:

1. **No Raw Sample Sharing:** Raw data matrices $\mathbf{S} \in \mathbb{R}^{N \times d}$ are never transmitted or pooled centrally.
2. **Private Variable Isolation:** Agent 0 never observes raw samples, sample moments, or edge predictions involving $Z_2$. Symmetrically, Agent 1 never observes $Z_1$.
3. **Disjoint Parameters & Sovereign Execution:** Parameter sharing across agents is strictly prohibited. Agent $k$ maintains its own private actor parameters $\theta_k$ and critic parameters $\phi_k$. No network weights, gradients, or latent representations are exchanged.

### Permitted Federated Communication
Agents coordinate strictly via two privacy-preserving mechanisms:
- **Boundary Summary Statistics:** Agents compute and exchange cross-covariance entries $\widehat{\text{Cov}}(X_1, X_2)$ over shared boundary nodes.
- **Peer Intervention Requests:** Agent $i$ can send a discrete request to Agent $j$ asking Agent $j$ to perform an intervention $do(X_j = c)$ within Agent $j$'s own domain.

---

## 5. Sequential Dec-POMDP & Hierarchical Action Space

Each episode runs for a horizon of $T \le T_{\max}$ steps. At each step $t$:

### 5.1 Observation Space
Each agent $k$ receives a local observation $o_{k, t} \in \mathbb{R}^{3d^2 + 1}$:
- Masked observational covariance $\Sigma_{\text{obs}} \odot \mathbf{M}_k^{\text{obs}}$ ($d \times d$).
- Running interventional covariance $\Sigma_{\text{running}} \odot \mathbf{M}_k^{\text{obs}}$ ($d \times d$).
- Interventional directional asymmetry tensor $\mathbf{A} \odot \mathbf{M}_k^{\text{obs}}$ ($d \times d$).
- Remaining budget scalar $B_{k, t}$.

### 5.2 2-Level Hierarchical Action Space
Agent $k$ selects a 2-level action $a_k = (c_k, t_k)$ in a single forward pass:

1. **Macro-Action Category $c_k \in \{0, 1, 2\}$:**
   - `0: LOCAL_INTERVENTION` — Perturb a node in local domain (Cost: $1.0$).
   - `1: PEER_REQUEST` — Request peer to perturb a boundary node (Cost: $1.0$).
   - `2: NOOP` — Conserve budget; perform no intervention (Cost: $0.0$).
2. **Micro-Action Target $t_k \in \{0, \dots, d-1\}$:**
   - Specifies which variable to intervene on or request.
   - **Hard Action Masking:** Forbidden actions (e.g. Agent 0 targeting $Z_2$, or interventions exceeding remaining budget) are masked with $-1\text{e}9$ prior to softmax action selection.

---

## 6. Architectural Inductive Bias: Skew-Symmetric Tournament Head

Standard neural network edge predictors attempt to predict directional edges using unconstrained MLPs over symmetric node features:

$$\text{Logit}(i \to j) = \text{MLP}([e_i, e_j])$$

Because observational covariance is symmetric ($\Sigma_{ij} = \Sigma_{ji}$), unconstrained MLPs suffer from **static prior memorization** and produce invalid 2-cycle conflicts ($X_i \rightleftarrows X_j$).

To solve this, we incorporate **Algebraic Skew-Symmetric Tournament Decomposition**:

$$\text{Logit}(i \to j) = S_{\theta}(e_i, e_j) + \frac{1}{2}\Big(\mathcal{O}_{\phi}(e_i, e_j, \mathbf{A}_{ij}) - \mathcal{O}_{\phi}(e_j, e_i, -\mathbf{A}_{ij})\Big) + \gamma \mathbf{A}_{ij}$$

where:
- $S_{\theta}(e_i, e_j) = S_{\theta}(e_j, e_i)$ is a symmetric skeleton head predicting un-directed edge presence.
- $\mathcal{O}_{\phi}(e_i, e_j, \mathbf{A}_{ij}) = -\mathcal{O}_{\phi}(e_j, e_i, -\mathbf{A}_{ij})$ is an anti-symmetric tournament head predicting directional orientation.
- $\mathbf{A}_{ij} = \left| 1 - \frac{\text{Var}(X_j \mid do(X_i))}{\text{Var}_{\text{obs}}(X_j)} \right| - \left| 1 - \frac{\text{Var}(X_i \mid do(X_j))}{\text{Var}_{\text{obs}}(X_i)} \right|$ is the interventional variance shift asymmetry tensor.

### Algebraic 2-Cycle Guarantee
Subtracting reverse logits yields:

$$\text{Logit}(i \to j) - \text{Logit}(j \to i) \equiv 2\mathcal{O}_{\phi}(i, j) + 2\gamma \mathbf{A}_{ij}$$

Bidirectional 2-cycles ($X_i \rightleftarrows X_j$) are **mathematically impossible by construction**.

---

## 7. Learning Objective & Optimization Targets

The agents are trained using **Disjoint Independent Proximal Policy Optimization (Disjoint IPPO)**:

$$\max_{\theta_k} \mathbb{E} \left[ L_k^{\text{CLIP}}(\theta_k) + \beta I_{k, t} - \lambda \mathcal{L}_k^{\text{graph}}(\theta_k) \right]$$

where:
1. **$L_k^{\text{CLIP}}(\theta_k)$**: Standard PPO clipped policy gradient objective.
2. **$I_{k, t} = \frac{1}{|O_k|} \| (\Sigma_t - \Sigma_{t-1}) \odot \mathbf{M}_k \|_F$**: Intrinsic Information-Gain Curiosity Reward driving active exploration over passive idling.
3. **$\mathcal{L}_k^{\text{graph}}(\theta_k)$**: Supervised Binary Cross-Entropy loss on local sub-DAG predictions.

### Global Stitching & Performance Metrics
At the end of an episode, local DAG predictions $\{\hat{G}_k\}_{k=1}^K$ are compiled into a global DAG $\hat{G}_{\text{stitched}}$ via **Differential Margin Thresholding**:

$$\hat{G}_{i \to j} = 1 \iff (P(i \to j) > 0.5) \land (P(i \to j) - P(j \to i) > \delta)$$

Evaluation metrics computed against true DAG $G^*$:
- **Structural Hamming Distance (SHD):** Sum of false positive, false negative, and reversed edges.
- **F1-Score:** Harmonic mean of precision and recall over directed causal edges.

---

## 8. Summary of Benchmark Challenges

1. **Multi-Topology Generalization:** Agents must generalize across all 8 Markov Equivalence Class (MEC) topologies without overfitting to a single graph.
2. **Budget Efficiency:** Agents must discover the correct DAG within $T \le 20$ steps while conserving budget $B \le 20.0$.
3. **Nonlinear ANM & Exogenous Noise:** Performance must remain robust under non-linear additive noise mechanisms ($f_i(x) = \tanh(W x)$) and high noise variance.
