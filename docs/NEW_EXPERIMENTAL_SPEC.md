# 🧪 Federated Active Causal Discovery: New Experimental Specification

*This document serves as the conceptual blueprint for the pivot towards studying privacy boundaries, economic costs, and optimal experiment design in Decentralized Causal Discovery.*

---

## 1. Environment & Causal Graph Setup
- **Meta-Learning Topologies**: The environment will randomize the true causal graph $G^*$ at the start of every episode to force the agents to learn a generalized active discovery policy rather than memorizing a fixed structure.
  - **4-Node Base Topologies**: Chain ($Z_1 \to X_1 \to X_2 \to Z_2$), Collider ($Z_1 \to X_1 \leftarrow X_2 \leftarrow Z_2$), Fork ($Z_1 \leftarrow X_1 \to X_2 \to Z_2$), and Fork+Collider. Node directionalities and SCM weights/noise will be randomized per episode.
- **Data Generation**: Continuous Linear-Gaussian Additive Noise Models (ANM).

## 2. Observability & Privacy Boundaries
- **Strict Local Visibility**: Agent $k$ can only observe data from its private internal nodes and exposed boundary nodes. 
  - *Example*: Agent 1 sees $\{Z_1, X_1\}$ (local) and $\{X_2\}$ (peer boundary). Agent 1 **never** sees $\{Z_2\}$.
- **Interventional Jurisdiction**: Agent $k$ can only physically intervene on its local nodes.
- **Data Representation (Algorithmic Aggregation)**: To preserve the Markov property without relying on unstable RNNs, the environment will maintain and provide a **Running Covariance/Correlation Matrix** of all data collected during the episode. At each step, a batch of $N$ samples will be drawn and aggregated to update this matrix.

## 3. Action Space Architecture
Agents use a **Hierarchical, Multi-Discrete Action Space** to select interventions:
- **Stage 1 (Macro)**: `[Local Intervention, Peer Request, NO-OP]`
- **Stage 2 (Micro Target)**: Node selection via a **Node Embedding + Shared Edge Scorer** architecture. This ensures the policy network can scale to any number of nodes dynamically.

## 4. Budgets, Economics, & "Peer Requests"
- **Budgets**: Agents start with a finite budget $B$.
- **Peer Requests**: Agent 1 can request Agent 2 to intervene on a boundary node (`REQ(X2)`).
  - Agent 2 automatically complies.
  - The cost (1 token) is deducted exclusively from Agent 1 (the requester).
- **Free-Rider Acceptance**: Agent 2 gets to observe the resulting data from the intervention on $X_2$ for free. This economic asymmetry is accepted as a natural dynamic of federated systems.
- **Episode Termination**: Occurs when BOTH budgets hit 0, or $t = T_{max}$.

## 5. Agent Architecture & PPO Training
- **Algorithm**: **Independent PPO (IPPO)**. Each agent has its own Actor and Critic.
- **Decentralization**: The Critic only evaluates the local observation state, maintaining strict decentralization (no global critics).
- **Dual-Head Continuous Refinement Model**: 
  - The neural network processes the observation and passes it to two heads simultaneously:
    1. **Action Head**: Outputs the multi-discrete intervention probabilities.
    2. **Graph Head**: Outputs a dense matrix of edge probabilities (the predicted local DAG).
  - The predicted DAG from step $t$ is fed back into the observation for step $t+1$, allowing the agent to continuously refine its structural belief.

## 6. Rewards, Stitching, & Penalties
- **Graph Stitching**: The environment deterministically stitches the locally predicted DAGs on the server side to compute the global reward. Conflicts on boundary edges are resolved via confidence thresholds (e.g., Stouffer's Z-score fusion).
- **Mixed Cooperative Reward**:
  - Agents receive shared rewards for correctly orienting boundary edges.
  - Agents receive exclusive local rewards for correctly orienting private edges.
- **Cycle Penalties**: If the stitched graph contains a cycle (contradictory boundary predictions), the agents involved receive a heavy penalty. This acts as an implicit communication channel.
- **Dense SHD Penalty (Mode 2)**: Structural Hamming Distance (SHD) is evaluated at every step. Because agents start with an empty/random graph at $t=0$, they will incur immediate heavy penalties. This creates a continuous time-penalty, forcing agents to intervene rapidly and optimally to minimize the accumulated loss over the episode.
