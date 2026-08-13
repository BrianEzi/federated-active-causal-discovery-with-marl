import enum
import chex
import jax.numpy as jnp

class MechanismType(enum.IntEnum):
    LINEAR = 0
    NONLINEAR_ANM = 1
    POST_NONLINEAR = 2

class NoiseType(enum.IntEnum):
    GAUSSIAN = 0
    GUMBEL = 1
    UNIFORM = 2

@chex.dataclass
class SCMConfig:
    d: int
    K: int
    mechanism_type: int  # MechanismType
    noise_type: int      # NoiseType
    noise_scale: float = 1.0

@chex.dataclass
class SCMParams:
    # W: continuous adjacency / weight matrix [d, d]
    # W[i, j] represents the structural weight of the edge from j to i
    W: chex.Array
    
    # MLP weights for NONLINEAR_ANM
    mlp_w1: chex.Array  # [d, d, hidden_dim]
    mlp_b1: chex.Array  # [d, hidden_dim]
    mlp_w2: chex.Array  # [d, hidden_dim, 1]
    mlp_b2: chex.Array  # [d, 1]

@chex.dataclass
class EnvState:
    true_adjacency: chex.Array    # [d, d]
    topological_order: chex.Array  # [d]
    scm_params: SCMParams
    budgets: chex.Array           # [K]
    step_count: int
    running_covariance: chex.Array # [d, d] (For algorithmic aggregation)
    total_samples: chex.Array      # [1] (To keep track of N for running average)
    
    # Dual-stream registers for empirical invariance testing
    obs_covariance: chex.Array    # [d, d] Baseline observational covariance (at NOOP / reset)
    int_covariance: chex.Array    # [d, d, d] Slice [k, :, :] is covariance measured under do(X_k)
    int_mask: chex.Array          # [d] Indicator of nodes intervened upon

    # Running mean, tracked alongside running_covariance with identical pooling semantics
    running_mean: chex.Array      # [d]

    # Raw per-step sample buffer with real intervention labels, feeding estimators
    # (e.g. AVICI) that need actual data rather than covariance-derived reconstructions.
    # Capacity is fixed at max_steps * sample_count -- the exact max an episode can ever
    # produce -- so no eviction/wraparound policy is needed; the buffer is reset each episode.
    raw_samples: chex.Array       # [capacity, d]
    raw_interv: chex.Array        # [capacity, d] binary: 1.0 where that sample's node was intervened on this step
    raw_count: chex.Array         # [1] write cursor -- number of valid rows written so far

    # Per-node intervention visit counts (shared across both agents, incremented whenever
    # either agent intervenes on that node), feeding the UCB-style target-selection bonus
    # -- see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md's greedy-policy-collapse fix. A
    # direct function of the agent's own action history, not environment response, so it's
    # immune to the running-covariance saturation that made purely reactive observation
    # channels converge to a near-fixed-point under a repeated action.
    node_intervention_counts: chex.Array  # [d]

class ActionCategory(enum.IntEnum):
    INTERVENE = 0
    NOOP = 1

@chex.dataclass
class AgentObservation:
    local_covariance: chex.Array   # [d, d]
    remaining_budget: chex.Array   # [1]
    global_graph_proxy: chex.Array # [d * d]

class InterventionType(enum.IntEnum):
    HARD = 0
    SOFT_SHIFT = 1
    SOFT_SCALE = 2

@chex.dataclass
class InterventionSpec:
    # d-dimensional array indicating if the node is actively intervened upon (1.0) or not (0.0)
    mask: chex.Array
    # d-dimensional array of InterventionType
    type: chex.Array
    # d-dimensional array of intervention values (shift, scale, or hard replacement value)
    value: chex.Array

# Standard 4-node federated topology: [Z1, X1, X2, Z2].
# Z1 (idx 0) and Z2 (idx 3) are private nodes, exclusive to Agent 0 and Agent 1 respectively.
# X1 (idx 1) and X2 (idx 2) are the shared boundary nodes.
STANDARD_LOCAL_MASKS = jnp.array([
    [1.0, 1.0, 0.0, 0.0],  # Agent 0 owns Z1, X1
    [0.0, 0.0, 1.0, 1.0]   # Agent 1 owns X2, Z2
])
STANDARD_BOUNDARY_MASK = jnp.array([0.0, 1.0, 1.0, 0.0])  # X1, X2
STANDARD_OBS_MASKS = jnp.array([
    [1.0, 1.0, 1.0, 0.0],  # Agent 0 observes Z1, X1, X2 (not Z2)
    [0.0, 1.0, 1.0, 1.0]   # Agent 1 observes X1, X2, Z2 (not Z1)
])

def compute_edge_authority_mask(local_mask: chex.Array, boundary_mask: chex.Array) -> chex.Array:
    """
    Canonical per-agent edge-authority mask (single source of truth).
    An agent may only assert a predicted edge (i, j) if BOTH i and j lie within its own
    local domain (its private node + its own boundary node), OR both lie on the shared
    boundary pair. This structurally forbids an agent from asserting an edge directly
    between its private node and the peer's domain (e.g. Z1 <-> X2) -- only the mutual
    boundary nodes (X1 <-> X2) may ever be jointly predicted by both agents.
    """
    return jnp.maximum(jnp.outer(local_mask, local_mask), jnp.outer(boundary_mask, boundary_mask))

def compute_edge_authority_masks(local_masks: chex.Array = STANDARD_LOCAL_MASKS,
                                  boundary_mask: chex.Array = STANDARD_BOUNDARY_MASK) -> chex.Array:
    """Returns [K, d, d] stacked edge-authority masks, one per agent."""
    return jnp.stack([compute_edge_authority_mask(local_masks[k], boundary_mask) for k in range(local_masks.shape[0])])

def compute_global_structural_mask(local_masks: chex.Array = STANDARD_LOCAL_MASKS,
                                    boundary_mask: chex.Array = STANDARD_BOUNDARY_MASK) -> chex.Array:
    """
    Union of every agent's edge-authority mask -- the full set of edges the true causal
    structure could ever contain (each agent's own local pair, plus the shared boundary
    pair), and nothing else. Unlike compute_edge_authority_mask (which restricts a single
    agent), this is for *centralized/server-side* graph hypotheses that combine
    information from all agents (e.g. FederatedCausalEnv.predict_graph_hypothesis) --
    those aren't a privacy violation the way one agent unilaterally asserting a
    cross-domain edge would be, but they must still respect the topology's actual
    structural constraints (e.g. no direct Z1 <-> Z2 edge, no direct Z1 <-> X2 edge).
    """
    return jnp.max(compute_edge_authority_masks(local_masks, boundary_mask), axis=0)
