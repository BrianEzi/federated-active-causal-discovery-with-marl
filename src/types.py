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
