import jax
import jax.numpy as jnp
from src.types import SCMConfig, SCMParams, MechanismType, NoiseType
from src.environment import init_env, get_observations, step_env

def test():
    # Setup static parameters
    d = 4
    K = 2
    
    config = SCMConfig(
        d=d,
        K=K,
        mechanism_type=MechanismType.LINEAR,
        noise_type=NoiseType.GAUSSIAN,
        noise_scale=0.1
    )
    
    # 0 -> 1 -> 2
    #       \-> 3
    # true_adjacency[i, j] = 1 if j -> i
    adj = jnp.array([
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 1., 0., 0.]
    ])
    
    topological_order = jnp.array([0, 1, 2, 3])
    
    # Let W be the same as adj for linear mechanism
    W = adj * 1.5
    
    scm_params = SCMParams(
        W=W,
        mlp_w1=jnp.zeros((d, d, 16)),
        mlp_b1=jnp.zeros((d, 16)),
        mlp_w2=jnp.zeros((d, 16, 1)),
        mlp_b2=jnp.zeros((d, 1))
    )
    
    # Agent 0 sees nodes 0, 1
    # Agent 1 sees nodes 2, 3
    agent_masks = jnp.array([
        [1., 1., 0., 0.],
        [0., 0., 1., 1.]
    ])
    
    budgets = jnp.array([10.0, 10.0])
    key = jax.random.PRNGKey(42)
    
    # 1. Initialize Env
    state = init_env(key, config, adj, scm_params, topological_order, agent_masks, budgets)
    print("Initial step count:", state.step_count)
    
    # 2. Get Observations (JIT compiled)
    @jax.jit
    def get_obs(state, key):
        return get_observations(state, config, agent_masks, 1000, key)
    
    obs_key, key = jax.random.split(key)
    obs = get_obs(state, obs_key)
    
    print("Agent 0 covariance shape:", obs.local_covariance[0].shape)
    print("Agent 0 remaining budget:", obs.remaining_budget[0])
    # Agent 0 shouldn't have covariance for node 2, 3
    print("Agent 0 cov[2,2]:", obs.local_covariance[0][2, 2])
    
    # 3. Step Env (JIT compiled)
    @jax.jit
    def step(state, joint_action, cost_vector):
        return step_env(state, joint_action, cost_vector)
    
    joint_action = jnp.zeros(K)
    cost_vector = jnp.array([1.0, 2.0])
    
    new_state, reward = step(state, joint_action, cost_vector)
    
    print("New step count:", new_state.step_count)
    print("New budgets:", new_state.budgets)
    print("Test passed!")

if __name__ == "__main__":
    test()
