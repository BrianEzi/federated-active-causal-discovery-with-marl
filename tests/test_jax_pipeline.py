import jax
import jax.numpy as jnp
from src.types import SCMConfig, EnvState, InterventionSpec, InterventionType, MechanismType, NoiseType
from src.environment import init_env, get_observations, step_env
from src.generators import generate_er_dag, generate_scm_params

def setup_env():
    d = 4
    K = 2
    
    config = SCMConfig(
        d=d,
        K=K,
        mechanism_type=MechanismType.LINEAR,
        noise_type=NoiseType.GAUSSIAN,
        noise_scale=0.1
    )
    
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    
    # 1. Generate ER DAG
    adj = generate_er_dag(k1, d, edge_prob=0.8)
    
    # 2. Generate SCM Params
    scm_params = generate_scm_params(k2, adj, MechanismType.LINEAR)
    
    topological_order = jnp.arange(d) # For an ER DAG built with triu, 0..d-1 is a valid topo order
    
    agent_masks = jnp.array([
        [1., 1., 1., 0.],
        [0., 1., 1., 1.]
    ])
    budgets = jnp.array([10.0, 10.0])
    
    state = init_env(key, config, adj, scm_params, topological_order, agent_masks, budgets)
    return config, state, agent_masks, key

def test_step_env_jit():
    """Test 1: Assert jax.jit(step_env) runs without raising ConcretizationTypeError."""
    config, state, agent_masks, key = setup_env()
    
    @jax.jit
    def step(s, a, c):
        return step_env(s, a, c)
        
    joint_action = jnp.zeros(config.K)
    cost_vector = jnp.ones(config.K)
    
    new_state, reward = step(state, joint_action, cost_vector)
    
    assert new_state.step_count == 1
    assert jnp.all(new_state.budgets == 9.0)

def test_vmap_get_observations():
    """Test 2: Assert jax.vmap(get_observations) successfully vectorizes over a batch."""
    config, state, agent_masks, base_key = setup_env()
    
    batch_size = 32
    keys = jax.random.split(base_key, batch_size)
    
    # Create 32 distinct environments (by slightly modifying the state's budget just as a dummy variation)
    # or just vmap over the keys. Since state is static, we can broadcast it, or replicate it.
    # We will just vmap over keys, meaning 32 distinct sampling runs.
    
    @jax.vmap
    def get_obs_batch(k):
        return get_observations(state, config, agent_masks, num_samples=100, key=k)
        
    # Compile and run
    batched_obs = jax.jit(get_obs_batch)(keys)
    
    # batched_obs.local_covariance should be [32, K, d, d]
    assert batched_obs.local_covariance.shape == (32, config.K, config.d, config.d)
    assert not jnp.isnan(batched_obs.local_covariance).any()

def test_hard_intervention_covariance():
    """Test 3: Verify that applying a HARD intervention on variable Xi reduces Cov(Xi, Parents(Xi)) to near zero."""
    config, state, agent_masks, key = setup_env()
    
    # We will manually call sample_scm instead of get_observations, because get_observations uses a dummy mask
    from src.scm import sample_scm
    
    num_samples = 5000
    
    # 1. Observational data
    obs_spec = InterventionSpec(
        mask=jnp.zeros(config.d),
        type=jnp.zeros(config.d, dtype=jnp.int32),
        value=jnp.zeros(config.d)
    )
    obs_samples = sample_scm(key, state, config, num_samples, obs_spec)
    
    # Let's find a node that has parents. Node 3 has parents if adj[:, 3] has ones.
    # Since we used edge_prob=0.8, node 3 likely has parents.
    target_node = config.d - 1
    
    # Calculate observational covariance
    obs_cov = jnp.cov(obs_samples, rowvar=False)
    
    # 2. Interventional data on target_node
    # We apply a HARD intervention to target_node
    mask = jnp.zeros(config.d).at[target_node].set(1.0)
    types = jnp.zeros(config.d, dtype=jnp.int32).at[target_node].set(InterventionType.HARD)
    values = jnp.zeros(config.d).at[target_node].set(5.0)
    
    int_spec = InterventionSpec(mask=mask, type=types, value=values)
    int_samples = sample_scm(key, state, config, num_samples, int_spec)
    
    # Calculate interventional covariance
    int_cov = jnp.cov(int_samples, rowvar=False)
    
    # The covariance between target_node and any of its structural parents should be near zero in int_cov
    parents = jnp.where(state.true_adjacency[:, target_node] > 0)[0]
    
    if len(parents) > 0:
        parent_idx = parents[0]
        
        # Check observational correlation is somewhat present (not guaranteed, but highly likely for linear ER)
        # Check interventional correlation is destroyed
        assert abs(int_cov[target_node, parent_idx]) < 0.05, f"Interventional covariance not destroyed: {int_cov[target_node, parent_idx]}"
