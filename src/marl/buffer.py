import numpy as np

class TrajectoryBuffer:
    def __init__(self, capacity: int, max_steps: int, state_dim: int, obs_dim: int, num_agents: int, num_actions: int):
        self.capacity = capacity
        self.max_steps = max_steps
        self.num_agents = num_agents
        
        self.states = np.zeros((capacity, max_steps, state_dim), dtype=np.float32)
        self.observations = np.zeros((capacity, max_steps, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, max_steps, num_agents), dtype=np.int32)
        self.rewards = np.zeros((capacity, max_steps, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, max_steps, 1), dtype=np.bool_)
        self.avail_actions = np.zeros((capacity, max_steps, num_agents, num_actions), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def add_episode(self, episode: dict):
        """
        episode should be a dictionary with lists of length <= max_steps.
        Keys: states, observations, actions, rewards, dones, avail_actions.
        """
        length = len(episode['states'])
        
        def pad(arr, pad_val=0):
            arr = np.array(arr)
            pad_width = [(0, self.max_steps - length)] + [(0, 0)] * (arr.ndim - 1)
            return np.pad(arr, pad_width, mode='constant', constant_values=pad_val)
            
        self.states[self.ptr] = pad(episode['states'])
        self.observations[self.ptr] = pad(episode['observations'])
        self.actions[self.ptr] = pad(episode['actions'])
        self.rewards[self.ptr] = pad(episode['rewards'])
        self.dones[self.ptr] = pad(episode['dones'], pad_val=True) 
        self.avail_actions[self.ptr] = pad(episode['avail_actions'])
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int):
        """Returns a random batch of padded trajectories."""
        # Ensure we don't sample more than we have
        if self.size < batch_size:
            indices = np.random.choice(self.size, self.size, replace=False)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
            
        return {
            'states': self.states[indices],
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'avail_actions': self.avail_actions[indices]
        }
