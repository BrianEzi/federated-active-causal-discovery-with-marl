import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, next_mask, done):
        self.buffer.append((state, action, reward, next_state, next_mask, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(next_masks),
            np.array(dones, dtype=np.float32)
        )
        
    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, obs_size: int, action_size: int, 
                 lr: float = 1e-3, gamma: float = 0.99, buffer_capacity: int = 10000):
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(obs_size, action_size).to(self.device)
        self.target_network = QNetwork(obs_size, action_size).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy action selection strictly respecting the action mask.
        """
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) == 0:
            # Fallback if somehow no actions are valid (e.g. fully constrained)
            return 0
            
        if random.random() < epsilon:
            return random.choice(valid_actions)
            
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).cpu().numpy()[0]
            
        # Mask out invalid actions by setting their Q-values to -infinity
        masked_q_values = np.where(mask == 1, q_values, -np.inf)
        
        # In case all Q-values are somehow -inf (e.g., highly negative initialization), fallback
        if np.all(masked_q_values == -np.inf):
            return random.choice(valid_actions)
            
        return int(np.argmax(masked_q_values))
        
    def update(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return 0.0
            
        states, actions, rewards, next_states, next_masks, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_masks = torch.FloatTensor(next_masks).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Double DQN logic
        # 1. Get action selection from Q-network
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            # Mask out invalid actions in the next state
            next_q_online = next_q_online.masked_fill(next_masks == 0, -1e9)
            next_actions = next_q_online.argmax(1, keepdim=True)
            
            # 2. Get Q-values from Target network using selected actions
            next_q_target = self.target_network(next_states).gather(1, next_actions)
            
            # Target Q-value
            target_q = rewards + self.gamma * (1 - dones) * next_q_target
            
        # Current Q-value
        current_q = self.q_network(states).gather(1, actions)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
