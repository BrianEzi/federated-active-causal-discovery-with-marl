import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.scm.generator import LinearGaussianSCM
from src.rl.env import CausalDiscoveryEnv
from src.rl.agent import DDQNAgent
from src.rl.metrics import compute_shd

def train_centralized_agent(episodes=50, max_steps=50, batch_size=32):
    print("Generating Centralized 10-Variable Dataset...")
    scm = LinearGaussianSCM(num_vars=10, random_seed=42)
    data = scm.generate_data(num_samples=1000)
    
    true_dag = scm.adjacency_matrix
    
    print("Initializing Environment and Agent...")
    env = CausalDiscoveryEnv(data, max_steps=max_steps, step_cost=0.01)
    
    # Obs size: 100 (10x10 binary matrix). Action size: 3 * 10 * 9 = 270.
    agent = DDQNAgent(obs_size=100, action_size=env.action_space.n, lr=1e-3)
    
    best_bic = float('inf')
    champion_dag = None
    
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.95
    epsilon = epsilon_start
    
    print("\nStarting Training Loop...")
    
    for episode in range(episodes):
        obs, info = env.reset()
        mask = info['action_mask']
        done = False
        
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        
        while not done:
            action = agent.select_action(obs, mask, epsilon)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = info['action_mask']
            
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.push(obs, action, reward, next_obs, next_mask, done)
            
            # Update network
            loss = agent.update(batch_size)
            episode_loss += loss
            episode_reward += reward
            
            obs = next_obs
            mask = next_mask
            steps += 1
            
            # Track champion graph
            if info['bic'] < best_bic:
                best_bic = info['bic']
                champion_dag = env.adjacency_matrix.copy()
                
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if (episode + 1) % 10 == 0 or episode == 0:
            current_shd = compute_shd(champion_dag, true_dag) if champion_dag is not None else -1
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Total Reward: {episode_reward:.2f} | "
                  f"Avg Loss: {episode_loss / steps:.4f} | "
                  f"Best BIC: {best_bic:.2f} | "
                  f"Champion SHD: {current_shd}")
            
        # Target network update every 5 episodes
        if (episode + 1) % 5 == 0:
            agent.update_target_network()

    print("\nTraining Complete.")
    final_shd = compute_shd(champion_dag, true_dag)
    print(f"Final Champion SHD against Ground Truth: {final_shd}")
    print(f"Ground Truth Edges: {np.sum(true_dag)}")
    print(f"Champion Edges: {np.sum(champion_dag)}")

if __name__ == "__main__":
    # We use a short training loop for the baseline test to ensure rapid validation.
    train_centralized_agent(episodes=20, max_steps=30, batch_size=16)
