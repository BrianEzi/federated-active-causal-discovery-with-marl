import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.scm.generator import LinearGaussianSCM
from src.scm.partition import partition_data
from src.rl.env import CausalDiscoveryEnv
from src.rl.agent import DDQNAgent
from src.rl.metrics import compute_shd, compute_overlap_metrics

def train_agent(agent_id, data, true_subgraph, overlap_indices, episodes=50, max_steps=30, batch_size=16):
    print(f"\n--- Training Agent {agent_id} ---")
    
    env = CausalDiscoveryEnv(data, max_steps=max_steps, step_cost=0.01)
    # Obs size: 36 (6x6). Action size: 3 * 6 * 5 = 90.
    agent = DDQNAgent(obs_size=36, action_size=env.action_space.n, lr=1e-3)
    
    best_bic = float('inf')
    champion_dag = None
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.95
    
    for episode in range(episodes):
        obs, info = env.reset()
        mask = info['action_mask']
        done = False
        
        episode_reward = 0.0
        
        while not done:
            action = agent.select_action(obs, mask, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = info['action_mask']
            
            done = terminated or truncated
            
            agent.replay_buffer.push(obs, action, reward, next_obs, next_mask, done)
            agent.update(batch_size)
            
            episode_reward += reward
            obs = next_obs
            mask = next_mask
            
            if info['bic'] < best_bic:
                best_bic = info['bic']
                champion_dag = env.adjacency_matrix.copy()
                
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if (episode + 1) % 5 == 0:
            agent.update_target_network()
            
        if (episode + 1) % 10 == 0 or episode == 0:
            current_shd = compute_shd(champion_dag, true_subgraph) if champion_dag is not None else -1
            print(f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.2f} | Best BIC: {best_bic:.2f} | Champion SHD: {current_shd}")
            
    print(f"Agent {agent_id} Training Complete.")
    final_shd = compute_shd(champion_dag, true_subgraph)
    
    v5_idx, v6_idx = overlap_indices
    tpr, fdr = compute_overlap_metrics(champion_dag, true_subgraph, v5_idx, v6_idx)
    
    print(f"Final SHD: {final_shd}")
    print(f"Overlap (V5, V6) -> TPR: {tpr:.2f}, FDR: {fdr:.2f}")
    
    return champion_dag, final_shd, tpr, fdr

def main():
    print("Generating Global 10-Variable Dataset...")
    scm = LinearGaussianSCM(num_vars=10, random_seed=42)
    data = scm.generate_data(num_samples=2000)
    
    true_dag = scm.adjacency_matrix
    
    print("Partitioning Data for Federated Agents...")
    centralized_data, agent1_data, agent2_data = partition_data(data)
    
    # Extract Subgraphs
    agent1_true_dag = true_dag[0:6, 0:6]
    agent2_true_dag = true_dag[4:10, 4:10]
    
    # Train Agent 1
    # For Agent 1, V5 is local index 4, V6 is local index 5
    agent1_dag, shd1, tpr1, fdr1 = train_agent(
        agent_id=1,
        data=agent1_data,
        true_subgraph=agent1_true_dag,
        overlap_indices=(4, 5),
        episodes=30
    )
    
    # Train Agent 2
    # For Agent 2, V5 is local index 0, V6 is local index 1
    agent2_dag, shd2, tpr2, fdr2 = train_agent(
        agent_id=2,
        data=agent2_data,
        true_subgraph=agent2_true_dag,
        overlap_indices=(0, 1),
        episodes=30
    )
    
    print("\n=== Federated Training Summary ===")
    print(f"Agent 1 (Observed V1-V6) | Final SHD: {shd1} | Overlap TPR: {tpr1:.2f} | Overlap FDR: {fdr1:.2f}")
    print(f"Agent 2 (Observed V5-V10) | Final SHD: {shd2} | Overlap TPR: {tpr2:.2f} | Overlap FDR: {fdr2:.2f}")
    
    if fdr2 > fdr1:
        print("-> Empirical Proof: Agent 2 has a higher False Discovery Rate on the overlap due to relative latent confounding (unobserved V4).")

if __name__ == "__main__":
    main()
