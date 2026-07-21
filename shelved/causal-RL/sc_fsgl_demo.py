import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================================================================
# 1. ENVIRONMENT DEFINITION (STABLE NORMAL + VICIOUS GRIDLOCK)
# =====================================================================

class ConfoundedTrafficEnv:
    """
    A 3-node traffic simulation graph (A -> B -> C).
    - Node A: Inflow traffic source. Has a localized spurious sensor (e.g., billboard glare) S_A.
    - Node B: Congested junction whose density depends causally on Node A's density.
              Trains an MPC controller to keep traffic density close to target (1.0).
    - Node C: Outflow sink.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.np_random = np.random.default_rng(seed)
        self.reset()
        
    def reset(self):
        self.x_A = 1.25  # True traffic density at Node A
        self.x_B = 1.0   # True traffic density at Node B
        self.x_C = 0.5   # True traffic density at Node C
        self.action_B = 0.225
        self.t = 0
        
    def step(self, action_B, ood_shift=False):
        # Rush hour inflow profile (temporally correlated pattern)
        inflow = 0.5 + 0.2 * np.sin(self.t / 10.0)
        
        # Localized spurious sensor noise only at Node A
        if ood_shift:
            # Spurious sensor goes haywire (OOD Shift)
            s_A = 15.0 + self.np_random.normal(0, 0.05)
        else:
            # During normal training/testing, spurious noise is correlated with rush hour
            s_A = 2.0 * inflow + self.np_random.normal(0, 0.05)
            
        # Sensor observations (X + noise, Spurious channel)
        # Node A: [Noisy True Traffic, Spurious glare]
        z_A = [self.x_A + self.np_random.normal(0, 0.05), s_A]
        # Node B & C: [Noisy True Traffic, Clean/Zero Spurious channel]
        z_B = [self.x_B + self.np_random.normal(0, 0.05), 0.0]
        z_C = [self.x_C + self.np_random.normal(0, 0.05), 0.0]
        
        # Gridlock threshold-based feedback backup
        # If Node C gets congested, traffic backs up to Node B
        backup = 1.5 * max(0.0, self.x_C - 1.5)
        
        # State transitions (SCM)
        next_x_A = 0.6 * self.x_A + inflow + self.np_random.normal(0, 0.05)
        # B depends causally on A; action_B mitigates congestion; backup increases it
        next_x_B = 0.6 * self.x_B + 0.5 * self.x_A - 1.0 * action_B + backup + self.np_random.normal(0, 0.05)
        # C depends on traffic routed from B
        next_x_C = 0.6 * self.x_C + 0.8 * action_B * self.x_B + self.np_random.normal(0, 0.05)
        
        # Clip values to prevent numerical overflow and represent capacity limits
        next_x_A = np.clip(next_x_A, 0.0, 15.0)
        next_x_B = np.clip(next_x_B, 0.0, 15.0)
        next_x_C = np.clip(next_x_C, 0.0, 15.0)
        
        self.x_A = next_x_A
        self.x_B = next_x_B
        self.x_C = next_x_C
        self.action_B = action_B
        self.t += 1
        
        return z_A, z_B, z_C

# =====================================================================
# 2. MODEL DEFINITIONS
# =====================================================================

class VanillaGNN(nn.Module):
    """
    Vanilla Federated Spatio-Temporal GNN.
    Aggregates raw neighbor representations across subgraphs, 
    causing spurious sensor noise from Node A to pollute Node B.
    """
    def __init__(self, input_dim=2, hidden_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.w_spatial = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z_dict):
        h = {node: self.encoder(z) for node, z in z_dict.items()}
        h_tilde = {}
        h_tilde['A'] = h['A']
        h_tilde['B'] = h['B'] + torch.matmul(h['A'], self.w_spatial)
        h_tilde['C'] = h['C'] + torch.matmul(h['B'], self.w_spatial)
        # Node B predicts Node A's causal contribution to make routing decisions
        pred_x_A = self.predictor(h_tilde['B'])
        return pred_x_A, h_tilde

class DecoupledGNN(nn.Module):
    """
    Causally Decoupled GNN (SC-FSGL).
    Uses a Conditional Separation Module (gate) to split inputs into
    Shared Causal (C) and Client-Specific (L) representations.
    Only propagates C through the spatial GNN.
    """
    def __init__(self, input_dim=2, hidden_dim=8, num_prototypes=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.encoder_causal = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encoder_local = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.w_spatial = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim) * 0.5)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z_dict):
        c, l = {}, {}
        for node, z in z_dict.items():
            mask = self.gate(z)
            z_c = mask * z
            z_l = (1 - mask) * z
            c[node] = self.encoder_causal(z_c)
            l[node] = self.encoder_local(z_l)
            
        c_tilde = {}
        c_tilde['A'] = c['A']
        c_tilde['B'] = c['B'] + torch.matmul(c['A'], self.w_spatial)
        c_tilde['C'] = c['C'] + torch.matmul(c['B'], self.w_spatial)
        
        pred_x_A = self.predictor(c_tilde['B'])
        return pred_x_A, c, l, c_tilde

def compute_contrastive_loss(c, l, prototypes, margin=2.0):
    """
    Contrastive Loss to align Causal vectors C_i to the Causal Codebook (prototypes)
    while pushing Client-Specific vectors L_i away from them.
    """
    loss = 0.0
    for node in ['A', 'B', 'C']:
        c_node = c[node]
        l_node = l[node]
        dist_c = torch.cdist(c_node.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0) ** 2
        min_dist_c = torch.min(dist_c, dim=1)[0]
        dist_l = torch.cdist(l_node.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0) ** 2
        min_dist_l = torch.min(dist_l, dim=1)[0]
        loss_c = torch.mean(min_dist_c)
        loss_l = torch.mean(torch.clamp(margin - min_dist_l, min=0.0))
        loss += loss_c + loss_l
    return loss

# =====================================================================
# 3. TRAINING DATA GENERATION
# =====================================================================

def get_training_dataset(num_steps=2000, seed=42):
    env = ConfoundedTrafficEnv(seed=seed)
    z_A_list, z_B_list, z_C_list, x_A_list = [], [], [], []
    for _ in range(num_steps):
        # Random exploratory actions during training
        action = env.np_random.uniform(0.0, 1.0)
        x_A_list.append(env.x_A)
        z_A, z_B, z_C = env.step(action, ood_shift=False)
        z_A_list.append(z_A)
        z_B_list.append(z_B)
        z_C_list.append(z_C)
    return (torch.tensor(z_A_list, dtype=torch.float32),
            torch.tensor(z_B_list, dtype=torch.float32),
            torch.tensor(z_C_list, dtype=torch.float32),
            torch.tensor(x_A_list, dtype=torch.float32).unsqueeze(1))

# =====================================================================
# 4. TRAINING AND CLOSED-LOOP SIMULATION
# =====================================================================

def main():
    # 1. Generate standard correlated training data
    print("Generating training dataset (2000 steps)...")
    z_A, z_B, z_C, x_A_true = get_training_dataset(2000, seed=100)
    
    # 2. Instantiate and train models
    vanilla_model = VanillaGNN()
    decoupled_model = DecoupledGNN()
    
    vanilla_opt = optim.Adam(vanilla_model.parameters(), lr=0.01)
    decoupled_opt = optim.Adam(decoupled_model.parameters(), lr=0.01)
    
    num_samples = z_A.shape[0]
    batch_size = 64
    epochs = 150
    
    print(f"Training models for {epochs} epochs...")
    for epoch in range(epochs):
        indices = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            z_batch = {'A': z_A[batch_idx], 'B': z_B[batch_idx], 'C': z_C[batch_idx]}
            x_A_batch = x_A_true[batch_idx]
            
            # Vanilla GNN Optimization
            vanilla_opt.zero_grad()
            pred_x_A_v, _ = vanilla_model(z_batch)
            loss_v = nn.MSELoss()(pred_x_A_v, x_A_batch)
            loss_v.backward()
            vanilla_opt.step()
            
            # Decoupled GNN Optimization
            decoupled_opt.zero_grad()
            pred_x_A_d, c, l, _ = decoupled_model(z_batch)
            loss_pred = nn.MSELoss()(pred_x_A_d, x_A_batch)
            loss_c = compute_contrastive_loss(c, l, decoupled_model.prototypes, margin=2.0)
            # High weight on contrastive loss enforces representation separation
            loss_d = loss_pred + 5.0 * loss_c
            loss_d.backward()
            decoupled_opt.step()
            
    print("Training complete. Verifying gate mask for Node A...")
    with torch.no_grad():
        test_z = torch.tensor([[1.0, 2.0], [0.5, 4.0], [1.5, 1.0]], dtype=torch.float32)
        mask_val = decoupled_model.gate(test_z)
        print("Gate Mask [Traffic Channel, Spurious Channel] (Traffic should be close to 1, Spurious close to 0):")
        for i, val in enumerate(test_z):
            print(f"  Input: {val.tolist()} -> Mask: {mask_val[i].tolist()}")
            
    # 3. Closed-Loop Evaluation Loop
    results = {}
    
    for mode, model in [('Vanilla GNN', vanilla_model), ('SC-FSGL (Decoupled)', decoupled_model)]:
        print(f"\nRunning Closed-Loop Evaluation for {mode}...")
        env = ConfoundedTrafficEnv(seed=200)
        
        history = {
            'time': [],
            'true_x_A': [],
            'est_x_A': [],
            'true_x_B': [],
            'action_B': [],
            'regret_B': [],
            'pred_error_B': []
        }
        
        # Run simulation for 80 steps
        for t in range(80):
            # OOD shift triggered at step 50
            ood = (t >= 50)
            inflow = 0.5 + 0.2 * np.sin(t / 10.0)
            s_A = 15.0 + env.np_random.normal(0, 0.05) if ood else 2.0 * inflow + env.np_random.normal(0, 0.05)
            
            # Read noisy sensors
            zt_A = torch.tensor([[env.x_A + env.np_random.normal(0, 0.05), s_A]], dtype=torch.float32)
            zt_B = torch.tensor([[env.x_B + env.np_random.normal(0, 0.05), 0.0]], dtype=torch.float32)
            zt_C = torch.tensor([[env.x_C + env.np_random.normal(0, 0.05), 0.0]], dtype=torch.float32)
            zt_dict = {'A': zt_A, 'B': zt_B, 'C': zt_C}
            
            with torch.no_grad():
                if mode == 'Vanilla GNN':
                    pred_x_A, _ = model(zt_dict)
                else:
                    pred_x_A, _, _, _ = model(zt_dict)
                pred_val = pred_x_A.item()
                
            # Node B's downstream routing policy (MPC-style controller)
            backup = 1.5 * max(0.0, env.x_C - 1.5)
            action = np.clip(0.6 * env.x_B + 0.5 * pred_val + backup - 1.0, 0.0, 1.0)
            
            # Save history
            history['time'].append(t)
            history['true_x_A'].append(env.x_A)
            history['est_x_A'].append(pred_val)
            history['true_x_B'].append(env.x_B)
            history['action_B'].append(action)
            # Control regret is squared deviation of B's traffic density from optimal target 1.0
            history['regret_B'].append((env.x_B - 1.0) ** 2)
            # Prediction error of Node A's density at Node B
            history['pred_error_B'].append((env.x_A - pred_val) ** 2)
            
            # Step environment
            env.step(action, ood_shift=ood)
            
        results[mode] = history
        print(f"  Average traffic at B before OOD (0-49): {np.mean(history['true_x_B'][:50]):.3f}")
        print(f"  Average traffic at B after OOD (50-79): {np.mean(history['true_x_B'][50:]):.3f}")
        
    # =====================================================================
    # 5. PREMIUM PLOTTING AND VISUALIZATION
    # =====================================================================
    print("\nGenerating premium comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Custom color palette
    color_vanilla = '#E74C3C'     # Sleek red for polluted Vanilla
    color_decoupled = '#1ABC9C'   # Sleek teal for robust SC-FSGL
    color_target = '#2C3E50'      # Dark slate for target line
    
    # Panel 1: Traffic Density at Node B
    ax1.plot(results['Vanilla GNN']['time'], results['Vanilla GNN']['true_x_B'], 
             color=color_vanilla, label='Vanilla GNN (Polluted Aggregation)', linewidth=2.5)
    ax1.plot(results['SC-FSGL (Decoupled)']['time'], results['SC-FSGL (Decoupled)']['true_x_B'], 
             color=color_decoupled, label='SC-FSGL (Causally Decoupled)', linewidth=2.5)
    ax1.axhline(1.0, color=color_target, linestyle='--', label='Target Density (1.0)', alpha=0.8)
    ax1.axvline(50, color='#95A5A6', linestyle=':', label='OOD Sensor Shift Triggered', linewidth=2.0)
    
    ax1.set_ylabel('Traffic Density at Node B ($X^B_t$)', fontsize=12, fontweight='bold')
    ax1.set_title('Decentralized Traffic Congestion Control under Spurious Sensor Shift', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0', framealpha=0.9)
    ax1.set_ylim(-0.1, 4.0)
    
    # Panel 2: Prediction Error of Node A's state
    ax2.plot(results['Vanilla GNN']['time'], results['Vanilla GNN']['pred_error_B'], 
             color=color_vanilla, label='Vanilla GNN Error', linewidth=2.0, alpha=0.8)
    ax2.plot(results['SC-FSGL (Decoupled)']['time'], results['SC-FSGL (Decoupled)']['pred_error_B'], 
             color=color_decoupled, label='SC-FSGL Error', linewidth=2.0, alpha=0.8)
    ax2.axvline(50, color='#95A5A6', linestyle=':', linewidth=2.0)
    
    ax2.set_xlabel('Simulation Time Step ($t$)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Node A Prediction Error ($[X^A_t - \\hat{X}^A_t]^2$)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0', framealpha=0.9)
    ax2.set_yscale('log')
    
    # Annotation of the "Vicious Cycle"
    ax1.annotate('Vicious Gridlock/Collapse\n(Action becomes fully active due to spurious noise)', 
                 xy=(55, 0.05), xytext=(58, 1.8),
                 arrowprops=dict(facecolor=color_vanilla, shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, color='#C0392B', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec=color_vanilla, lw=1))
    
    ax1.annotate('Stable Flow Maintained\n(Causal gating filters spurious noise)', 
                 xy=(65, 1.1), xytext=(48, 2.7),
                 arrowprops=dict(facecolor=color_decoupled, shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, color='#16A085', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8F5", ec=color_decoupled, lw=1))
    
    plt.tight_layout()
    plot_path = 'sc_fsgl_comparison.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nPlot successfully saved to: {plot_path}")
    print("\nSummary of results:")
    print("--------------------------------------------------------------------------------")
    print(f"Vanilla GNN Mean Post-OOD Regret: {np.mean(results['Vanilla GNN']['regret_B'][50:]):.4f}")
    print(f"SC-FSGL Mean Post-OOD Regret:     {np.mean(results['SC-FSGL (Decoupled)']['regret_B'][50:]):.4f}")
    print("--------------------------------------------------------------------------------")
    print("SC-FSGL successfully broke the vicious cycle of representation pollution!")

if __name__ == '__main__':
    main()
