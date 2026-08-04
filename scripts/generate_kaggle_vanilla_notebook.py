import json
import os

def create_kaggle_vanilla_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(content):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")]
        })

    def add_code(content):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in content.split("\n")]
        })

    # CELL 1: Header
    add_md(r"""# ⚡ Vanilla Minimal Baseline Training Notebook (Kaggle Fast Edition)
### Flat Discrete(4) Action Space & Minimal Causal Discovery Baseline
**Author / Researcher**: Brian Ezi | MSc Thesis Project  
**Branch**: `feat/vanilla-minimal-baseline`

---

### 🔬 Overview & Purpose
This notebook provides a fast, minimal benchmark training script for the **Vanilla Baseline Agent**:
- **Action Space**: Flat `Discrete(4)`: $0 \to do(Z_{\text{local}})$, $1 \to do(X_{\text{local}})$, $2 \to req(X_{\text{peer}})$, $3 \to \text{NO-OP}$.
- **Graph Estimator**: Statistical correlation and invariance asymmetry thresholding.
- **Execution Speed**: Completes full 50-100 episode training and analytics in under **10 seconds**.""")

    # CELL 2: Setup
    add_md(r"""## Step 1: Install Dependencies & Verify Accelerator""")
    add_code(r"""import os
import sys
import subprocess

def setup_env():
    print("=== [Step 1] Installing dependencies & verifying accelerator ===")
    pkgs = ["wandb", "optax", "flax", "chex", "distrax", "dm-haiku", "matplotlib", "pandas", "pytest"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
    
    import jax
    devices = jax.devices()
    print(f"\n✓ JAX Accelerator Platform: {devices[0].platform.upper()}")
    print(f"✓ Detected Devices: {devices}")
    return devices[0].platform

setup_env()""")

    # CELL 3: Repository Setup
    add_md(r"""## Step 2: Sync Repository (Branch: `feat/vanilla-minimal-baseline`)""")
    add_code(r"""def sync_repository(branch: str = "feat/vanilla-minimal-baseline"):
    working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd()
    os.chdir(working_dir)
    repo_name = "federated-active-causal-discovery-with-marl"
    repo_path = os.path.join(working_dir, repo_name) if not os.path.exists(os.path.join(os.getcwd(), "src")) else os.getcwd()
    
    if not os.path.exists(os.path.join(repo_path, "src")):
        print(f"Cloning repository...")
        subprocess.check_call(["git", "clone", "https://github.com/BrianEzi/federated-active-causal-discovery-with-marl.git", repo_path])
        os.chdir(repo_path)
    else:
        os.chdir(repo_path)
        
    subprocess.call(["git", "fetch", "origin"])
    subprocess.call(["git", "checkout", branch])
    subprocess.call(["git", "reset", "--hard", f"origin/{branch}"])
    subprocess.call(["git", "pull", "origin", branch])
    
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
        
    print(f"✓ Active Branch: {branch} in {os.getcwd()}")

sync_repository()""")

    # CELL 4: Run Fast Training
    add_md(r"""## Step 3: Run Fast Vanilla Baseline Training (50 Episodes)""")
    add_code(r"""import time
import pandas as pd
import matplotlib.pyplot as plt

def run_vanilla_training(episodes: int = 50, fixed_graph: int = 0):
    print(f"\n🚀 Launching Vanilla Baseline Training ({episodes} episodes, Fixed Graph {fixed_graph})...")
    start_t = time.time()
    
    output_dir = os.path.abspath("vanilla_results")
    cmd = [
        sys.executable, "-m", "src.train",
        "--agent_type", "vanilla",
        "--num_episodes", str(episodes),
        "--fixed_graph", str(fixed_graph),
        "--output_dir", output_dir,
        "--save_file"
    ]
    subprocess.check_call(cmd)
    
    elapsed = time.time() - start_t
    print(f"✅ Training completed in {elapsed:.2f} seconds!")
    
    csv_path = os.path.join(output_dir, "training_metrics.csv")
    df = pd.read_csv(csv_path)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["train/episode"], df["eval/shd"], color="#e74c3c", lw=2)
    axes[0].set_title("Structural Hamming Distance (SHD) ↓", fontweight='bold')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("SHD")
    
    axes[1].plot(df["train/episode"], df["eval/f1"], color="#2ecc71", lw=2)
    axes[1].set_title("Edge Discovery F1-Score ↑", fontweight='bold')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("F1 Score")
    
    plt.tight_layout()
    plt.show()
    
    print("\n--- Final Metric Summary (Last 10 Episodes) ---")
    tail = df.tail(10)
    print(f"Mean SHD: {tail['eval/shd'].mean():.2f} ± {tail['eval/shd'].std():.2f}")
    print(f"Mean F1 Score: {tail['eval/f1'].mean():.2f} ± {tail['eval/f1'].std():.2f}")
    print(f"Mean Reward: {tail['train/episode_reward'].mean():.2f} ± {tail['train/episode_reward'].std():.2f}")
    return df

df_metrics = run_vanilla_training(episodes=50, fixed_graph=0)""")

    out_path = os.path.join("notebooks", "kaggle_vanilla_baseline.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated notebook at {out_path}")

if __name__ == "__main__":
    create_kaggle_vanilla_notebook()
