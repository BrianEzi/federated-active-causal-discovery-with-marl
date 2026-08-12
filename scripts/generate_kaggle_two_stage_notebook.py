import json
import os

def create_kaggle_two_stage_notebook():
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

    # CELL: Header
    add_md(r"""# Two-Stage Active Causal Discovery: Soft-Shift Disjoint IPPO (Kaggle GPU Edition)
### The "New Paradigm": Decoupled Target Policy + Personal/Shared Boundary Rewards
**Author / Researcher**: Brian Ezi | MSc Thesis Project
**Branch**: `feat/vanilla-minimal-baseline`

---

### Overview & Purpose
This notebook trains the current full-featured Disjoint IPPO architecture:
- **Unified Intervention Action Space**: Agents select `INTERVENE` (on any node within their local domain or the shared boundary) or `NOOP`, replacing the earlier local/peer-request split.
- **Soft-Shift Interventions**: $X_i := f_i(\mathbf{Pa}_i) + \epsilon_i + \delta_i$ with $\mu_\delta = 2.0$, preserving variance instead of hard-clamping boundary nodes.
- **Personal Local & Shared Boundary Rewards**: Each agent is penalized for its own private-node SHD errors plus a shared penalty on boundary ($X_1 \leftrightarrow X_2$) errors.
- **Observation Feedback & 3-Stage Curriculum**: Agents observe their own previous predicted DAG slice, and topology sampling ramps from Graph 0 -> Chain MEC pair -> all 8 topologies.
- **Graph Structure Estimation**: The predicted causal DAG (used for SHD/F1 evaluation) comes from the fixed analytic invariance scorer over the server-stitched covariance, not from a learned graph head -- the actor networks now only learn intervention targeting. `--use_inductive_graph_head` is still accepted for checkpoint/CLI compatibility but is currently architecturally a no-op (the graph-head auxiliary network was removed).

This mirrors the configuration used for the 1000-episode empirical run on the UCL Myriad HPC cluster, so results here are directly comparable to that baseline.""")

    # CELL: Setup
    add_md(r"""## Step 1: Install Dependencies & Verify Accelerator""")
    add_code(r"""def install_dependencies_and_check_gpu():
    import subprocess
    import sys

    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "wandb", "optax", "flax", "chex", "distrax", "dm-haiku", "matplotlib", "pandas", "networkx", "pytest"
    ])

    import jax
    devices = jax.devices()
    print(f"\n[GPU Check] JAX Accelerator Devices: {devices}")
    platform = devices[0].platform if devices else "none"
    print(f"[GPU Check] Default Backend Platform: {platform.upper()}")
    return devices

devices = install_dependencies_and_check_gpu()""")

    # CELL: Repository Setup
    add_md(r"""## Step 2: Clone Repository & Switch to Branch (`feat/vanilla-minimal-baseline`)""")
    add_code(r"""def setup_repository(branch: str = "feat/vanilla-minimal-baseline", repo_url: str = "https://github.com/BrianEzi/federated-active-causal-discovery-with-marl.git") -> str:
    import os
    import sys
    import subprocess

    working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd()
    os.chdir(working_dir)

    repo_dir_name = "federated-active-causal-discovery-with-marl"
    repo_path = os.path.join(working_dir, repo_dir_name) if not os.path.exists(os.path.join(os.getcwd(), "src")) else os.getcwd()

    if not os.path.exists(os.path.join(repo_path, "src")):
        print(f"Cloning repository from {repo_url}...")
        subprocess.check_call(["git", "clone", repo_url, repo_path])
        os.chdir(repo_path)
    else:
        os.chdir(repo_path)

    print(f"Working in repository root: {os.getcwd()}")

    subprocess.call(["git", "fetch", "origin"])
    subprocess.call(["git", "checkout", branch])
    subprocess.call(["git", "reset", "--hard", f"origin/{branch}"])
    subprocess.call(["git", "pull", "origin", branch])

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    active_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    print(f"[Git] Active Branch: {active_branch}")
    return repo_path

repo_path = setup_repository(branch="feat/vanilla-minimal-baseline")""")

    # CELL: WandB auth
    add_md(r"""## Step 3: Authenticate Weights & Biases (WandB)""")
    add_code(r"""def authenticate_wandb() -> bool:
    import os
    import wandb

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api_key = None
        for key_name in ["wandb_api_key", "WANDB_API_KEY", "wandb"]:
            try:
                wandb_api_key = user_secrets.get_secret(key_name)
                if wandb_api_key:
                    break
            except Exception:
                pass
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login(key=wandb_api_key)
            print("[WandB] Authenticated successfully via Kaggle Secret!")
            return True
        else:
            print("[WandB] Kaggle Secret not found. Enabling anonymous logging mode...")
            wandb.login(anonymous="allow")
            return True
    except Exception as e:
        print(f"[WandB] Kaggle Secret unavailable ({e}). Falling back to anonymous logging...")
        wandb.login(anonymous="allow")
        return True

authenticate_wandb()""")

    # CELL: Verification tests
    add_md(r"""## Step 4: Run Verification Test Suite""")
    add_code(r"""def run_verification_tests() -> bool:
    import subprocess
    import sys

    print("Running full verification test suite (pytest)...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=False)
    if res.returncode == 0:
        print("\n[Test Suite] ALL TESTS PASSED (100% SUCCESS)!")
        return True
    else:
        print("\n[Test Suite] WARNING: Some unit tests encountered issues. Review log above.")
        return False

run_verification_tests()""")

    # CELL: Training
    add_md(r"""## Step 5: Train the Two-Stage Soft-Shift Disjoint IPPO Architecture (1000 Episodes)
Defaults below reproduce the "new paradigm" configuration: soft-shift interventions, dense personal/shared SHD rewards,
the analytic invariance graph estimator (frozen), observation feedback, the anti-symmetric tournament graph head, and
the 3-stage topology curriculum -- matching the `submit_job_cpu.sh` / `submit_job.sh` HPC configuration.""")
    add_code(r"""def train_two_stage_ippo(
    num_episodes: int = 1000,
    num_variables: int = 4,
    num_agents: int = 2,
    batch_size: int = 32,
    initial_budget: float = 20.0,
    action_cost: float = 1.0,
    learning_rate: float = 3e-4,
    eval_freq: int = 10,
    allowed_topologies: str = None,       # e.g. "0,1" or "0,2,6"; leave None to use the curriculum schedule
    use_inductive_graph_head: bool = True, # currently a no-op: the graph-head network was removed from the actor; kept for CLI/checkpoint compatibility
    intervention_type: str = "soft_shift", # "soft_shift" or "hard"
    soft_shift_val: float = 2.0,
    estimator_type: str = "analytic",      # "analytic" or "avici"
    freeze_graph_estimator: bool = True,
    obs_feedback: bool = True,
    reward_density: str = "dense",         # "dense" or "sparse"
    intrinsic_coef: float = 0.05,
    impact_coef: float = 0.0,
    boundary_margin: float = 0.10,
    curriculum: bool = True,
    checkpoint_dir: str = None,
    output_dir: str = None,
    eval_temperature: float = 0.0,
    use_wandb: bool = True,
    wandb_project: str = "federated-causal-marl-kaggle"
) -> dict:
    # Launches src.train with the two-stage action loop / soft-shift / personal-shared-reward architecture.
    import os
    import sys
    import subprocess

    if checkpoint_dir is None:
        checkpoint_dir = "/kaggle/working/checkpoints" if os.path.exists("/kaggle/working") else "checkpoints"
    if output_dir is None:
        output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.train",
        "--agent_type", "ippo",
        "--num_variables", str(num_variables),
        "--num_agents", str(num_agents),
        "--num_episodes", str(num_episodes),
        "--batch_size", str(batch_size),
        "--initial_budget", str(initial_budget),
        "--action_cost", str(action_cost),
        "--learning_rate", str(learning_rate),
        "--eval_freq", str(eval_freq),
        "--checkpoint_dir", checkpoint_dir,
        "--output_dir", output_dir,
        "--eval_temperature", str(eval_temperature),
        "--intervention_type", intervention_type,
        "--soft_shift_val", str(soft_shift_val),
        "--estimator_type", estimator_type,
        "--freeze_graph_estimator", "true" if freeze_graph_estimator else "false",
        "--obs_feedback", "true" if obs_feedback else "false",
        "--reward_density", reward_density,
        "--intrinsic_coef", str(intrinsic_coef),
        "--impact_coef", str(impact_coef),
        "--boundary_margin", str(boundary_margin),
        "--save_file"
    ]

    if allowed_topologies is not None:
        cmd.extend(["--allowed_topologies", str(allowed_topologies)])
        cmd.append("--no_curriculum")
    elif curriculum:
        cmd.append("--curriculum")
    else:
        cmd.append("--no_curriculum")

    if use_inductive_graph_head:
        cmd.append("--use_inductive_graph_head")
    else:
        cmd.append("--no_inductive_graph_head")

    if use_wandb:
        cmd.extend(["--use_wandb", "--wandb_project", wandb_project])

    print(f"\n[Training] Launching Two-Stage Soft-Shift IPPO (Inductive Head = {use_inductive_graph_head})...")
    print(f"[Training] Command: {' '.join(cmd)}\n")

    # Kaggle GPU kernels preallocate ~90% of JAX GPU memory; a subprocess trying to
    # allocate its own share on top of that OOMs immediately. Force dynamic allocation.
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    subprocess.check_call(cmd, env=env)

    return {
        "checkpoint_path": os.path.join(checkpoint_dir, "best_ippo_params.pkl"),
        "trace_path": os.path.join(output_dir, "evaluation_trace.json"),
        "metrics_path": os.path.join(output_dir, "training_metrics.csv")
    }

train_results = train_two_stage_ippo(num_episodes=1000)""")

    # CELL: Checkpoint inspection
    add_md(r"""## Step 6: Inspect Best Model Checkpoint & Verify Architecture""")
    add_code(r"""def inspect_checkpoint(checkpoint_path: str = None) -> dict:
    import os
    import pickle

    if checkpoint_path is None:
        ckpt_dir = "/kaggle/working/checkpoints" if os.path.exists("/kaggle/working") else "checkpoints"
        checkpoint_path = os.path.join(ckpt_dir, "best_ippo_params.pkl")

    if not os.path.exists(checkpoint_path):
        print(f"[Checkpoint] No checkpoint file found at: {checkpoint_path}")
        return {}

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    print(f"==================================================")
    print(f"  BEST IPPO MODEL CHECKPOINT: {checkpoint_path}")
    print(f"==================================================")
    print(f"- Keys in Checkpoint: {list(ckpt.keys())}")
    head_type = "InductiveIPPOActor class (architecturally identical to IPPOActor -- graph head removed)" if ckpt.get("use_inductive_graph_head") else "IPPOActor"
    arch_type = "Recurrent GRU (IPPORNNActor)" if ckpt.get("use_rnn") else "Feedforward"
    print(f"- Graph Head Architecture: {head_type}")
    print(f"- Sequence Model: {arch_type}")
    print(f"- Number of Sovereign Actor Parameter Sets: {len(ckpt.get('actor_list', []))}")
    print(f"- Number of Sovereign Critic Parameter Sets: {len(ckpt.get('critic_list', []))}")
    print(f"==================================================")
    return ckpt

checkpoint_data = inspect_checkpoint()""")

    # CELL: Training curve plots
    add_md(r"""## Step 7: Plot Training Curves (Reward, SHD, F1, Budgets)""")
    add_code(r"""def plot_training_curves(metrics_path: str = None):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    if metrics_path is None:
        metrics_path = os.path.join(out_dir, "training_metrics.csv")

    if not os.path.exists(metrics_path):
        print(f"[Training Curves] Metrics file not found at: {metrics_path}")
        return None

    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(df["train/episode"], df["train/episode_reward"], color="#3498db", lw=1.5)
    axes[0, 0].set_title("Episode Reward", fontweight="bold")
    axes[0, 0].set_xlabel("Episode")

    axes[0, 1].plot(df["train/episode"], df["eval/shd"], color="#e74c3c", lw=1.5)
    axes[0, 1].set_title("Structural Hamming Distance (SHD) [down]", fontweight="bold")
    axes[0, 1].set_xlabel("Episode")

    axes[1, 0].plot(df["train/episode"], df["eval/f1"], color="#2ecc71", lw=1.5)
    axes[1, 0].set_title("Edge Discovery F1-Score [up]", fontweight="bold")
    axes[1, 0].set_xlabel("Episode")

    axes[1, 1].plot(df["train/episode"], df["agent_0_budget"], label="Agent 0", color="#9b59b6")
    axes[1, 1].plot(df["train/episode"], df["agent_1_budget"], label="Agent 1", color="#f39c12")
    axes[1, 1].set_title("Remaining Budget at Episode End", fontweight="bold")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    tail = df.tail(max(1, len(df) // 10))
    print("\n--- Final-Decile Metric Summary ---")
    print(f"Mean SHD: {tail['eval/shd'].mean():.2f} +/- {tail['eval/shd'].std():.2f}")
    print(f"Mean F1 Score: {tail['eval/f1'].mean():.2f} +/- {tail['eval/f1'].std():.2f}")
    print(f"Mean Reward: {tail['train/episode_reward'].mean():.2f} +/- {tail['train/episode_reward'].std():.2f}")
    return df

df_metrics = plot_training_curves()""")

    # CELL: Trace visualization
    add_md(r"""## Step 8: Visualize Post-Training Evaluation Trace (All 8 Topologies)""")
    add_code(r"""def visualize_training_trace(trace_file: str = None, output_dir: str = None):
    import os
    from src.visualize_trace import parse_and_visualize_trace
    from IPython.display import Image, display

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    if trace_file is None:
        trace_file = os.path.join(out_dir, "evaluation_trace.json")
    if output_dir is None:
        output_dir = os.path.join(out_dir, "plots")

    if not os.path.exists(trace_file):
        print(f"[Visualization] Trace file not found at: {trace_file}")
        return

    print(f"Visualizing evaluation trace from: {trace_file}")
    parse_and_visualize_trace(trace_path=trace_file, output_dir=output_dir)

    summary_img = os.path.join(output_dir, "all_topologies_summary.png")
    if os.path.exists(summary_img):
        print("\n[Plot Summary] 8-Topology Evaluation Grid:")
        display(Image(filename=summary_img))

visualize_training_trace()""")

    # CELL: Temperature sweep
    add_md(r"""## Step 9: Multi-Temperature Evaluation & Stochastic Comparison Sweep
Evaluates the trained model under stochastic policy sampling ($T \in [0.0, 0.2, 0.5, 1.0]$) to analyze discovery robustness.""")
    add_code(r"""def evaluate_and_compare_temperatures(
    checkpoint_path: str = None,
    temperatures: list = [0.0, 0.2, 0.5, 1.0],
    output_dir: str = None
):
    import os
    from src.visualize_trace import compare_temperatures_and_visualize
    from IPython.display import Image, display

    ckpt_dir = "/kaggle/working/checkpoints" if os.path.exists("/kaggle/working") else "checkpoints"
    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

    if checkpoint_path is None:
        checkpoint_path = os.path.join(ckpt_dir, "best_ippo_params.pkl")
    if output_dir is None:
        output_dir = os.path.join(out_dir, "plots", "temperature_comparison")

    if not os.path.exists(checkpoint_path):
        print(f"[Temperature Sweep] Checkpoint file not found: {checkpoint_path}")
        return

    print(f"\n=== Running Multi-Temperature Comparison across T = {temperatures} ===")
    compare_temperatures_and_visualize(
        ckpt_path=checkpoint_path,
        temperatures=temperatures,
        output_dir=output_dir
    )

    mean_plot = os.path.join(output_dir, "temperature_scale_mean_comparison.png")
    if os.path.exists(mean_plot):
        print("\n[Plot Summary] Temperature Comparison Mean SHD Curve:")
        display(Image(filename=mean_plot))

evaluate_and_compare_temperatures(temperatures=[0.0, 0.2, 0.5, 1.0])""")

    out_path = os.path.join("notebooks", "kaggle_two_stage_ippo.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated notebook at {out_path}")

if __name__ == "__main__":
    create_kaggle_two_stage_notebook()
