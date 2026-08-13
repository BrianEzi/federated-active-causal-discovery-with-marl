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
- **Recurrent Policy (GRU)**: Actor and Critic networks default to GRU-based recurrent memory (`IPPORNNActor`/`IPPORNNCritic`) rather than a plain feedforward MLP. Each episode is a sequence of up to `max_steps` interventions where every step's observation is only the *current* covariance/mask state with no explicit memory of earlier steps -- an MLP re-decides from scratch each step, while the GRU carries that within-episode history forward. Pass `--no_rnn` to `src.train` (or `use_rnn=False` to `train_two_stage_ippo` below) to reproduce the earlier feedforward baseline.

**Note on the Myriad HPC baseline**: that 1000-episode empirical run used the earlier feedforward-MLP default, before the switch to GRU above -- results from this notebook are **not** directly comparable to it anymore unless you pass `use_rnn=False`.

**Optional (Steps 10-12)**: trains a second comparison run using AVICI, a pretrained learned graph estimator, instead of the analytic formula -- useful for checking whether the analytic estimator's plateauing SHD is a limitation of the heuristic itself, or something deeper in the training loop. See Step 10 for honestly-stated caveats about this integration (sample reconstruction from covariance, no interventional labels yet, heavier optional dependencies).""")

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
    use_rnn: bool = True,                  # GRU recurrent Actor/Critic to track within-episode history; MLP re-decides from scratch each step
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
    wandb_project: str = "federated-causal-marl-kaggle",
    env_overrides: dict = None
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

    if use_rnn:
        cmd.append("--use_rnn")
    else:
        cmd.append("--no_rnn")

    if use_wandb:
        cmd.extend(["--use_wandb", "--wandb_project", wandb_project])

    print(f"\n[Training] Launching Two-Stage Soft-Shift IPPO (RNN = {use_rnn}, Inductive Head = {use_inductive_graph_head})...")
    print(f"[Training] Command: {' '.join(cmd)}\n")

    # Kaggle GPU kernels preallocate ~90% of JAX GPU memory; a subprocess trying to
    # allocate its own share on top of that OOMs immediately. Force dynamic allocation.
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print("\n=== SUBPROCESS ERROR LOG ===")
        print(result.stderr)
        print("============================\n")
        raise RuntimeError(f"Training script failed with exit code {result.returncode}. See log above.")

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

    # CELL: AVICI install (optional, isolated)
    add_md(r"""## Step 10 (Optional): Install AVICI for a Learned-Estimator Comparison
The run above used the fast closed-form "analytic invariance" graph estimator (`--estimator_type analytic`).
AVICI ([Lorch et al.](https://github.com/larslorch/avici), Amortized Variational Inference for Causal Discovery)
is a pretrained transformer that predicts causal structure directly from data, rather than a hand-derived
formula -- this section trains a second run using it, so you can see whether a learned estimator does better
where the analytic one plateaus.

**Known caveats, stated honestly rather than glossed over:**
- AVICI's model expects `[n, d]` raw observation samples. This environment only tracks *aggregated covariance*
  (not raw per-step samples), so samples are synthesized from `N(0, running_covariance)` before being passed in.
  This is exact for the default `LINEAR` mechanism (covariance is then a complete sufficient statistic of the
  SCM's stationary distribution) -- it is **not** faithful for `NONLINEAR_ANM` / `POST_NONLINEAR`.
- Per-sample intervention labels aren't reconstructed (`interv=None` is passed) -- AVICI currently sees a
  purely observational-looking batch, so it isn't using its full interventional-data capability yet.
- AVICI pulls in a much heavier dependency chain than the rest of this notebook (`tensorflow`,
  `tensorflow-datasets`, `pyarrow==10.0.1`, `igraph`, `deepdiff`, `huggingface-hub`). The install is isolated
  to this cell specifically so a failure here can't affect the Step 5 run above. `pyarrow==10.0.1` is an old
  pin and may need a source build on some Python versions.
- **Confirmed, not hypothetical**: AVICI's own code (`avici/pretrain.py`) does
  `from jax.sharding import PositionalSharding` unconditionally -- an API removed in newer JAX releases.
  AVICI declares only `jax>=0.3.17` with no upper bound, so it breaks against whatever (newer) JAX
  Kaggle's base image ships. `jax==0.4.30` (this project's own pinned version, per `requirements.txt`)
  still has `PositionalSharding` -- confirmed by downloading its real wheel and checking. But Step 1
  above already imported the newer jax into this kernel, and Python caches modules by name, so a plain
  `pip install jax==0.4.30` in this cell would change what's on disk without changing what's already
  loaded -- it would **not** fix the import on its own. So the AVICI comparison run below installs its
  own self-contained jax-ecosystem stack (`jax==0.4.30`/`jaxlib==0.4.30` plus matching pinned
  `dm-haiku`/`optax`/`chex`/`numpy`/`scipy`, from this project's own `requirements.txt` -- confirmed
  empirically that isolating jax/jaxlib alone isn't enough, since `haiku` would still resolve from Step
  1's newer, unpinned install and itself need JAX APIs that don't exist in 0.4.30) into an isolated
  directory, and runs the comparison as a genuinely separate subprocess (via `python -m src.train`, the
  same CLI entry point used for the analytic run above) with that directory first on `PYTHONPATH`. This
  never touches the jax already loaded in this kernel, so Steps 1-9 are unaffected either way.
  Trade-off: the isolated `jaxlib==0.4.30` here is CPU-only (no CUDA wheel requested), so combined with
  AVICI's own transformer forward pass running on every environment step, this comparison run will
  likely be noticeably slower than the GPU-accelerated run above -- if it's taking too long, interrupt
  and re-run with a smaller `num_episodes` first to gauge speed before committing to the full
  comparison.""")
    add_code(r"""def install_avici():
    import subprocess
    import sys
    import os

    print("Installing AVICI (heavier dependency set: tensorflow, pyarrow, igraph, huggingface_hub)...")
    env = os.environ.copy()
    # avici's sdist chain pulls in the deprecated 'sklearn' PyPI shim, which modern pip
    # refuses to build without this explicit opt-in.
    env["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"

    def pip_install(args, label, target_env=None):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + args,
            env=target_env if target_env is not None else env, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"--- pip stdout (tail, {label}) ---")
            print(result.stdout[-3000:])
            print(f"--- pip stderr (tail, {label}) ---")
            print(result.stderr[-3000:])
            raise subprocess.CalledProcessError(result.returncode, [sys.executable, "-m", "pip", "install"] + args)

    try:
        # avici (confirmed from its real PyPI metadata) pins pyarrow==10.0.1 exactly. That
        # version has no prebuilt wheel for Kaggle's Python, so a normal `pip install avici`
        # forces a source build that fails at "Preparing metadata (pyproject.toml)" -- it
        # needs the Arrow C++ toolchain, which no pip flag alone provides (confirmed on a
        # real Kaggle run). avici itself ships as a pure-Python wheel with no C extensions,
        # so instead of fighting that pin: install every OTHER dependency avici actually
        # imports at runtime, deliberately skipping jax/jaxlib/dm-haiku/optax (this
        # notebook's Step 1 already installs those at versions avici's own, looser
        # constraints are satisfied by -- touching them here risks upgrading jax underneath
        # the training that already ran in Steps 1-9), then install avici itself with
        # --no-deps so pip never resolves or builds pyarrow==10.0.1 at all.
        pip_install(["--upgrade", "pyarrow"], "pyarrow")
        pip_install([
            "tensorflow", "tensorflow-datasets~=4.3.0", "imageio", "jupyter", "matplotlib",
            "pandas", "igraph", "scikit-learn", "tqdm", "psutil", "deepdiff", "huggingface-hub",
        ], "avici runtime deps")
        pip_install(["--no-deps", "avici"], "avici")

        # avici's own code needs jax.sharding.PositionalSharding, which Kaggle's (newer)
        # base-image JAX has removed -- confirmed empirically on a real Kaggle run. This
        # kernel already imported that newer jax during Step 1, and Python caches modules
        # by name, so installing an older jax here wouldn't change what THIS process has
        # already loaded. Instead: install a self-contained jax-ecosystem stack into an
        # isolated directory, and verify + later run AVICI in genuinely separate
        # subprocesses with that directory first on PYTHONPATH -- never touching the jax
        # already loaded here, so Steps 1-9 stay unaffected regardless of outcome.
        #
        # This isolated stack must be jax==0.4.30/jaxlib==0.4.30 (this project's own pinned
        # version, per requirements.txt; confirmed to still have PositionalSharding) PLUS
        # matching pinned dm-haiku/optax/chex/numpy/scipy from that same requirements.txt --
        # NOT the unpinned dm-haiku etc. Step 1 installed into the main environment. Confirmed
        # empirically: the isolated jax==0.4.30 alone still resolved `import haiku` from the
        # main environment's newer, unpinned dm-haiku, which needs a jax.core API
        # (take_current_trace) that doesn't exist in 0.4.30, and crashed. requirements.txt's
        # jax/jaxlib/dm-haiku/optax/chex/numpy/scipy pins are a set already validated together
        # for this project's own training code, so isolating that whole set (not just
        # jax/jaxlib) keeps everything mutually compatible.
        isolated_dir = "/kaggle/working/_avici_jax_env" if os.path.exists("/kaggle/working") else "_avici_jax_env"
        os.makedirs(isolated_dir, exist_ok=True)
        pip_install([
            "--target", isolated_dir,
            "jax==0.4.30", "jaxlib==0.4.30", "dm-haiku==0.0.12", "optax==0.2.2",
            "chex==0.1.86", "numpy==1.26.4", "scipy==1.13.1",
        ], "isolated jax-ecosystem stack")

        # avici/__init__.py unconditionally does `from .buffer import Sampler`, and
        # avici/buffer.py does `import pyarrow.plasma as plasma` at module level. Plasma
        # was removed from modern pyarrow entirely (confirmed on a real Kaggle run:
        # ModuleNotFoundError: No module named 'pyarrow.plasma'), and no pyarrow version
        # both has a prebuilt wheel for Kaggle's Python and still ships Plasma. But
        # Sampler (and the simulate_data() it backs) is avici's own synthetic
        # training-data generator -- confirmed by reading avici's real source that
        # avici/pretrain.py (load_pretrained, AVICIModel) and avici/model.py, the only
        # code this project actually calls, have zero references to plasma, Sampler, or
        # buffer. So the crash is an import-time-only dead weight, not a real capability
        # gap for inference. A sitecustomize.py in the isolated directory (auto-imported
        # by Python at startup for anything using this PYTHONPATH, so it applies to both
        # the import check below and the actual training subprocess in Step 11) stubs
        # out the pyarrow.plasma module before avici ever imports it.
        with open(os.path.join(isolated_dir, "sitecustomize.py"), "w") as f:
            f.write(
                "import sys, types\n"
                "if 'pyarrow.plasma' not in sys.modules:\n"
                "    sys.modules['pyarrow.plasma'] = types.ModuleType('pyarrow.plasma')\n"
            )

        isolated_env = os.environ.copy()
        isolated_env["PYTHONPATH"] = isolated_dir + os.pathsep + isolated_env.get("PYTHONPATH", "")
        check = subprocess.run(
            [sys.executable, "-c", "import avici; print('AVICI_IMPORT_OK')"],
            env=isolated_env, capture_output=True, text=True,
        )
        if "AVICI_IMPORT_OK" not in check.stdout:
            print("--- AVICI isolated import check stdout ---")
            print(check.stdout[-3000:])
            print("--- AVICI isolated import check stderr ---")
            print(check.stderr[-3000:])
            if "PositionalSharding" in check.stderr:
                print("Still hit the PositionalSharding issue even inside the isolated jax==0.4.30 "
                      "environment -- something didn't isolate cleanly (e.g. a stray system jaxlib "
                      "shadowing the isolated one). See the raw output above.")
            if "pyarrow.plasma" in check.stderr:
                print("Still hit the pyarrow.plasma issue even with the sitecustomize.py stub in "
                      "place -- check that sitecustomize.py actually landed in the isolated directory "
                      "and that Python is picking it up (site processing must not be disabled). See "
                      "the raw output above.")
            raise RuntimeError("avici import check failed inside the isolated jax==0.4.30 subprocess")

        print("AVICI installed and importable (verified in an isolated jax==0.4.30 subprocess).")
        return True, {"PYTHONPATH": isolated_env["PYTHONPATH"]}
    except Exception as e:
        print(f"AVICI install failed ({type(e).__name__}: {e}).")
        print("Skipping the AVICI comparison run -- Steps 1-9 above are unaffected.")
        return False, None

avici_available, avici_env_overrides = install_avici()""")

    # CELL: AVICI comparison training run
    add_md(r"""## Step 11 (Optional): Train with the AVICI Estimator""")
    add_code(r"""def train_avici_comparison(num_episodes: int = 1000):
    import os

    if not avici_available:
        print("AVICI is not available in this environment -- skipping comparison run.")
        return None

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    print(f"[AVICI] Training for {num_episodes} episodes as an isolated subprocess (jax==0.4.30, "
          f"CPU-only jaxlib) via 'python -m src.train --estimator_type avici' -- AVICI's transformer "
          f"forward pass runs every environment step, so expect this to be noticeably slower than the "
          f"GPU-accelerated analytic run above. If it's taking too long, interrupt and re-run with a "
          f"smaller num_episodes first to gauge speed.")
    return train_two_stage_ippo(
        num_episodes=num_episodes,
        estimator_type="avici",
        checkpoint_dir=os.path.join(out_dir, "checkpoints_avici"),
        output_dir=os.path.join(out_dir, "avici_run"),
        use_wandb=True,
        wandb_project="federated-causal-marl-two-stage",
        env_overrides=avici_env_overrides,
    )

avici_results = train_avici_comparison(num_episodes=1000)""")

    # CELL: Comparison plot
    add_md(r"""## Step 12 (Optional): Compare Analytic vs AVICI SHD/F1 Trajectories""")
    add_code(r"""def plot_analytic_vs_avici_comparison():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    analytic_path = os.path.join(out_dir, "training_metrics.csv")
    avici_path = os.path.join(out_dir, "avici_run", "training_metrics.csv")

    if not os.path.exists(analytic_path) or not os.path.exists(avici_path):
        print("Need both the Step 5 (analytic) and Step 11 (AVICI) runs to compare.")
        print(f"  analytic: {analytic_path} (exists={os.path.exists(analytic_path)})")
        print(f"  avici:    {avici_path} (exists={os.path.exists(avici_path)})")
        return

    df_analytic = pd.read_csv(analytic_path)
    df_avici = pd.read_csv(avici_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_analytic["train/episode"], df_analytic["eval/shd"], label="Analytic", color="#e74c3c", alpha=0.8)
    axes[0].plot(df_avici["train/episode"], df_avici["eval/shd"], label="AVICI", color="#3498db", alpha=0.8)
    axes[0].set_title("SHD: Analytic vs AVICI Estimator", fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("SHD")
    axes[0].legend()

    axes[1].plot(df_analytic["train/episode"], df_analytic["eval/f1"], label="Analytic", color="#e74c3c", alpha=0.8)
    axes[1].plot(df_avici["train/episode"], df_avici["eval/f1"], label="AVICI", color="#3498db", alpha=0.8)
    axes[1].set_title("F1: Analytic vs AVICI Estimator", fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    tail_analytic = df_analytic.tail(max(1, len(df_analytic) // 10))
    tail_avici = df_avici.tail(max(1, len(df_avici) // 10))
    print("\n--- Final-Decile Mean SHD (lower is better) ---")
    print(f"Analytic: {tail_analytic['eval/shd'].mean():.2f} +/- {tail_analytic['eval/shd'].std():.2f}")
    print(f"AVICI:    {tail_avici['eval/shd'].mean():.2f} +/- {tail_avici['eval/shd'].std():.2f}")

plot_analytic_vs_avici_comparison()""")

    out_path = os.path.join("notebooks", "kaggle_two_stage_ippo.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated notebook at {out_path}")

if __name__ == "__main__":
    create_kaggle_two_stage_notebook()
