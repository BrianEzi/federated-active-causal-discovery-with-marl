import json
import os

def create_kaggle_agent_estimator_matrix_notebook():
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
    add_md(r"""# Agent-vs-Estimator Learning: Isolating Intervention Skill from Graph-Estimator Memorization (Kaggle GPU Edition)
### Companion run to the Myriad HPC diagnostic matrix, same day
**Author / Researcher**: Brian Ezi | MSc Thesis Project
**Branch**: `investigate/graph-head-regression` (unmerged -- exploratory, not yet on `feat/vanilla-minimal-baseline`)

---

### Why this notebook exists
A prior investigation (`docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md`) found that a new `--estimator_type learned`
(a small trainable graph-structure estimator, separate from the RL policy, trained online every step) dramatically
improved SHD convergence versus the frozen `analytic` and `avici` estimators. That raised a real question: is the
*policy* learning to intervene better, or is the *estimator* just memorizing the small set of training topologies
(only 8 possible ground-truth DAGs exist in this environment)?

This notebook runs the same diagnostic matrix as the Myriad HPC companion run, so the two can be compared directly:

- **Estimators**: `learned` (can memorize), `avici` (frozen, fixed today to use real per-step samples + real
  intervention labels instead of a buggy synthetic reconstruction -- see Step 6 below), `analytic` (frozen,
  well-characterized formula)
- **Reward density**: `dense` (per-step SHD-shaped) vs `sparse` (terminal-only) -- tests whether the dense reward
  was "doing the agent's thinking for it"
- **3 seeds** (42, 7, 13) per combination = **18 runs total**, 200 episodes each, `--fixed_graph 0`/soft-shift,
  matching the Myriad run exactly for apples-to-apples comparison.

Six new **agent-side** metrics (not just end-to-end SHD) are logged per episode -- see `src/episode_metrics.py` and
`src/train.py`'s episode loop: interventions required to reach SHD=0, AUC of the within-episode SHD curve (both raw
and baseline-relative), edge-orientation-yield per intervention, a closed-form Gaussian information-gain estimate
(kept out of the reward pipeline -- diagnostic only), variance-shift trend, and agent coordination/redundancy. None
of these touch `compute_ippo_rewards`, so they stay trustworthy as independent evidence rather than something the
policy is directly optimized against.

**GPU note, stated honestly**: the `learned` and `analytic` arms run in this kernel's main process and will use
whatever accelerator Kaggle assigns (GPU if selected). The `avici` arms run in a genuinely separate, isolated
subprocess with its own `jax==0.4.30` (needed to dodge a real JAX/AVICI incompatibility -- see Step 6) -- that
isolated jax is **CPU-only** here, matching the Myriad HPC comparison run exactly, so the two stay comparable. GPU
selection speeds up 12 of the 18 runs, not all 18.""")

    # CELL: Setup
    add_md(r"""## Step 1: Install Dependencies & Verify Accelerator""")
    add_code(r"""def install_dependencies_and_check_gpu():
    import subprocess
    import sys

    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "wandb", "optax", "flax", "chex", "distrax", "dm-haiku", "matplotlib", "pandas", "networkx", "pytest", "scipy"
    ])

    import jax
    devices = jax.devices()
    print(f"\n[GPU Check] JAX Accelerator Devices: {devices}")
    platform = devices[0].platform if devices else "none"
    print(f"[GPU Check] Default Backend Platform: {platform.upper()}")
    return devices

devices = install_dependencies_and_check_gpu()""")

    # CELL: Repository Setup
    add_md(r"""## Step 2: Clone Repository & Switch to Branch (`investigate/graph-head-regression`)
This branch is unmerged, exploratory work -- it has the new `learned` estimator, the AVICI fix, and the new
agent-side metrics this notebook depends on. It does not yet exist on `feat/vanilla-minimal-baseline`.""")
    add_code(r"""def setup_repository(branch: str = "investigate/graph-head-regression", repo_url: str = "https://github.com/BrianEzi/federated-active-causal-discovery-with-marl.git") -> str:
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

repo_path = setup_repository(branch="investigate/graph-head-regression")""")

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
    add_md(r"""## Step 4: Run Verification Test Suite
81 tests as of this branch (70 pre-existing + 11 new for the agent-side metrics). A good sanity check before
committing to an ~18-run, multi-hour matrix.""")
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

    # CELL: Training wrapper
    add_md(r"""## Step 5: Training Wrapper (`python -m src.train` subprocess launcher)
Same pattern as the baseline notebook's `train_two_stage_ippo` -- shells out to the real CLI entry point (the same
one used for the Myriad HPC baseline) rather than reimplementing the training loop, so this notebook can never
drift from the actual training code. `env_overrides` lets the AVICI arms redirect to the isolated jax==0.4.30
subprocess set up in Step 6, without touching the arms that don't need it.""")
    add_code(r"""def train_two_stage_ippo(
    num_episodes: int = 200,
    num_variables: int = 4,
    num_agents: int = 2,
    batch_size: int = 32,
    initial_budget: float = 20.0,
    action_cost: float = 1.0,
    learning_rate: float = 3e-4,
    eval_freq: int = 5,
    fixed_graph: int = 0,
    allowed_topologies: str = None,
    use_rnn: bool = True,
    use_inductive_graph_head: bool = True,
    intervention_type: str = "soft_shift",
    soft_shift_val: float = 2.0,
    estimator_type: str = "analytic",      # "analytic", "avici", or "learned"
    freeze_graph_estimator: bool = True,
    obs_feedback: bool = True,
    reward_density: str = "dense",         # "dense" or "sparse"
    intrinsic_coef: float = 0.05,
    impact_coef: float = 0.0,
    boundary_margin: float = 0.10,
    curriculum: bool = True,
    seed: int = 42,
    checkpoint_dir: str = None,
    output_dir: str = None,
    eval_temperature: float = 0.0,
    use_wandb: bool = True,
    wandb_project: str = "federated-causal-marl-two-stage",
    run_name: str = None,
    env_overrides: dict = None
) -> dict:
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
        "--seed", str(seed),
        "--save_file"
    ]

    if fixed_graph is not None:
        cmd.extend(["--fixed_graph", str(fixed_graph)])
        cmd.append("--no_curriculum")
    elif allowed_topologies is not None:
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
        if run_name:
            cmd.extend(["--run_name", run_name])

    print(f"\n[Training] {run_name or 'run'} -- estimator={estimator_type}, reward_density={reward_density}, seed={seed}")
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
        print(result.stderr[-4000:])
        print("============================\n")
        raise RuntimeError(f"Training script failed with exit code {result.returncode}. See log above.")

    return {
        "checkpoint_path": os.path.join(checkpoint_dir, "best_ippo_params.pkl"),
        "trace_path": os.path.join(output_dir, "evaluation_trace.json"),
        "metrics_path": os.path.join(output_dir, "training_metrics.csv")
    }

print("train_two_stage_ippo defined.")""")

    # CELL: AVICI install (isolated)
    add_md(r"""## Step 6: Install AVICI (Isolated Subprocess)
AVICI needs `jax.sharding.PositionalSharding`, which Kaggle's base-image JAX has removed. Fixed the same way as the
earlier baseline notebook: a self-contained `jax==0.4.30`/`jaxlib==0.4.30` + matching pinned `dm-haiku`/`optax`/
`chex`/`numpy`/`scipy` stack in an isolated directory, run as a genuinely separate subprocess via `PYTHONPATH`, plus
a `sitecustomize.py` stub for a separate `pyarrow.plasma` removal issue (avici's own synthetic-data tooling, unused
by the inference path this project actually calls).

**New today** (see `docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md`'s "Morning session" section): the AVICI branch of
`predict_graph_hypothesis` now feeds it real per-step samples with real intervention labels (previously: fake
zero-mean synthetic samples with `interv=None`, discarding all mean-shift and intervention-identity signal). Also
fixed a serious performance regression this caused: without capping how much history gets fed per call
(`--avici_max_context`, default 400), every step's growing input shape forced JAX to re-JIT-compile AVICI's
internals from scratch, blowing a ~6-10 min run up to a projected ~4.4 hours. Capped context brings it back to a
practical ~30 min/200-episode run.""")
    add_code(r"""def install_avici():
    import subprocess
    import sys
    import os

    print("Installing AVICI (heavier dependency set: tensorflow, pyarrow, igraph, huggingface_hub)...")
    env = os.environ.copy()
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
        pip_install(["--upgrade", "pyarrow"], "pyarrow")
        pip_install([
            "tensorflow", "tensorflow-datasets~=4.3.0", "imageio", "jupyter", "matplotlib",
            "pandas", "igraph", "scikit-learn", "tqdm", "psutil", "deepdiff", "huggingface-hub",
        ], "avici runtime deps")
        pip_install(["--no-deps", "avici"], "avici")

        isolated_dir = "/kaggle/working/_avici_jax_env" if os.path.exists("/kaggle/working") else "_avici_jax_env"
        os.makedirs(isolated_dir, exist_ok=True)
        pip_install([
            "--target", isolated_dir,
            "jax==0.4.30", "jaxlib==0.4.30", "dm-haiku==0.0.12", "optax==0.2.2",
            "chex==0.1.86", "numpy==1.26.4", "scipy==1.13.1",
        ], "isolated jax-ecosystem stack")

        with open(os.path.join(isolated_dir, "sitecustomize.py"), "w") as f:
            f.write(
                "import sys, types\n"
                "if 'pyarrow.plasma' not in sys.modules:\n"
                "    sys.modules['pyarrow.plasma'] = types.ModuleType('pyarrow.plasma')\n"
            )

        isolated_env = os.environ.copy()
        isolated_env["PYTHONPATH"] = isolated_dir + os.pathsep + isolated_env.get("PYTHONPATH", "")
        # On a GPU-enabled Kaggle kernel, the main environment has a CUDA JAX plugin
        # (jax_plugins.xla_cuda12) installed. JAX auto-discovers jax_plugins.* namespace
        # packages across the *entire* sys.path at import time, not just the isolated
        # directory -- so even though the isolated jax==0.4.30/jaxlib==0.4.30 load
        # correctly, JAX still finds and tries to initialize that newer CUDA plugin, which
        # speaks an incompatible PJRT API version to the older jaxlib and crashes with
        # "Mismatched PJRT plugin PJRT API version" before any training code runs.
        # Confirmed on a real Kaggle GPU run. JAX_PLATFORMS=cpu skips backend/plugin
        # discovery entirely -- which is also what we already intend here, since this
        # isolated environment is CPU-only by design (matching the Myriad HPC comparison
        # run), so this just makes that explicit instead of relying on discovery to fail
        # to a CPU fallback safely (it doesn't -- it crashes instead).
        isolated_env["JAX_PLATFORMS"] = "cpu"
        check = subprocess.run(
            [sys.executable, "-c", "import avici; print('AVICI_IMPORT_OK')"],
            env=isolated_env, capture_output=True, text=True,
        )
        if "AVICI_IMPORT_OK" not in check.stdout:
            print("--- AVICI isolated import check stdout ---")
            print(check.stdout[-3000:])
            print("--- AVICI isolated import check stderr ---")
            print(check.stderr[-3000:])
            raise RuntimeError("avici import check failed inside the isolated jax==0.4.30 subprocess")

        print("AVICI installed and importable (verified in an isolated jax==0.4.30 subprocess).")
        return True, {"PYTHONPATH": isolated_env["PYTHONPATH"], "JAX_PLATFORMS": "cpu"}
    except Exception as e:
        print(f"AVICI install failed ({type(e).__name__}: {e}).")
        print("Skipping avici arms of the matrix -- learned/analytic arms are unaffected.")
        return False, None

avici_available, avici_env_overrides = install_avici()""")

    # CELL: Run the matrix
    add_md(r"""## Step 7: Run the 18-Run Diagnostic Matrix
`{learned, avici, analytic} x {dense, sparse reward} x 3 seeds (42, 7, 13)`, 200 episodes each,
`--fixed_graph 0`/soft-shift -- identical to the Myriad HPC companion run. Each run's exceptions are caught
individually so one flaky run doesn't abort the rest of an unattended multi-hour session; failures are recorded
and printed at the end rather than raised.""")
    add_code(r"""def run_estimator_reward_matrix(num_episodes: int = 200):
    import os
    import time

    estimators = (["learned"] * 6) + (["analytic"] * 6) + (["avici"] * 6)
    rewards = (["dense", "dense", "dense", "sparse", "sparse", "sparse"]) * 3
    seeds = [42, 7, 13] * 6

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    results = {}
    failures = []

    for est, rwd, seed in zip(estimators, rewards, seeds):
        if est == "avici" and not avici_available:
            print(f"Skipping avici/{rwd}/seed{seed} -- AVICI unavailable in this environment.")
            continue

        run_name = f"matrix_{est}_{rwd}_s{seed}"
        env_overrides = avici_env_overrides if est == "avici" else None

        start = time.time()
        try:
            result = train_two_stage_ippo(
                num_episodes=num_episodes,
                estimator_type=est,
                intervention_type="soft_shift",
                reward_density=rwd,
                fixed_graph=0,
                seed=seed,
                checkpoint_dir=os.path.join(out_dir, "matrix_runs", run_name, "checkpoints"),
                output_dir=os.path.join(out_dir, "matrix_runs", run_name),
                run_name=run_name,
                use_wandb=True,
                wandb_project="federated-causal-marl-two-stage",
                env_overrides=env_overrides,
            )
            elapsed = time.time() - start
            print(f"[{run_name}] done in {elapsed:.0f}s")
            results[run_name] = result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[{run_name}] FAILED after {elapsed:.0f}s: {type(e).__name__}: {e}")
            failures.append(run_name)

    print(f"\n=== Matrix complete: {len(results)}/18 succeeded, {len(failures)} failed ===")
    if failures:
        print("Failed runs:", failures)
    return results

matrix_results = run_estimator_reward_matrix(num_episodes=200)""")

    # CELL: Analysis
    add_md(r"""## Step 8: Compare Agent-Side Metrics Across the Matrix
Loads every run's `training_metrics.csv` and plots the new agent-side metrics (not just SHD) grouped by estimator
and reward density -- the actual test of whether the policy is learning real intervention skill (visible under the
frozen-but-fair `avici`) versus the estimator adapting on its own (only visible under `learned`).""")
    add_code(r"""def compare_matrix_results():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    matrix_dir = os.path.join(out_dir, "matrix_runs")

    estimators = ["learned", "analytic", "avici"]
    rewards = ["dense", "sparse"]
    seeds = [42, 7, 13]
    metrics_to_plot = [
        "eval/shd", "eval/interventions_to_shd0", "eval/shd_auc_normalized",
        "eval/orientation_precision_a0", "eval/redundancy_rate", "eval/entropy_gain_episode",
    ]

    frames = {}
    for est in estimators:
        for rwd in rewards:
            dfs = []
            for seed in seeds:
                run_name = f"matrix_{est}_{rwd}_s{seed}"
                path = os.path.join(matrix_dir, run_name, "training_metrics.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df["seed"] = seed
                    dfs.append(df)
            if dfs:
                frames[(est, rwd)] = pd.concat(dfs, ignore_index=True)

    if not frames:
        print("No matrix results found -- did Step 7 complete?")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    colors = {"learned": "#2ecc71", "analytic": "#e74c3c", "avici": "#3498db"}
    styles = {"dense": "-", "sparse": "--"}

    for ax, metric in zip(axes, metrics_to_plot):
        for (est, rwd), df in frames.items():
            if metric not in df.columns:
                continue
            grouped = df.groupby("train/episode")[metric].mean()
            ax.plot(grouped.index, grouped.values, color=colors[est], linestyle=styles[rwd],
                     alpha=0.85, label=f"{est}/{rwd}")
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.show()

    print("\n=== Summary (mean over all episodes, averaged across 3 seeds) ===")
    for (est, rwd), df in sorted(frames.items()):
        row = {m: df[m].mean() for m in metrics_to_plot if m in df.columns}
        print(f"{est:10s} / {rwd:6s} -- " + ", ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in row.items()))

compare_matrix_results()""")

    return nb

if __name__ == "__main__":
    notebook = create_kaggle_agent_estimator_matrix_notebook()
    output_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "kaggle_agent_estimator_matrix.ipynb")
    output_path = os.path.abspath(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated notebook at {output_path}")
