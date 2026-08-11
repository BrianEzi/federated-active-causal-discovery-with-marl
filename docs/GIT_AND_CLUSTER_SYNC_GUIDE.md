# 🔄 Git & Cluster Synchronization Guide

This guide establishes the operational protocol for synchronizing code, hyperparameter configs, and trained artifacts across three environments:
1. **Local Development Machine** (`c:\Workspace\MSc Project`)
2. **GitHub Central Repository** (`origin/main`)
3. **UCL Myriad HPC Cluster** (`ucabbse@myriad.rc.ucl.ac.uk:~/marl_causal`)

---

## 🧠 Synchronization Decision Matrix

Future AI agents and researchers must consult this matrix before initiating any code synchronization:

| Development Scenario | Strategy | Primary Command | Rationale |
| :--- | :--- | :--- | :--- |
| **New Feature, Refactor, or Benchmark Code** | **Strategy 1: Standard Git Pipeline** | `git push origin main`<br>`ssh myriad "cd ~/marl_causal && git pull"` | Guarantees code is version-controlled, reviewable, and reproducible across environments. |
| **Standard Training Job Deployment** | **Strategy 2: 1-Liner Automated Sync** | `git push origin main; ssh myriad "cd ~/marl_causal && git pull origin main"` | Combines local push and remote pull into a single seamless action. |
| **Rapid Iteration, Print Debugging, or Quick Tweaks** | **Strategy 3: Direct `rsync` Mirroring** | `rsync -avz --exclude='.git' --exclude='envs' ... ./ myriad:~/marl_causal/` | Avoids polluting Git commit history with temporary "test fix" commits while testing rapid edits. |
| **Retrieving Experiment Results & Checkpoints** | **Strategy 4: Remote Artifact Extraction** | `scp myriad:~/marl_causal/training_metrics.csv ./`<br>`scp -r myriad:~/marl_causal/checkpoints/ ./` | Downloads generated CSVs, JSON evaluation traces, and model weights from Myriad to local repo for analysis. |

---

## 📋 Strategy Protocols

### Strategy 1: Standard Git Pipeline (Local → GitHub → Myriad)
Use this strategy for all formal code changes, architectural refactors, and feature additions.

1. **Commit and Push Locally**:
   ```bash
   git add .
   git commit -m "feat: add inductive graph head architecture"
   git push origin main
   ```

2. **Pull on Myriad**:
   ```bash
   ssh myriad "cd ~/marl_causal && git pull origin main"
   ```

---

### Strategy 2: 1-Liner Automated Sync (Local → Myriad)
Use this strategy for quick, single-command deployment when you know the code is ready for cluster runs.

Run directly in PowerShell / Terminal:
```powershell
git push origin main; ssh myriad "cd ~/marl_causal && git pull origin main"
```

---

### Strategy 3: Direct `rsync` / `scp` Mirroring (Local → Myriad)
Use this strategy when making minor print statement tweaks, debugging environment bugs, or testing parameter variations without creating Git commits.

#### PowerShell Command (Using `scp` via SSH Host Alias):
```powershell
scp -r src scripts tests requirements.txt submit_job.sh myriad:~/marl_causal/
```

#### Linux / WSL / Bash Command (Using `rsync`):
```bash
rsync -avz \
  --exclude='.git' \
  --exclude='envs' \
  --exclude='logs' \
  --exclude='wandb' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  ./ myriad:~/marl_causal/
```

> [!CAUTION]
> **Safety Exclusion**: Never omit `--exclude='envs'` or `--exclude='logs'` when running `rsync`. Overwriting Myriad's virtual environment (`~/envs/marl_env`) or active SGE log files will break running jobs.

---

### Strategy 4: Remote Artifact Extraction (Myriad → Local)
Use this strategy after a training or evaluation job completes on Myriad to pull metrics, plots, and checkpoints back to your local repository for deep analysis.

```powershell
# Extract CSV metrics and evaluation traces
scp myriad:~/marl_causal/training_metrics.csv ./
scp myriad:~/marl_causal/evaluation_trace.json ./

# Extract saved model checkpoints
scp -r myriad:~/marl_causal/checkpoints/ ./
```

---

## 🛡️ Operating Safeguards for AI Agents

1. **Git Hygiene**:
   Never commit virtual environments (`envs/`), cluster logs (`logs/`), WandB telemetry (`wandb/`), or raw binary checkpoints (`*.pkl`) to Git. Verify `.gitignore` is respected before pushing.

2. **Module Execution Rule**:
   Always execute code on Myriad using Python module syntax from the project root (`python -m src.train`) to prevent `src/types.py` from shadowing standard library `types`.

3. **Pre-Submission Verification**:
   Before launching long-running SGE jobs, execute the unit test suite on Myriad:
   ```bash
   ssh myriad "source ~/envs/marl_env/bin/activate && cd ~/marl_causal && pytest tests/ -v"
   ```
