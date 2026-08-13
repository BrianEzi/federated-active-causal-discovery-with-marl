# UCL Myriad HPC & GPU Cluster Guide

This guide documents the setup, execution, and troubleshooting workflows for running the **Federated Active Causal Discovery with MARL** training pipeline on the **UCL Myriad Cluster**.

---

## 1. Connecting via SSH (Passwordless Authentication)

### SSH Configuration (`~/.ssh/config`)
Add the following configuration block to your local SSH config (`~/.ssh/config` on Linux/macOS or `C:\Users\<user>\.ssh\config` on Windows):

```ssh-config
Host knuckles
    HostName knuckles.cs.ucl.ac.uk
    User ezinwoke
    IdentityFile ~/.ssh/id_ed25519

Host myriad
    HostName myriad.rc.ucl.ac.uk
    User ucabbse
    ProxyJump knuckles
    IdentityFile ~/.ssh/id_ed25519
```

### SSH Key Setup
Generate a local SSH key pair and register it on both `knuckles` and `myriad`:

```powershell
# In local PowerShell:
ssh-keygen -t ed25519 -C "ezinwoke@ucl.ac.uk"

# Copy key to jump host:
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | ssh ezinwoke@knuckles.cs.ucl.ac.uk "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# Copy key to Myriad:
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | ssh myriad "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Once registered, connect seamlessly with:
```bash
ssh myriad
```

---

## 2. Python Environment Setup on Myriad

### Environment Modules (`~/.bashrc`)
To ensure Python 3.11 loads without library conflicts, append the module load command to your `~/.bashrc` on Myriad:

```bash
source /shared/ucl/apps/bin/defmods
module load python3/3.11
```

### Virtual Environment Creation
```bash
# Create isolated environment
python3 -m venv ~/envs/marl_env
source ~/envs/marl_env/bin/activate
cd ~/marl_causal

# Install pinned pre-compiled wheels (avoids C/Cython source compilation)
pip install --prefer-binary -r requirements.txt
```

### JAX CUDA Setup (For GPU Execution)
If submitting to GPU queues (`#$ -l gpu=1`), install the CUDA 12 JAX build inside your virtual environment:

```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## 3. Submitting & Managing Jobs (Sun Grid Engine)

### Submission Script (`~/marl_causal/submit_job.sh`)
```bash
#!/bin/bash -l

# SGE Directives
#$ -N marl_causal_train
#$ -cwd
#$ -l h_rt=04:00:00
#$ -pe smp 4
#$ -o logs/
#$ -e logs/

# Request GPU (optional)
# #$ -l gpu=1

mkdir -p logs
source /home/ucabbse/envs/marl_env/bin/activate
cd /home/ucabbse/marl_causal

# Run MARL training via module invocation
python -m src.train --num_agents 2 --batch_size 32 --num_episodes 1000 --save_file
```

### Essential SGE Commands
| Command | Action |
| :--- | :--- |
| `qsub submit_job.sh` | Submit job to SGE queue |
| `qstat -u ucabbse` | View active jobs (`qw` = queued, `r` = running) |
| `qstat -g c` | Check node capacity and available core slots |
| `qdel <JOB_ID>` | Cancel a pending or running job |
| `tail -f logs/marl_causal_train.o<JOB_ID>` | View live standard output log |
| `tail -f logs/marl_causal_train.e<JOB_ID>` | View live error log |
| `qacct -j <JOB_ID>` | Inspect job accounting and resource usage |

---

## 4. Troubleshooting & Known Caveats

1. **Avoid `src/types.py` Module Shadowing**:
   Executing `python src/train.py` from within `src/` causes Python to import `./src/types.py` instead of the standard library `types` module, resulting in `ImportError: cannot import name 'GenericAlias'`. Always execute training using module syntax from the repository root:
   ```bash
   python -m src.train [ARGS]
   ```

2. **SGE Log Output Syntax**:
   Use `#$ -o logs/` and `#$ -e logs/` in submission scripts. Do not escape variable names (e.g. `\$JOB_NAME`) inside SGE `#$` directives.

3. **WandB Offline Syncing**:
   If compute nodes block external HTTP requests, set `export WANDB_MODE=offline` in `submit_job.sh` and sync logs manually from the login node:
   ```bash
   wandb sync wandb/latest-run
   ```
