---
name: UCL Myriad HPC & GPU Workflow
description: Essential Operational Guide for SSH passwordless access, Python 3.11 module setup, JAX/CUDA wheels, SGE batch submission, and WandB tracking on UCL Myriad Cluster.
---

# 🚀 UCL Myriad HPC & GPU Operational Protocol

This skill provides step-by-step instructions for AI agents and human researchers running training pipelines on the UCL Myriad High-Performance Computing (HPC) cluster.

---

## 1. SSH Connection & Jump Host Protocol

Myriad requires hopping through the UCL CS bastion jump host (`knuckles.cs.ucl.ac.uk`).

### SSH Config Setup (`~/.ssh/config` on local machine)
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

### One-Touch Connection
```bash
ssh myriad
```

---

## 2. Environment & Module Requirements

Myriad runs CentOS 7. To prevent C compiler / GLIBC mismatch errors during `pip install`, follow strict module discipline.

### Module Configuration (`~/.bashrc` on Myriad)
Ensure `~/.bashrc` loads Python 3.11 cleanly without conflicting compiler modules:
```bash
source /shared/ucl/apps/bin/defmods
module load python3/3.11
```

### Virtual Environment Path & Activation
```bash
source /home/ucabbse/envs/marl_env/bin/activate
cd /home/ucabbse/marl_causal
```

### Pre-Compiled Wheel Rule (Preventing Source Build Failures)
NEVER allow `pip` to download `.tar.gz` source packages that require Meson, Cython, or Go compilers. Always use exact wheel pins or `--prefer-binary`:

```txt
jax==0.4.30
jaxlib==0.4.30
flax==0.8.4
orbax-checkpoint==0.5.3
optax==0.2.2
chex==0.1.86
numpy==1.26.4
scipy==1.13.1
dm-haiku==0.0.12
pytest==7.4.4
wandb==0.16.6
```

### Installing JAX with CUDA Support (GPU Nodes)
When running GPU-accelerated jobs, install JAX CUDA 12 wheels inside the virtualenv:
```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## 3. SGE Job Submission Protocol (Sun Grid Engine)

### Standard Batch Script Template (`submit_job.sh`)
```bash
#!/bin/bash -l

# SGE Directives
#$ -N marl_causal_train
#$ -cwd
#$ -l h_rt=04:00:00
#$ -pe smp 4
#$ -o logs/
#$ -e logs/

# For GPU Jobs, uncomment the line below:
# #$ -l gpu=1

mkdir -p logs
source /home/ucabbse/envs/marl_env/bin/activate
cd /home/ucabbse/marl_causal

# IMPORTANT: Always use `python -m src.train` from root directory to prevent
# `src/types.py` from shadowing Python standard library `types`.
python -m src.train --num_agents 2 --batch_size 32 --num_episodes 1000 --save_file
```

### Job Management Commands
- **Submit Job**: `qsub submit_job.sh`
- **Check Queue**: `qstat -u ucabbse`
- **Check Cluster Capacity**: `qstat -g c`
- **Delete Job**: `qdel <JOB_ID>`
- **Live Output Stream**: `tail -f logs/marl_causal_train.o<JOB_ID>`
- **Job Accounting**: `qacct -j <JOB_ID>`

---

## 4. Operational Best Practices & Troubleshooting

1. **Avoid `src/types.py` Shadowing**:
   Never run `python src/train.py` from inside `src/`. Always invoke via module syntax (`python -m src.train`) from project root.

2. **SGE Output Directives**:
   Use `#$ -o logs/` and `#$ -e logs/` in submission scripts. Do NOT use backslashed variables like `\$JOB_NAME.\$JOB_ID.out` inside `#$` directives.

3. **Short Runtime Priority**:
   Requesting `-l h_rt=04:00:00` or shorter allows SGE to schedule jobs significantly faster into backfill slots.

4. **WandB Offline Syncing**:
   If compute nodes lose external internet access, set `export WANDB_MODE=offline` in your job script, and sync logs from the login node afterwards using `wandb sync wandb/latest-run`.

---

## 5. Code Synchronization Protocols

Refer to `docs/GIT_AND_CLUSTER_SYNC_GUIDE.md` for full strategy definitions.

| Scenario | Strategy | Command |
| :--- | :--- | :--- |
| **New Feature / Refactor** | **Strategy 1: Full Git Flow** | `git push origin main` -> `ssh myriad "cd ~/marl_causal && git pull"` |
| **1-Click Deployment** | **Strategy 2: 1-Liner Sync** | `git push origin main; ssh myriad "cd ~/marl_causal && git pull origin main"` |
| **Quick Debugging / Print Tweaks** | **Strategy 3: `rsync`/`scp` Mirroring** | `scp -r src scripts tests requirements.txt submit_job.sh myriad:~/marl_causal/` |
| **Extracting Metrics / Weights** | **Strategy 4: Artifact Retrieval** | `scp myriad:~/marl_causal/training_metrics.csv ./` |

