# Data Analysis Guide

This guide details how to programmatically extract and analyze your training logs from Weights & Biases (W&B) for the Federated Active Causal Discovery project.

## 1. Extracting Data via W&B API

You can fetch your run histories directly into Pandas DataFrames using the `wandb` Python library.

```python
import wandb
import pandas as pd

api = wandb.Api()
# Replace with your actual entity and project name
project_name = "your-entity/federated-causal-marl-kaggle"
runs = api.runs(project_name)

all_metrics = []
for run in runs:
    # Filter runs by state or name if needed
    if run.state != "finished":
        continue
        
    # Fetch history (metrics over steps/episodes)
    history = run.history()
    history['run_id'] = run.id
    history['agent_type'] = run.config.get('agent_type', 'unknown')
    all_metrics.append(history)

df = pd.concat(all_metrics)
print(df.head())
```

## 2. Analyzing Performance (SHD & F1 curves)

To plot the learning curves and compare your IPPO agent against baselines:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='train/episode', y='eval/shd', hue='agent_type')
plt.title("Structural Hamming Distance (SHD) over Training")
plt.xlabel("Episode")
plt.ylabel("SHD (Lower is better)")
plt.show()
```

## 3. Viewing the DAG Visualisations

Since we integrated visual DAG logging, you can view the causal graphs directly in the W&B dashboard under the **Media** tab.
To programmatically download the images for a report:

```python
for run in runs:
    for file in run.files():
        if "media/images" in file.name:
            file.download(replace=True)
            print(f"Downloaded {file.name}")
```

## 4. Troubleshooting Actor Loss Collapse

If `eval/shd` remains flat and `train/episode_reward` is stuck at a large negative value (e.g. `-80`), it indicates the agent is predicting an invariant graph (often an empty graph, SHD=3). Check the `train/actor_loss` scale. If you notice massive `actor_loss` values early on, ensure that PPO Generalized Advantage Estimation (GAE) normalization is active in `src/marl/ppo_trainer.py`.
