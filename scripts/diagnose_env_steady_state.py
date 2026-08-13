"""Phase 0 diagnostic (see the state-representation-fix plan): does the environment's
own observation-relevant statistics settle into a steady state under a repeated fixed
action, independent of how the running covariance is aggregated?

No training or checkpoint needed -- forces a fixed action sequence (agent_0 always
intervenes on node 0, agent_1 always on node 2, matching the concrete collapse example
in docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md) and logs the per-step L2 delta of each
covariance-derived observation channel plus the predicted DAG.

Usage: python -m scripts.diagnose_env_steady_state
"""
import numpy as np
import jax

from src.types import SCMConfig, MechanismType, NoiseType, ActionCategory
from src.evaluator_env import FederatedCausalEnv, compute_invariance_asymmetry_matrix


def main():
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    action_costs = np.array([1.0, 1.0])
    env = FederatedCausalEnv(
        config, action_costs, initial_budget=20.0, fixed_graph=True,
        max_steps=20, intervention_type="hard", estimator_type="analytic",
    )

    key = jax.random.PRNGKey(0)
    obs_dict, info = env.reset(key, force_idx=0)

    prev = {
        "obs_covariance": np.array(env.jax_state.obs_covariance),
        "running_covariance": np.array(env.jax_state.running_covariance),
        "asymmetry": np.array(compute_invariance_asymmetry_matrix(
            env.jax_state.obs_covariance, env.jax_state.int_covariance, env.jax_state.int_mask
        )),
        "predicted_dag": np.array(env.last_predicted_dag),
    }

    print(f"{'step':>4} | {'d(obs_cov)':>12} | {'d(run_cov)':>12} | {'d(asym)':>12} | {'d(pred_dag)':>12}")
    deltas_by_channel = {k: [] for k in prev}

    fixed_action = {
        "agent_0": (int(ActionCategory.INTERVENE), 0),
        "agent_1": (int(ActionCategory.INTERVENE), 2),
    }

    for step in range(env.max_steps):
        step_key = jax.random.fold_in(key, step)
        obs_dict, rewards, terminated, step_info = env.step(fixed_action, predicted_dags=None, key=step_key)

        curr = {
            "obs_covariance": np.array(env.jax_state.obs_covariance),
            "running_covariance": np.array(env.jax_state.running_covariance),
            "asymmetry": np.array(compute_invariance_asymmetry_matrix(
                env.jax_state.obs_covariance, env.jax_state.int_covariance, env.jax_state.int_mask
            )),
            "predicted_dag": np.array(env.last_predicted_dag),
        }

        row = []
        for k in prev:
            delta = float(np.linalg.norm(curr[k] - prev[k]))
            deltas_by_channel[k].append(delta)
            row.append(delta)
        print(f"{step:>4} | {row[0]:>12.6f} | {row[1]:>12.6f} | {row[2]:>12.6f} | {row[3]:>12.6f}")

        prev = curr
        if terminated:
            break

    print("\n=== Summary: does each channel flatten to near-zero before step 20? ===")
    for k, deltas in deltas_by_channel.items():
        early = np.mean(deltas[:3])
        late = np.mean(deltas[-3:])
        ratio = late / early if early > 1e-9 else float("nan")
        print(f"{k:20s}  early(steps 0-2) mean delta={early:.6f}  late(last 3) mean delta={late:.6f}  "
              f"ratio={ratio:.4f}  {'FLATTENED' if late < 0.05 * early else 'still moving'}")


if __name__ == "__main__":
    main()
