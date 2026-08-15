"""The overnight experiment matrix, as data.

Emits one JSON object per configuration. The submit script indexes into this list and the
analysis reads the same file, so what was *intended* to run and what is *reported* cannot
drift apart -- a failure mode this project has already paid for.

Design: **one-factor-at-a-time around a fixed baseline**, not a full grid. A full grid over
13 levers is astronomically large; OFAT answers the question actually being asked, which is
"what does each lever do, and does the conclusion depend on it?" It cannot detect
interactions between levers, and that limitation is stated in the write-up rather than
papered over.

Each configuration runs all of its seeds inside a single array task. References (random,
greedy, edge-marginal-greedy, no-intervention) cost ~6 minutes at d=5 and depend only on
the *environment* levers, so computing them once per configuration rather than once per
seed saves several hours across the matrix.

Usage:
    python -m scripts.sweep_configs            # human-readable listing
    python -m scripts.sweep_configs --json     # the machine-readable matrix
    python -m scripts.sweep_configs --count    # number of array tasks
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

# The reference point every one-factor-at-a-time arm is measured against. d=5 is the
# primary reporting size: its greedy-vs-random gap is 1.61 interventions against 0.78 at
# d=4, so the same amount of noise costs far less resolution here.
BASELINE: Dict = dict(
    d=5,
    observation="edge_marginals",
    train_episodes=6000,
    eval_episodes=300,
    budget=20,
    n_obs=1000,
    n_int=100,
    identify_threshold=0.7,
    prior="erdos_renyi",
    prior_p=0.5,
    intervene_scale=2.0,
    entropy_coef=0.003,
    lr=3e-4,
    step_cost=0.05,
    hidden=128,
    gamma=0.99,
    episodes_per_update=32,
)

CORE_SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [0, 1, 2]

# Every one-factor arm: (lever, [values]), each value producing one configuration.
# The baseline value is deliberately absent from each list -- it is already measured by
# the core arm, and re-running it would waste a task while inviting two slightly different
# numbers for the same setting.
LEVERS: List = [
    # -- agent ---------------------------------------------------------------------
    # Entropy is the lever that reproduced the previous project's collapse signature:
    # at 0.01 the policy stayed near-uniform and argmax was arbitrary. 0.0 tests the
    # opposite failure (premature determinism), 0.03 pushes the known failure further.
    ("entropy_coef", [0.0, 0.001, 0.01, 0.03]),
    ("lr", [1e-4, 1e-3]),
    # step_cost=0.0 removes the incentive to be quick, leaving only "identify eventually".
    # If gap-closed survives that, the metric is not measuring what it claims to.
    ("step_cost", [0.0, 0.02, 0.15]),
    ("train_episodes", [2000, 12000]),
    ("hidden", [64, 256]),
    ("gamma", [0.9, 1.0]),
    ("episodes_per_update", [16, 64]),
    # -- environment ----------------------------------------------------------------
    ("budget", [10, 40]),
    # Observational sample count sets how sharply the equivalence class is pinned down
    # before the agent acts. At 200 the class itself may still be uncertain.
    ("n_obs", [200, 5000]),
    # Predicted to be a dead lever: a 20x change moved the previous measurement by 0.3.
    # Included precisely because that is a falsifiable prediction worth testing.
    ("n_int", [50, 400]),
    # 0.5 is a deliberate NEGATIVE CONTROL, not a candidate setting: a Markov equivalence
    # class of size 2 caps each member at exactly 0.5, so this threshold can declare an
    # unbroken tie "identified". Solve rates are expected to inflate. If they do not, the
    # identification criterion is not doing what it is documented to do.
    ("identify_threshold", [0.5, 0.9]),
    # p=0.5 is exactly the uniform-over-DAGs prior, so it is the baseline rather than a
    # separate arm; these are genuinely sparser.
    ("prior_p", [0.2, 0.35]),
    ("prior", ["scale_free"]),
    # How hard the intervention pushes. Too weak and it carries little information; the
    # question is whether the agent's advantage depends on the intervention being strong.
    ("intervene_scale", [1.0, 5.0]),
]


def build_matrix() -> List[Dict]:
    configs: List[Dict] = []

    # -- core: the headline comparison, 5 seeds -------------------------------------
    # d x observation. The A-vs-B gap (exact posterior vs edge marginals) is the measured
    # cost of compressing belief into something that scales, which is a result in itself.
    for d in (4, 5):
        for observation in ("posterior", "edge_marginals"):
            configs.append({
                **BASELINE, "d": d, "observation": observation,
                "seeds": CORE_SEEDS, "arm": "core",
                "tag": f"core_d{d}_{observation}",
            })

    # -- one-factor-at-a-time arms, 3 seeds -----------------------------------------
    for lever, values in LEVERS:
        for value in values:
            configs.append({
                **BASELINE, lever: value,
                "seeds": SWEEP_SEEDS, "arm": lever,
                "tag": f"{lever}_{value}",
            })

    return configs


def to_cli(config: Dict) -> str:
    """Render a configuration as arguments for scripts.run_experiment."""
    skip = {"seeds", "arm", "tag"}
    parts = [f"--{k} {v}" for k, v in config.items() if k not in skip]
    parts.append("--seeds " + " ".join(str(s) for s in config["seeds"]))
    parts.append(f"--tag {config['tag']}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--cli", type=int, default=None,
                        help="print the CLI arguments for one 1-indexed task")
    args = parser.parse_args()

    matrix = build_matrix()
    if args.count:
        print(len(matrix))
    elif args.cli is not None:
        print(to_cli(matrix[args.cli - 1]))
    elif args.json:
        print(json.dumps(matrix, indent=2))
    else:
        total_runs = sum(len(c["seeds"]) for c in matrix)
        print(f"{len(matrix)} configurations, {total_runs} (config, seed) runs\n")
        arm = None
        for i, c in enumerate(matrix, 1):
            if c["arm"] != arm:
                arm = c["arm"]
                print(f"\n[{arm}]")
            print(f"  {i:>3}  {c['tag']:<28} seeds={len(c['seeds'])}")


if __name__ == "__main__":
    main()
