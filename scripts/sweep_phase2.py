"""Phase 2 lever sweep: E1 (per-node) and E2 (flat), same configurations.

The overnight sweep characterised 13 levers around a baseline that we now know was
**broken**: the flat network cannot express "score every node the same way", so every
lever was measured against a model that could not do the task whatever the lever was set
to. Those results describe the failure mode, not the levers.

So this repeats the sweep around the configuration that actually works -- per-node scorer,
lr=1e-3, hidden=256, episodes_per_update=16, action memory on -- and then repeats it again
with `arch=flat` and everything else identical. E1 versus E2 on the same axes separates
two things that were previously confounded:

  * a lever that matters for the TASK -- moves the number under both architectures;
  * a lever that mattered only because the NETWORK was broken -- moves it under flat and
    goes flat under per-node, or vice versa.

Only `arch` differs between the two arms. Every other setting, including action memory, is
held identical, so any difference is attributable.

n_obs=5000 throughout, because GATE 1 does not pass at d=5 below that -- the overnight
sweep's baseline of 1000 was measured on an environment that did not require intervening.
One arm deliberately keeps n_obs=1000 as a NEGATIVE CONTROL: it should fail G5 loudly and
show what an invalid environment does to the headline number. It is tagged as such so it
can never be read as a normal result.

Usage:
    python -m scripts.sweep_phase2 --count
    python -m scripts.sweep_phase2 --cli 7
    python -m scripts.sweep_phase2            # human-readable listing
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

# The configuration that beat the greedy oracle at d=4 (+1.283), d=5 (+1.233) and d=6
# (+1.098). Everything below is measured as a deviation from this point.
BASELINE: Dict = dict(
    d=5,
    observation="edge_marginals",
    train_episodes=6000,
    eval_episodes=300,
    budget=20,
    n_obs=5000,
    n_int=100,
    identify_threshold=0.7,
    prior="erdos_renyi",
    prior_p=0.5,
    intervene_scale=2.0,
    entropy_coef=0.003,
    lr=1e-3,
    step_cost=0.05,
    hidden=256,
    gamma=0.99,
    episodes_per_update=16,
    layers=1,
    include_counts=True,
)

SEEDS = [0, 1, 2]

# Values differing from BASELINE. The baseline value is deliberately absent from each list:
# it is already measured by the baseline arm, and re-running it would waste a task while
# inviting two slightly different numbers for the same setting.
LEVERS: List = [
    # -- agent -----------------------------------------------------------------------
    ("entropy_coef", [0.0, 0.001, 0.01, 0.03]),
    # 3e-4 was the OLD baseline. Included so the sweep contains the point the overnight
    # conclusions were drawn at, measured now on a working network.
    ("lr", [1e-4, 3e-4]),
    ("step_cost", [0.0, 0.02, 0.15]),
    ("train_episodes", [2000, 12000]),
    ("hidden", [64, 128]),
    ("gamma", [0.9, 1.0]),
    ("episodes_per_update", [32, 64]),
    # -- environment -------------------------------------------------------------------
    ("budget", [10, 40]),
    ("n_int", [50, 400]),
    # 0.5 is a NEGATIVE CONTROL, not a candidate: a class of size 2 caps each member at
    # exactly 0.5, so this threshold can declare an unbroken tie "identified".
    ("identify_threshold", [0.5, 0.9]),
    ("prior_p", [0.2, 0.35]),
    ("prior", ["scale_free"]),
    ("intervene_scale", [1.0, 5.0]),
]

# Levers that are flags rather than values, plus the two arms that need special handling.
BOOLEAN_LEVERS: List = [
    # Action memory. Measured overnight as buying STABILITY rather than capability: without
    # it, seeds ran +1.043 to -1.766. G4 should fire on this arm; if it does not, that
    # earlier reading was wrong.
    ("include_counts", False),
    # Removing the pass action entirely.
    ("no_pass", True),
]

SPECIAL: List = [
    # Potential-based shaping on posterior entropy. Policy-invariant in theory (Ng, Harada
    # & Russell 1999), so a large effect here would mean the implementation is not
    # potential-based.
    {"shaping_coef": 0.1, "tag": "shaping_coef_0.1"},
    # Gate-invalid environment, kept ON PURPOSE as a negative control.
    {"n_obs": 1000, "tag": "NEGCONTROL_n_obs_1000_gate1_fails"},
]

ARCHES = ("pernode", "flat")


def build_matrix() -> List[Dict]:
    configs: List[Dict] = []
    for arch in ARCHES:
        base = {**BASELINE, "arch": arch}
        configs.append({**base, "seeds": SEEDS, "arm": "baseline",
                        "tag": f"{arch}_baseline"})
        for lever, values in LEVERS:
            for value in values:
                configs.append({**base, lever: value, "seeds": SEEDS, "arm": lever,
                                "tag": f"{arch}_{lever}_{value}"})
        for lever, value in BOOLEAN_LEVERS:
            configs.append({**base, lever: value, "seeds": SEEDS, "arm": lever,
                            "tag": f"{arch}_{lever}_{value}"})
        for special in SPECIAL:
            overrides = {k: v for k, v in special.items() if k != "tag"}
            configs.append({**base, **overrides, "seeds": SEEDS,
                            "arm": list(overrides)[0],
                            "tag": f"{arch}_{special['tag']}"})
    return configs


# Flags that `run_experiment` takes as store_true rather than as a value.
FLAGS = {"include_counts", "no_pass"}


def to_cli(config: Dict, out_dir: str = "results/phase2") -> str:
    skip = {"seeds", "arm", "tag"}
    parts = []
    for key, value in config.items():
        if key in skip:
            continue
        if key in FLAGS:
            # A False flag is simply absent; emitting `--include_counts False` would be
            # parsed as the flag being SET, which is the opposite of the intent.
            if value:
                parts.append(f"--{key}")
        else:
            parts.append(f"--{key} {value}")
    parts.append("--seeds " + " ".join(str(s) for s in config["seeds"]))
    parts.append(f"--tag {config['tag']}")
    parts.append(f"--out {out_dir}/{config['tag']}.json")
    # Gate 1 is recorded for every run, but never allowed to abort: the negative-control
    # arm exists precisely to produce a gate-failing result, and the canaries are what
    # make that legible rather than silent.
    parts.append("--gate1_episodes 200")
    parts.append("--wandb_project sa-phase2")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--cli", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="results/phase2")
    args = parser.parse_args()

    matrix = build_matrix()
    if args.count:
        print(len(matrix))
    elif args.cli is not None:
        print(to_cli(matrix[args.cli - 1], args.out_dir))
    elif args.json:
        print(json.dumps(matrix, indent=2))
    else:
        per_arch = len(matrix) // len(ARCHES)
        print(f"{len(matrix)} configurations "
              f"({per_arch} per architecture x {len(ARCHES)}), "
              f"{sum(len(c['seeds']) for c in matrix)} (config, seed) runs\n")
        arm = None
        for i, c in enumerate(matrix, 1):
            if (c["arch"], c["arm"]) != arm:
                arm = (c["arch"], c["arm"])
                print(f"\n[{arm[0]} / {arm[1]}]")
            print(f"  {i:>3}  {c['tag']:<48} seeds={len(c['seeds'])}")


if __name__ == "__main__":
    main()
