"""Stage 3: attack the measured failure directly.

Stage 1 measured it: the agent learns **not to pass** within ~1500 episodes and then never
learns **where to intervene**, settling at exactly random-policy cost (2.44 against random's
2.44) with entropy stalled at 1.34-1.39 of a 1.609 maximum.

The suspected mechanism is that pass-versus-act is a large, consistent contrast in return
while which-node is a small one, and both share a single batch-wide advantage
normalisation. Stages 1 and 2 tune the reward's *shape*. This stage changes what the agent
is asked to learn at all, in the two ways that follow directly from that mechanism:

- **no_pass** removes the large contrast from the action space entirely. If the mechanism
  is right, the small contrast should then dominate the normalised advantage. This is the
  cleanest falsification test available: if removing `pass` does not help, the explanation
  is wrong and the difficulty lies in the which-node signal itself, not in competition
  between the two.

- **shaping** adds a dense per-step reward for sharpening the posterior, which is credit
  assignment for the which-node decision specifically -- the thing the sparse terminal
  reward cannot deliver. Potential-based, so policy-invariant whatever the coefficient
  (Ng, Harada & Russell 1999): it cannot manufacture a policy the unshaped objective
  would not also prefer. That guarantee is what makes it admissible when the original plan
  ruled out reward shaping generally.

Both, and their combination, are run against the stage-1 baseline so the comparison is
like-for-like.

**Reading caveat, recorded before the numbers exist**: an agent without `pass` cannot
under-act, so its `no_under_acting` check passes by construction and is VACUOUS. It must
not be counted as evidence. That is the same trap that produced a retracted
oracle-agreement figure earlier in this project, and it is why the check is listed here
rather than discovered later.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from scripts.sweep_configs import BASELINE, to_cli

SEEDS = [0, 1, 2]

# (tag suffix, overrides) -- each is a hypothesis, not a hyperparameter guess.
ARMS = [
    # Does removing the competing large contrast let the small one be learned?
    ("nopass", {"no_pass": True}),
    # Does dense which-node credit assignment help, and how much is needed?
    ("shape0.3", {"shaping_coef": 0.3}),
    ("shape1.0", {"shaping_coef": 1.0}),
    ("shape3.0", {"shaping_coef": 3.0}),
    # Both fixes together.
    ("nopass_shape1.0", {"no_pass": True, "shaping_coef": 1.0}),
    # Both, plus letting the policy actually sharpen.
    ("nopass_shape1.0_e0", {"no_pass": True, "shaping_coef": 1.0, "entropy_coef": 0.0}),
    ("nopass_e0", {"no_pass": True, "entropy_coef": 0.0}),
    # Shaping combined with the stronger step cost from stage 2.
    ("shape1.0_cost0.15", {"shaping_coef": 1.0, "step_cost": 0.15}),
]


def build_matrix() -> List[Dict]:
    configs = []
    for suffix, overrides in ARMS:
        configs.append({
            **BASELINE, **overrides,
            "seeds": SEEDS, "arm": "diagnostic",
            "tag": f"s3_{suffix}",
        })
    return configs


def _to_cli(config: Dict) -> str:
    """Like `sweep_configs.to_cli`, but `no_pass` is a store_true flag with no value."""
    no_pass = config.pop("no_pass", False)
    rendered = to_cli(config)
    return rendered + (" --no_pass" if no_pass else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--cli", type=int, default=None)
    args = parser.parse_args()

    matrix = build_matrix()
    if args.count:
        print(len(matrix))
    elif args.cli is not None:
        print(_to_cli(matrix[args.cli - 1]))
    elif args.json:
        print(json.dumps(matrix, indent=2))
    else:
        print(f"{len(matrix)} configurations, {len(matrix) * len(SEEDS)} runs")
        for i, c in enumerate(matrix, 1):
            print(f"  {i:>3}  {c['tag']}")


if __name__ == "__main__":
    main()
