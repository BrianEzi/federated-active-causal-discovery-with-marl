"""Train a NEW SEED of an existing run, with the flags read back from that run's own config.

WHY THIS EXISTS. Adding seeds to a ladder rung means reproducing a training command that was
typed days ago and never written down: `cluster/` has no script for the window ladder, and
the flags are recoverable only from the `config` block inside each result file. Retyping them
is the failure mode this project has already paid for twice in one night -- `--n_obs 60`
where the dial used 1000, and `gnn` where the ladder used `gnn_portable`. Both produced
numbers that looked entirely plausible and were comparing different experiments.
`scripts/rescore_from_config.py` removed that class for EVALUATION; this removes it for
TRAINING.

HOW IT IS VERIFIED, which is the point of the file. Reconstructing flags is only useful if
the reconstruction is checked, so `--verify` (on by default) runs `scripts/ma_train.py`'s own
`main` with training and evaluation stubbed out, and compares the `config` block that comes
back against the source file's, key by key. A mismatch aborts before any compute is spent.
It is the same argument as re-scoring from the config rather than from memory: the check runs
through the real code path, not through a second copy of the mapping.

WHAT IT CANNOT CHECK, and this is a genuine hole rather than a caveat.
`scripts/ma_train.py::_config_record` does not record the PPO hyperparameters --
`entropy_coef`, `orthogonal_init`, `gnn_layers`, `turn_aware_credit`, `mask_pass_updates` --
nor `--max_edges`. They are therefore NOT recoverable from an old result file, and a new seed
matches the old ones only if those were left at their defaults. This script assumes the
defaults, prints that assumption, and accepts explicit overrides. For runs written after
2026-08-28 the fields ARE recorded and are checked like everything else, because
`_config_record` was extended in the same commit that added this file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

# The PPO settings that no pre-2026-08-28 result file records, with the defaults
# `scripts/ma_train.py` would have applied. A run that used anything else cannot be matched
# from its result file alone -- say so rather than silently assuming.
UNRECORDED_DEFAULTS = {
    "entropy_coef": 0.01,
    "orthogonal_init": False,
    "gnn_layers": 2,
    "turn_aware_credit": False,
    "mask_pass_updates": 0,
    "max_edges": None,
}

# Fields that `_config_record` gained AFTER some result files were written, with the value
# that means "the behaviour this run actually had". A key absent from the source is accepted
# only when the reconstruction lands on the inert value here -- absent plus a non-default is
# a real mismatch, because it would mean training a new seed with a knob the old ones never
# had. Anything not in this table cannot be silently forgiven.
POST_HOC_INERT = dict(UNRECORDED_DEFAULTS,
                      difference_reward=False,
                      difference_reward_mode="both",
                      reward_scale=1.0)


def topology_flags(topology: dict) -> List[str]:
    """Invert `federated_topology(n_agents, private_size, n_shared)` from the recorded sets.

    Refuses uneven private sets rather than guessing: `federated_topology` cannot express
    them, so a topology that does not round-trip is one this script must not pretend to
    reproduce. The `--verify` pass would catch it anyway; failing here says why.
    """
    private = topology["private"]
    sizes = {len(p) for p in private}
    if len(sizes) != 1:
        raise SystemExit(f"private sets of differing sizes {sorted(sizes)} -- "
                         "federated_topology cannot build this, reconstruct by hand")
    return ["--n_agents", str(len(private)),
            "--private_size", str(sizes.pop()),
            "--n_shared", str(len(topology["exposed"]))]


def flags_from_config(config: dict, seed: int, out: str,
                      train_episodes: Optional[int] = None,
                      eval_episodes: Optional[int] = None) -> List[str]:
    """The argv that reproduces `config`, at a new seed.

    Every value is read from the config block. Nothing defaults to what a neighbouring
    experiment used, which is the whole reason this function exists rather than a shell
    variable.
    """
    argv = ["--seed", str(seed), "--out", out]
    argv += topology_flags(config["topology"])
    argv += ["--n_obs", str(config["n_obs"]),
             "--n_int", str(config["n_int"]),
             "--budget", str(config["budget"]),
             "--rule", config["rule"],
             "--turn_order", config["turn_order"],
             "--graph_model", config.get("graph_model", "er"),
             "--sf_m", str(config.get("sf_m", 2)),
             "--backend", config["belief_backend"],
             "--policy_arch", config["policy_arch"],
             "--cb_n_boot", str(config.get("cb_n_boot", 12)),
             "--episode_mix", config.get("episode_mix", "any"),
             "--train_episodes", str(train_episodes
                                     if train_episodes is not None
                                     else config.get("train_episodes", 4000))]
    if eval_episodes is not None:
        argv += ["--eval_episodes", str(eval_episodes)]

    # `prior_p` is DERIVED from d when the flag is unset, and it was changed on 2026-08-22.
    # Passing the recorded value back is what keeps a new seed on the same prior as the old
    # ones instead of on today's derivation.
    if config.get("prior_p") is not None:
        argv += ["--prior_p", repr(float(config["prior_p"]))]
    if config.get("claim_bar") is not None:
        argv += ["--claim_bar", repr(float(config["claim_bar"]))]
    if config.get("reward_criterion"):
        argv += ["--reward_criterion", config["reward_criterion"]]
    if config.get("step_cost"):
        argv += ["--step_cost", repr(float(config["step_cost"]))]
    if config.get("potential_shaping"):
        argv += ["--potential_shaping", repr(float(config["potential_shaping"]))]
    for key in ("vs_evidence", "vs_evidence_alpha"):
        if key in config:
            argv += [f"--{key}", str(config[key])]

    modes = list(config["action_modes"])
    if modes == ["vary"]:
        argv += ["--vary_only"]
    elif modes == ["clamp"]:
        argv += ["--clamp_only"]

    # Store-true flags. `claims_require_all_types` is recorded POSITIVELY and its flag is the
    # negative one, so it inverts.
    for key, flag in (("disclose_regime", "--disclose_regime"),
                      ("per_agent_reward", "--per_agent_reward"),
                      ("observe_belief_channels", "--observe_belief_channels"),
                      ("observe_partner_counts", "--observe_partner_counts"),
                      ("mode_by_role", "--mode_by_role"),
                      ("oracle_obs_structure", "--oracle_obs"),
                      ("difference_reward", "--difference_reward")):
        if config.get(key):
            argv += [flag]
    if not config.get("claims_require_all_types", True):
        argv += ["--legacy_claim_exemption"]
    if config.get("difference_reward_mode", "both") != "both":
        argv += ["--difference_reward_mode", config["difference_reward_mode"]]
    if config.get("reward_scale", 1.0) != 1.0:
        argv += ["--reward_scale", repr(float(config["reward_scale"]))]
    return argv


# Keys that are ALLOWED to differ between the source run and the reconstruction, with why.
IGNORED = {
    "train_episodes",   # deliberately overridable
    "identify_threshold", "intervene_scale", "cb_alpha", "claim_penalty",
    # ^ no CLI flag exists for any of these; they come from MAConfig's defaults on both
    #   sides, so a difference would mean the DEFAULT moved -- which `verify` reports below
    #   rather than ignores. They are listed so the reason is written down, not to skip them.
}


def verify(source: dict, argv: List[str]) -> Dict[str, tuple]:
    """Run `ma_train.main` with training and evaluation stubbed, and diff the config blocks.

    Goes through ma_train's own construction rather than rebuilding MAConfig here. A second
    copy of the mapping would agree with this file's bugs and disagree with the experiment.

    REDIRECTS `--out` to a throwaway directory. `ma_train.main` writes its report and its
    checkpoint wherever `--out` points, so verifying against the real target path left a
    stub result file and an untrained `.pt` sitting exactly where a reader would take them
    for the finished run. Caught on this file's first invocation.
    """
    import tempfile

    import ma.policy
    import scripts.ma_train as ma_train

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="verify_config_"))
    argv = list(argv)
    argv[argv.index("--out") + 1] = str(scratch / "verify.json")

    class _StubPPO:
        first_success_episode = None

        def __init__(self, env, cfg):
            self.env = env

        def train(self, **kwargs):
            return []

        def policies(self, deterministic=False):
            return {a: (lambda env, result: 0) for a in self.env.topology.agents}

        def save(self, path):
            pass

    stub_arm = {"success": 0.0, "success_ci": (0.0, 0.0), "mean_steps": 0.0,
                "clamp_fraction": 0.0}
    real = (ma_train.IndependentPPO, ma_train.run_arm, ma_train._wandb_run)
    ma_train.IndependentPPO = _StubPPO
    ma_train.run_arm = lambda *a, **k: dict(stub_arm)
    ma_train._wandb_run = lambda *a, **k: None
    try:
        report = ma_train.main(argv + ["--no_wandb"])
    finally:
        (ma_train.IndependentPPO, ma_train.run_arm, ma_train._wandb_run) = real

    rebuilt = report["config"]
    differences = {}
    for key in sorted(set(source) | set(rebuilt)):
        if key in IGNORED:
            continue
        missing = key not in source
        a, b = source.get(key, "<absent>"), rebuilt.get(key, "<absent>")
        # A field `_config_record` gained after the source run was written is absent on one
        # side only. Forgive it only where the reconstruction lands on the inert value --
        # see POST_HOC_INERT. Absent plus a live setting is a genuine mismatch.
        if missing and key in POST_HOC_INERT and b == POST_HOC_INERT[key]:
            continue
        if json.dumps(a, sort_keys=True, default=str) != json.dumps(b, sort_keys=True,
                                                                    default=str):
            differences[key] = (a, b)
    return differences


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="an existing result .json to copy the settings from")
    ap.add_argument("--seed", type=int, required=True, help="the NEW seed to train")
    ap.add_argument("--out", required=True, help="where to write the new result .json")
    ap.add_argument("--train_episodes", type=int, default=None,
                    help="override the source's episode count (default: match it)")
    ap.add_argument("--eval_episodes", type=int, default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="flags appended verbatim, for settings no result file records "
                         "(--entropy_coef, --gnn_layers, --max_edges, ...)")
    ap.add_argument("--no_verify", action="store_true",
                    help="skip the config round-trip check. Do not.")
    ap.add_argument("--run", action="store_true",
                    help="actually train. Without it this prints the command and verifies.")
    args = ap.parse_args(argv)

    source = json.loads(pathlib.Path(args.source).read_text())
    config = source["config"]
    flags = flags_from_config(config, args.seed, args.out,
                              args.train_episodes, args.eval_episodes)

    print("reconstructed from", args.source)
    print("  python -m scripts.ma_train " + " ".join(flags + list(args.extra)))
    missing = [k for k in UNRECORDED_DEFAULTS if k not in config]
    if missing:
        print("\n  NOT RECORDED by the source run, assuming ma_train's defaults:")
        for key in missing:
            print(f"    {key} = {UNRECORDED_DEFAULTS[key]!r}")
        print("  A new seed matches the old ones only if the original run used these.")

    if not args.no_verify:
        differences = verify(config, flags)
        if differences:
            print("\nCONFIG MISMATCH -- refusing to train:")
            for key, (was, now) in differences.items():
                print(f"  {key}: source={was!r} rebuilt={now!r}")
            return 1
        print("\n  verified: the rebuilt config matches the source on every recorded key")

    if not args.run:
        print("\n  dry run -- pass --run to train")
        return 0

    import scripts.ma_train as ma_train
    ma_train.main(flags + list(args.extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
