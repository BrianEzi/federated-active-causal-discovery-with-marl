"""The epsilon-greedy control: is the learned policy a dithered myopic rule?

WHY (Brian, 5 Sep). The learned policy's argmax derivative is much worse than its sampled
form, so its stochasticity is load-bearing -- which is consistent with the hypothesis that
training merely recovered "myopic plus exploration noise". If an untrained epsilon-greedy
arm matches the learned arm, the contribution reframes; if it does not, the objection dies
with data. Either answer must be known before submission.

DESIGN. The arm takes the myopic uncertainty rule's action with probability 1-eps and a
uniform vary-mode action with probability eps, per agent per round, seeded. It is evaluated
on the SAME episode seeds as every stored measurement (env.reset(seed*100_000 + episode)),
so its per-episode vectors pair against the stored learned/greedy rows. Strongest-opponent
convention: a GRID of eps is measured and the best one is the reported baseline; the
selection favours the baseline and is said aloud wherever quoted.

The agent lives in this script rather than ma/baselines.py: nothing in the frozen core
changes two days before submission.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                      # noqa: E402

from ma.baselines import make_baselines, RandomAgent                    # noqa: E402
from ma.evaluate import evaluate_episode, global_graph_report           # noqa: E402
from ma.policy import IndependentPPO                                    # noqa: E402
from scripts.rescore_from_config import env_from_config                 # noqa: E402


class EpsilonGreedy:
    """Myopic uncertainty rule with probability 1-eps, uniform vary action with eps."""

    def __init__(self, greedy, rand, eps: float, seed: int):
        self.greedy, self.rand, self.eps = greedy, rand, float(eps)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed):
        for p in (self.greedy, self.rand):
            if hasattr(p, "reset"):
                p.reset(seed)
        self._rng = np.random.default_rng(seed * 7919 + int(self.eps * 1000))

    def __call__(self, env, result):
        pick_random = self._rng.random() < self.eps
        return (self.rand if pick_random else self.greedy)(env, result)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="run JSONs supplying config+seed per cell")
    ap.add_argument("--eps", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--base", default="greedy", choices=["greedy", "argmax", "sampled"],
                    help="the arm under the epsilon treatment. 'greedy' is the original "
                         "control; 'argmax'/'sampled' load the run's learned policy at "
                         "--checkpoint and dither IT -- the 5 Sep follow-up: if argmax+eps "
                         "recovers sampled performance, the policy is its own argmax plus "
                         "noise; if not, the sampled distribution's shape is load-bearing.")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    payload = []
    for path in args.results:
        path = pathlib.Path(path)
        report = json.loads(path.read_text())
        config, seed = report["config"], report.get("seed", 0)
        for eps in (float(v) for v in args.eps.split(",")):
            env = env_from_config(config, seed=seed)
            if args.base == "greedy":
                builders = {a: make_baselines(env, a, seed) for a in env.topology.agents}
                base = {a: builders[a]["greedy_uncertainty"] for a in env.topology.agents}
            else:
                import torch
                if args.checkpoint == "best":
                    ck = path.with_name(path.stem + "_best.pt")
                elif args.checkpoint == "final":
                    ck = path.with_suffix(".pt")
                else:
                    ck = path.with_name(f"{path.stem}_{args.checkpoint}.pt")
                ppo = IndependentPPO.load(str(ck), env)
                # Seeded like global_shd_paired.play: the evaluation is a pure function of
                # (checkpoint, seed, episodes, convention).
                torch.manual_seed(seed)
                base = ppo.policies(deterministic=(args.base == "argmax"))
            policies = {a: EpsilonGreedy(base[a],
                                         RandomAgent(a, seed, allow_clamp=False),
                                         eps, seed + a)
                        for a in env.topology.agents}
            for p in policies.values():
                p.reset(seed)
            hard, soft, resolved, succ = [], [], [], []
            for episode in range(args.episodes):
                # The exact expression global_shd_paired.play and run_arm use: same worlds.
                result = env.reset(seed=seed * 100_000 + episode)
                while not result.done:
                    result = env.step({a: policies[a](env, result)
                                       for a in env.topology.agents})
                g = global_graph_report(env)
                hard.append(g["global_hard_shd"])
                soft.append(g["global_soft_shd"])
                resolved.append(g["global_resolved_fraction"])
                succ.append(float(evaluate_episode(env)["success"]))
            payload.append({
                "source": str(path), "seed": seed, "eps": eps, "base": args.base,
                "checkpoint": args.checkpoint if args.base != "greedy" else None,
                "episodes": args.episodes, "eval_evidence": config.get("vs_evidence"),
                "means": {"hard": float(np.mean(hard)), "soft": float(np.mean(soft)),
                          "resolved": float(np.mean(resolved)),
                          "success": float(np.mean(succ))},
                "rows": {"hard": hard, "soft": soft, "resolved": resolved,
                         "success": succ}})
            print(f"{path.stem} base={args.base} eps={eps:.2f}  hard {np.mean(hard):.5f}  "
                  f"success {np.mean(succ):.3f}  resolved {np.mean(resolved):.3f}",
                  flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out} ({len(payload)} entries)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
