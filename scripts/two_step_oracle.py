"""Does looking two experiments ahead beat looking one ahead? Measures the CEILING.

Every single-agent result so far compares a learned agent against a MYOPIC oracle. But
greedy is provably optimal on the final intervention of an episode -- one-step lookahead is
the whole remaining problem there -- and measured episode lengths are:

    d=5:  0 interventions 7%, one 36%, two 39%, three or more 18%
    d=7:  0 interventions 6%, one 29%, two 42%, three or more 23%

So in ~77-82% of episodes greedy is optimal or near-optimal BY CONSTRUCTION, and the only
room for planning is the remaining fifth. That raises the possibility that the experiment
could never have produced its own success criterion.

This measures it directly, without any learned agent in the loop. If a two-step lookahead
oracle cannot beat the one-step oracle, then no policy can, and "the agent matched greedy"
was the only available outcome.

The objective must be EXPECTED NUMBER OF INTERVENTIONS, not total information gathered.
A first version of this script maximised EIG(v) + E[max_w EIG(w)] and came out WORSE than
the myopic oracle (4.53 against 1.60 at d=4), which is impossible for correct lookahead.
The bug was the objective: total information over two steps is bounded by the current
entropy, so an action that identifies immediately and one that defers identification score
almost identically, and the argmax can prefer to defer. Information is not the goal;
finishing is.

So the depth-2 rule minimises expected cost instead:

    cost(belief) = 0                                       if already concentrated
                 = 1 + SUM_g P(g) * cost(belief | g)       otherwise

truncated at depth 2 with a terminal penalty of 1 for "still not finished". "Concentrated"
uses max posterior mass >= threshold, which is what an oracle can actually evaluate -- it
cannot see which graph is true.

The partition model is the same one the myopic oracle uses, so the two policies differ only
in lookahead depth.

PRE-REGISTERED, before the numbers exist:
    If two-step and one-step reach the same mean number of interventions within noise,
    there is no planning value in this environment and the single-agent design cannot pose
    the question it was built to answer. I expect a small gain at most -- perhaps 0.05
    interventions -- concentrated entirely in the >=3-intervention episodes.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from sa.backend import Backend
from sa.env import EnvConfig
from sa.oracle import InterventionOracle, _partition_entropy


class TwoStepOracle:
    """Exhaustive depth-2 lookahead over the same partition model as the myopic oracle."""

    def __init__(self, space, threshold: float = 0.7, seed: int = 0):
        self.base = InterventionOracle(space)
        self.d = space.d
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _one_step_cost(self, posterior: np.ndarray) -> float:
        """Expected cost of the best single further intervention, with a penalty of 1 for
        outcomes that still leave the belief unconcentrated."""
        sig = self.base.signatures
        best = np.inf
        for w in range(self.d):
            groups = sig[:, w]
            cost = 1.0
            for g in range(self.base.n_groups[w]):
                mask = groups == g
                mass = posterior[mask].sum()
                if mass <= 1e-12:
                    continue
                if (posterior[mask].max() / mass) < self.threshold:
                    cost += mass          # still not finished -> one more step at least
            best = min(best, cost)
        return best

    def scores(self, posterior: np.ndarray) -> np.ndarray:
        """Negated expected cost, so that argmax is still the right choice."""
        sig = self.base.signatures
        out = np.zeros(self.d)
        for v in range(self.d):
            groups = sig[:, v]
            cost = 1.0
            for g in range(self.base.n_groups[v]):
                mask = groups == g
                mass = posterior[mask].sum()
                if mass <= 1e-12:
                    continue
                sub = np.zeros_like(posterior)
                sub[mask] = posterior[mask] / mass
                if sub.max() >= self.threshold:
                    continue              # finished after this one intervention
                cost += mass * self._one_step_cost(sub)
            out[v] = -cost
        return out

    def __call__(self, env, result) -> int:
        s = self.scores(result.posterior)
        best = np.flatnonzero(s >= s.max() - 1e-9)
        return int(self.rng.choice(best))


def run(policy, env, cfg, episodes, seed):
    lens, solved = [], 0
    for ep in range(episodes):
        r = env.reset(seed=seed * 10_000 + ep)
        if hasattr(policy, "reset"):
            policy.reset(seed=seed * 977 + ep)
        n = 0
        while not r.done and n < cfg.budget:
            r = env.step(policy(env, r)); n += 1
        lens.append(n); solved += bool(r.identified)
    return np.array(lens), solved / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--out", default="results/planning/two_step.json")
    args = ap.parse_args()

    rows = []
    for d in args.d:
        cfg = EnvConfig(d=d, n_obs=args.n_obs, budget=args.budget)
        backend = Backend(cfg, seed=0)
        env = backend.make_env()
        one = backend.make_baselines(seed=0)["greedy_oracle"]
        two = TwoStepOracle(backend.space, threshold=cfg.identify_threshold, seed=0)

        t0 = time.perf_counter()
        l1, s1 = run(one, env, cfg, args.episodes, seed=1)
        l2, s2 = run(two, env, cfg, args.episodes, seed=1)
        # Paired comparison: identical episodes, so the difference is per-episode.
        diff = l1 - l2
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        deep = l1 >= 3
        row = {
            "d": d, "episodes": args.episodes,
            "one_step_mean": float(l1.mean()), "two_step_mean": float(l2.mean()),
            "one_step_solved": s1, "two_step_solved": s2,
            "mean_saving": float(diff.mean()), "se": float(se),
            "ci": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
            "saving_on_long_episodes": float(diff[deep].mean()) if deep.any() else None,
            "n_long_episodes": int(deep.sum()),
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"d={d}: one-step {l1.mean():.3f}  two-step {l2.mean():.3f}  "
              f"saving {diff.mean():+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]  "
              f"(on >=3-step episodes: {row['saving_on_long_episodes']}, "
              f"n={row['n_long_episodes']})  [{row['seconds']:.0f}s]", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
