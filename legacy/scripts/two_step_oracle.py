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

truncated at depth 2. TERMINATION IS PROBABILISTIC, and getting that wrong was the second
bug in this script. The environment ends an episode when the posterior mass on the TRUE
graph reaches the threshold, and no policy can see which graph is true. What a policy can
compute is that at most one graph can hold mass >= 0.7, so

    P(episode ends | belief b)  =  max(b)   if max(b) >= threshold, else 0

-- NOT the indicator max(b) >= threshold. A belief concentrated at 0.8 on some graph ends
the episode only 80% of the time; the other 20% it concentrated on the wrong one and the
episode continues. Treating that as certain termination made the deeper search prefer
actions that concentrate mass anywhere, including onto a wrong graph, and produced the
impossible result of two-step lookahead scoring WORSE than one-step (-0.277 at d=5).

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

    def __init__(self, space, threshold: float = 0.7, depth: int = 2, seed: int = 0):
        self.base = InterventionOracle(space)
        self.d = space.d
        self.threshold = threshold
        self.depth = depth
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _p_terminate(self, belief: np.ndarray) -> float:
        top = float(belief.max())
        return top if top >= self.threshold else 0.0

    def _value(self, belief: np.ndarray, depth: int) -> float:
        """Expected further interventions from `belief`, looking `depth` moves ahead."""
        alive = 1.0 - self._p_terminate(belief)
        if alive <= 1e-12:
            return 0.0
        if depth <= 0:
            return alive            # crude but consistent terminal estimate
        sig = self.base.signatures
        best = np.inf
        for w in range(self.d):
            groups = sig[:, w]
            total = 0.0
            for g in range(self.base.n_groups[w]):
                mask = groups == g
                mass = belief[mask].sum()
                if mass <= 1e-12:
                    continue
                sub = np.zeros_like(belief)
                sub[mask] = belief[mask] / mass
                total += mass * self._value(sub, depth - 1)
            best = min(best, total)
            if best == 0.0:
                break
        return alive * (1.0 + best)

    def scores(self, posterior: np.ndarray) -> np.ndarray:
        """Negated expected cost, so argmax remains the right choice."""
        sig = self.base.signatures
        out = np.zeros(self.d)
        for v in range(self.d):
            groups = sig[:, v]
            total = 0.0
            for g in range(self.base.n_groups[v]):
                mask = groups == g
                mass = posterior[mask].sum()
                if mass <= 1e-12:
                    continue
                sub = np.zeros_like(posterior)
                sub[mask] = posterior[mask] / mass
                total += mass * self._value(sub, self.depth - 1)
            out[v] = -total
        return out

    def __call__(self, env, result) -> int:
        s = self.scores(result.posterior)
        best = np.flatnonzero(s >= s.max() - 1e-9)
        return int(self.rng.choice(best))


def run(policy, env, cfg, episodes, seed):
    lens, solved, mec = [], 0, []
    for ep in range(episodes):
        r = env.reset(seed=seed * 10_000 + ep)
        if hasattr(policy, "reset"):
            policy.reset(seed=seed * 977 + ep)
        # A property of the TRUE graph, fixed before either policy acts. Used for
        # conditioning instead of episode length -- see the note in main().
        mec.append(int(r.info["mec_size"]))
        n = 0
        while not r.done and n < cfg.budget:
            r = env.step(policy(env, r)); n += 1
        lens.append(n); solved += bool(r.identified)
    return np.array(lens), solved / episodes, np.array(mec)


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
        l1, s1, mec = run(one, env, cfg, args.episodes, seed=1)
        l2, s2, _ = run(two, env, cfg, args.episodes, seed=1)
        # Paired comparison: identical episodes, so the difference is per-episode.
        diff = l1 - l2
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        # CONDITIONING MATTERS. Selecting episodes where the ONE-STEP arm took >=3 moves
        # conditions on that arm doing badly, so regression to the mean inflates the
        # apparent two-step gain. Reported, but flagged as biased. The honest conditioning
        # uses a property of the true graph fixed before either policy moves: the size of
        # its Markov equivalence class, which is what determines how much orientation work
        # is left after observation.
        deep = l1 >= 3
        big_mec = mec >= 4
        row = {
            "d": d, "episodes": args.episodes,
            "one_step_mean": float(l1.mean()), "two_step_mean": float(l2.mean()),
            "one_step_solved": s1, "two_step_solved": s2,
            "mean_saving": float(diff.mean()), "se": float(se),
            "ci": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
            "saving_on_long_episodes_BIASED": float(diff[deep].mean()) if deep.any() else None,
            "n_long_episodes": int(deep.sum()),
            "saving_on_large_mec": float(diff[big_mec].mean()) if big_mec.any() else None,
            "n_large_mec": int(big_mec.sum()),
            "mean_mec_size": float(mec.mean()),
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"d={d}: one-step {l1.mean():.3f}  two-step {l2.mean():.3f}  "
              f"saving {diff.mean():+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]  "
              f"| MEC>=4: {row['saving_on_large_mec']} (n={row['n_large_mec']}) "
              f"| >=3-step [biased]: {row['saving_on_long_episodes_BIASED']} "
              f"(n={row['n_long_episodes']})  [{row['seconds']:.0f}s]", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
