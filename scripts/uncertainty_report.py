"""Is the H(E) / H(G|E) split actually useful, or merely well defined?

A metric can satisfy every identity and still tell you nothing. Four things it has to do to
earn its place, each a separate check below:

  U1  SEPARATE THE TWO KINDS OF DATA. More observational data should shrink H(E) and leave
      H(G|E) alone. If both move together the split is not carving anything.

  U2  RANK POLICIES. A greedy agent should remove more addressable bits per intervention
      than a random one. If it cannot tell them apart it is useless for evaluation.

  U3  PREDICT DIFFICULTY. Episodes that start with more addressable uncertainty should need
      more interventions. This is what makes it a GRADED version of GATE 1 rather than a
      pass/fail, and it is the property the old skeleton/orientation measures never had.

  U4  BE BASELINE-FREE. It must be computable without reference to an oracle, since the
      d=7 oracle turned out to be degraded and every score defined against it inherited
      that. This one is true by construction and is noted rather than measured.

PRE-REGISTERED, before the numbers exist:
    U1 and U4 I expect to hold outright. U2 I expect to hold but with a modest margin,
    since a random policy on a 4-node graph still hits useful targets often. U3 is the one
    I am least sure of -- if addressable bits at start barely vary across episodes, it
    cannot predict anything, and that would ITSELF be the finding: it would say the task
    has almost no variation in difficulty, which is consistent with everything else found
    about the planning horizon.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from sa.backend import Backend
from sa.baselines import make_baselines, no_intervention_policy
from sa.env import EnvConfig
from sa.uncertainty import decompose, episode_trace, summarise_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--n_obs", type=int, nargs="+", default=[200, 1000, 20000])
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--out", default="results/uncertainty/report.json")
    args = ap.parse_args()

    report = {"u1": [], "u2": [], "u3": []}

    # ---- U1: does observational data move H(E) but not H(G|E)? --------------------
    print("U1  effect of observational data on each component (step 0)")
    print(f"{'d':>3} {'n_obs':>7} {'H(G)':>8} {'H(E)':>8} {'H(G|E)':>8} {'classes':>8}")
    for d in args.d:
        for n_obs in args.n_obs:
            env = Backend(EnvConfig(d=d, n_obs=n_obs, budget=args.budget), seed=0).make_env()
            vals = [decompose(env.reset(seed=ep).posterior, env.space)
                    for ep in range(args.episodes)]
            row = {k: float(np.mean([v[k] for v in vals]))
                   for k in ("h_total", "h_class", "h_within", "n_classes_with_mass")}
            row.update({"d": d, "n_obs": n_obs})
            report["u1"].append(row)
            print(f"{d:>3} {n_obs:>7} {row['h_total']:>8.3f} {row['h_class']:>8.3f} "
                  f"{row['h_within']:>8.3f} {row['n_classes_with_mass']:>8.1f}", flush=True)

    # ---- U2: can it rank policies? ------------------------------------------------
    print("\nU2  bits of addressable uncertainty removed, by policy")
    print(f"{'d':>3} {'policy':>16} {'start':>7} {'removed':>8} {'per-int':>8} "
          f"{'ints':>6} {'solved':>7}")
    for d in args.d:
        cfg = EnvConfig(d=d, n_obs=1000, budget=args.budget)
        backend = Backend(cfg, seed=0)
        env = backend.make_env()
        base = backend.make_baselines(seed=0)
        policies = {"greedy_oracle": base["greedy_oracle"], "random": base["random"],
                    "no_intervention": no_intervention_policy}
        for name, policy in policies.items():
            s = [summarise_trace(episode_trace(env, policy, seed=1000 + ep))
                 for ep in range(args.episodes)]
            row = {"d": d, "policy": name,
                   "start": float(np.mean([x["addressable_bits_at_start"] for x in s])),
                   "removed": float(np.mean([x["addressable_bits_removed"] for x in s])),
                   "per_intervention": float(np.mean(
                       [x["addressable_bits_per_intervention"] for x in s])),
                   "interventions": float(np.mean([x["interventions"] for x in s])),
                   "solved": float(np.mean([x["identified"] for x in s])),
                   "class_bits_removed": float(np.mean(
                       [x["class_bits_removed"] for x in s]))}
            report["u2"].append(row)
            print(f"{d:>3} {name:>16} {row['start']:>7.3f} {row['removed']:>8.3f} "
                  f"{row['per_intervention']:>8.3f} {row['interventions']:>6.2f} "
                  f"{row['solved']:>7.3f}", flush=True)

    # ---- U3: does starting uncertainty predict how much work is needed? -----------
    print("\nU3  does addressable uncertainty at step 0 predict interventions needed?")
    for d in args.d:
        cfg = EnvConfig(d=d, n_obs=1000, budget=args.budget)
        backend = Backend(cfg, seed=0)
        env = backend.make_env()
        policy = backend.make_baselines(seed=0)["greedy_oracle"]
        starts, needed = [], []
        for ep in range(args.episodes):
            t = episode_trace(env, policy, seed=2000 + ep)
            starts.append(t["rows"][0]["h_within"])
            needed.append(t["interventions"])
        starts, needed = np.array(starts), np.array(needed)
        r = float(np.corrcoef(starts, needed)[0, 1]) if starts.std() > 1e-9 else float("nan")
        row = {"d": d, "correlation": r,
               "start_mean": float(starts.mean()), "start_sd": float(starts.std(ddof=1)),
               "start_min": float(starts.min()), "start_max": float(starts.max()),
               "needed_mean": float(needed.mean())}
        report["u3"].append(row)
        print(f"  d={d}: corr(addressable bits at start, interventions used) = {r:+.3f}"
              f"   start bits mean {starts.mean():.3f} sd {starts.std(ddof=1):.3f} "
              f"range [{starts.min():.2f}, {starts.max():.2f}]", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "report": report}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
