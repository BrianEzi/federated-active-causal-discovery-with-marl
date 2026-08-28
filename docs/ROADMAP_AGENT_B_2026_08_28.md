# Roadmap — second agent — 28 to 31 August

High level by design. Brian will take the detail with you directly. Freeze Monday 31 Aug,
dissertation 8 Sep.

**Read [`docs/OBJECTIVE.md`](OBJECTIVE.md) first.** It is new, and it is now the document
every other one answers to. It states the top-line goal, what an exceptional version of this
project demonstrates, and the verified boundary of the novelty claim.

---

## 1. The split

**You own the results we already have.** Take them from "measured" to "thesis-ready":
seeds, protocol, figures, and the honest bounding of every claim. This is the track that
guarantees a dissertation.

**I move to the attribution avenue**, which is closer to the true objective and higher
variance. If it lands before the freeze, it becomes the headline; if not, your track is what
ships. Treat my work as strictly optional upside and do not wait on it.

**File boundaries.** I will work in `cb/attribution*`, a new experiment module, and
profiling. **I will not touch `ma/env.py`, `ma/policy.py` or `ma/baselines.py` without saying
so first** — your results depend on them.

Things I already landed in those shared files, so they do not surprise you:
- `ma/env.py` — `difference_reward`, `difference_reward_mode`, `reward_scale` (all default to
  the previous behaviour), plus `difference_credit()` and a per-episode `_touched_by` tracker.
- `ma/baselines.py` — `PartitionedGreedyAgent`, offered as `greedy_partitioned`.
- `scripts/ma_train.py` — matching flags.
- Full `ma` suite green at 92.

## 2. What is settled — do not spend compute re-deciding

| question | answer |
|---|---|
| Agent-count collapse — more training? | **No.** 4x training still leaves I(S;A)/H at 0.109. |
| Entropy bonus the cause? | **No.** 10x cut changed nothing and scored worse. |
| Turn-aware credit the cause? | **No** — your own retraction, confirmed independently here (`deltaonly` 0.077 against plain 0.110). |
| Credit assignment the cause? | **No** — withdrawn on a pre-registered criterion. See below. |
| **Reward SCALE the cause?** | **Yes, apparently.** Plain reward x 0.214 takes eight agents from 0.110 to **0.653** across two seeds, beating greedy for the first time. Per-agent return grows 1.66 to 11.86 from 2 to 8 agents; value loss is MSE so it grows as the square and swamps the policy gradient through the shared trunk. |
| Greedy `bar` handicap? | Not on the factored ladder — 0.7 and 1.0 identical, verified on the agent config directly. |
| Clamp on hub-heavy graphs? | Refuted. Clamp-only 0.233 against vary-only 0.589. |
| GRPO? | Refuted for this environment. A group mean is a state baseline and cannot reorder advantages within a group. |

## 3. Live warnings — these change claims you may be about to write

**The window ladder is CLEAN on the scale confound.** Return magnitude sits flat at 1.6–3.5
across k while the agent axis spans 7x. The main scaling figure is not contaminated. Checked.

**"Argmax as primary" is RETRACTED.** It held on the window ladder and reverses on the agent
ladder — up to −0.153 +/- 0.040. Probable cause: this is a covering task and a deterministic
policy can re-pick a node it already used, which does nothing, while sampling diversifies
coverage. Do not put the general recommendation in the thesis.

**A two-line convention beats the learner past three agents.** `PartitionedGreedyAgent`
scores 0.880 at eight agents against the learner's 0.627, duplicating 23x less shared work.
Learned wins at 2 and 3 agents (0.487 vs 0.213, 0.800 vs 0.613) and loses at 6 and 8. **The
six-agent row is not yet fair** — it uses `a06_s0`, which failed the MI gate. A scaled `a06`
retrain is needed before that comparison is quotable, and it may move.

**Transfer to sampled evidence fails.** Learned beats greedy under oracle (0.610 vs 0.470) and
ties under sampled (0.874 vs 0.868). Bound this honestly; do not claim transfer.

**The MI gate is a FLOOR, not a quality measure.** Near-zero voids a number outright. Above
the floor it measures commitment to the *training* objective, so it must never be compared
across arms with different objectives — a mistake I made and had to withdraw.

## 4. Your track, in priority order

1. **Three training seeds on every headline window rung.** Currently one. This is the single
   weakest thing about the main result and the cheapest to fix.
2. **The budget-collapse plot.** Plot joint success against `budget / required cover` rather
   than against k. The forced cover is computable in closed form (tails of directed edges,
   both endpoints of confounded pairs) and falls from 0.757k at k=4 to 0.542k at k=30 — so
   the current iso-budget normalisation quietly favours large windows. Eval-only on existing
   checkpoints, and it closes the last open objection to the main figure.
3. **Scaled retrains on the agent ladder** at 2/3/6/8 with `--reward_scale`, so the agent axis
   is reported as a tuning artifact with a mechanism rather than as a mystery. Also makes the
   coordination crossover a fair comparison.
4. **Return normalisation instead of a magic constant.** 0.214 is not a contribution; the
   principled version is normalising returns before the value loss. If it reproduces the gain
   with no hand-picked constant, the mechanism is confirmed. If it does not, the scale story
   is wrong too and we need to know.
5. **One sampled-evidence replication** of a window rung, reported as a limitation.
6. Figures and write-up.

**Cut:** Erdos-Renyi, GRPO, the alpha-blend, architecture work.

## 5. What I am doing, so you can ignore it safely

Attribution — determining **which participant owns** the latent behind a bidirected edge, which
the 2026 federated causal discovery survey confirms no existing method does.

Scoped around a metric that makes it cheap: **regret against a computable optimum.** The
attribution-required experiment set is forced, exactly as the structural one is — to attribute
a latent, its owner must intervene on the private variable causing both endpoints — so the
minimum is closed-form and regret is exact rather than bounded.

Four phases, each able to kill the effort cheaply, criteria fixed in advance:
- **Phase 0**, hours, no training: is there headroom over the best heuristic at all? If mean
  regret is already under ~1 experiment, I stop and say so.
- **Phase 1**: profile and make iteration 5–10x cheaper. Sampled evidence rebuilds the belief
  from scratch every round and is the likely bottleneck.
- **Phase 2**: minimal decisive experiment — 2–3 agents, small windows, sampled evidence.
  Deliberately NOT building factored attribution: "it scales" is a claim your ladder already
  carries, and the enumerated backend is exact at small k.
- **Phase 3**: train, only if Phase 0 says there is room. MI gate first. If it does not beat
  the strongest heuristic by 2 SE on 3 seeds by **Sunday midday**, your track ships.
