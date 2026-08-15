# Single-Agent Active Causal Discovery — Plan and Success Criteria

Supersedes the initial plan (`~/.claude/plans/cozy-floating-scott.md`), which predates the
gate results and the reframed goal. Lives in the repo so it is versioned alongside
`SA_EXPERIMENT_LOG.md` (measurements) and `THEORY_NOTES.md` (citations).

---

## Why this exists

The two-agent codebase accumulated ~15 interacting levers and a chain of measurement bugs
that invalidated most of its conclusions. Three findings forced the rebuild: the task was
solvable without acting (~50% of episodes, against a theoretical 0%); most comparisons had
error bars as large as their effects (±30pp at 8 episodes per condition); and the
four-estimator comparison turned out to have evaluated one estimator four times.

This rebuild is single-agent, minimal, and gated: nothing is trained until the environment
is shown to pose a real problem.

## The goal

**Validate that the training technique works — not beat greedy EIG.** Greedy is expected
to win at these sizes. What is being established is that the RL setup reliably learns to
approach a known-good policy on a well-posed task.

**Matching greedy reliably IS the result.** This is stated up front so it cannot be
reframed as a consolation prize after the numbers arrive.

Scaling ladder, each rung attempted only once the one below is trusted:

1. one agent, few nodes — does the method work at all?  ← **we are here**
2. one agent, more nodes — does it survive scale?
3. two agents, few nodes — add decentralisation, scale held fixed
4. two agents, more nodes

---

## Status

**Phases 0–2 complete. Both gates pass.** 98 tests, `sa/` imports nothing from `src/`.

| | result |
|---|---|
| GATE 1 (task requires intervening) | d=3 **14.67%** vs 16.00% predicted; d=4 **10.0%** vs 10.87% |
| GATE 2 (choices matter) | d=4 oracle **1.38** vs random **2.53**; d=5 **1.80** vs **3.54** |
| Edge marginals viable | costs +3% (d=3), +5% (d=4), +11% (d=5) |

Built: `graphs` (enumeration, Markov equivalence classes), `score` (BGe + BIC),
`posterior` (exact, local-score cached), `scm`, `env`, `priors`, `oracle` (Shannon EIG),
`baselines`, `gates`.

---

## Success criteria — fixed before any agent exists

### The metric

**Gap closed** = `(random − agent) / (random − greedy)`, in interventions to identify.
1.0 = matches greedy, 0.0 = no better than random. Normalised so it stays comparable as
`d` changes.

Calibrated against an epsilon-greedy oracle so the thresholds mean something behavioural:
**0.80 ≈ choosing correctly ~70% of the time**, **0.90 ≈ ~80–90%**.

### PRIMARY — must pass

> **Gap closed ≥ 0.80**, under deterministic (temperature-0) evaluation, on **≥ 4 of 5**
> training seeds, at **d = 5**, over **≥ 300** evaluation episodes with bootstrap intervals
> — and not materially worse at d = 3 and d = 4.

Why 0.80 and not 0.95: at these sample sizes a higher bar measures seed luck, and risks
the criterion driving the work rather than describing it. Why report the **minimum** across
seeds rather than the mean: a mean hides a lucky run, which is the failure mode the
previous project never caught.

**d = 5 is the primary reporting size, deliberately.** The greedy–random gap is 0.78
interventions at d=4 but 1.61 at d=5, so identical noise is twice as damaging at d=4 —
measured, a quantity that should have read exactly 0.00 came out at −0.29 at d=4 versus
−0.02 at d=5. Treat d=4 as supporting evidence with a wider band.

### HARD FAILS — regardless of gap closed

- **Under-acting > 10%** — episodes where the agent passed while still unidentified.
  This is the NOOP collapse that killed the previous round. It needs its own criterion
  because it is *not* captured by the primary metric: an early pass leaves the episode
  unsolved, excluding it from the mean over solved episodes, so giving up on hard
  instances can silently **improve** the headline number.
- **Deterministic collapse** — gap closed under temperature-0 evaluation more than **0.10**
  below the sampled-evaluation value. The previous project trained fine and collapsed when
  evaluated greedily; this is the specific check for that.

Over-acting is deliberately *not* a separate criterion. It is already fully penalised —
gap closed counts interventions, so every wasted one costs directly. And over-acting on an
already-identified graph is impossible by construction, since `reset` ends the episode when
the posterior already clears the threshold.

### DIAGNOSTICS — always reported, never pass/fail

- **Mean regret on informative steps** (nats). Judges each action against how much
  uncertainty was actually present, so near-misses are not binary failures. **Aggregate
  only over steps where the oracle had a preference** — averaging in steps where every
  option ties is what produced the retracted 99.4% agreement figure.
- **Gap closed stratified by Markov equivalence class size.** Class size drives difficulty
  (correlation 0.56 at d=4 vs 0.29 for edge count). A single average mixes episodes needing
  zero interventions with episodes needing three, so it cannot distinguish a good agent
  from an easy draw. Also separates two skills: knowing when not to act (singleton classes)
  and choosing well when you do (large classes).
- **Exact-posterior vs edge-marginal difference** — the cost of a scalable belief.
- **Interventions used**, tracked from the start. Budget is currently loose (10, against an
  oracle needing ~1.4), so waste is invisible in pass/fail until it is tightened.

### Measurement protocol

Two frozen-evaluation passes per checkpoint, both from the same recorded traces:

| metric | source |
|---|---|
| gap closed | per-episode interventions, solved episodes only |
| under-acting | per-episode: passed while `identified=False` |
| mean regret | per-step `oracle.score_choice`, `informative` only |
| stratified gap | per-episode `info["mec_size"]` |
| collapse check | second pass, sampled rather than temperature-0 |

The **deterministic** pass produces every pass/fail number — it is the deployment condition
and the one that collapsed before. The **sampled** pass exists only to detect the gap.
≥300 episodes, bootstrap intervals on everything.

---

## Phase 3 — the agent

`sa/policy.py`: PPO. Reward **+1 on identification, small cost per step** — a shortest-path
objective needing no shaping analysis.

Two observation conditions, run as a deliberate comparison:

- **A — exact posterior.** A sufficient statistic, so this is a proper MDP and a
  feedforward network suffices. No recurrence. This structurally removes the saturating
  running-average state previously diagnosed as the cause of the greedy collapse.
- **B — edge marginals** (`d(d−1)` values). Scales to any `d`, but lossy. Compared against
  `EdgeMarginalGreedy`, **not** the full-posterior oracle — otherwise the comparison
  conflates a worse policy with a lossier belief.

Order: d=3 for debugging, then d=5 as the reporting size, then d=4 to fill in.

## Phase 4 — results

Agent vs greedy vs random, both conditions, all diagnostics, intervals throughout. The
A−B difference is what licenses trusting edge-marginal results at d ≥ 6, where the exact
posterior cannot be computed (d=6 is 3.7M DAGs).

---

## Deferred, with reasons

- **Budget tightening.** Stays loose until the agent demonstrably learns greedy; only then
  tighten to create genuine planning pressure, where greedy is provably suboptimal.
- **ER vs scale-free comparison.** Sparsity is near-vacuous below d≈8 — ER-1 is *denser*
  than uniform at d≤5, and five nodes cannot host a hub. Belongs to the large-d phase.
- **Anything beyond d=5 with an exact posterior.** Not tractable; needs condition B plus
  approximate inference.
- **Second agent.** Rung 3, not before.

## Known limitation, stated not fixed

The oracle is **myopic** — the best single next experiment, not the best sequence. Greedy
is the standard tractable choice, and the `(1−1/e)` guarantee of Golovin & Krause (2011)
requires adaptive submodularity, which expected information gain does not satisfy in
general. That gap is the headroom a planning agent could eventually exploit; it is not
what this phase is testing.
