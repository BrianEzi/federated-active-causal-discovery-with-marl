# Single-Agent Rebuild — Experiment Log

Running record of every measurement, decision, and correction, written **as it happens**.
Numbers here are what was actually observed, not what was expected. Mistakes are kept in
with their corrections rather than edited away — the previous investigation lost weeks to
a defect that was visible in a metadata field nobody re-read, and the nuance of *why* a
number moved is the part that gets lost first.

Convention: **[MEASURED]** = ran it and these are the numbers. **[DECIDED]** = a choice
made, with its reason. **[CORRECTED]** = something earlier here was wrong.

---

## 2026-08-14 — Framing

**[DECIDED] The research goal is to validate the training technique, not to beat greedy EIG.**
Stated by the user, and it changes the success criteria materially. Greedy EIG is expected
to win at these sizes. What matters is establishing that the RL setup can *reliably learn
to approach* it on a well-posed task. The scaling ladder is deliberate:

1. one agent, few nodes → does the method work at all?
2. one agent, many nodes → does it survive scale?
3. two agents, few nodes → add decentralisation, holding scale fixed
4. two agents, many nodes

Each rung is only worth climbing once the one below is trusted. This supersedes the
earlier plan's success criteria, which treated "matches greedy" as a consolation prize and
"beats greedy" as the result. **Matching greedy reliably IS the result** at this stage.

---

## 2026-08-14 — Phase 0: graph foundations

**[MEASURED] Enumeration matches known counts.** DAGs / Markov equivalence classes:
25 / 11 at d=3, 543 / 185 at d=4, 29,281 / 8,782 at d=5. (OEIS A003024, A007984.)

**[MEASURED] Singleton fractions — the GATE 1 target.** The share of DAGs alone in their
equivalence class, i.e. identifiable without intervening: **16.00%** (d=3), **10.87%**
(d=4), **8.93%** (d=5). Computing this number is the whole point: the previous codebase
ran for weeks with a 50% observational solve rate against a theoretical 0%, undetected.

**[MEASURED] The old scorer inverts the answer once graphs differ in size.** The two-agent
estimator used a profile likelihood, valid there only because all 8 candidates had exactly
3 edges. Over all 25 DAGs at d=3, the six densest tie at the top holding **67% of
posterior mass** while the true 2-edge graph ranks **9th of 25**. Not blurring — inversion.

**[DECIDED] BGe as the default score**, with BIC retained as an independently-verifiable
cross-check. Both are score-equivalent, which is the formal statement that observational
data cannot separate DAGs within a class.

**[CORRECTED] A real bug in BICScore, caught by the score-equivalence test on first run.**
It centred `y` in the no-parents branch but fitted without an intercept in the parents
branch. Within-class spread 7e-2; after centring both sides, ~1e-13. BGe was correct as
written (5.7e-14). The test earned its place immediately.

**[MEASURED] BGe vs BIC characterised rather than assumed.** BGe concentrates more slowly
(67% on the true class at n=1000 vs BIC's 92%) but is *more* conservative on genuinely
independent data (85–98% on the empty graph vs 68–89%). Residual mass on dense graphs in a
chain is not a defect: nodes 0 and 2 are genuinely marginally dependent, so excluding that
edge needs the subtler conditional-independence signal. Standardising changes nothing;
raising `alpha_w` makes it worse (d+6 puts 88–94% on the densest). BGe stays default.

**[MEASURED] Cost.** Local-score caching means `d·2^(d-1)` terms, not one per DAG: 12 at
d=3, 32 at d=4, 80 at d=5. Posterior recomputation is 4ms (d=3), 12ms (d=4), 39ms (d=5)
per step. Space construction is one-off: 0.1s at d=4, 6.4s at d=5.

---

## 2026-08-14 — Phase 1: environment and GATE 1

**[DECIDED] `identify_threshold = 0.7`.** A class of size k caps each member's posterior
mass at 1/k, so a size-2 class reaches exactly 0.5 — a 0.5 threshold could declare an
*unbroken tie* "identified". 0.7 is unreachable while any tie remains, which is what makes
the criterion mean something. Deliberately not `argmax == truth`: equivalent DAGs tie to
machine precision, so argmax would report success at random.

**[DECIDED] `n_obs = 1000`**, calibrated against GATE 1 so the agent starts with the
equivalence class essentially pinned and its job is cleanly to orient within it.

**[MEASURED] GATE 1 PASSES.** Observational-only identification: d=3 **14.67%**
(CI 11.0–19.0) vs 16.00% target; d=4 **10.00%** (CI 6.0–14.0) vs 10.87%. Both intervals
contain the theoretical value.

**[CORRECTED] I had the leak backwards.** I wrote a guard asserting equal noise scales
would reinstate the observational shortcut. **They do not** — 16.7% against a 16.0%
target. BGe is score-equivalent *by construction*, so it cannot separate Markov-equivalent
DAGs however the noise is distributed. The leak lives in the **estimator**, not the data:
it requires a scorer assuming a known or shared variance. Consequence: per-node noise
scales in `sa/scm.py` are defence in depth, **not** the load-bearing fix — the load-bearing
fix was replacing the scorer. This matters the moment a non-score-equivalent estimator
(AVICI, or anything learned) is swapped back in.

**[MEASURED] The guard, rebuilt against the real failure mode.** `KnownVarianceScore`
reproduces the old scorer; with equal noise it leaks **29.5%** against the 16.0% target,
and GATE 1 catches it. Both this and BGe's immunity are now pinned by tests.

---

## 2026-08-14 — Phase 2: oracle, baselines, GATE 2

**[DECIDED] Shannon rather than Gini for the oracle's scoring rule.** The outcome (a
node's descendant set) is a deterministic function of the graph, so `H(outcome|graph)=0`
and therefore `I(graph;outcome) = H(outcome)`. Maximising outcome entropy **is** expected
information gain exactly (Lindley 1956). The previous Gini/Simpson version was the
Tsallis-2 analogue needing a defence via generalised uncertainty measures. One line
removes the approximation.

**[CORRECTED] GATE 2 was measuring the wrong quantity.** Originally specified on
identification *rate*, which saturates — both policies identify essentially always (100%
vs 99.3% at d=3), so the gate failed for the wrong reason. Efficiency is where the
difference lives, and it is what the agent is being asked to improve. Now measured as
interventions-to-identify over **solved episodes only**, since an unsolved episode's count
is censored at the budget and would reward giving up.

**[MEASURED] GATE 2 PASSES.** Interventions to identify, 200 episodes:
d=3 oracle **1.12** (1.03–1.22) vs random **1.55** (1.39–1.74);
d=4 oracle **1.38** (1.27–1.49) vs random **2.53** (2.28–2.80). Disjoint at both.

---

## 2026-08-15 — Choosing the difficulty axis

**[MEASURED] `n_int` is a dead lever.** A **20× cut** (100 → 5 samples per intervention)
moves the oracle by only ~0.3 interventions at every d:

| d | n_int=100 | n_int=25 | n_int=10 | n_int=5 |
|---|---|---|---|---|
| 3 | 1.05 | 1.08 | 1.22 | 1.36 |
| 4 | 1.63 | 1.61 | 1.71 | 1.78 |
| 5 | 1.86 | 1.91 | 2.04 | 2.11 |

Reason: a hard intervention on the right node cuts the equivalence class *structurally*,
regardless of sample size, provided there is enough data to see the shift — and 5 samples
suffices for a shift that large. Cutting `n_int` adds statistical noise, not decision
depth. **My earlier suggestion to use `n_int` as the difficulty lever was wrong.**

**[MEASURED] `d` is the axis that moves.** Oracle 1.05 → 1.86 across d=3→5, and the gap
over random widens from 0.57 to 1.52 — the gap grows faster than the absolute count, which
is what matters for having something measurable.

**[MEASURED] Difficulty is driven by equivalence-class size, not graph size.**
Correlation with interventions needed: MEC size **0.56** (d=4), **0.36** (d=5); edge count
0.29 (d=4), **0.04** (d=5). At d=4, size-24 class instances take 2.6 interventions,
singletons take 0.33.

**[DECIDED] Scale `d`; drop `n_int` as a lever.** User's call, supported by the above.

**[NOTED] Hard ceiling on the exact posterior.** d=6 is 3.7M DAGs, d=7 ~1.1 billion.
Exact enumeration caps at **d=5**. This makes the two observation conditions load-bearing
rather than academic: d=3–5 runs both and measures the gap, and that measured gap is what
licenses trusting edge-marginal results at d≥6 where the exact posterior is unavailable.

**[NOTED] Sparsity trades against difficulty.** Sparse graphs have small equivalence
classes (1-edge graph → size 2; empty → size 1), dense ones have the large ones. So a
realistic sparse prior makes the task *easier*, shortening the horizon. Uniform-over-DAGs
was accidentally supplying the harder instances. Accepted knowingly: realism and research
comparability win, and lost difficulty is recovered by raising `d`.

---

## 2026-08-15 — Graph priors

**[MEASURED] Uniform-over-DAGs IS Erdős–Rényi with p = 0.5.** `P(G) ∝ p^|E|
(1-p)^(pairs-|E|)` collapses to the constant `0.5^pairs` at p=0.5, giving every DAG equal
weight. Confirmed numerically at d=4: p=0.5 reproduces the uniform figures exactly
(E[edges] 3.71, E[MEC] 5.46, singleton fraction 0.1087). **So the setup was never off the
research standard — it was ER at an unrealistically high p.** A much smaller correction
than "switch graph families".

**[MEASURED] Sparsity is nearly vacuous at d <= 5.** The literature's ER-1 convention
(expected edges = d) means p = 2/(d-1), which is *denser* than uniform at these sizes:
p=1.0 at d=3 (the complete graph), 0.667 at d=4, 0.5 at d=5. There is no room to be sparse
when C(d,2) is close to d. At expected-edges = d-1, expected class sizes are 3.97 (d=3),
5.46 (d=4), 5.96 (d=5) — barely distinguishable from uniform.

**[DECIDED] `p = 0.5` stays the default at d <= 5, with `p` a first-class parameter.**
Sparsity becomes a real lever around d >= 8 where C(d,2) >> d — which is also where the
exact posterior is unavailable. The ER-vs-scale-free comparison therefore belongs to the
large-d, edge-marginal phase; five nodes cannot host a hub, so running it now would
compare two near-identical distributions and prove nothing.

**[DECIDED] The prior is now used for BOTH generation and inference.** Previously the graph
was drawn uniformly while the posterior assumed uniform — consistent by accident. Pairing a
sparse generator with a uniform prior would be a misspecification showing up as
over-confidence in dense graphs, easily mistaken for an estimator bug.

**[CORRECTED] The leaky-estimator figure was a small-sample point estimate.** Logged
earlier as 29.5% (200 episodes). At 1000 episodes it is **26.0%**, against BGe's 13.8% and
a 16.0% target. The leak is real (~10pp), but the guard test needed 600 episodes rather
than 150 to resolve it reliably — at 150 the sampling noise could drop it under tolerance
and make the test flaky.

---

## 2026-08-15 — Edge marginals vs the exact posterior

**[MEASURED] The scalable belief representation costs very little at these sizes.** Greedy
EIG computed from an independent-edge reconstruction of the posterior — seeing only the
d(d-1) edge marginals, all cross-edge correlations discarded — versus the same policy on
the exact posterior. 120 episodes, interventions to identify:

| d | random | greedy (exact posterior) | greedy (edge marginals) | cost |
|---|---|---|---|---|
| 3 | 1.37 (1.18-1.57) | 1.01 (0.88-1.14) | 1.04 (0.92-1.18) | +3% |
| 4 | 2.56 (2.23-2.89) | 1.58 (1.39-1.79) | 1.66 (1.45-1.90) | +5% |
| 5 | 3.54 (3.06-4.16) | 1.80 (1.61-2.01) | 2.00 (1.73-2.33) | +11% |

Intervals overlap heavily at every d, and both greedy variants are far clear of random.
**The lossy representation is viable** — the assumption the entire scaling story rests on,
now tested rather than assumed.

The cost grows with d (3% -> 5% -> 11%), the expected direction: more edges means more
discarded correlations. Re-measure at each new d rather than extrapolating — this trend is
what licenses trusting edge-marginal results at d >= 6, where the exact posterior cannot be
computed at all.

**[NOTED] `EdgeMarginalGreedy` should become a permanent baseline, not a one-off.** An
agent restricted to edge marginals compared against a full-posterior oracle conflates two
different deficits: being a worse policy, and holding a lossier belief. The edge-marginal
greedy policy is the correct opponent for the condition-B agent.

---

## 2026-08-15 — Calibrating the success metric

**[DECIDED] Primary metric is "gap closed":** `(random - agent) / (random - greedy)`,
measured in interventions to identify. 1.0 means matching greedy, 0.0 means no better than
random. Normalising this way keeps the number comparable as `d` changes, which raw
intervention counts are not.

**[MEASURED] What gap-closed values actually mean.** An epsilon-greedy oracle (the greedy
policy, acting uniformly at random a fraction `eps` of the time), 300 episodes:

| eps (fraction of random actions) | gap-closed, d=4 | gap-closed, d=5 |
|---|---|---|
| 0.0 | 1.00 | 1.00 |
| 0.1 | 0.92 | 0.95 |
| 0.2 | 0.92 | 0.88 |
| 0.3 | 0.80 | 0.82 |
| 0.5 | 0.61 | 0.70 |
| 0.75 | 0.21 | 0.51 |

So **gap-closed >= 0.8 corresponds to choosing correctly roughly 70% of the time**, and
>= 0.9 to roughly 80-90%. That grounds any threshold in behaviour rather than taste.

**[MEASURED] d=4 is a noisy regime for this metric; d=5 is much better.** The eps=1.0 row
should read exactly 0.00 by construction, and instead came out at **-0.29 at d=4** versus
**-0.02 at d=5** over 300 episodes. The cause is the denominator: the greedy-random gap is
only 0.78 interventions at d=4 but 1.61 at d=5, so identical absolute noise is twice as
damaging at d=4. Practical consequence: **d=5 should be the primary reporting size**, and
any d=4 number needs either many more episodes or a wider tolerance band. This is the same
class of error as the previous project's +/-30pp noise floor, and worth catching before it
sets a threshold rather than after.

---

## 2026-08-15 — Phase 3: the agent, and two metric bugs it exposed

**[CORRECTED] The primary metric was gameable by failing.** The first smoke agent scored
gap-closed **2.04** -- apparently twice as good as greedy -- while agreeing with the oracle
**6%** of the time and solving only 65% of episodes. It solved easy episodes quickly and
let hard ones hit the budget; the solved-only average then excluded exactly the episodes it
was bad at.

I had guarded against this via *passing* (`under_acting_rate`), but budget exhaustion
causes identical censoring and that rate read 0.00. Fixed by charging unsolved episodes at
the full budget (`episode_costs`), plus a solve-rate hard fail as belt and braces. The same
agent now scores **-16.2** and correctly fails every check.

**[CORRECTED] The metric's anchors were wrong.** Random must score exactly 0.0 and greedy
exactly 1.0 by definition. They read **0.233** and **1.067**, because stateful baseline
policies carried RNG state that advanced between the reference run and the evaluation run
-- so evaluating the same policy twice gave different answers. Baselines are now
resettable and `run_episodes` resets them; anchors are exact.

**[CORRECTED] Observation features were on incompatible scales.** The budget feature was a
raw count sitting at 20.0 while posterior entries averaged 0.04 -- a ~500x mismatch that
saturated the tanh trunk and drowned out the belief the agent acts on. Normalised to [0,1],
with a regression test.

**[MEASURED] The greedy-collapse signature reproduced from an entropy bonus alone.** With
`entropy_coef = 0.01`, training looked healthy (solve rate 1.0, mean length 1.3 against
greedy's 1.0) but entropy plateaued at **1.09 against a 1.386 maximum**. The policy never
sharpened, so argmax was arbitrary -- the deterministic policy picked node 2 regardless of
belief, while the sampled policy performed fine. That is precisely the previous project's
"trains well, collapses when evaluated greedily" failure, and it arose here purely from an
exploration bonus that never decayed. Lowered to 0.003; entropy now falls 1.39 -> 1.08 over
1500 episodes. Worth remembering as a candidate explanation whenever that pattern recurs.

**[MEASURED] d=3 is a poor training testbed.** Random costs 1.31 against greedy's 1.00, so
the entire learnable advantage is 0.31 interventions and there is almost no gradient signal.
Consistent with GATE 2 (gap 0.57 at d=3 vs 1.52 at d=5). Do not read much into any d=3
training number; d=5 remains the primary reporting size.

**[DECIDED] Move experiments to Myriad.** Local runs are single-threaded on CPU and seeds
are fully independent, so an SGE array job gives near-linear speedup. A separate
`~/envs/sa_env` was built rather than reusing the shared `marl_env`, since installing extra
dependencies into that environment previously caused a protobuf/wandb conflict, and `sa/`
is meant to stay isolated. Worktree at `~/marl_sa`.

---

## 2026-08-15 — overnight lever sweep

**[DECIDED] One-factor-at-a-time around a fixed baseline, not a grid.** Thirteen levers is
far too many for a full factorial, and the question actually being asked is "what does each
lever do, and does the conclusion depend on it?" — which OFAT answers directly. The cost is
that it cannot detect interactions between levers; that limitation is stated in the results
rather than papered over. Baseline: `d=5, edge_marginals, 6000 episodes, budget 20,
entropy 0.003, lr 3e-4, step_cost 0.05, hidden 128, n_obs 1000, n_int 100, threshold 0.7,
ER p=0.5`. 34 configurations, 110 (config, seed) runs. Matrix lives in
`scripts/sweep_configs.py` as data, read by both the submit script and the analysis, so
intent and report cannot drift apart.

**[DECIDED] d=3 dropped, d=6 added.** d=3's learnable advantage is 0.31 interventions — no
signal worth a night of compute. d=5 stays the primary reporting size.

**[DECIDED] Two deliberate negative controls in the matrix.** `identify_threshold=0.5`
should *inflate* solve rates, because a Markov equivalence class of size 2 caps each member
at exactly 0.5 and the threshold can therefore declare an unbroken tie identified. And
`step_cost=0.0` removes any incentive to be quick, leaving only "identify eventually" — if
gap-closed survives that unchanged, the metric is not measuring what it claims to. Both are
predictions that can fail.

**[DECIDED] `n_int` included despite being a predicted dead lever.** A 20x change moved the
previous measurement by 0.3. It is in the matrix precisely because that is falsifiable.

**[MEASURED] d=6 is reachable after vectorising three construction loops.** Enumerating its
DAGs one candidate at a time takes ~28 minutes, paid again by every job. Rewritten as array
operations over blocks of graphs (Kahn's algorithm, transitive closure, and MEC signatures
packed into integer bit-codes), d=6 now builds in **37 seconds** and reproduces both
externally known counts exactly: **3,781,503 DAGs** (OEIS A003024) and **1,067,825
equivalence classes** (A007984). Its singleton fraction — the GATE 1 target at d=6 — is
**8.10%**, continuing the decline from 16.00% (d=3), 10.87% (d=4), 8.93% (d=5).

The enumeration *order* is preserved byte-for-byte against the per-graph implementation,
asserted by test at d=2..5. This is not cosmetic: a DAG's index is its identity everywhere
in the codebase, so a reordering would silently renumber every graph while every count
still looked correct.

**[MEASURED] d=6 costs ~0.7s per posterior update.** That puts the four reference policies
at roughly an hour, which cannot be repeated per seed — hence `--ref_cache`, which stores a
fingerprint of the entire environment config and *refuses to load* when it differs. Sharing
baselines across seeds is only safe because they are deterministic given the config and the
fixed seed 99; silently reusing references from another environment would look like a
result rather than a bug.

**[DECIDED] d=6 runs `edge_marginals` only.** The exact-posterior observation is 3,781,504
numbers wide at d=6 — a 484M-parameter first layer. Condition A is structurally out of
reach at this size, which is exactly the cost the A-vs-B comparison at d=4 and d=5 exists
to quantify.

**[DECIDED] Raw training history is now saved.** Entropy and solve-rate trajectories are how
a collapse is diagnosed after the fact, and re-running a night of jobs to recover a curve is
not an acceptable cost. Results also carry provenance (git commit, package versions, host,
UTC time) and the reference-policy metrics, so a row is self-contained.

**[CORRECTED] The cluster and the laptop are not on the same torch.** Myriad's package index
tops out at `torch 2.6.0+cpu`; the laptop has `2.10.0+cpu`. numpy and scipy were pinned to
match exactly (1.26.4 / 1.13.1), but torch could not be. Now recorded in provenance so a
numerical difference between environments cannot be invisible.
