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

**[MEASURED] The agent learns the easy half of the action space and none of the hard half.**
First stage-1 result, `core_d4_edge_marginals`, 5 seeds. Reading the training trajectory
rather than the endpoint:

| episodes | solve rate | mean length | entropy |
|---|---|---|---|
| 32 | 0.66 | 1.31 | 1.609 |
| 800 | 0.97 | 1.97 | 1.427 |
| 1568 | 1.00 | 2.19 | 1.389 |
| 5984 | 0.97 | 2.44 | 1.337 |

The early state is not "good and getting worse". A uniform policy over `d+1 = 5` actions
passes 20% of the time, and passing ends the episode immediately — hence solve 0.66 at
length 1.31. What the agent then learns, over ~1500 episodes, is **not to pass**: solve
rate goes to 1.00. What it never learns is *which node to target* — mean length settles at
~2.4, and random costs **2.44**. It ends at exactly random-policy quality.

Entropy stalls at 1.34–1.39 against a maximum of ln(5) = 1.609, so the deterministic
policy's argmax is essentially arbitrary, which is why deterministic gap-closed (−4.9 to
−9.9) is so much worse than sampled (+0.08). Same collapse signature as before, from the
same cause: a policy that never sharpened.

**[CORRECTED] My first explanation of this was wrong, and the error is worth recording.**
I initially wrote that the learnable signal is "5% of the reward scale" — step cost 0.05
against a +1 terminal bonus — and that the gradient is therefore tiny. That reasoning does
not survive contact with the algorithm. Advantages are normalised to unit standard
deviation before the policy update, and the near-constant +1 is absorbed by the value
baseline, so the *absolute* scale of the reward cancels. `value_loss` falling cleanly from
0.38 to 0.01 confirms the critic is doing exactly that job. Likewise `policy_loss ≈ 0.005`
is not evidence of a weak signal: with normalised advantages the clipped surrogate is
near zero at the start of every epoch by construction. I misread a structural constant as
a symptom.

The defensible version is *relative*, not absolute: the pass-versus-act contrast has a
large and consistent effect on return (+1 against 0), while the which-node contrast has a
small one (fractions of a step). Both share one batch-wide normalisation, so the
which-node signal is a small share of the normalised advantage. That predicts precisely
what was measured — the large contrast gets learned, the small one does not.

This leaves the stage-2 grid well-motivated but for a corrected reason. `step_cost` is not
a no-op under normalisation, because it changes the *relative* worth of acting versus
passing rather than the overall scale; `gamma` changes how much finishing sooner is worth;
`entropy_coef` sets how hard the policy is held toward uniform. All three move the same
underlying quantity, which is why they are gridded rather than swept one at a time.

### Stage 1 results — all 34 configurations fail, and the reason is not a hyperparameter

**[MEASURED] Every arm fails, and the failure is qualitatively the same.** Gap-closed ranges
from −2.1 to −18.3; not one configuration passed a single seed. The uniformity is the
finding: no lever rescues the run, so the problem is not in the region any of them explore.

**[MEASURED] The deterministic agent solves LESS often than random.** Deterministic solve
rate is 0.25–0.59 across arms while greedy solves 0.99 and random solves ~1.00 at budget 20.
Inefficiency cannot produce this — a merely clumsy policy still identifies the graph
eventually. `optimal_rate` of **0.02–0.10 against a chance level near 0.29** says the same
thing more sharply: the agent is *systematically anti-correlated* with the oracle, not
simply unhelpful.

Both are what a policy that has stopped reading its observation looks like. Its argmax is
constant, so it re-intervenes on the same node every step, gathers almost no new structural
information, and exhausts the budget.

**[CORRECTED] The entropy bonus is not the cause, and my earlier suspicion of it was wrong.**
`entropy_coef=0.0` — the bonus switched off entirely — still ends at final entropy **1.596**
against a maximum of ln(6)=1.792, and gap-closed −6.53. Across 0.0 / 0.001 / 0.01 / 0.03 the
median gap moves only from −6.53 to −6.42. If the policy stays near-uniform with no bonus at
all, the bonus was never what held it there. This retires the explanation carried over from
the previous project.

**[MEASURED] Which levers move anything (none rescue it).** `lr=1e-3` gives −5.35 against
−8.60 at 1e-4, and the lowest final entropy of any arm (1.495) — the clearest sign of a
policy actually sharpening. `hidden=256` gives −5.08 against −8.98 at 64. `episodes_per_update=16`
gives −5.04. Capacity and step size matter; reward shape barely does.

**[CORRECTED] `budget` is largely a metric artifact and must not be read as a lever.**
`budget_10` scores −2.79 and `budget_40` scores −17.97, which looks like a huge effect. It
is mostly definitional: `episode_costs` charges unsolved episodes at the full budget, so
raising the budget multiplies the penalty for the same underlying failure. The comparison
is not wrong — it is exactly what the metric is designed to do — but it measures the cost
of failing, not sensitivity to the budget.

**[MEASURED] The observation carries a decodable answer; PPO is not extracting it.** A
supervised probe (`scripts/probe_observation.py`) trained on the agent's *own architecture*
to predict the oracle's tied-best target from the agent's *own observation* reaches **0.42**
at d=4 against a chance level of 0.287 and a majority-class baseline of 0.26. Well short of
perfect, but decisively above both. So explanation "the representation does not contain the
answer" is ruled out at d=4; the failure is in the learning, not the input.

Notably the exact posterior reaches only **0.46** against edge marginals' 0.42 — the
sufficient statistic is barely better than the lossy summary. That points at the *decoding
computation* being the hard part, not the information content: recovering the oracle's
choice means reconstructing a posterior over DAGs, computing each node's descendant-set
partition, and taking its entropy. That is a lot to ask of a two-layer MLP, whichever
representation it starts from.

**[DECIDED] Stage 4 follows from the diagnosis, not from tuning.** The observation contains
the posterior and remaining budget but **not which nodes have already been intervened on**.
Formally it does not need them — the posterior is sufficient, and if an intervention taught
nothing then the same target really is still best. But that argument is about the *optimal*
policy; a deterministic network whose output barely varies with its input has no way out of
the loop at all. `include_counts` adds them, `repeat_rate`/`distinct_targets` measure whether
the loop is real, and arm 6 (best settings, no counts) is the control that stops the learning
rate taking credit for the observation change.

### The architecture was the bottleneck

**[MEASURED] A supervised probe localises the failure to the network, not the task.**
Trained on the agent's own observation to predict the oracle's tied-best target at d=4,
with abundant labels and no exploration problem (chance 0.279, majority 0.271):

| observation | architecture | probe accuracy |
|---|---|---|
| edge marginals | flat MLP | 0.528 |
| **edge marginals** | **per-node scorer** | **0.814** |
| exact posterior | flat MLP | 0.618 |

The per-node scorer reading the *lossy* summary beats the flat network reading the *exact
sufficient statistic*. That is the cleanest possible localisation: the difficulty is not the
reward, not the exploration, not the information content of the observation. It is the flat
network's ability to express the mapping at all. Fifty-four configurations across stages 1-4
were tuning things that could not have mattered.

**[DECIDED] `PerNodeActorCritic`, and why this shape.** The oracle's score for node i is a
function of i's own descendant structure — *the same function for every i*. A dense layer
from d(d−1) marginals to d logits must instead learn each node's scorer separately and
rediscover from data that the nodes are interchangeable. The new network embeds each
neighbour pair (i→j, j→i), pools over neighbours, and scores node i from its own pooled
summary, with one shared scorer serving all d nodes. Value and the pass logit read a
mean-pooled summary, so they are permutation-*invariant* — correct, since how good a state
is does not depend on node labels.

Two consequences beyond accuracy: the policy is permutation-**equivariant**, which the
oracle is and the flat network structurally cannot be; and the parameter count no longer
grows with d, so the identical model form carries to d=6 rather than needing a new one.

**[CORRECTED] My first version of this class was not equivariant, and the test caught it.**
Node i's features were its neighbours' marginals *in index order*. Relabelling the nodes
reorders that vector, so the network was equivariant only under permutations that happened
to preserve neighbour ordering — which is to say, not equivariant. I had written the test
asserting the property before believing it held, and it failed with a max absolute
difference of 1.9e-3 on logits of order 1e-3, i.e. completely. Fixed by pooling over
neighbours (mean and max concatenated) instead of concatenating them in order, following
Deep Sets (Zaheer et al. 2017); mean and max together because a single statistic collapses
distinctions the score depends on. The property now holds to 1e-5.

Worth keeping as a general lesson: the inductive bias I *intended* and the one I *wrote*
differed, and nothing about the code's appearance revealed it. Only asserting the
mathematical property directly did.

**[DECIDED] d=6 runs the per-node architecture.** It is the only configuration with evidence
behind it, and its d-independent parameter count means d=6 continues the same experiment
rather than starting a separate one. The reference cache fingerprint now excludes
`include_counts`, which is safe for a specific and checkable reason: that flag only changes
what `env.observation()` returns, and no reference policy calls it — random draws from its
RNG, no_intervention is constant, and both greedy variants read `result.posterior` directly.
Every other field can move the references and every other field stays in the fingerprint.

### Stage 5 — the agent beats the greedy oracle at d=5

**[MEASURED] The winning configuration, and it needs all three ingredients.** At d=5,
gap-closed per seed (0 = random, 1 = greedy oracle, >1 = beating it):

| configuration | arch | seeds | min gap | entropy | verdict |
|---|---|---|---|---|---|
| `pernode_best_counts_shape` | per-node | 3 | **+1.276** | 0.57 | **3/3 pass** |
| `pernode_best_counts` | per-node | 4 | **+1.116** | 0.52 | all beat greedy |
| `pernode_best` (no memory) | per-node | 4 | −1.766 | — | unstable |
| `pernode` alone | per-node | 3 | −3.696 | 1.11 | fail |
| `flat_control` (same settings) | flat | 3 | −1.553 | 1.32 | fail |

`pernode_best_counts` reads +1.284, +1.116, +1.181, +1.304 across its four seeds — every
one beating the myopic oracle, with a spread of 0.19. That is criterion **S2** from
docs/SA_PLAN.md, the outcome recorded in advance as "the result".

**[MEASURED] The architecture is worth ~2.7 gap-closed, isolated by control.**
`flat_control` runs *identical* settings — lr 1e-3, hidden 256, episodes_per_update 16,
`include_counts` — on the old dense network and scores −1.553 against +1.116. The control
was put in the matrix precisely so the learning rate could not be credited for this, and it
earned its place.

**[MEASURED] Action memory is what makes it *stable*, not what makes it *work*.** The same
architecture without `include_counts` swings from +1.043 to −1.766 across seeds. With it,
the four seeds span 0.19. This is consistent with the collapse mechanism: a deterministic
policy that cannot see which nodes it has already targeted has no way to break out of
re-picking one, and whether it does becomes a lottery over initialisations. Note the
posterior is *formally* sufficient, so this is not an information fix — it is a fix to the
policy's ability to act on information it already had.

**[MEASURED] Entropy is the single clearest tell across the whole night.** Every failing
configuration ended between 1.21 and 1.61 nats against a 1.79 maximum; every passing one
ended between 0.52 and 0.60. Across 60-plus configurations this separated pass from fail
more reliably than any hyperparameter.

**[CORRECTED] My "the architecture is the bottleneck" claim was initially overstated, then
confirmed on better evidence.** I first compared flat (0.528) against per-node (0.814) at
600 episodes of probe data while the cluster's flat-only run reached 0.766 at 3000 — so
part of the original gap was data quantity. Re-run at matched sizes, per-node dominates at
every point and has a higher ceiling:

| episodes | flat | per-node |
|---|---|---|
| 300 | 0.430 | 0.840 |
| 1,000 | 0.648 | 0.856 |
| 3,000 | 0.766 | 0.872 |
| 9,000 | 0.791 | 0.890 |

Per-node at 300 episodes beats flat at 9,000 — roughly a 30x sample-efficiency advantage.
The conclusion survived, but only because the matched comparison was run rather than
assumed.

**[CORRECTED] The raw results were not committed when I first said they were.** The repo
carries a blanket `*.json` ignore rule, so the archiving commit added the README and none of
the 61 result files. `git add` on ignored paths is a silent no-op and the commit succeeded
normally. Now exempted explicitly in `.gitignore`; 71 files tracked. Worth remembering that
"the commit succeeded" is not evidence the files are in it.

### GATE 1 was pinned once, at d=3, and silently stopped holding

**[MEASURED] The observational-only rate falls below its theoretical target as d grows.**
GATE 1 requires the no-intervention identification rate to equal the fraction of DAGs alone
in their Markov equivalence class — a number computable exactly from the graph space. It was
checked at d=3 (14.67% against 16.00%) and at d=4, passed both times, and was then assumed.
Measured properly across d and n_obs, 200 episodes each, with bootstrap CIs:

| d | target | n_obs=1000 | n_obs=5000 | n_obs=20000 |
|---|---|---|---|---|
| 4 | 0.1087 | 0.085 OK | 0.085 OK | 0.090 OK |
| 5 | 0.0893 | **0.040 MISSES** | 0.060 OK | 0.075 OK |
| 6 | 0.0810 | **0.025 MISSES** | **0.050 MISSES** | 0.060 OK |

The default `n_obs=1000` therefore fails the gate at d=5 — the primary reporting size — and
fails it badly at d=6. Larger graphs have more parameters to estimate from the same 1000
samples, so the posterior never concentrates enough to identify even the graphs that are
observationally identifiable in principle.

**[CORRECTED] Every d=5 result tonight ran in an under-powered environment, including the
headline.** The agent began each episode from a blurrier belief than the design intends, and
some episodes that should have been solvable without intervening were not.

What this does *not* break: gap-closed is measured against random and greedy baselines
evaluated in the *same* environment, so the ranking, the flat-versus-per-node comparison,
and the ablation all stand. What it does break: the claim that the environment matches its
specification, and any comparison of *absolute* difficulty across d.

**[DECIDED] Stage 6 re-runs the winner where the gate passes** (d=5 at n_obs 5000 and
20000, with the flat control alongside), rather than shipping the result with a footnote.

**[DECIDED] GATE 1 should be a per-configuration precondition, not a one-off.** This is the
same failure shape that cost this project its previous round: a check performed once, under
one setting, and thereafter assumed. The check is cheap — 200 no-intervention episodes — and
the target is free to compute. It belongs in `run_experiment.py` as a guard that refuses to
train when the environment does not match its specification.

### Stage 6 — the result survives the GATE 1 correction

**[MEASURED] Re-run where the gate passes, the headline holds and tightens.**

| configuration | d | n_obs | GATE 1 | seeds | min gap |
|---|---|---|---|---|---|
| `s6_d4_nobs5000` | 4 | 5000 | passes | 3/3 | **+1.283** |
| `s6_d5_nobs5000` | 5 | 5000 | passes | 3/3 | **+1.233** |
| `s5_pernode_best_counts` | 5 | 1000 | **fails** | 5/5 | +1.116 |
| `s6_d5_nobs5000_flat` | 5 | 5000 | passes | 0/2 | −1.858 |

The winning configuration scores **+1.233** on its worst seed in the specification-compliant
environment, against +1.116 in the under-powered one — better, not worse. The flat control
run in that same valid environment still fails at −1.858, so the architecture comparison
also survives.

This matters more than the small numerical improvement: the headline no longer rests on an
environment that did not match its own specification. It was worth the extra hour rather
than shipping the result with a footnote.

**[MEASURED] More observational data helps the agent substantially at d=4 too.** At
n_obs=1000 the d=4 per-node arm scored min +0.239 with 1/5 seeds passing; at n_obs=5000 it
scores min +1.283 with 3/3. Sharper starting beliefs make the which-node decision easier to
learn, not just the environment more correct.

### Where this leaves the single-agent case

The scaling ladder's first rung is done: **the agent beats the myopic information-gain
oracle at d=4 and d=5, on every seed, in a gate-valid environment.** That is criterion S2
from docs/SA_PLAN.md, recorded in advance as "the result".

The recipe is three things together, none sufficient alone:
1. a permutation-equivariant per-node scorer (worth ~2.7 gap-closed against its control),
2. tuned optimiser settings (lr 1e-3, hidden 256, episodes_per_update 16),
3. intervention counts in the observation — which buys *stability*, not capability.

Open items for the next session, in priority order:
- **Make GATE 1 a per-run precondition** in `run_experiment.py`. This is the highest-value
  change: the same "checked once, then assumed" failure has now cost this project twice.
- **d=6 is unresolved.** Its runs used n_obs=1000, where the gate misses widest; a valid
  d=6 needs n_obs=20000, and training cost there is ~7h/seed.
- Re-run the d=5 lever sweep at n_obs=5000 — every stage-1 conclusion was measured in the
  under-powered environment and may not transfer.
- Only then move to two agents.

### d=6 — one seed, encouraging, not established

**[MEASURED] The agent beats the oracle at d=6 too, on the one seed that finished.**
gap-closed **+1.145** (sampled +1.054), solve rate 0.98 against greedy's 0.98, cost 2.57
against greedy's 2.77, final entropy 0.73. Oracle agreement on informative steps is
**0.426**, against 0.02–0.10 for every configuration that failed — the agent is now
positively correlated with the oracle rather than anti-correlated.

**[CORRECTED] This number is not gate-valid and must not be reported as equivalent to the
d=4/d=5 results.** It ran at `n_obs=1000`, where GATE 1 misses by the widest margin of any
setting measured (0.025 against a 0.081 target). A valid d=6 needs `n_obs=20000`, and at
~6.4s per episode that is roughly seven hours per seed before the extra sampling cost.

**[MEASURED] d=6 costs about 4.6 hours per seed at n_obs=1000.** Seed 2 took 16,737s. The
other two tasks were still running at 9h30m CPU against a 10-hour walltime and are likely to
be killed before writing output — a sizing mistake on my part: I estimated 3.5h per seed
from a small sample and set the walltime with too little headroom. The completed seed's
result file was copied off the cluster before the deadline.

**[DECIDED] Two-thirds of a d=6 run is worth keeping as one caveated data point** rather
than discarded, but it does not change the headline, which rests on d=4 and d=5 where the
environment is verified.

### Standing recommendation

`GATE 1` now runs as a precondition on every training run (`scripts/run_experiment.py`),
prints its verdict, records it in the output JSON, and can be made fatal with
`--require_gate1`. Tests assert it fails when it should. This is the single most valuable
change of the night: the same "checked once, then assumed" failure has now cost this
project twice, and it is cheap to prevent — 200 no-intervention episodes against a target
that is free to compute.

**[CORRECTED] All three d=6 seeds finished; my walltime worry was wrong.** I predicted two
of three would be killed before writing output. They completed at 4.6-4.8 hours each,
comfortably inside the 10-hour limit. The estimate was wrong in both directions across the
night: 3.5h predicted, ~4.7h actual, and then an unfounded fear of overrun on top of it.

Final d=6 results, all three seeds passing their per-seed criteria:

| seed | gap | solve | oracle agreement | entropy | repeat rate |
|---|---|---|---|---|---|
| 0 | +1.098 | 0.99 | 0.492 | 0.59 | 0.158 |
| 1 | +1.098 | 0.99 | 0.445 | 0.96 | 0.173 |
| 2 | +1.145 | 0.98 | 0.426 | 0.73 | 0.155 |

Repeat rate of 0.15-0.17 is worth noting against the diagnostic's purpose: the collapse it
was built to detect would push this toward 1.0. The agent revisits targets occasionally,
which is legitimate — more samples do sharpen a posterior — rather than pathologically.

**[MEASURED] The identical +1.098 on seeds 0 and 1 is a coincidence, not a duplicated run.**
Both used exactly 400 interventions across 150 episodes, so `mean_cost` is 400/150 for both.
They differ in regret (0.111 vs 0.136), final entropy (0.59 vs 0.96) and oracle agreement
(0.492 vs 0.445). Episode costs are integers, so exact ties in the mean are unremarkable —
but worth checking rather than assuming, since a duplicated run would look identical.

**[CORRECTED] The per-task "OVERALL: FAIL" lines in the d=6 logs are an artefact.** Each
array task ran a single seed, and `summarise_seeds` requires 4 passing seeds by default, so
a one-seed summary can never pass regardless of its result. The per-seed verdicts are all
`passed: True`. Anyone reading those logs directly would draw the opposite conclusion; the
aggregate in `results/all_runs.csv` is the number that means anything.

**[NOTE] The d=6 result files carry `gate1: None`** because they predate the precondition.
Their GATE 1 failure is known from the separate audit, not from the runs themselves.

## 2026-08-15 — Hot-path optimisation before the next phase

[MEASURED] Profiled the environment step before committing compute to Phase 2. The cost
structure was not what the plan assumed. Milliseconds per `step` + `edge_marginals`
observation, laptop CPU:

| config | before | after | speedup |
|---|---|---|---|
| d=4, n_obs=1000  |   22.2 |  16.9 | 1.31x |
| d=4, n_obs=5000  |   29.4 |  17.7 | 1.66x |
| d=4, n_obs=20000 |   57.3 |  27.9 | 2.05x |
| d=5, n_obs=1000  |   57.6 |  38.3 | 1.50x |
| d=5, n_obs=5000  |   72.5 |  42.1 | 1.72x |
| d=5, n_obs=20000 |  137.2 |  51.2 | 2.68x |
| d=6, n_obs=1000  | 1470.5 | 791.7 | 1.86x |
| d=6, n_obs=20000 | 1850.6 | 845.7 | 2.19x |

[CORRECTED] The plan's "known risk" that E1/E2 would run ~2x slower at n_obs=5000 was
wrong, in two ways.

1. The n_obs dependence was an implementation artefact, not intrinsic. `BGeScore` depends
   on the data only through (n, column means, centred scatter), and the statistics for a
   subset of columns are submatrices of the full-column ones. The old code nevertheless
   re-sliced and re-centred all n rows once per (node, parent set) pair -- 160 passes over
   the data per posterior at d=5. Hoisting that to one pass per node makes the score table
   nearly n-independent. Measured at d=5: 49.0 / 48.0 / 49.3 ms at n_obs 1000 / 5000 /
   20000 -- flat.
2. Consequently E1/E2 at n_obs=5000 (42.1 ms/step) are now *faster* than the overnight
   runs they extend, which were d=5 n_obs=1000 at 57.6 ms/step. The risk is not merely
   absent, it is reversed.

[MEASURED] At d=6 the bottleneck was never sample count at all: it was two n-independent
reductions over the 3.78 million enumerated DAGs. Original split at n_obs=1000 --
score table 90 ms, score gather 384 ms, edge marginals 517 ms. This is what made d=6 cost
4.7 h/seed, and it explains why the earlier runtime extrapolation from smaller d missed.

Four changes, each an exact restatement rather than an approximation:

- **Sufficient statistics** (`sa/score.py`): `BGeScore.sufficient_stats` /
  `local_score_from_stats`. Scorers without them (BIC, KnownVariance) take the old path.
- **Flat gather index** (`sa/posterior.py`): the old `table[rows, parent_set_ids]`
  broadcast a [1, d] index against [N, d] and cast int32 to intp on every call.
  Precomputing the flat intp index is bit-identical: 384 -> 193 ms at d=6, ~180 MB against
  a 12 GB request.
- **Edge marginals by bincount** (`sa/posterior.py`): edge i->j exists exactly when i is
  in j's parent set, so d bincounts over the parent-set ids replace a weighted sum over
  N x d x d floats (which upcast the int8 DAG array to float64 in chunks). 517 -> 245 ms,
  agreeing to 2.5e-15.
- **Two accidental O(N)-per-step costs**: `space.is_singleton` is a property that
  materialised a 3.8M-element bool array to read one element each step; and the posterior
  re-took the log of a prior that never changes. Both hoisted.

[DECIDED] Stopped here. The remaining d=6 cost is Python-level overhead across ~750 small
slogdet calls per step; removing it needs a batched rewrite of the score table, which is a
real correctness risk for one arm (E4, 3 seeds). Not worth it.

[MEASURED] Consequences for the plan. E4 (d=6, n_obs=20000, the arm flagged as possibly
unaffordable) is now ~845.7/1470.5 x 4.7 h = **~2.7 h/seed**, comfortably inside the 10 h
walltime -- and that is at 20x the sample count of the run that took 4.7 h. The timing
probe stays in the plan, because this is an extrapolation from laptop CPU to Myriad and
the last two d=6 runtime predictions were both wrong.

[MEASURED] Verification: `tests/test_optimisations.py`, 23 tests, pins every fast path
against the slow one it replaced -- including the pre-optimisation BGe marginal inlined
verbatim as a reference, so the comparison cannot drift into the new code checking itself.
Full suite 268 passed (and got faster: 277 s -> 182 s).

## 2026-08-15 — Phase 0 complete: instrumentation

[DECIDED] Built before any Phase 2 compute, because instrumentation added afterwards
cannot explain runs that have already happened.

**WandB** (`sa/tracking.py`, `scripts/sync_wandb.py`). Off unless `--wandb_project` is
passed; offline by default; never fatal. The third property is the one that mattered to
get right: a compute node has no outbound internet, so an online `wandb.init()` there does
not fail fast, it hangs -- burning the entire walltime of a job that was otherwise going
to succeed. Every call is wrapped, so a missing package, a full disk, a permissions
problem or 34 concurrent array writers all degrade to a warning and a no-op tracker.
`tests/test_tracking.py` breaks WandB eleven different ways and asserts the caller
continues, including failure on the first `log` mid-training -- later and more expensive
than a clean failure at startup.

Training curves are replayed to WandB after `train()` returns rather than streamed from
inside the PPO loop, which keeps `sa/policy.py` free of any tracking dependency. The
curves are identical; only their arrival time differs, and nothing watches a batch job
live.

**Canaries G1-G5** (`sa/gates.py`), recorded in every result JSON and printed at the end
of every run. Distinct from GATE 1/2, which qualify an environment beforehand; these
travel with the numbers so a result cannot be read without its checks.

[CORRECTED] G2 as first written was decoration. It asserted that gap closed evaluates to
0 at the random reference and 1 at greedy -- but that is an algebraic identity of the
formula, which defines its own endpoints, so it holds even when the two references are
swapped. The test asserting "G2 fires on swapped references" passed while the canary
stayed silent; it was passing on an incidental assertion about costs, not on the canary.
Fixed by also checking the ordering (greedy cannot cost more than random), which is what
actually detects a swap. Recording this because it is the exact failure mode the canaries
exist to prevent, reproduced while building them.

[MEASURED] End-to-end verification, d=4, 2 seeds, deliberately under-trained at 200
episodes. Two canaries fired and three stayed quiet, which is correct for a run this
short:

- G1 fired: final entropy 1.603 nats = 100% of the ln(5) = 1.609 ceiling.
- G4 fired: gap closed spanned 1.191 across two seeds (-2.830 to -1.638).
- G2 quiet: anchors exact, random -> 0.0e+00, greedy -> 1.000000 (costs 2.500 vs 1.717).
- G3 quiet: 100% of scored actions informative.
- G5 quiet: GATE 1 rate 0.0700 against singleton fraction 0.1087.

Result JSON contains all five records; two offline WandB run directories were written and
`scripts/sync_wandb.py --dry_run` lists both. `wandb/` added to `.gitignore` -- the JSON
files remain the record.

[MEASURED] Full suite 302 passed (was 268 after the optimisation commit, 158 before this
phase). The 34 new tests are `test_canaries.py` (22) and `test_tracking.py` (11), plus
one G2 test rewritten to assert the canary itself rather than a side condition.

## 2026-08-15 — Phases 1 and 2 launched

[DECIDED] Depth added to the per-node scorer as `layers`, repeating the neighbour
aggregation round k times. The constraint that shaped the implementation: `layers=1` has
to reproduce the network behind the d=4/5/6 results *exactly*, not merely have the same
shape. Every `nn.Linear` draws from the torch RNG at construction, so creating the extra
modules anywhere but last would shift the initialisation of everything after them -- and
the resulting divergence would look like ordinary seed variance, not like a bug. The round
modules are therefore built last and only when `layers > 1`, and `tests/test_depth.py`
asserts state-dict and output equality at d=4/5/6, with and without action memory.

Equivariance re-checked at depth 2 and 3. The added rounds gather neighbour embeddings by
index, which is exactly the mistake an earlier version of this class made -- it pooled
neighbours in index order and so was equivariant only under permutations that happened to
preserve that order. Messages are pooled mean+max, as in the first round.

[MEASURED] Local pilot, d=4, 800 episodes, one seed: depth 1/2/3 gave 0.802 / 0.788 /
0.799. No signal for depth, but that is one cell. Noted here because it is the prior the
full grid will either confirm or overturn, and recording it now prevents reading the grid
as confirmation of something already believed.

An earlier pilot at 150 episodes returned 0.592 for all three depths, which looked like
the parameter being ignored. It was a 49-sample tie -- test accuracy has 1/49 granularity
there. Confirmed wired in by re-running at 800 episodes, where the depths separate.

[DECIDED] Probe grid restructured from the plan's 24 (d x size x depth) tasks to 24
(d x size x seed) tasks, each training all three depths on the SAME collected data. Two
reasons: collection dominates the cost, so this is ~3x cheaper; and it makes "at matched
data size" in the decision rule literal rather than approximate. The three seeds are the
important half of the change -- the decision threshold is 0.03 and the pilot showed 0.014
of spread between depths on identical data, so a single-seed grid would have been applying
a fine rule to noise.

[DECIDED] Phase 2 baseline moved to the configuration that actually won (per-node,
lr=1e-3, hidden=256, episodes_per_update=16, action memory on, n_obs=5000). The overnight
sweep characterised 13 levers around a network that could not express the task whatever
the lever was set to, so those numbers describe the failure mode rather than the levers.
E2 repeats all 33 configurations with `arch=flat` and everything else byte-identical; a
test asserts the two halves differ in `arch` and nothing else, since that is the whole
basis of the comparison.

[DECIDED] Added a deliberate negative control to both arms: n_obs=1000 at d=5, where GATE
1 does not pass. Not in the original plan, which specified gate-passing n_obs throughout.
Included because it costs one task per architecture and converts "the gate matters" from
an assertion into a measurement. Verified locally before submission: G5 fires with
"observational-only rate 0.0267 against a singleton fraction of 0.0893 -- GATE 1 FAILED",
recorded in the result file rather than left for someone to notice later.

[MEASURED] Jobs submitted to Myriad, all queued:

| job | what | tasks |
|---|---|---|
| 146493 | P1 depth probe | 24 |
| 146525 | E1 + E2 lever sweep (33 configs x 2 arches x 3 seeds) | 66 |
| 146526 | E4 d=6 timing probe, n_obs=20000 | 1 |

E4's full run is deliberately NOT submitted yet. The projection from the hot-path work is
~2.7 h/seed, but that extrapolates a laptop benchmark to a cluster node, and d=6 runtime
has been mis-predicted twice already. The probe measures it on a real node first.

[MEASURED] Test suite 335 passed (158 at the start of this phase).
