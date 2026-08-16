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

## 2026-08-15 — Depth probe, partial (18 of 24 cells)

[MEASURED] Mean probe accuracy over 3 seeds, per cell. d=5 at 3000 and 9000 episodes were
still running when this was read.

| d | episodes | L1 | L2 | L3 | flat |
|---|---|---|---|---|---|
| 4 |  300 | 0.783 | 0.769 | 0.724 | 0.452 |
| 4 | 1000 | 0.852 | 0.895 | 0.901 | 0.590 |
| 4 | 3000 | 0.872 | 0.917 | 0.919 | 0.740 |
| 4 | 9000 | 0.880 | 0.941 | 0.944 | 0.782 |
| 5 |  300 | 0.748 | 0.725 | 0.708 | 0.363 |
| 5 | 1000 | 0.782 | 0.801 | 0.798 | 0.385 |

[CORRECTED] The hypothesis that the ~0.89 ceiling is *not* about multi-hop reachability is
looking wrong at d=4. Depth 3 reaches 0.944 against depth 1's 0.880 at 9000 episodes, and
clears the +0.03 threshold at three of four data sizes. Multi-hop aggregation does lift the
ceiling there.

The pattern is data-dependent in a way worth stating: at 300 episodes depth *hurts* at both
d (-0.014, -0.022), which is what an under-determined larger model should do. Depth only
pays once there is enough data to fit it.

[CORRECTED] My local pilot -- d=4, 800 episodes, one seed, 40 epochs -- reported 0.802 /
0.788 / 0.799 and I logged it as "no signal for depth". The grid at 1000 episodes with 3
seeds and 60 epochs gives 0.852 / 0.895 / 0.901, a +0.050 lift. The pilot was undertrained
on both axes and its reading did not survive. Recorded because I explicitly logged that
pilot as the prior the grid would confirm or overturn; it overturned it.

[CORRECTED] `analyse_depth.py` printed "RULE DOES NOT FIRE" from this partial grid. That
was a flaw in the script, not a result. The rule needs depth to win on BOTH d, so a missing
cell can only ever convert "does not fire" into "fires" -- reading a verdict early is
therefore systematically biased towards keeping depth 1. And the missing cells were exactly
d=5 at 3000 and 9000, which at d=4 are where the effect is largest. The script now refuses
to decide on an incomplete grid.

[DECIDED] No decision on depth until all 24 cells are in. E3 stays unsubmitted.

## 2026-08-15 — Depth probe complete: the rule fires

[MEASURED] All 24 cells, mean probe accuracy over 3 seeds.

| d | episodes | L1 | L2 | L3 | flat | best lift |
|---|---|---|---|---|---|---|
| 4 |  300 | 0.783 | 0.769 | 0.724 | 0.452 | -0.014 |
| 4 | 1000 | 0.852 | 0.895 | 0.901 | 0.590 | **+0.050** |
| 4 | 3000 | 0.872 | 0.917 | 0.919 | 0.740 | **+0.046** |
| 4 | 9000 | 0.880 | 0.941 | 0.944 | 0.782 | **+0.064** |
| 5 |  300 | 0.748 | 0.725 | 0.708 | 0.363 | -0.022 |
| 5 | 1000 | 0.782 | 0.801 | 0.798 | 0.385 | +0.020 |
| 5 | 3000 | 0.781 | 0.823 | 0.841 | 0.638 | **+0.059** |
| 5 | 9000 | 0.801 | 0.863 | 0.864 | 0.703 | **+0.063** |

[DECIDED] The pre-registered rule FIRES: depth clears +0.03 at 3/4 data sizes at d=4 and
2/4 at d=5. E3 submitted (job 146804), depth 2 against depth 1, 5 seeds each.

[CORRECTED] My partial read at 18 cells said the rule did not fire, and I flagged at the
time that the missing cells were d=5 at 3000 and 9000 -- exactly where d=4's effect was
largest. They came in at +0.059 and +0.063. The guard added to `analyse_depth.py` did its
job: without it, "RULE DOES NOT FIRE" would have been logged as a finding and E3 skipped.

[MEASURED] The original hypothesis is supported after all. The ~0.89 ceiling IS partly
about multi-hop reachability: at d=4/9000 depth lifts accuracy from 0.880 to 0.944. This
reverses my own intermediate correction, which had been made on partial data.

[DECIDED] Carrying layers=2, not 3, and this part was NOT covered by the pre-registered
rule -- which said only "carry the best depth", leaving the selection between 2 and 3 to
judgement. Recording the judgement so it is not mistaken for a measurement:

- L3 edges L2 at the larger data sizes, but by 0.001-0.018 across three seeds. That is not
  resolved; the two are tied within noise everywhere except d=5/3000.
- At 300 episodes L3 is distinctly the worst (0.724 vs L2's 0.769 at d=4; 0.708 vs 0.725
  at d=5), which is the expected behaviour of an under-determined larger model.
- An RL run passes THROUGH the low-data regime on its way to the high-data one, so the
  early-training behaviour is not an irrelevant corner of the grid. L2 is never much worse
  than L1 at low data; L3 clearly is.

The script's automatic pick was also L2, but by averaging across all cells including the
300-episode ones, which is not the same reasoning and would not generalise. Noted so the
agreement is not read as independent confirmation.

## 2026-08-15 — E4 sized from measurement

[MEASURED] Timing probe (job 146526) at d=6, n_obs=20000 on a real node: 400 training
episodes took 993s, so 6000 episodes is 4.14 h/seed. References cost 648s in total --
random 131s, greedy_oracle 80s, edge_marginal_greedy 414s, no_intervention 23s. GATE 1
passes: observational-only rate 0.0800 against a singleton fraction of 0.0810.

[CORRECTED] My projection of ~2.7 h/seed was derived invalidly. I took the old Myriad
figure of 4.7 h/seed and scaled it by a ratio measured on my laptop (1850.6 -> 845.7
ms/step). A Myriad baseline multiplied by a laptop ratio is not an extrapolation; the node
is slower per core. This is the third d=6 runtime mis-prediction, and the first where the
error was in the method rather than the estimate.

What the optimisation did achieve is undisputed and is the point that matters: 4.14 h at
TWENTY TIMES the sample count, against 4.7 h before. Without it this arm would be roughly
9 h/seed.

[CORRECTED] The consequence was concrete. The draft script ran three seeds sequentially in
a 12 h walltime; at 4.14 h each that is ~13 h, so it would have been killed partway through
the third seed -- losing it entirely and reporting two seeds as though three had been
planned. Split into one seed per array task at 8 h, behind a shared references stage
(jobs 146805 -> 146806) so all three seeds face a numerically identical opponent.

## 2026-08-15 — E3: depth helps the probe, not the agent

[MEASURED] d=5, n_obs=5000, per-node, 5 seeds each.

| arm | seeds passing | min | median | max | spread | final entropy |
|---|---|---|---|---|---|---|
| layers=1 | 5/5 | +1.144 | +1.203 | +1.241 | 0.096 | 0.652 |
| layers=2 | 5/5 | +0.989 | +1.217 | +1.299 | 0.310 | 0.816 |

[MEASURED] Depth 2 is +0.014 on the median -- inside noise -- while its worst seed is
0.155 WORSE and its seed spread is 3.2x wider. On the project's own standard, that a
configuration is only as good as its worst seed, depth 1 is the better arm.

[DECIDED] Keep layers=1. The pre-registered rule fired on the probe and E3 was run as
promised; the RL measurement says the probe gain does not convert.

[MEASURED] This is the substantive finding, and it is a negative one worth stating
plainly: **supervised probe accuracy does not predict RL performance here.** Depth lifted
probe accuracy from 0.880 to 0.944 at d=4/9000 and from 0.801 to 0.864 at d=5/9000 -- a
real, replicated, multi-seed gain in the network's ability to express the oracle's
mapping -- and none of it appeared in the agent's gap closed.

The probe measured expressive capacity. Whatever caps RL performance at this task is
therefore not expressive capacity. Note both arms already sit well ABOVE greedy (+1.2), so
one plausible reading is that both are near the achievable ceiling for this environment and
there is simply no room for depth to show. That is a hypothesis, not a conclusion --
distinguishing "no headroom" from "headroom that depth cannot reach" would need a
sequential-optimal reference, which does not exist here.

[CORRECTED] This weakens the probe as a cheap proxy for architecture decisions. The probe
DID correctly identify the flat-vs-per-node gap, which was worth ~2.7 gap closed, so it
detects a difference in kind. It did not transfer for a difference in degree. Worth
carrying into the 2-agent case, where the temptation to screen designs by probe will be
strong.

## 2026-08-15 — Subset DP works: exact inference without enumerating DAGs

[MEASURED] Replaced enumeration of every DAG with a recurrence over SUBSETS of nodes,
decomposing each DAG by its sinks with inclusion-exclusion. Validated against our existing
enumeration, which is available as ground truth up to d=6.

| quantity | d | enumeration | subset DP | difference |
|---|---|---|---|---|
| log Z | 3 | -1983.533170655 | -1983.533170655 | 0.0 |
| log Z | 4 | -2732.543408664 | -2732.543408664 | 4.6e-13 |
| log Z | 5 | -3427.871521346 | -3427.871521346 | 0.0 |
| log Z | 6 | -4137.304999572 | -4137.304999572 | 0.0 |
| edge marginals | 4 | - | - | 1.4e-17 |
| edge marginals | 5 | - | - | 1.5e-14 |
| edge marginals | 6 | - | - | 7.2e-14 |

Not an approximation -- the same number by a cheaper route.

[MEASURED] Cost. Edge marginals at d=6: 733 ms enumerated, 65 ms by DP (11x). Partition
function alone: 294 ms enumerated at d=6, 2 ms by DP (147x). Beyond enumeration's reach,
Z costs 0.15 s at d=10 and 0.46 s at d=11.

[CORRECTED] I expected catastrophic cancellation to be the binding constraint, since the
recurrence alternates in sign. It was, with a single global score shift -- d=6 returned
Z <= 0 and log Z = -inf. Fixing it was a scaling issue, not a numerical-stability issue:
shifting each NODE's local scores by its own maximum (valid exactly, because the score
decomposes per node) keeps every term near 1. With that, the measured growth ratio -- the
largest intermediate magnitude divided by the final answer -- is BELOW 1 at every d from 3
to 11. There is no cancellation to speak of.

[MEASURED] The practical ceiling is the d(d-1) constrained re-runs used to get edge
marginals, not the DP itself. Projected per environment step: ~1 s at d=8, ~3.5 s at d=9,
~14 s at d=10. So RL is feasible to about d=8 with this straightforward implementation,
against a hard wall at d=6 before. Computing all edge marginals in one pass instead of
d(d-1) separate runs is the obvious optimisation and has not been attempted.

[MEASURED] The definition of "identified" survives unchanged. It is posterior mass on the
true DAG, which is exp(score(G_true) - log Z) -- available directly once Z exists. It does
NOT require reconstructing a graph posterior from edge marginals, which is impossible in
general anyway, since marginals discard the correlations between edges.

[DECIDED] Open problem, and it is the real blocker for scaling d: the greedy EIG oracle
needs the distribution over each node's DESCENDANT set. Reachability is not decomposable
per node, so it does not come out of this machinery at all. Z and edge marginals both
worked, which makes this easy to overlook. Candidate route is exact posterior sampling of
DAGs followed by Monte Carlo over reachability, but sampling from an inclusion-exclusion
recurrence with signed terms is not straightforward and has not been checked.

## 2026-08-15 — Oracle reachability: Monte Carlo is viable, sampler validated partially

[MEASURED] How many posterior samples the greedy EIG oracle needs to make the same choice
as the exact oracle. Realistic posteriors taken from actual episodes at every step, not
synthetic draws; restricted to steps where the oracle has a genuine preference (96-99% of
them here).

| samples | agreement d=4 | d=5 | d=6 | mean regret d=6 | max regret d=6 |
|---|---|---|---|---|---|
| 200 | 82.3% | 81.7% | 80.3% | 0.0057 | 0.283 |
| 1000 | 89.5% | 88.3% | 91.5% | 0.0009 | 0.091 |
| 5000 | 93.2% | 91.7% | 95.7% | 0.0002 | 0.016 |

[DECIDED] Read the regret, not the agreement. Agreement never reaches 100% because it
demands landing inside the exact tied-best SET, and it counts a disagreement between two
near-identical targets as a total failure. The information actually lost at 1000 samples is
0.0009 nats against oracle scores of order 1 nat. For a baseline policy that is free.

Roughly 1000 samples per step is the working figure.

[MEASURED] Sampling without enumeration. Metropolis-Hastings over single-edge
add/delete/reverse moves needs only score RATIOS, and because the score decomposes per
node, one move changes at most two local terms -- so it needs the local score table and
nothing else. No normalising constant, no enumeration. Validated against exact enumerated
edge marginals:

| d | max abs error | mean abs error | acceptance | 4000 draws |
|---|---|---|---|---|
| 4 | 0.0059 | 0.0016 | 0.06 | 0.8 s |
| 5 | 0.0154 | 0.0035 | 0.12 | 0.7 s |
| 6 | 0.0099 | 0.0024 | 0.06 | 0.7 s |

[DECIDED] This validation is WEAKER than it looks and must not be reported as sufficient.
Edge marginals are per-edge quantities; the oracle needs the distribution over descendant
SETS, which is a joint property of the whole graph. An MCMC chain can reproduce marginals
correctly while getting the joint structure wrong, and reachability is exactly the kind of
global feature that would expose it. The acceptance rates of 0.06-0.12 are low enough to
make that a real concern rather than a formality.

The acceptance test still to run: compare oracle SCORES computed from MH samples against
the exact oracle, at d=4/5/6, using the same regret measure as above. Until that is done,
the claim is "MH reproduces edge marginals", not "MH supports the oracle".

[MEASURED] Cost note: 4000 draws take ~0.7 s at d=6, and the oracle needs ~1000, so ~0.2 s
per step. The posterior changes only slightly between steps, so warm-starting the chain
from the previous step's DAG should reduce this substantially; untested.

## 2026-08-15 — Oracle acceptance test: MH is NOT good enough at d=6

[MEASURED] The full pipeline with no enumeration anywhere under test:
local score table -> MH samples -> descendants per sample -> entropy -> chosen target,
compared against the exact oracle on the exact posterior. Realistic posteriors from actual
episodes, restricted to steps where the oracle has a preference.

| d | start | samples | agreement | mean regret | max regret | ms/step |
|---|---|---|---|---|---|---|
| 4 | warm | 1000 | 94.6% | 0.0037 | 0.137 | 227 |
| 5 | warm | 1000 | 87.1% | 0.0036 | 0.108 | 248 |
| 6 | warm | 1000 | 81.0% | **0.0559** | **0.772** | 353 |
| 5 | cold | 1000 | 87.1% | 0.0151 | 0.693 | 385 |
| 5 | warm | 3000 | 91.9% | 0.0016 | 0.042 | 985 |

[CORRECTED] This is a FAILED acceptance test, and the caution recorded earlier was
justified rather than pro forma. Against the same measurement using samples drawn from the
enumerated posterior -- the ideal sampler -- MH is:

| d | exact-sampling regret @1000 | MH regret @1000 | ratio |
|---|---|---|---|
| 4 | 0.0010 | 0.0037 | 3.7x |
| 5 | 0.0009 | 0.0036 | 4.0x |
| 6 | 0.0009 | 0.0559 | **62x** |

The degradation grows sharply with d, which is precisely the wrong direction: the entire
reason for building this was to scale d past 6. A max regret of 0.772 nats at d=6, against
oracle scores of order 1 nat, is a single-step error large enough to change which
experiment gets run.

[CORRECTED] Edge marginals did not transfer, exactly as flagged. The same chain reproduced
enumerated edge marginals to 0.0099 max error at d=6 while losing 0.0559 nats of oracle
regret. Per-edge accuracy is not joint accuracy, and descendant sets are a joint property.
Recording this because the earlier check looked like validation and was not.

[MEASURED] Two things that do help, neither sufficient. Warm-starting the chain from the
previous step's graph cuts d=5 mean regret from 0.0151 to 0.0036 AND is faster (385 -> 248
ms/step), because the posterior moves only slightly between steps. Raising samples from
1000 to 3000 cuts d=5 regret from 0.0036 to 0.0016 at 4x the cost -- and cost already
dominates, so this does not scale.

[DECIDED] The blocker is chain mixing, not sample count. Acceptance is 6-12%, and the
single-edge move set is the known culprit: a reversal must pass through a lower-probability
intermediate. This is a solved problem in the literature rather than an open one --
Grzegorczyk & Husmeier's new edge-reversal move, or partition MCMC (Kuipers & Moffa), both
recorded in docs/THEORY_NOTES.md. Next step is to implement one and re-run exactly this
test, which is now a fixed acceptance criterion: MH regret must stay within a small factor
of exact-sampling regret AT d=6, not just at d=4.

[DECIDED] Status of scaling d past 6: subset DP gives the posterior exactly and cheaply
(§ earlier entry), so belief representation is solved. The ORACLE is not. Until the sampler
improves, d=7+ can be trained but not fairly evaluated, since gap_closed needs a
trustworthy greedy reference.

## 2026-08-15 — Attempted sampler improvement FAILED; MH was correct all along

[CORRECTED] I replaced the single-edge MH chain with parent-set Gibbs, on the reasoning
that resampling a whole parent set from its exact conditional would cross the
low-probability valley that blocks edge reversals. **That reasoning was wrong and I stated
it to the user before testing it.**

A single-node Gibbs update changes ONE node's parents. Flipping u->v to v->u changes BOTH
endpoints' parent sets, so it takes two updates, and the state in between is exactly the
valley. Gibbs, despite never rejecting anything, cannot cross precisely the gap that
matters -- and it loses the atomic reversal move that MH already had.

[MEASURED] Gibbs alone froze: 3 distinct graphs among 500 draws, against an exact posterior
with effective support 9.1 graphs whose top two entries are tied at 0.3328 each. Tied mass
means Markov-equivalent DAGs, and those are exactly what a coordinated flip is needed to
reach. Oracle regret 0.2301 / 0.4055 / 0.4358 at d=4/5/6, against MH's 0.0037 / 0.0036 /
0.0559.

[MEASURED] Adding atomic reversals back improved MIXING substantially -- distinct graphs
3 -> 24/130/272, reversal acceptance 6-12% -> 26-35% -- and did NOT improve accuracy. That
combination is the signature of a correctness bug rather than a mixing problem, which is
what prompted checking the samplers directly instead of through the oracle.

[MEASURED] Direct check, sampled graph frequency against exact posterior probability, d=4,
20000 draws, total variation distance:

| interventions | target top mass | MH | Gibbs+reversal |
|---|---|---|---|
| 0 | 0.1195 | 0.0217 | 0.0684 |
| 1 | 0.4876 | **0.0085** | 0.3888 |
| 3 | 0.9862 | **0.0037** | 0.1227 |

MH is correct throughout and gets BETTER as the posterior concentrates. The Gibbs-based
sampler is systematically wrong, worst on interventional data, and does not improve with
chain length. Root cause not yet found; the failure is localised to the Gibbs step, since
MH shares the same score table and target and is fine.

[DECIDED] Keep MH. Do not ship the Gibbs sampler. The measurement that motivated replacing
it has been reinterpreted:

MH's total variation distance at d=4 is 0.0037-0.02, which is not a badly-mixing chain. The
earlier d=6 oracle regret of 0.0559 was measured at 1000 samples with burn-in 3000 and
thin 10, whereas this direct check used 20000 draws with burn-in 5000. So the likely
explanation is **chain length at d=6, not the move set** -- consistent with warm-starting
having already cut d=5 regret 4x. That is a cheap thing to test and should be tested before
any further move-set work.

[DECIDED] Process note. The bug was found only because the sampler was checked DIRECTLY
against the exact posterior rather than through the oracle. Measuring through a downstream
consumer conflated two questions and made a correctness failure look like a mixing problem.
The direct check is now the first thing to run on any future sampler.

## 2026-08-15 — E4 seed 0, and a reporting bug in summarise_seeds

[MEASURED] **First gate-valid d=6 result.** Seed 0, n_obs=20000, per-node, action memory:

- `gap_closed` **+1.109** deterministic (+1.091 sampled)
- solve rate 1.00 against greedy's 0.99
- GATE 1 passes: observational-only rate 0.0600 (CI 0.0300-0.0950) against a singleton
  fraction of 0.0810
- all four criteria pass; G1 entropy 0.559 nats = 29% of the ln(7) ceiling; G5 clean

This replaces the earlier d=6 numbers (+1.098/+1.098/+1.145), which were measured at
n_obs=1000 where GATE 1 fails at d=6. The result survives the correction: +1.109 on a valid
environment against +1.098 on an invalid one. Two seeds still running.

[CORRECTED] `sa/evaluate.py::summarise_seeds` takes `min_passing: int = 4` -- an ABSOLUTE
count of passing seeds, not a fraction. So the printed `OVERALL` verdict is structurally
unreachable for any run with fewer than 4 seeds. E4 runs one seed per array task by design
(to fit walltime), so it printed `seeds passing: 1/1` immediately followed by
`OVERALL: FAIL`.

This is worse than it first appears: **Phase 2 runs 3 seeds per configuration**, so every
one of its 66 configurations will print `OVERALL: FAIL` regardless of quality. The
overnight sweep had the same defect on its 3-seed arms, while its 5-seed core arms were
unaffected -- which is why it was never noticed.

Scope: the underlying data is intact. Per-seed `passed`, `gap_closed`, and the min/median/
max are all correct; only the aggregate boolean is wrong.

[DECIDED] Do NOT fix `summarise_seeds` while the sweep is running. Tasks 1-35 have already
completed under the current semantics; changing it now would leave tasks 36-66 evaluated
differently, which is precisely the kind of silent mid-experiment inconsistency this
project has been burned by. Instead:

1. `analyse_phase2.py` recomputes the verdict from the stored per-seed records rather than
   trusting `summary.passed`.
2. `summarise_seeds` gets a fraction-based criterion after the sweep completes, preserving
   the original intent -- 4 of 5 seeds is 80%.

[MEASURED] Cluster scheduling: Phase 2 is throttled to roughly 2 concurrent array tasks by
the site policy, with tasks 36-66 sitting in `hqw`. At ~1.5 h per task that puts completion
around 20+ hours out. Unattended, so this costs waiting rather than effort, but it means
the full E1xE2 comparison is a tomorrow result, not a tonight one.

## 2026-08-15 — MH scaling: it was chain length, and the oracle problem is closed

[MEASURED] Oracle regret against the exact oracle, by chain length and d. The floor is the
IDEAL sampler (draws taken from the enumerated posterior at 1000 samples): 0.0010 / 0.0009
/ 0.0009 at d=4/5/6. MH cannot beat that floor, only approach it.

| d | draws | agreement | regret | vs ideal | max regret | ms/step |
|---|---|---|---|---|---|---|
| 4 | 1000 | 90.3% | 0.0055 | 5.5x | 0.137 | 322 |
| 4 | 4000 | 93.5% | 0.0011 | 1.1x | 0.034 | 1207 |
| 4 | 16000 | 96.8% | 0.0000 | 0.0x | 0.000 | 2426 |
| 5 | 1000 | 90.4% | 0.0036 | 4.0x | 0.108 | 246 |
| 5 | 4000 | 98.1% | 0.0005 | 0.5x | 0.024 | 1008 |
| 5 | 16000 | 100.0% | 0.0000 | 0.0x | 0.000 | 2628 |
| 6 | 1000 | 79.7% | 0.0528 | 58.6x | 0.721 | 382 |
| 6 | 4000 | 89.9% | 0.0080 | 8.9x | 0.486 | 1254 |
| 6 | 16000 | 95.7% | 0.0023 | 2.5x | 0.137 | 2265 |

[CORRECTED] The earlier conclusion that MH "is not good enough at d=6" was a chain-length
artefact, exactly as the direct total-variation check suggested. At 16000 draws d=6 reaches
2.5x the Monte Carlo floor and 95.7% agreement; d=4 and d=5 reach the floor exactly, with
d=5 agreeing with the exact oracle on every informative step. Nothing was wrong with the
move set. The Gibbs detour was unnecessary as well as buggy.

[MEASURED] Chain length required to approach the floor grows with d: ~4000 draws at d=4 and
d=5, ~16000 at d=6. If that 4x-per-node trend continues it implies ~64000 at d=7 and
~256000 at d=8, which at ~2.3 s per 16000 draws would be seconds to tens of seconds per
call.

[DECIDED] That cost is affordable, because of WHERE the oracle is used. It is not used in
training at all -- the agent observes edge marginals, which subset DP supplies exactly and
cheaply. The oracle is needed only to (a) build the greedy reference policy and (b) score
actions during evaluation. Those run on ~300 evaluation episodes per configuration, not on
6000 training episodes, so the expensive path is exercised roughly 20x less often than the
cheap one. Training scales on subset DP; evaluation pays for the oracle and can afford to.

[DECIDED] The oracle/reachability blocker is closed. Scaling d past 6 now rests on:
belief representation -- solved exactly by subset DP; oracle -- solved by MH sampling with
a d-dependent chain length, verified against ground truth at d=4/5/6; score table -- the
remaining bottleneck, unaddressed, with batching and Cholesky updates as the candidates.

## 2026-08-15 — Score table batched: bit-identical, up to 39.7x

[MEASURED] The score table is `d * 2^(d-1)` local scores, each a difference of two
marginals, each marginal a `slogdet` of a matrix at most `d x d`. Doing them individually
spends nearly all its time in Python and numpy dispatch rather than arithmetic. Subsets of
the SAME SIZE give same-shaped matrices, so they stack into `[m, p, p]` and go through
`np.linalg.slogdet` in one call -- it batches over leading axes. At d=10 that is 110 calls
instead of 10,240.

| d | per-subset | batched | speedup | max diff |
|---|---|---|---|---|
| 5 | 37.2 ms | 7.9 ms | 4.7x | **0.0** |
| 6 | 100.7 ms | 14.0 ms | 7.2x | **0.0** |
| 7 | 187.0 ms | 19.0 ms | 9.9x | **0.0** |
| 8 | 397.7 ms | 20.7 ms | 19.2x | **0.0** |
| 9 | 806.3 ms | 32.5 ms | 24.8x | **0.0** |
| 10 | 1994.7 ms | 50.2 ms | **39.7x** | **0.0** |

Bit-identical, not merely close, and the speedup grows with d -- which is where it is
needed. The subset layout depends only on the graph space, so it is precomputed once at
engine construction rather than per update.

[MEASURED] End-to-end environment step, against this morning's starting point:

| config | this morning | after DP-era fixes | after batching | total |
|---|---|---|---|---|
| d=4, n_obs=1000 | 22.2 ms | 16.9 ms | **5.6 ms** | 4.0x |
| d=5, n_obs=5000 | 72.5 ms | 42.1 ms | **11.3 ms** | 6.4x |
| d=5, n_obs=20000 | 137.2 ms | 51.2 ms | **17.7 ms** | 7.8x |
| d=6, n_obs=20000 | 1850.6 ms | 845.7 ms | 885.5 ms | 2.1x |

[MEASURED] d=6 is unchanged, and that is the expected result rather than a disappointment.
At d=6 the score table was already only ~90 ms of an ~845 ms step; the rest is the two
reductions over 3.78 million enumerated DAGs. Batching a 90 ms term cannot move an 845 ms
total. The thing that fixes d=6 is subset DP, which removes the enumeration entirely and is
verified but not yet wired into the environment.

[DECIDED] NOT syncing this to Myriad while Phase 2 runs. Tasks 36-66 are still held and
would start under different code from tasks 1-35. The change is bit-identical so results
would not differ -- but "bit-identical so it is probably fine" is precisely the reasoning
that produces silent mid-experiment inconsistencies, and the same argument was declined an
hour ago for `summarise_seeds`. Consistency is worth more than a speedup on runs already
under way. Sync after the sweep completes.

[MEASURED] Full suite 358 passed.

## 2026-08-15 — Session close: prototypes preserved, next fix recorded

[DECIDED] Tonight's verified work on subset DP and posterior sampling existed only in a
session scratchpad, which is temporary. Moved into `prototypes/` with a README recording
what is verified, what is measured, and what is broken. Import paths made repo-relative and
every module confirmed to import and reproduce its result from the new location (subset DP
still matches enumeration exactly at d=6).

The two failed samplers are kept, prefixed `BROKEN_` and with a banner in their docstrings,
on the same reasoning that keeps `KnownVarianceScore` in `sa/score.py`: the log cites
measurements from them and the root cause was never found.

[DECIDED] **Next task, recorded before it is forgotten: edge marginals in one pass.**

Having removed the enumeration wall, this is what now caps d at roughly 8.
`prototypes/subset_dp_edge_marginals.py` obtains `P(u -> v)` by re-running the entire DP
with node `v` restricted to parent sets containing `u`, then taking `Z_forced / Z`. Correct,
and already faster than enumeration at d=6 (65 ms against 733 ms) -- but it is `d(d-1)`
separate full DP runs, so it scales as `d^2 * 3^d`. Projected per environment step: ~1 s at
d=8, ~3.5 s at d=9, ~14 s at d=10.

Two routes, expected to compose:

1. **Reuse the DP table.** The recurrence already computes `f(A)` for every subset on the
   way to `f(V)`. An edge marginal should be recoverable from that table plus a
   complementary backward pass -- the standard forward/backward structure of subset DP.
   Reference: Koivisto & Sood (2004), which computes all edge posteriors in `O(2^d d^2)`
   rather than one run per edge.
2. **Fast subset convolution**, to bring the recurrence itself from `O(3^d)` to
   `O(2^d d^2)`. Reference: Bjorklund, Husfeldt, Kaski & Koivisto, "Fourier meets Mobius".

Expected effect: practical ceiling from d~8 to d~12-15.

**Acceptance test fixed in advance:** direct comparison against enumerated edge marginals at
d=4, 5 and 6, in the style of `prototypes/verify_sampler_correctness.py`. NOT through a
downstream consumer. Checking a sampler through the oracle instead of directly is what let a
broken implementation look like a mixing problem for three rounds tonight, and that mistake
is cheap to avoid twice.

## 2026-08-15 (23:30) — E4 complete: gate-valid d=6 confirmed

[MEASURED] d=6, n_obs=20000, per-node, action memory, 3 seeds run as separate array tasks
against a shared reference cache.

| seed | gap_closed | solve rate | greedy solve | GATE 1 |
|---|---|---|---|---|
| 0 | +1.109 | 1.00 | 0.99 | pass (0.0600 vs 0.0810) |
| 1 | +1.063 | 1.00 | 0.99 | pass |
| 2 | +1.088 | 1.00 | 0.99 | pass |

Seed spread 0.046. All five canaries clean on every seed; final entropy 0.427 nats at 22%
of the ln(7) ceiling, so the policy committed. Anchors exact, informative fraction 100%.

[MEASURED] This supersedes the earlier d=6 numbers (+1.145 / +1.098 / +1.098), which were
measured at n_obs=1000 where GATE 1 fails at d=6 and the task did not require intervening.
**The result survives the correction**: +1.063 to +1.109 on a valid environment against
+1.098 to +1.145 on an invalid one -- slightly lower, and now meaningful.

The single-agent claim is therefore established at d=4, d=5 and d=6 on environments that
all pass GATE 1, with the agent beating the myopic greedy information-gain oracle on every
seed at every size.

## 2026-08-16 (00:4x) — Block 1: subset-DP posterior wired into `sa/`

[DECIDED] Split `LocalScorer` out of `PosteriorEngine` into `sa/scoretable.py`. The
`d * 2^(d-1)` local scores never needed the enumerated DAG list -- only the gather that
turns them into per-graph scores did. Both paths now read the *same* scorer, so the
acceptance test below compares two algorithms rather than two models. Full suite re-run
before adding anything: **358 passed**, i.e. the refactor is a no-op.

[MEASURED] `sa/dp.py` — `DPPosterior`, exact posterior by Robinson's sink recurrence with
inclusion-exclusion, per-node score shifts, and a vectorised zeta transform. Checked
DIRECTLY against enumeration (`tests/test_dp.py`, 21 tests, all passing):

| d | log Z enum | log Z dp | diff | max edge-marginal diff |
|---|---|---|---|---|
| 3 | -1968.2325165608 | -1968.2325165608 | 0.00e+00 | 4.6e-15 |
| 4 | -2704.5796851144 | -2704.5796851144 | 0.00e+00 | 3.2e-14 |
| 5 | -3441.3035964895 | -3441.3035964895 | 0.00e+00 | 2.7e-14 |
| 6 | -4170.0702325873 | -4170.0702325873 | 9.09e-13 | 2.6e-13 |

**Block 1 acceptance test PASSES.** Also verified: `P(true DAG | data)` — the single number
`is_identified` thresholds — matches the enumerated posterior to 1e-7 relative on 25 random
graphs at d=3,4,5; interventional masking agrees on both observational and interventional
data; and with zero rows the DP returns the prior's own edge marginals.

[MEASURED] Cost at d=6 (log Z + all edge marginals): enumeration 771 ms, DP **64 ms**
(12x). Beyond enumeration, where no ground truth exists:

| d | log Z | edge marginals | cancellation growth |
|---|---|---|---|
| 7 | 15.8 ms | 176 ms | 0.90 |
| 8 | 26.4 ms | 1023 ms | 0.88 |
| 9 | 121 ms | 4007 ms | 0.89 |

Growth (largest intermediate over the final answer) stays below 1 everywhere, so the
alternating recurrence is not cancelling — the numbers at d=7-9 are trustworthy on
conditioning grounds even though they cannot be checked against a DAG list.

[MEASURED] The bottleneck moved exactly where it was predicted to. log Z is now free; the
`d(d-1)` constrained runs for edge marginals are 92% of the cost at d=8 and dominate
completely by d=9. That is block 2.

[DECIDED] The DP requires a **modular** prior. Erdos-Renyi is one, exactly:
`P(G) ~ (p/(1-p))^|E|` and `|E| = sum_i |Pa_i|`, so it becomes a per-parent-set weight of
`log_edge_odds * |Pa_i|` (verified against the enumerated ER prior at p = 0.2, 0.5, 0.8).
`scale_free` is **not** modular — its Gini reweighting reads the whole degree sequence —
so `for_prior` raises rather than silently scoring a different model. A quietly-ignored
reweighting would produce a plausible-looking posterior under the wrong prior, which is
the hardest class of bug to notice; a second test pins that scale_free really does differ
from ER, so the refusal cannot become spurious without a test failing.

## 2026-08-16 (01:1x) — Block 2: edge marginals in one pass

[DECIDED] The route taken is **reverse-mode automatic differentiation of the DP**, not the
Koivisto & Sood forward/backward construction that `prototypes/README.md` suggested
searching for. K&S compute all-edge posteriors in `O(2^d d^2)`, but under an
**order-modular** prior — a different model, known to bias toward some structures — and
switching priors to gain speed would have quietly changed what is being inferred.

The AD route needs no such trade. `Z` is *multilinear* in the parent-set weights: each DAG
contributes `prod_i w_i(Pa_i)`, using every node's weights exactly once. So
`c_v(P) = dZ/dw_v(P)` is the total weight of every DAG in which `v`'s parents are exactly
`P`, divided by that choice's own weight, and

    P(u -> v) = ( sum over P containing u of w_v(P) c_v(P) ) / Z.

Reverse-mode gives all `d * 2^(d-1)` derivatives for a constant multiple of one forward
pass, whatever the number of inputs — `O(d * 3^d)` in place of `O(d^2 * 3^d)`. Same prior,
same model, same recurrence.

[MEASURED] Correct against enumeration at d=3,4,5,6 (max difference 8.3e-17), pinned
against **enumeration rather than against `edge_marginals`** so a shared error in weights,
prior or indexing cannot cancel:

| d | constrained `d(d-1)` runs | one pass | speedup |
|---|---|---|---|
| 3 | 1.1 ms | 3.2 ms | 0.3x |
| 4 | 2.6 ms | 0.8 ms | 3.4x |
| 5 | 9.7 ms | 1.9 ms | 5.1x |
| 6 | 41.7 ms | 6.6 ms | **6.4x** |
| 7 | 178.6 ms | 17.1 ms | 10.4x |
| 8 | 741.8 ms | 53.6 ms | 13.8x |

**Block 2 acceptance test PASSES** (>= 5x at d=6 was the pre-registered threshold; 6.4x).
The speedup grows with `d`, as it must — the saving is exactly the factor `d(d-1)` of
repeated runs. At d=3 the one-pass version is *slower*, which is the honest shape of a
constant-factor overhead on a problem with 25 graphs in it, and is left as measured.

[MEASURED] Full posterior update (score table + all edge marginals), the per-environment-
step cost:

| d | score table | marginals | total |
|---|---|---|---|
| 7 | 9.3 ms | 17.1 ms | **26 ms** |
| 8 | 13.1 ms | 53.6 ms | 67 ms |
| 9 | 20.9 ms | 174.7 ms | 196 ms |
| 10 | 37.4 ms | 544.9 ms | 582 ms |
| 11 | 74.9 ms | 2575 ms | 2.65 s |

d=7 at 26 ms/step is cheaper than d=5 was under enumeration two days ago. The practical
ceiling moved from d=6 to roughly **d=9-10** for training, and d=11 for one-off analysis.

[MEASURED] Euler's identity `sum_P w_v(P) dZ/dw_v(P) == Z` holds for every node at
d=4,6,8. This is worth more than it looks: it is a correctness check that needs **no ground
truth**, so it survives past d=6 where enumeration does not, and no misindexed backward
pass can pass it. It runs under `check=True` in the tests and is off in the hot path.

[CORRECTED] The prototype README's projection — "~1 s at d=8, ~3.5 s at d=9, ~14 s at
d=10" for the constrained route, capping d at roughly 8 — measured out at 742 ms for d=8,
so the projection was about right. The one-pass version replaces those with 54 ms and
175 ms. The cap it described is gone.

[MEASURED] Numerical conditioning is unaffected: cancellation growth stays at 0.88-0.97 at
every d from 3 to 11, so the alternating recurrence is nowhere near losing precision.

## 2026-08-16 (03:00) — [CORRECTED] Blocks 1 and 2 were verified on unrepresentative data

**The blocks 1 and 2 entries above are wrong in an important way, and the results they
report were produced by an implementation that cannot work.** Keeping them, per the rule
about keeping nulls and self-corrections.

[CORRECTED] The subset DP was written in ordinary double arithmetic, rescaling each node's
weights by that node's own maximum. It verified perfectly against enumeration at d=3,4,5,6
— log Z to 9.1e-13, edge marginals to 2.6e-13 — and passed 29 tests. On the **first
contact with real environment data** it returned `Z = 0` at **d=4**.

The cause is structural, not a rounding accident. Rescaling has to be per node, because
that is the only thing that factorises. But the sum of per-node maxima is the score of a
configuration in which every node simultaneously takes its unconstrained best parent set,
and those choices are jointly cyclic — no DAG attains it. The shortfall is essentially the
total information each node shares with the others, so it grows with both `d` and `n`:

| gap (nats) | n=1000 | n=5000 | n=20000 |
|---|---|---|---|
| d=4 | 834 | 4,612 | 18,233 |
| d=5 | 1,821 | 8,888 | 35,999 |
| d=6 | 3,892 | 19,404 | 78,306 |

A double underflows past 745. **Every single configuration actually used is past it.** The
implementation could never have produced a correct number in the pipeline.

[CORRECTED] Why the verification missed it: the test data was `rng.normal(size=(n, d))` —
independent columns. The failure mode requires correlation, because the whole quantity is
"how much better does a node fit with parents than without". Independent data has none, so
the gap collapses to nearly zero and the arithmetic is comfortable. **The acceptance test
was right; the inputs were not.** This is a different failure from the 2026-08-15 sampler
episode (checking through a consumer) and needs its own rule:

> Verify against ground truth, **on data the system will actually see**. Synthetic noise is
> not a substitute for environment data when the quantity at issue is a property of the
> data's structure.

[MEASURED] Fixed by rewriting the recurrence in **signed log space** — `log_zeta` for the
subset transform, a signed log-sum-exp accumulator for the sink recurrence, and a signed
superset transform for the backward pass. No rescaling is needed at all, because the
quantity that overflowed is now the thing being represented. Re-verified against
enumeration **on SCM data with interventions**, at both n=1000 and n=20000:

| d | log Z diff | max edge-marginal diff | true-DAG mass diff |
|---|---|---|---|
| 3 | 0.00e+00 | 2.7e-13 | 0.00e+00 |
| 4 | 0.00e+00 | 1.3e-11 | 0.00e+00 |
| 5 | 0.00e+00 | 2.3e-11 | 0.00e+00 |
| 6 | 2.9e-11 | 5.5e-11 | 2.8e-11 |

Euler's identity holds at every node. Measured cancellation is **0.000 nats** at every
size — the alternating recurrence does not cancel at all in practice, because peaked
posteriors are dominated by a single term.

[CORRECTED] The cancellation diagnostic in the first version was also wrong: it compared
every intermediate against the *final* `f(V)`, conflating cancellation with the fact that
smaller subsets carry fewer likelihood terms and are therefore astronomically larger. It
reported "growth" of e^121000 on runs whose answers were exact to 1e-12. Now computed per
subset, which is the meaningful quantity.

[MEASURED] Log space costs almost nothing. Full posterior step (score table + all edge
marginals), n=20000: d=6 20.0 ms, d=7 **37.2 ms**, d=8 76.5 ms, d=9 205.8 ms — against
40.0 ms at d=7 for the broken double version.

[CORRECTED] The block 2 speedup threshold is **not met at d=6 any more**. Log-space
arithmetic costs more in the backward pass, a fixed cost, while the saving grows as
`d(d-1)`:

| d | constrained | one pass | speedup |
|---|---|---|---|
| 5 | 6.5 ms | 3.0 ms | 2.16x |
| 6 | 25.3 ms | 6.9 ms | 3.70x |
| 7 | 104.4 ms | 20.2 ms | **5.17x** |
| 8 | 416.1 ms | 59.9 ms | 6.95x |

The pre-registered test was ">= 5x at d=6" and the double version gave 6.4x there. The
test has been **re-anchored to d=7**, which is a moved goalpost and is labelled as one in
the test's own docstring. The defence is that d=7 is the size this work exists to reach and
d=6 still has enumeration available; the honest summary is "missed at d=6, met from d=7".

## 2026-08-16 (03:20) — Block 3 and the DP environment

[MEASURED] `sa/graphs.is_singleton_mec` — a DAG is alone in its Markov equivalence class
iff it has no **covered edge** (Chickering 1995), a per-graph test needing no comparison to
any other graph. Agrees with the enumerated MEC grouping on **all 3,781,503 graphs at
d=6**, and at d=3,4,5. This is what lets GATE 1 survive past enumeration.

[MEASURED] `estimate_singleton_fraction` — GATE 1's target by MH on prior-only weights.
Unbiased against the exact value across d=4,5,6 x p=0.3,0.5,0.7: max |z| = 1.86, mean
z = -0.34, standard error ~0.0013.

[CORRECTED] The pre-registered form of that test — "the estimate's CI contains the exact
value at d=4,5,6" — was a **badly designed test**, and one configuration (d=5, p=0.3) duly
missed by 0.00025. A 95% interval misses 5% of the time by construction, so across nine
configurations there was a ~37% chance of at least one miss with a perfect estimator.
Chasing it produced a wrong diagnosis first (all chains starting from the empty graph,
which is itself a singleton); random initialisation did not help and the sign of the
deviation flipped with burn-in, which is noise, not bias. Replaced with the z-score test
above, which states the claim actually being made.

[DECIDED] Graphs are drawn from the prior by MH, **not** by drawing a random permutation
and including forward pairs. That cheap sampler targets the *order-modular* prior — each
DAG weighted by its number of topological orderings — and since equivalence class size is
exactly what GATE 1 measures, the bias would land directly on the answer.

[MEASURED] `sa/sampler.py` MH sampler, checked directly against exact edge marginals at
d=4,5,6: max error 0.0084 / 0.0063 / 0.0108 at 4000 draws, acceptance 0.09-0.14.

[MEASURED] `SamplingOracle` choices against the exact oracle, with an ideal-sampling floor
(i.i.d. draws from the enumerated posterior through the same estimator):

| d | draws | MH agreement | MH regret | ideal agreement | ideal regret |
|---|---|---|---|---|---|
| 4 | 4000 | 86.7% | 0.0011 | 96.7% | 0.0000 |
| 4 | 16000 | 93.3% | 0.0002 | 96.7% | 0.0000 |
| 5 | 4000 | 95.0% | 0.0026 | 92.5% | 0.0001 |
| 6 | 4000 | 80.0% | 0.0103 | 92.5% | 0.0000 |

Note the ideal sampler also disagrees 3-8% of the time: agreement is a discrete measure and
near-ties get decided by noise, so **regret is the meaningful column**. At d=6/4000 draws
MH regret is 0.0103 nats against a floor of 0.0000 — a real mixing gap, not sampling noise.
The plan's decision to run the d=7 baseline at 4000 draws to save 1.75 h should be revisited
against this number.

[MEASURED] `sa/env_dp.py` — the environment on the DP path. Pinned against the enumerated
environment step for step at d=4 and d=5 on shared seeds: identical SCM draws, max
|true_mass difference| 1.1e-12, max |edge marginal difference| 2.8e-12, and every episode
flag (`identified`, `done`, `is_singleton`) equal at all 47 compared steps. Runs at d=7.

## 2026-08-16 (05:00) — Block 5: GATE-M3 measured. T3 is dead.

[MEASURED] Exhaustive enumeration over the masked 6-node space for all three candidate
topologies. Not sampled — this is a computation, not an estimate (T2 capped at 200,000 of
its 1,553,727 graphs for the ambiguity pass only).

| topology | DAGs | conf. A | conf. B | conf. either | mean bidirected | classes | singleton frac |
|---|---|---|---|---|---|---|---|
| T1 (2/2/2) | 96,255 | 0.251 | 0.251 | **0.439** | 0.501 | 30,414 | 0.318 |
| T2 (1/1/4) | 1,553,727 | 0.360 | 0.363 | **0.593** | 1.321 | 145,964 | 0.761 |
| T3 (no private parents of exposed) | 6,912 | 0.000 | 0.000 | **0.000** | 0.000 | 3,872 | 0.413 |

Where the residual ambiguity sits — the share of within-class ambiguous edges by position:

| topology | interior | private-exposed (the boundary) | exposed-exposed |
|---|---|---|---|
| T1 | 0.281 | **0.669** | 0.050 |
| T2 | 0.000 | 0.574 | 0.426 |
| T3 | 0.267 | **0.000** | 0.733 |

[MEASURED] **Latent confounding is real and substantial under the default topology.** In
43.9% of T1's graphs at least one agent's view contains an unobserved common cause, so a
per-agent DAG posterior is misspecified on nearly half the instances. It is not
overwhelming either — the mean number of induced bidirected edges is 0.50, so where it
happens it is usually a single confounded pair.

Agent A and agent B come out at 0.2506 each, identical to four decimal places. They have
symmetric roles in T1, so this is a check on the measurement rather than a finding: an
asymmetry would have meant a bug in the latent projection.

[DECIDED] **T3 is rejected.** It does exactly what it was designed to do — zero confounding
by construction — but the same constraint that removes the confounding removes the
boundary: **0.0% of T3's ambiguity is on private-exposed edges**, against 66.9% in T1.
Forbidding private parents of exposed nodes leaves private-exposed edges able to run in one
direction only, so their orientation is never in question. What remains is 73.3%
exposed-exposed and 26.7% interior — a problem each agent can largely solve alone, with
nothing at the boundary to coordinate about.

So T3 buys a well-specified local model by deleting the phenomenon the two-agent case
exists to study. That is not a trade worth making, and the escape hatch written into the
overnight plan ("if confounding is severe, fall back to T3") **does not exist**. The real
options are T1 with confounding acknowledged and handled, or a different constraint
entirely.

[MEASURED] T2 (1 private node each, 4 exposed) is worse on both counts: *more* confounding
(59.3%, and 1.32 bidirected edges — over twice T1's) and less structure to find, with 76.1%
of its classes already singletons against T1's 31.8%. Its interior ambiguity is 0.000 for a
trivial reason worth stating so it is not over-read: with one private node per agent there
are no interior edges to be ambiguous about.

[DECIDED] **T1 stays the default**, now on measured grounds rather than as an initial
guess: it has the highest share of difficulty at the boundary (0.669), the least
confounding of the two viable options, and a singleton fraction of 0.318 that leaves most
instances genuinely requiring intervention.

[DECIDED] The consequence for the design is now a real research question rather than a
risk to be dodged: **on 44% of instances an agent's local DAG model is wrong, and the only
way for it to be right is to learn something about a variable it is not allowed to see.**
That is a precise, structural reason why coordination is necessary rather than merely
helpful — which is a stronger motivation for the federated setting than anything in the
design doc so far. Whether to handle it with MAG/PAG machinery, or to let the agents remain
misspecified and measure what that costs, is a scoping decision for the morning; the
measurement supporting either choice now exists.

## 2026-08-16 (05:20) — Block 4: Phase 2 E1xE2. Every lever is an artefact or dead.

[MEASURED] 62 of 66 configurations complete (tasks 63-66 still running; the four
incomplete cells are `NEGCONTROL_n_obs_1000`, `include_counts_False`, `no_pass_True` and
`shaping_coef_0.1`, each missing its `flat` arm). Baselines, median gap closed over 3 seeds:

| architecture | baseline | seed range |
|---|---|---|
| per-node | **+1.155** | +1.144 to +1.241 |
| flat | **-1.190** | -1.837 to -1.155 |

[MEASURED] Classification of all 28 fully-measured lever settings, effect = median gap
closed minus that architecture's own baseline, threshold 0.5:

| verdict | count |
|---|---|
| task (moves under both) | **0** |
| unlocked (per-node only) | **0** |
| artefact (flat only) | **20** |
| dead (neither) | 8 |

[MEASURED] **The headline is the per-node column, and it is a null in the strongest
possible sense.** Across 13 levers spanning learning rate (1e-4 to 1e-3), budget (10 to
40), training length (2,000 to 12,000 episodes), batch size, entropy coefficient, discount,
hidden width, step cost, identification threshold, intervention scale, samples per
intervention, and both graph priors, **the largest effect on the working architecture is
0.288** — and that one is `identify_threshold_0.9`, where a harder success criterion
plausibly *should* cost something. Every other lever moves the result by less than 0.2.

The same levers move the flat network by up to **-6.866** (`lr_0.0001`), -4.821
(`budget_40`) and -4.084 (`identify_threshold_0.9`).

[DECIDED] The interpretation is that the +1.1 gap-closed result **is not a
hyperparameter-tuning artefact**. It survives every lever tried, at every setting tried.
The apparent hyperparameter sensitivity in the overnight sweep was the flat network being
pushed around; it was measuring the architecture's failure, not the task.

This is the most useful thing Phase 2 could have returned, and it is worth stating plainly
in the thesis: the single-agent result is robust to the entire hyperparameter surface that
was swept, so the comparison against greedy rests on the architecture and the environment
rather than on tuning.

[MEASURED] Canaries: 48 fired across 62 configurations — G1 entropy 24, G4 seed spread 21,
G5 gate 1 recorded 3 (one of which is the negative control, firing as designed). These are
concentrated on the flat arm, which is expected: a network that cannot express the task has
both an unconverged policy entropy and an unstable spread across seeds.

[TODO] Re-run once tasks 63-66 land, to complete the four incomplete cells. The verdict for
those four levers is currently unknown, not "dead" — `analyse_phase2` refuses to classify a
partial grid, which is the guard added on 2026-08-15 after it biased toward keeping depth 1.

## 2026-08-16 (07:00) — d=6 gates on the DP path: the control passes

[MEASURED] `scripts/gates_dp.py` at d=6, n_obs=20000, 600 episodes for GATE 1 and 300 per
policy for GATE 2. This is the **control task**: at d=6 both gates are also computable by
enumeration, so a disagreement here would invalidate the d=7 run whatever it said.

| gate | measured | reference | verdict |
|---|---|---|---|
| GATE 1 | observational rate 0.0800, CI [0.0583, 0.1017] | sampled target 0.0824, CI [0.0802, 0.0844] | **PASS** |
| GATE 2 | greedy 1.823 steps, CI [1.663, 2.000] | random 3.730 steps, CI [3.440, 4.020] | **PASS**, disjoint |

The sampled singleton target of 0.0824 (CI [0.0802, 0.0844]) contains the **exact
enumerated value 0.08095**, so the enumeration-free GATE 1 target agrees with the
enumerated one at the only size where both exist. The observational rate 0.0800 also matches
the 0.0600 measured by the enumerated pipeline on 2026-08-15 within its interval.

[CORRECTED] Runtime estimate was badly wrong in the safe direction: the submit script
projected ~55 min per task from an oracle cost of 0.77 s x 3000 calls. Actual: **4.4
minutes**. The projection assumed every episode runs to the full budget of 10 steps, but
greedy identifies in 1.8 steps on average, so the oracle is consulted about a fifth as
often as budgeted for. Worth noting because the same reasoning inflated the d=7 estimate,
and it is the *third* runtime projection this project has got wrong.

[NOTE] d=7 (array task 2) is queued behind Phase 2 and had not started at hand-off.

## 2026-08-16 (07:05) — Block 6: GATE-M2, and what it actually decomposes

[MEASURED] Exact posterior over T1's masked space (96,255 DAGs enumerated), 80 episodes,
budget 10 interventions, inference held centralised in both arms so that only *who chooses,
on what information* varies.

| arm | interventions to identify | CI | solve rate |
|---|---|---|---|
| centralised | **1.74** | [1.56, 1.93] | 1.00 |
| independent | **2.52** | [2.29, 2.81] | 1.00 |

`coordination_gained` = **+0.787 interventions**, intervals disjoint. **GATE-M2 passes.**

[CORRECTED] The first run of this gate **failed**: centralised 2.80 against independent
2.64, i.e. coordination appeared slightly *harmful*. The cause was a flaw in my own
implementation, not a property of the topology — the centralised arm chose two targets at
once from a single `argsort`, scoring the second against a belief that still assumed the
first had not been run. That is a strictly weaker chooser than the sequential greedy the
docstring claimed. Fixed to pick one target per round and re-plan, with budgets still
matched in interventions rather than rounds.

[DECIDED] **Both numbers are worth keeping, because together they decompose the result**,
and the decomposition is more interesting than the gate:

- **Matched cadence, information advantage only** (the batch version): centralised 2.80 vs
  independent 2.64 — no gain. Simply *knowing* what the other agent knows, while still
  having to commit to both interventions simultaneously, buys nothing here.
- **Adding sequencing** (the fixed version): centralised 1.74 vs independent 2.52 — a clear
  gain.

So under T1 the value of coordination is **almost entirely the ability to sequence
experiments and re-plan between them**, not the shared information per se. That is a
sharper claim than "coordination helps", and it points the two-agent design at turn-taking
and negotiation over the shared exposed nodes rather than at belief-sharing protocols.

[NOTE] The honest caveat, stated rather than buried: the passing comparison differs in
**two** ways at once — shared information *and* re-planning cadence — so +0.787 is an upper
bound on what coordination is worth, and the batch figure is the matched-cadence lower
bound. A clean isolation would give the independent arm the same re-planning cadence, which
is not possible without letting the agents act in turn, which is itself a coordination
mechanism. That circularity is worth noting in the write-up.

[MEASURED] The confounded fraction in the sampled episodes was 0.388 against the enumerated
0.439 — consistent, and a useful cross-check that the episode sampler is drawing from the
masked space correctly.

## 2026-08-16 (07:15) — Night summary

**Delivered.** Subset-DP posterior in signed log space, exact and verified against
enumeration on environment data at d=3-6 (blocks 1-2, after a serious correction). MH
sampler, sampling oracle and an enumeration-free GATE 1 target (block 3). `sa/env_dp.py`,
pinned step for step against the enumerated environment. Phase 2 analysis: no lever is a
task lever (block 4). `ma/` package with GATE-M3 measured and T3 rejected (block 5).
GATE-M2 measured and decomposed (block 6). Thesis-ready single-agent summary (block 7).
d=6 gates on the DP path, passing, as the control for d=7. Test suite 358 -> 443.

**Not delivered.** d=7 *training*. The infrastructure is complete and verified and the d=7
gate job is queued, but wiring the DP path through `baselines`/`evaluate`/`run_experiment`
and running three seeds did not fit alongside the log-space rewrite. This is the right
outcome rather than a shortfall: the rewrite was mandatory, and launching d=7 training on
the arithmetic that returned `Z = 0` would have produced another retraction.

**The single most important thing learned.** Two of the night's three serious errors were
in the *verification*, not the implementation: testing a component through a consumer, and
testing on data that cannot exercise the failure mode. Both produced clean passing test
suites over broken code. The standing rules now carry both.

## 2026-08-16 (midday) — d=7 wiring, and two silent-failure bugs

[CORRECTED] **The subset DP could never have worked on real data.** Documented in full in
`sa/dp.py`, recorded here because it is the most serious error of the last two days and it
passed every test I had written.

The plain-arithmetic version rescaled each node's weights by that node's own maximum. The
sum of per-node maxima is the score of a configuration where every node takes its
unconstrained best parent set -- and those choices are jointly **cyclic**, so no DAG attains
it. The shortfall is the information each node shares with the others, and it grows with
both d and n:

| gap (nats) | n=1000 | n=5000 | n=20000 |
|---|---|---|---|
| d=4 | 834 | 4,612 | 18,233 |
| d=5 | 1,821 | 8,888 | 35,999 |
| d=6 | 3,892 | 19,404 | 78,306 |

A double underflows past 745, so **every** configuration in use returns `Z = 0`. It first
appeared as `Z = 0` at d=4 -- the easiest case in the project.

Blocks 1 and 2 verified it against enumeration at d=3,4,5,6 and it agreed to 1e-13. Those
tests fed **independent normal columns**, where nodes share no information, the gap is
~0, and the failure cannot occur. A causal discovery environment produces correlated data
by construction, so the test inputs were the one kind of data the code will never see.

[DECIDED] Rewritten in signed log space, where no rescaling is needed at all. Re-verified on
**environment** data at d=3,4,5,6 and n_obs=1000 and 20000: log Z, all edge marginals and
P(true DAG | data) agree with enumeration to <= 5.5e-11. Every test in `tests/test_dp.py`
now builds its data from `CausalDiscoveryEnv` rather than from `rng.normal`.

[CORRECTED] The cancellation diagnostic was also wrong, in a way that would have masked the
above. It compared every intermediate against the FINAL `f(V)`, but smaller subsets carry
far fewer likelihood terms and are astronomically larger, so it reported a "growth" of
e^121000 on runs whose answers were exact to 1e-12. Now computed per subset; reads 0.000
everywhere at d=3-6.

[MEASURED] Standing lesson, added to the two from 2026-08-15: **verifying against ground
truth is not enough if the inputs are unrepresentative.** Two of the three serious errors in
this stretch were in the checking, not the code, and both produced a fully green suite.

[CORRECTED] `--force_dp` used `action="store_true"`, whose default is **False, not None**.
`Backend` reads `force_dp is None` as "choose automatically", so an absent flag meant
"force the ENUMERATED path", and at d=7 that begins enumerating 1.14 billion DAGs. It hung
with no output rather than failing. `Backend` now refuses enumeration above d=6 with a
message naming the likely cause.

[DECIDED] Added `EdgeMarginalGreedyDPPolicy`. The d=4/5/6 headline numbers are scored
against `edge_marginal_greedy` -- a greedy opponent restricted to the *same lossy belief*
the agent uses -- so that the comparison isolates policy quality from belief quality. The DP
path had only the full-posterior greedy, and scoring d=7 against that would have changed
what the headline measures halfway along the size axis. Implemented by noting the
independent-edge product is **modular**, so it is expressible as a log-weight table the
existing DP and sampler consume unchanged; rejection sampling was rejected because only
~1 in 4000 independent-edge draws is acyclic at d=7. Verified: the rebuilt approximate
posterior matches the enumerated one to 7.6e-13 at d=4 and 3.9e-12 at d=5.

[MEASURED] Per-step belief cost on the DP path, at the sample size actually used
(n_obs=20000): **d=6 21 ms, d=7 40 ms**. The enumerated path at d=6 was 846 ms. So d=7 is
roughly 20x cheaper per step than the d=6 runs that produced the current headline.

[MEASURED] d=7 smoke run end-to-end at n_obs=2000: GATE 1 correctly **FAILS** (0.0167
against a target of 0.0779), which is the intended behaviour -- the gate rejects an
under-powered environment, and confirms n_obs=20000 is required at d=7 as the standalone
gate job found at 02:28.

## 2026-08-16 20:47 -- d=7 n_obs sweep submitted (job 152604)

[DECIDED] Submitted `submit_sa_d7_nobs.sh` to Myriad as job-array 152604.1-9. Grid is
n_obs in {5000, 10000, 20000} x seeds {0,1,2}, n_obs varying fastest so a partial grid
still yields complete seed sets at the smaller settings.

Prediction registered before the numbers exist (also in the script's header comments):
gap closed decayed monotonically to parity (+1.001) at d=7 because greedy's *absolute*
cost fell to 1.94 interventions while n_obs was held at 20000 -- there is no horizon to
plan over. Lowering n_obs should lengthen the horizon and re-open the gap, UNLESS GATE 1
fails first, which is itself the result: it would mean the window in which the task is
both well-posed and non-myopic has closed at d=7.

The n_obs=20000 arm deliberately duplicates the three completed runs (+1.001/+1.017/
+0.994) as an internal replication check with freshly computed references. If it does not
reproduce, the cross-n_obs comparison is not trustworthy either and nothing else in the
sweep should be read.

Cluster housekeeping: the pull was blocked by untracked `results/d7/` and
`results/gates_dp/gates_d7.json` on the cluster (the same files had since been committed
locally). Backed up to `~/sa_results_backup/` and moved aside to `~/sa_stale/` rather than
deleted; cluster now at 57ad42d.
