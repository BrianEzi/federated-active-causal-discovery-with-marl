# Turn budget, signalling and termination — spec

**Agreed 21 August 2026, before implementation.** Written because four bugs on 20–21 August
were all silent *design* decisions made mid-implementation, each caught only by an expensive
training grid. Every decision below was settled in discussion first; this file is the record,
not a proposal.

Supersedes the budget and termination rules in `docs/REDESIGN_2026_08_20.md`.
**Hard experiment freeze: 31 August 2026.**

---

## 1. The problem this fixes

Two failures, one root.

**The step cost made passing optimal.** Measured: at `step_cost = 0.05` over ~7.7 steps, a
random-level policy has expected value **−0.255** against **0.000** for passing. Every
"collapse" recorded against the learner was the agent being *correct*.

**Turn-taking made silence free.** With a per-agent budget counting only *interventions*, an
agent that declines spends nothing. Combined with shared samples and a shared reward, that is
a public-goods game in which free-riding is the rational play. Measured under the first
turn-taking grid: 5/10 seeds collapsed into passing at `mean_steps 1.11`, against 0/10 for
the same settings under simultaneous play.

---

## 2. Budget — a shared pool of rounds

**`budget` is the total number of ROUNDS for the system.** Every round consumes one unit,
whether the active agent intervenes or declines.

Rationale, in order of weight:

1. **It internalises free-riding.** A round wasted by A is a round B does not get. Since the
   terminal reward is shared, that cost lands back on A. Free-riding stops being *free* —
   which is the right strength of fix. It remains *possible*, and §7 measures it rather than
   forbidding it.
2. **It is one parameter, not two**, and it matches how experimental budgets actually work:
   the money is spent by the project, not per-scientist.
3. **Under round-robin it is exactly equivalent** to a per-agent budget of `rounds /
   n_agents`, because the alternation is fixed and an agent cannot take extra turns — only
   waste its own. The two designs diverge only under **random** turn order, where a shared
   pool lets one agent draw more turns by chance. The shared pool is the more natural object
   there too.

**Starting value: 10 rounds** (≈ 5 turns each at two agents).

## 3. Pass is a forfeit, and it generates data

Declining consumes the round and **generates an observational batch** (`n_int` rows, no
intervention), delivered to both agents — samples are shared, so there is no asymmetry.

This makes `pass` a real "observe and wait" action with a genuine cost, rather than a null.
It also has a side benefit worth stating: **total data volume becomes constant** at
`n_obs + budget × n_int`, so data quantity no longer varies with how much a policy acted.
That confound is present in every number this project has produced to date.

## 4. Termination

An episode ends when **the round budget is exhausted** or **all agents have identified**.

**There is no voluntary termination.** It buys nothing once the step cost is zero — the
episode already auto-terminates on joint success, and stopping early with no solution scores
0 while continuing might still score a discounted +1. Removing it also deletes the entire
class of rule that produced the 20 August collapse.

## 5. Step cost is zero

Efficiency pressure comes from the finite round budget and from `gamma = 0.99` discounting:
with a terminal +1, solving at round 3 is worth more than solving at round 9.

**An explicit per-turn penalty was considered and rejected.** Under a fixed round budget it is
very nearly a constant offset — the same rounds are spent either way — differing from
discounting only in shape (linear against geometric). Not worth a parameter or the two-arm
training comparison that would be needed to separate them, with ten days left.

**Load-bearing dependency, do not break it silently:** free early stopping and a zero step
cost are linked. Re-introducing a step cost without also re-introducing a termination
mechanism, or vice versa, re-opens the collapse. Anyone changing one must read this section.

## 6. Signalling — three broadcast categories

At each **round boundary**, every agent broadcasts one categorical, free of charge:

    intervened in the SHARED set  |  intervened in its OWN PRIVATE set  |  NO INTERVENTION

**Broadcast, not action.** Signalling costs no round. Under the alternative — declare only on
your own turn, at the cost of that turn — establishing "we are both finished" costs one turn
per agent and takes `n` rounds. That is pure waste, and it grows with the number of agents.

**Advisory, never binding.** Others may ignore it. With a shared reward there is no incentive
to lie; if agents nonetheless learn to ignore the signal, that is a finding.

**What it does and does not reveal.** It names no variable and carries no value. "I intervened
in my private set" is close to the existing regime bit (which is sharper: it says *clamped*,
which is what scoring needs). "I intervened in the shared set" is weaker than the existing
`disclose_shared_targets`, which names the node — defensible because shared columns are
visible to both, so a partner could infer it from the data anyway.

**Grounding.** Grimsman, Ali, Hespanha & Marden (2018) bound distributed greedy's quality by
how much of the *other agents' decisions* each agent can see, with performance degrading in
the size of the largest group deciding independently. This channel is exactly that
information, and the literature says what it is worth.

**PROVISIONAL ON THE SUPERVISOR.** Mirco's constraints are no private-variable information
and no central server; a categorical about one's own action type violates neither, and it is
peer-to-peer. He is to confirm. Implement so it can be removed with a single flag.

## 7. The "done" bit — logged, not acted on

**It is computed from the agent's OWN POSTERIOR CONCENTRATION** — entropy, or mass on its most
likely equivalence class.

**It must NOT be the credit-set mass.** The credit set is defined relative to the TRUE graph
(`credit_candidates(window, truth)` pins private-incident edges to the truth), so it is an
ORACLE quantity. It is already computed every step for the reward, so passing it to the agent
would be free — and would be leaking the answer. Free is not the same as legitimate.

**Logged only.** No termination mechanism reads it. That preserves the deployment story — "a
practitioner could stop when the agents say they are done" — as a reportable diagnostic
(*in what fraction of episodes did both agents signal done within one round of actually being
finished?*) without creating an incentive surface.

**Calibration must be measured, not assumed.** A badly calibrated agent is confidently wrong:
0.85 posterior mass on answers that are right 60% of the time. Such an agent broadcasting
"done" tells its partner to stop helping while it still needs clamps. The check is a
reliability curve — bin episodes by the agent's own confidence, plot against how often it was
right. Expected to be good, for a reason worth putting in the thesis: the posterior is
**exact** under a **correctly specified** model, which is precisely where Bayesian confidence
should be calibrated, and it is a concrete advantage of exact inference over the learned
estimators this project began with.

## 8. Logging

Log the RAW QUANTITY, never the verdict. The `d=6` mess required a full retrain purely
because runs stored `identified: false` instead of the posterior mass — and no policy
checkpoints existed. **25.7 hours of cluster compute is unrecoverable for that reason.**

**Free-riding** — per agent, never the max across agents:
`interventions_spent`, `rounds_forfeited`, and `min/max` interventions as a single index.
`mean_steps` currently takes the max, which hides an idle agent inside an average.

**Behaviour:** clamp fraction **split by target type — own-private against shared**. This is
the whole altruism signal, and the aggregate cannot distinguish them: clamping a shared node
does nothing for a partner, only the private clamp does. Plus the full action histogram over
`(node, mode)` per agent, and the clean rounds *delivered to* each agent (what landed, not
what was intended).

**Raw scores, so a criterion change never forces a re-run again:** posterior mass on the true
DAG *and* on the credit set, per agent per round; the per-agent `confounded` flag; and the
true graph's own statistics.

**Graph structure:** `connected` (single component) or not. **Every multi-agent metric is
reported split by this.** A disconnected graph makes the agents' subproblems independent —
no cross-boundary paths, no confounding, nothing to coordinate about — so those episodes
cannot test what we are building. Connected is the headline; disconnected stays as the
grounding case.

**Health:** policy entropy and pass rate per agent, `first_success_episode`, and every value
per seed rather than aggregated.

**Scaling:** wall-clock split across belief update, scoring, and policy forward pass. Cheap
now, and it is what tells us where to optimise for five agents instead of guessing.

## 9. Guard — the observational leak

At budget 10 this adds `10 × n_int = 1000` observational rows on top of `n_obs = 1000`. This
project exists because of a leak in which episodes were solvable *without acting*, so the
risk must be named.

**The argument that we are protected:** observational data cannot break Markov-equivalence
ties, and the credit set demands private-incident edges *exactly*. Extra rows sharpen the
posterior *within* what is identifiable; they cannot make an unidentifiable orientation
identifiable. So the pass-only baseline should improve only on graphs that were already
observationally identifiable.

**That is a testable claim, not a hope. The guard: track the pass-only baseline's success
rate directly.** If it climbs beyond the observationally-identifiable fraction, the leak is
back, and the fix is to cut `n_int` for forfeits or reduce `n_obs`.

`PassAgent` accordingly stops being a null arm and becomes a real **observation-only
baseline** — how far you get by never intervening at all.

## 10. Baselines

| baseline | what it is | what it shows |
|---|---|---|
| pass-only | never intervenes | the observation-only floor, and the §9 leak guard |
| random | uniform over legal targets | what pure exploration buys |
| selfish greedy | own window only — what exists today | the honest myopic floor |
| sequential greedy | conditions on partners' disclosed choices (SGA) | the real decentralised baseline |
| joint greedy | one oracle scoring ALL agents' posteriors | upper bound; deliberately violates federation |

Beating sequential greedy makes the claim mean something; approaching joint greedy shows the
federation costs little. References and the caveat that their submodularity guarantees do
**not** transfer to expected information gain are in `docs/BIBLIOGRAPHY.md` §15.

## 11. What this invalidates

**Every two-agent number produced before this spec.** Budget means something different, the
step cost is gone, data volume is constant, and termination has changed. This is a clean
break, not a comparison. Simultaneous-action results remain reproducible via
`turn_order="simultaneous"` and are reported as a superseded protocol.

## 12. Acceptance tests

Written before the code, and each maps to a decision above.

1. A round is consumed whether the active agent acts or declines.
2. A forfeited round generates exactly `n_int` observational rows, visible to both agents.
3. No single agent can end an episode; episodes end only on exhausted rounds or joint success.
4. Signalling consumes no round.
5. The done bit is absent from the observation vector's credit-set-derived fields — asserted
   directly, because "free to compute" made this an easy mistake to make.
6. Per-agent intervention counts are logged separately and are not derivable only as a max.
7. Clamp fraction is split by own-private against shared target.
8. Clean rounds remain reachable under every turn order (existing guard, must keep passing).
9. `step_cost = 0` is the default, with §5's dependency noted at the definition site.
