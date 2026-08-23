# Brief: can constraint-based causal discovery replace our Bayesian engine?

**You are starting cold. Read this whole file before touching anything.**

Worktree `.claude/worktrees/constraint-based`, branch `explore/constraint-based`, cut from
`main` at `c58543e` on 2026-08-23.

**This is a time-boxed EXPLORATION, not a rewrite.** The Bayesian path continues in parallel in
`.claude/worktrees/ma-disclosure`. Your job is to find out whether the constraint-based route is
viable, and to say so honestly if it is not. **A well-evidenced "no" is a successful outcome**
and goes in the discussion chapter either way.

---

## 1. The project in one page

MSc thesis: **federated active causal discovery with multi-agent reinforcement learning.**

Several agents each observe part of a causal system. Each has **private** variables nobody else
sees and **shared** (exposed) variables everyone sees. Each round, an agent spends budget to
**intervene** on a variable it has authority over, everyone observes the resulting samples on
the variables they can see, and each agent updates its belief about the causal structure of its
own window. The research question:

> Can agents learn to *choose experiments* better than a myopic greedy baseline, and does
> cooperative, privacy-preserving information sharing between them help?

The active/interventional part is the contribution. Nobody in the federated causal discovery
literature does experiment *selection* — see `docs/BIBLIOGRAPHY.md` §17.

**Hard constraints, from the supervisor:**
- **No central server.** Any coordinator-based algorithm is disqualified.
- **No information about private variables** may cross the boundary. One exception was
  negotiated and approved on 2026-08-23 — see `docs/DISCLOSURE_DESIGN.md`.

**Deadlines: experiment freeze 31 August 2026, dissertation due 8 September 2026.**

## 2. Why you are being asked this

Our belief engine is **exact Bayesian**: a BGe marginal likelihood (Geiger & Heckerman 2002)
over linear-Gaussian structural equation models, summed over all DAGs by a subset dynamic
program (`sa/dp.py`, Robinson's sink recurrence in signed log space).

The student's assessment, 2026-08-23, and it is correct:

> The only reason we are still using this exact Bayesian method is that it was good for
> comparison while we were working single-agent and testing ideas. It is clearly not the
> preferred route in the literature — it does not scale, and it does not generalise past
> linear Gaussian graphs.

Three specific problems:

**2.1 The confounding layer explodes.** Measured 2026-08-23,
`scripts/ma_disclosure_scaling.py`, results in `results/disclosure_scaling.json`:

    |X|  pairs  assignments   one belief update
      3      3           25         0.028 s
      4      6          543         0.577 s
      5     10       29,281        60.139 s

`|X|` is the shared-set size. The assignment count is **the number of DAGs over the shared
set** — an assignment picks {absent, u->v, v->u} per pair subject to acyclicity, which *is* a
DAG. Super-exponential: `|X|=6` is 3.78 million. **Dead at `|X|=5`.**

**2.2 The window is capped near `k=15`** by the `O(k 2^k)` dynamic program.

**2.3 BGe is linear-Gaussian only.** No nonlinearity, no discrete variables, no mixed types.

## 3. What we already have that you should reuse

**Do not rebuild these.** All are tested and on `main`.

- **`ma/projection.py`** — real MAG (maximal ancestral graph) machinery. `d_separated()` by
  moralisation, `ancestor_matrix()`, `latent_projection()` returning a MAG as edge codes, and
  `bidirected_pairs()`. Definitions follow Richardson & Spirtes (2002). It is a **verification
  tool**, not an inference engine — but the primitives are correct and tested.
- **`scripts/ma_structural_ceiling.py`** — computes the full conditional-independence signature
  of a graph restricted to an agent's window, and asks whether any latent-free DAG reproduces
  it. Read the docstring; it explains the observational identifiability limit carefully.
- **`ma/topology.py`** — n-agent topologies, who-observes-what, who-may-intervene-on-what, the
  jointly-visible edge rule.
- **`ma/env.py`** — the environment. Turn-taking, clamp interventions, shared budget. **Note its
  guard:** it refuses topologies hiding more than one node from any agent. That guard is
  deliberate and correct for the *Bayesian* path; a constraint-based engine may not need it.

## 4. The literature — verified 2026-08-23

Full entries and prose in `docs/BIBLIOGRAPHY.md` §16 and §17. The ones that matter to you:

- **Tillman & Spirtes (2011)**, AISTATS — IOD (Integration of Overlapping Datasets). Learns
  equivalence classes from multiple datasets over *overlapping variable sets*, with latent and
  selection variables. **This is our partition.**
- **Triantafillou & Tsamardinos (2015)**, JMLR 16 — **COmbINE**. Overlapping variable sets *plus
  multiple interventions* plus latents. Converts dependencies and independencies into path
  constraints and solves the combination as a boolean satisfiability instance. **Closest
  published match to our problem.** arXiv:1403.2150.
- **Hahn, Zajak, Heider & Ribeiro (2026)**, arXiv:2603.05149 — **fedCI / fedCI-IOD**. Federates
  IOD. Claims to be the first federated causal discovery under latent confounding across
  heterogeneous datasets. Purely observational. Has a public Python package. **Read this first.**
- **Zhang (2008)** — completeness of FCI's orientation rules for PAGs (partial ancestral graphs).
- **Colombo et al. (2012)** — RFCI, the scalable FCI variant. *[unverified — check it]*

## 5. The question you are actually answering

**Primary, and it is one question:**

> Can a constraint-based engine give us what the active-learning loop needs — a usable measure
> of *uncertainty over structure*, to drive experiment selection — while scaling past `|X|=5`
> and handling latent confounding natively?

**This is the crux and it is not obvious.** Constraint-based methods output an **equivalence
class** (a PAG), not a posterior. Our agents choose actions by expected information gain, which
needs a distribution over hypotheses. An equivalence class is not one.

Candidate answers to investigate, in rough order of promise:

1. **Bootstrap the independence tests.** Resample data, rerun FCI, get a distribution over
   PAGs. Empirical, embarrassingly parallel, no new theory. Probably the pragmatic answer.
2. **Score the equivalence class by size.** Information gain becomes "how much does this
   intervention shrink the class". Cheap, crude, and arguably closer to what the oracle
   already does.
3. **Bayesian constraint-based hybrids** — Claassen & Heskes have work on putting probabilities
   on constraint-based inference. *[find and verify]*
4. **Keep the Bayesian engine for action selection, use constraint-based for the final
   answer.** Ugly but might be defensible if the two agree at small sizes.

**Secondary questions, all of which matter:**

- **Does it survive "no central server"?** IOD-style integration combines constraints from
  several sites. Who does the combining? If it needs a coordinator, can it be made peer-to-peer,
  and at what cost? This is a **hard constraint**, so an algorithm that fails it is out.
- **What does it leak?** Sharing conditional-independence facts about *shared* variables is
  probably fine. Verify that a shared CI fact cannot be inverted to reveal private structure.
  fedCI shares regression sufficient statistics, which is **more** than we currently allow —
  check what the minimum is.
- **How does it use interventions?** Interventional data gives constraint types observational
  data cannot. COmbINE handles this; understand how, because it is our core setting.
- **Does it actually scale here?** Sparse-graph polynomial claims often hide a bad constant.
  Measure it on our topologies rather than trusting the asymptotics.

## 6. How to work

**Read `CLAUDE.md` and the standing practices before starting.** The ones that will bite you:

- **Spec before coding.** Agree the design with the student before implementing. Four bugs in
  one day were all silent design decisions made mid-implementation.
- **Log incrementally.** Append `[MEASURED]` / `[DECIDED]` / `[CORRECTED]` entries to
  `docs/logs/SA_EXPERIMENT_LOG.md` as they happen. Keep nulls and self-corrections.
- **Record references** into `docs/BIBLIOGRAPHY.md` with VERIFIED / STANDARD / UNVERIFIED status.
  **Verify by actually searching, not from memory** — this was enforced on 2026-08-23 after a
  reference was cited for the wrong argument.
- **Verify directly, not through a consumer.** A broken sampler once looked like a mixing problem
  for three rounds because it was measured through the oracle.
- **Verify on representative data.** A subset dynamic program passed 29 tests on `rng.normal`
  then returned zero on the first real episode.
- **Test that metrics can be EARNED.** A reported metric was once structurally unearnable, and
  529 tests passed anyway.

**Do not:**
- touch `main`, or the `ma-disclosure` worktree
- change anything in `ma/` or `sa/` that the Bayesian path depends on — add alongside instead
- install heavyweight dependencies without asking. AVICI already cost this project days of
  environment debugging; see the memory on Kaggle isolation.

## 7. Deliverable

**`docs/CONSTRAINT_BASED_ASSESSMENT.md`**, answering §5's primary question with evidence, plus:

- a recommendation: adopt, hybrid, or reject — with reasons
- what it would cost to adopt, in days
- what it would give up
- whichever measurements you made, with the scripts that produced them
- what remains unknown

**Prototype only as far as needed to answer the question.** A measured "this is a 3-week job and
we have 8 days, here is exactly why" is a complete and useful answer. So is "this works, here is
a working prototype on our topologies". Both beat a half-built engine.

**First thing to report back:** whether the uncertainty problem in §5 has a workable answer. If
it does not, everything else is moot and the student should know within a day, not a week.
