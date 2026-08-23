# Strip scope for the constraint-based worktree

**Revision 2, 2026-08-23. NOT EXECUTED. Needs sign-off.**

Supersedes revision 1. Extended after a second lookthrough and four further instructions:
untrack `results/` entirely, dissolve `sa/`, converge overlapping modules, and scope a minimal
test suite. Everything is measured from the tree.

---

## 1. The multi-hidden-node guard — what it is, and whether we need it

`ma/env.py.__init__` refuses any topology where an agent has **more than one node hidden from
it**:

    widest_hidden = max(len(topology.hidden_from(a)) for a in topology.agents)
    if widest_hidden > 1:
        raise NotImplementedError(...)

**What goes wrong without it.** Each batch of rows carries a `clean` value per agent — a
**scalar fraction**, `n_clamped / len(hidden)`. The belief mixes a clean and a dirty score table
with weight `q = 1 - fraction`.

With one hidden node the fraction is 0 or 1 and everything is exact. With **two** hidden nodes
`h1` and `h2`, clamping only `h1` gives `fraction = 0.5` — and every confounding hypothesis is
then scored with the same `q = 0.5`. The mixture knows **how many** hidden nodes were clamped
but not **which**. So "h1 confounds this pair" and "h2 confounds this pair" score *identically*,
even though the data is clean with respect to `h1` and dirty with respect to `h2`.

`_assignment_weights` receives only that scalar per row batch. There is no per-node identity
anywhere in its input.

**So the guard is necessary — but it guards a representation, not the problem.** It is a
limitation of the scalar-fraction encoding, not of the science.

**Two ways to lift it inside the Bayesian engine:**
1. Make `clean` a `[n_rows, n_hidden]` mask instead of a scalar, so each hypothesis reads the
   column for the node it is about. This is the real fix; the cost is that `regime_tables`
   currently caches on unique scalar fractions and would need rekeying.
2. Permit only all-or-nothing clamping of the hidden set. Exact, but restrictive.

**For a constraint-based engine it is not needed at all.** There is no clean/dirty score
mixture. Independence tests condition on the actual per-row intervention regime, so the
"fraction" abstraction that loses node identity never exists.

> **This matters more than it looks. Rung 1 — three agents — is blocked TODAY by this guard,
> because three agents with one private node each hides two nodes from every agent. The
> constraint-based route lifts it for free.** That is an argument for the switch that had not
> been made, and it is probably the strongest one: it unblocks the scaling ladder, not just the
> runtime.

**Do not simply delete the guard.** A subagent removed it once and inverted its regression test,
with 474 tests green throughout. It should be *replaced* by a backend-specific capability check:
the exact backend declares it cannot handle `widest_hidden > 1`, the constraint backend declares
it can, and the env asks the backend rather than hard-coding the rule.

## 2. `results/` — untrack entirely

**Currently 646 files, 74.8 MB, deliberately tracked.** `.gitignore` blanket-ignores `*.json`
and then explicitly re-exempts results, with this rationale in the file:

> *"Experiment results are DATA, not build artefacts: the blanket `*.json` rule above would
> silently drop every raw result, which is exactly what the reproducibility requirement needs
> kept."*

That rationale was deliberate, so overriding it should be deliberate too. **The concern is real
and needs answering, not ignoring:** `docs/logs/SA_EXPERIMENT_LOG.md` records findings, not raw
data. Untrack the results and the numbers behind every reported figure exist in exactly one
place — a single working directory on one laptop.

**Recommended, and this is a package rather than a single change:**

1. `git rm -r --cached results/` and drop the exemption lines from `.gitignore`.
2. **Archive the current tree outside the repo before deleting anything** — a dated zip on a
   second location. "Stored safely locally" is one disk failure from "not stored".
3. Track a **`results/MANIFEST.md`** — file names, sizes, SHA-256, and which document cites
   each. Small, text, diffable. Provenance survives even when the bytes live elsewhere.
4. Note in `docs/STATE_OF_TRUTH.md` where the archive lives.

Without (2) and (3) this trades clutter for an unrecoverable loss of thesis evidence, and the
loss would not surface until someone asked for a number during the viva.

## 3. Dissolving `sa/` — everything natively multi-agent

`ma/` pulls exactly ten symbols from `sa/`:

    from sa.dp import DPPosterior                          -> crosscheck
    from sa.score import BGeScore                          -> crosscheck
    from sa.scm import sample, sample_multi, sample_scm_params
    from sa.priors import connectivity_prior_p
    from sa.graphs import build_graph_space, descendants, is_acyclic, mec_signature
    from sa.gates import bootstrap_ci                      -> one function
    from sa.oracle import _partition_entropy               -> one PRIVATE function
    from sa.posterior import PosteriorEngine, is_identified -> only by ma/coordination (DEAD)

**Target layout:**

    ma/                    natively multi-agent; no `sa` import anywhere
      topology.py          unchanged
      projection.py        absorbs ma/confounding.py -- see section 4
      scm.py               <- sa/scm.py
      graphs.py            <- sa/graphs.py
      priors.py            <- sa/priors.py
      stats.py             <- bootstrap_ci + _partition_entropy, both promoted to public
      nets.py              <- PerNodeActorCritic, extracted from sa/policy.py
      env.py               + the backend boundary
      policy.py  baselines.py  evaluate.py

    crosscheck/            reference implementations, frozen, never extended
      belief_dp.py  dp.py  score.py  scoretable.py

    cb/                    the new engine
      citest.py  skeleton.py  orient.py  bootstrap.py  backend.py

**Deleted outright** (unreachable from `ma/`, single-agent-only): `sa/env.py`, `sa/env_dp.py`,
`sa/evaluate.py`, `sa/baselines.py`, `sa/uncertainty.py`, `sa/tracking.py`, `sa/dag_samplers.py`,
`sa/sampler.py`, `sa/posterior.py`, and `sa/policy.py` **after** `PerNodeActorCritic` is
extracted. `sa/oracle.py` and `sa/gates.py` go once their one function each is moved.

**`crosscheck/` as a directory name is the point.** It cannot be imported by accident and read as
production code, which is exactly the confusion the strip exists to prevent.

## 4. `confounding.py` and `projection.py` — they are NOT the same

They look redundant. They are not, and merging them silently would change a reported number.

**`confounding.latent_projection_pairs(adjacency, observed, hidden)`** — pairs of observed nodes
reachable from a **common hidden source** through hidden intermediates. A sufficient condition
for a bidirected edge.

**`projection.bidirected_pairs(adjacency, observed)`** — builds the full MAG. A pair is
bidirected only when (a) no subset of the observed set d-separates them, **and** (b) neither is
an ancestor of the other.

**The difference is condition (b).** If `h` confounds `u` and `v` *and* a real edge `u -> v`
exists, then `u` **is** an ancestor of `v`, so the MAG carries `u -> v` — directed, not
bidirected. `confounding` reports that pair; `projection` does not. **`confounding` over-reports
relative to the textbook MAG definition.**

**Converge onto `projection.py`.** It is the stricter, textbook-correct definition (Richardson &
Spirtes 2002), it has zero internal dependencies, and — decisively — it is what
`ma/env._confounded_positions` already uses to score identification, and what
`scripts/ma_structural_ceiling.py` used for the 2.3% headline. `projection` is already
authoritative in everything that reports a number.

**Keep** `measure_topology` and `ambiguity_location` (used by `tests/test_ma_topology.py`), moved
across. **Write a test asserting exactly where the two definitions diverge** before deleting the
loser, so the difference is recorded rather than lost.

## 5. Minimal test suite

Currently 35 files, 3,764 lines.

**KEEP — core**

    tests/test_projection.py          the ground-truth validator. Highest value in the tree.
    tests/test_ma_topology.py         topology, visibility, intervention authority
    tests/ma/test_env.py              environment
    tests/ma/test_evaluate.py         metrics
    tests/ma/test_metric_reachability.py   standing rule: metrics must be EARNABLE
    tests/test_env_turns.py           turn-taking protocol
    tests/test_env_turn_budget.py     shared budget
    tests/ma/test_greedy_tiebreak.py  baseline
    tests/test_canaries.py            every canary must FIRE on its own failure
    tests/test_optimisations.py       fast paths pinned against slow ones -- prune to kept paths
    tests/test_depth.py               PerNodeActorCritic layers=1 reproducibility
    tests/sa/test_graphs.py  ->  tests/test_graphs.py
    tests/sa/test_priors.py  ->  tests/test_priors.py

**KEEP — crosscheck**

    tests/ma/test_belief_dp.py  tests/ma/test_belief_crosscheck.py
    tests/test_dp.py            tests/sa/test_score.py  ->  tests/crosscheck/
    tests/fixtures/ma_reference_posteriors.npz    the frozen reference. CRITICAL -- do not delete.

**DELETE**

    tests/sa/test_env.py  test_evaluate.py  test_posterior.py
    tests/sa/test_oracle_and_baselines.py  test_exact_sampler.py  test_gate1_precondition.py
    tests/test_env_dp.py  test_uncertainty.py  test_tracking.py  test_sampling_oracle.py
    tests/test_score_regimes.py               module being deleted
    tests/ma/test_block_confounding.py        tests score_regimes

**CHECK FIRST:** `tests/sa/test_policy.py` — if it covers `PerNodeActorCritic`, keep those tests
and move them with the net. Do not delete unread.

**NEW:** `tests/cb/`.

Estimated result: roughly 20 files from 35, and the fast loop well under a minute.

## 6. What the second lookthrough turned up

**`requirements.txt` is stale and wrong.** It pins `jax`, `jaxlib`, `flax`, `chex`, `dm-haiku`
and an unpinned `avici` — the AVICI stack that cost this project days and is not installed. And
it **does not list `torch`**, which `ma/policy.py` imports and which every training run needs. A
fresh environment built from this file cannot run the code. Rewrite it.

**`tests/fixtures/ma_reference_posteriors.npz`** is the frozen reference the belief cross-check
is held to. It is the single artefact that would catch a silent regression in a new engine.
Flagged explicitly because it is a binary in a fixtures directory and reads as incidental.

**`.agents/`** holds `AGENTS.md`, commit conventions, versioning and branching rules, and two
skills (federated causal discovery, UCL Myriad HPC). Operating instructions — **keep**.

**`ma/coordination.py` is the only true dead module** — 236 lines, zero references. Its deletion
also removes the last consumer of `sa/posterior.py`, which is why `posterior.py` can go.

## 7. Execution order

Tests green between every stage; stop at the first red.

    0  archive results/ outside the repo, write MANIFEST.md         (no code change)
    1  untrack results/, fix .gitignore                             (no code change)
    2  delete legacy/  -- 145 files, verified self-contained
    3  delete ma/coordination.py, then ma/score_regimes.py + its tests
    4  converge confounding -> projection, with the divergence test
    5  extract PerNodeActorCritic -> ma/nets.py; bootstrap_ci + _partition_entropy -> ma/stats.py
    6  move sa/{scm,graphs,priors}.py -> ma/; rewrite imports; delete the rest of sa/
    7  move belief_dp + dp + score + scoretable -> crosscheck/
    8  the backend boundary, and convert the guard to a backend capability check
    9  prune tests; rewrite requirements.txt
    10 cb/ scaffold

Stages 0–1 are the biggest win for the least risk. Stage 8 is the one that actually prevents the
confusion this strip is for.

## 8. Merge-back — unchanged and still important

Deletions stay in **their own commits**, separate from additions. If the exploration succeeds,
**cherry-pick `cb/` and the boundary onto `main`; do not merge this branch**, or the deletions
propagate and `main` loses the single-agent path along with them.

## 9. Risks

**The `sa/` dissolution touches every import in the tree.** It is mechanical but wide, and a
missed import fails at collection time rather than silently — which is the good failure mode.
Run the suite after each stage rather than at the end.

**Extracting `PerNodeActorCritic` must not perturb the RNG draw.** `tests/test_depth.py` exists
precisely because `layers=1` has to reproduce the d=4/5/6 results *exactly*, not merely have the
same shape. Move the class verbatim; do not tidy it in transit.

**Untracking results is irreversible in effect if step 0 is skipped.** Archive first.
