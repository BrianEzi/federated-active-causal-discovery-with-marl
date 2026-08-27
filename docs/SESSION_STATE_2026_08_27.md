# Session state — 27 August 2026, 21:30

> **READ THIS BOX FIRST.** The second agent found, late on 27 August (`dd6131d`), that
> `UncertaintyGreedyAgent` takes `bar=0.7` while these backends GRADE at `claim_bar=1.0`, so
> greedy stopped targeting claims that were 70% settled and was blind to a band of open
> questions by construction. On the attributed 3-agent task, fixing it REVERSES the
> headline: learned minus greedy goes from +0.142 ± 0.024 to **−0.091 ± 0.025**.
>
> **Verified here, and the scope is narrower than "everything".** The handicap can only bite
> where claim frequencies actually land in the 0.7–1.0 band. Measured, 30 episodes each:
>
> | backend | claim mass in the blind band | affected? |
> |---|---|---|
> | `factored` | **0 / 2879 = 0.000** | **NO** |
> | `version_space` | 17 / 1299 = 0.013 | yes |
> | `attributed` | 24 / 1505 = 0.016 | yes |
>
> The factored belief leaves 1, 2 or 3 candidates per pair, so its frequencies are quantised
> to {1.00, 0.50, 0.33} and NOTHING can fall in the band. Re-running greedy at bar 1.0 on
> the ladder configuration gives numbers identical to three decimals.
>
> **Therefore: §5's SCALING LADDER STANDS (it is `factored`). The ATTRIBUTION result in §5
> and the older `version_space` headlines DO NOT — re-run those against greedy at the graded
> bar before quoting them.**

**Resume point for a fresh session. Read this file completely before touching anything.**
Companions: `docs/FINDINGS_2026_08_27.md` (the measurements and every withdrawn claim),
`docs/HANDOVER_2026_08_27.md` (the second agent's brief, sections 8–9 written by them and
by us respectively), `docs/logs/SA_EXPERIMENT_LOG.md` (the trail).

**Freeze 31 August. Dissertation 8 September.** Branch `explore/constraint-based`,
HEAD `e82a2a0`, pushed. 337+ tests pass.

Supervisor meeting happened this afternoon and went well. Mirco is happy with the direction
and asked for **more data on deterministic training and how it scales**, as a thesis figure
comparing configurations. That is what the ladder below is for.

---

## 1. What the project is, in four sentences

Several agents each see a slice of one causal system — their own private variables plus a
shared set everyone observes. Each may experiment only on what it can see, they share a
budget of rounds, and none shares raw data. The question is whether they LEARN to divide
experiments better than a sensible rule, and — since 26 August — whether they can go further
and **attribute a hidden confounder to the partner who owns it**, which is a question
single-agent causal discovery cannot pose.

---

## 2. Setup and the three things that bite

```bash
cd federated-active-causal-discovery-with-marl
.venv/bin/python -m pytest tests/ -q          # ~337 passed, ~140 s
```

- **No bare `python`.** Use `.venv/bin/python`.
- **Everything needs `PYTHONPATH=.`** from the repo root.
- **Cap threads on parallel runs.** `OMP_NUM_THREADS=1` etc. Note torch keeps 4 threads per
  process regardless; `torch.set_num_threads(1)` would be the real fix and is not applied.
  Throughput was fine at 9 processes on 10 cores — measure throughput, do not read the load
  average, which is inflated by torch's pool.

---

## 3. The four belief engines, and which to use

| backend | representation | reaches | keeps |
|---|---|---|---|
| `constraint` | bootstrapped PC/FCI on real data | k≈8–10, slow | realism |
| `version_space` | explicit list of whole-window structures | k ≤ 6 | exactness, and an EXACT ceiling/optimum |
| `factored` | one small candidate set PER PAIR | **k = 30** | soundness, conservatively |
| `attributed` | version space × who owns each hidden cause | **3 agents only** | the novel target |

`version_space` and `factored` take `vs_evidence="oracle"` (prune by true ancestry, exact at
any distance) or `"sampled"` (prune by what finite DATA shows). **Sampled makes `n_int` a
noise dial on one environment family** instead of two incomparable worlds. Validated:
converges to the oracle at 4000 rows, 99.2% truth retention at `vs_evidence_alpha=1e-3`,
which is the measured optimum — **stricter is NOT safer** (see FINDINGS §4).

**The factored belief is a conservative RELAXATION, verified**: over 1,920 pairs it is
sometimes less decisive than the enumerated belief and never once contradicts it.

**Price of `factored`:** no joint constraints, and no exact ceiling or optimal-rounds figure
(those enumerate). **Exact headroom at k ≤ 6, honest bounds above.** Write it that way.

---

## 4. Policy architectures — four arms, and what each is for

| `policy_arch` | networks | decentralised? | use |
|---|---|---|---|
| `gnn` | one per agent, window-shaped | training and execution | every pre-27-Aug headline |
| `gnn_portable` | ONE shared across agents | execution only | scale ladder, shared arm |
| `gnn_solo` | one per agent, portable | **fully** | scale ladder, primary arm |
| `gnn_hybrid` | shared trunk, private heads | execution only | ablation |

**There has never been CTDE in this codebase** — no centralised critic; every value head
reads only its own agent's observation. Be precise with Mirco: parameter sharing is a
separate and narrower thing, and it only ever appeared in the portable arm. Sharing is
implementable peer-to-peer via decentralised/gossip SGD (Lian et al. 2017; Zhang et al. 2018
for networked MARL; BrainTorrent for medical FL) — **references recalled, NOT verified;
verify before they enter the bibliography.** Our implementation uses synchronous pooling, so
the honest claim is "implementable peer-to-peer", not "implemented".

**Portability comes from the ARCHITECTURE, not from sharing.** Every learned width is
per-node or per-pair, and partner blocks are pooled. One 94,851-parameter checkpoint runs at
observation sizes 158 → 1,852 unchanged.

---

## 5. Results — what is established

### The scaling ladder (the thesis figure), 3 seeds, `factored`, scale-free

**Window ladder, 4 agents.** Learning WINS as windows grow:

| window k | nodes | shared | solo | greedy |
|---|---|---|---|---|
| 4 | 7 | 0.813 ± 0.050 | 0.462 | 0.760 |
| 6 | 12 | 0.582 ± 0.136 | 0.271 | 0.609 |
| **8** | 20 | **0.342 ± 0.049** | 0.167 | 0.211 |
| **12** | 30 | **0.227 ± 0.040** | 0.120 | 0.138 |

**Agent ladder, window fixed at 6.** Learning COLLAPSES past 3 agents:

| agents | nodes | shared | solo | greedy |
|---|---|---|---|---|
| 2 | 8 | 0.431 ± 0.016 | 0.447 | 0.209 |
| 3 | 10 | **0.800 ± 0.019** | 0.658 | 0.569 |
| 6 | 16 | 0.169 ± 0.029 | 0.067 | **0.598** |
| 8 | 20 | 0.080 ± 0.031 | 0.038 | **0.520** |

Metric is joint success (every agent correct).

**THE AGENT LADDER AT 6 AND 8 IS NOT A RESULT — THOSE POLICIES NEVER TRAINED.** Measured
after the fact, mutual information between observation and action, I(S;A)/H, exact from the
policy rather than estimated:

| run | I(S;A)/H | final entropy | max entropy |
|---|---|---|---|
| w08 (learning wins) | **0.616** | 0.910 | 2.20 |
| w12 (learning wins) | **0.513** | 1.094 | 2.56 |
| a08 (learning "fails") | **0.034** | 1.800 | 1.946 |

At 8 agents the policy barely reads its observation and sits a hair below uniform entropy —
a fixed mixture wearing a network, the same signature the second agent found in the
attribution learners (0.035). So "learning collapses between 3 and 6 agents" was written up
here as a property of the task, with three seeds and error bars, and it is a TRAINING
FAILURE. Nothing about coordination was measured at 6 or 8 agents.

**What stands:**
1. **Bigger windows are fine** — ~1.6x over greedy at k=8 and k=12, and those policies
   demonstrably learned (I(S;A)/H 0.51–0.62).
2. **Agent scaling is UNDETERMINED above 3.** Consistent with credit-assignment noise: at 8
   agents each acts in 3 of 24 rounds while its reward depends on a window 7 partners
   disturb, so the gradient never moves it off initialisation. `a08long` (16,000 episodes)
   is the decisive test.
3. **Sharing beats solo about 2:1** — but check I(S;A)/H per arm before trusting any cell,
   since an untrained policy's number says nothing about sharing.

**RUN THE MUTUAL-INFORMATION CHECK BEFORE REPORTING ANY LEARNED RESULT.**
`mi_check.py` in the session scratchpad; it is exact and takes two minutes.

### Attribution — the novel contribution, 3 agents only

Fixed policies, 3 agents × 2 private + 3 shared, budget 12:

| policy | structure | attribution | identified |
|---|---|---|---|
| shared only | 0.864 | **0.000** | 0.000 |
| private only | 0.891 | 0.981 | 0.292 |
| probe then work | 0.926 | 0.907 | **0.658** |
| random | 0.861 | 0.407 | 0.275 |

Attribution evidence arrives ONLY from a partner's private experiment — 0.000 is not
"rarely". Learned, 3 seeds on identical episodes: **0.670 ± 0.005**, MATCHING the
theory-derived schedule (0.667) and beating greedy (0.582), under a per-agent reward.
Say "found a strategy as good as the one theory predicts", NOT "beat the baseline".

### Transfer fails, with the mechanism demonstrated

Deterministic-trained → statistical: 0.171 against greedy 0.229 and random 0.208. It loses
to RANDOM, so it is not confusion — a confused policy drifts toward random, not past it.

**Cause: unfamiliar belief inputs.** Blurring ONLY what the policy sees, inside the
environment it has mastered, reproduces the failure profile almost exactly (budget share
0.727 vs transfer's 0.691; shared coverage 0.53 vs 0.55) and drops identification
0.988 → 0.633. Caveat: the blur shrinks frequencies toward 0.5, so it shows information LOSS
reproduces it, not that "differently shaped" would.

### Other established results

- **Every bidirected edge in a window joins two SHARED variables**, and its latent lies in
  exactly one named agent's block. Attribution is a choice among suspects.
- **Scale-free gives 5.3x more hidden-common-cause triangles at matched density.** It is the
  right generator for this thesis.
- **Confounding resolves CLIQUE-WISE in the idealisation** (full triangle 0.600 vs pairwise
  0.611, 2 partial in 60) and **PAIRWISE in the statistical engine** (full 0.000, 33 partial).
- `latent_projection` searched every conditioning subset; replaced with inducing paths,
  **1440/1440 exact agreement**. That, not the belief, was the scaling wall.
- Budget: at one round per agent, 25 of 40 episodes are unsolvable by ANY assignment.
  Budget is now ~3 moves per agent and the ceiling is reachable.

---

## 6. Running at 21:30, and what each decides

| job | decides | state |
|---|---|---|
| `a08long` — 8 agents, 16,000 episodes | is the agent-count failure UNDERTRAINING? | update 790/1000, entropy **1.800 → 1.473**, window rate **0.852** — moving off initialisation, which the 4,000-episode run never did. Looking like YES. |
| `mode_at_scale.py` | does CLAMP earn its keep at 3 private vars on scale-free, in the statistical AND sampled engines? | running, no output yet |
| ladder `w20`, `w30` | last two points of the window ladder | w20 shared done, w30 shared training, solo behind |
| cluster (second agent) | attribution at 20,000 episodes — the decisive rerun | theirs |

Logs in the session scratchpad `logs/`. Results: `results/ladder/`, `results/vs_scale/`,
`results/vs_attr/`, `results/attr_bar1/` and `results/vs_generator/` (both theirs),
`results/transfer/`.

### The second agent's findings, merged and confirmed

1. **The greedy baseline was configured at bar 0.7 while grading happens at 1.0** — see the
   box at the top of this file for the verified scope. Real, and it inverts every
   `attributed` and `version_space` margin.
2. **The attribution learners never converged** — I(S;A)/H of 0.035–0.291 against greedy's
   1.000. So the attribution result is **UNDETERMINED, not negative**. 20,000 episodes
   decides it and is running on the cluster.
3. **The objective mismatch is NOT the cause.** Their attribution-aware greedy
   (`ma/attribution_greedy.py`) is WORSE than plain greedy (0.784 vs 0.824), because unsure
   groups' children are always shared nodes, so the term steers greedy off the private
   probes whose structural pruning was resolving attribution anyway. A clean negative that
   rules out the obvious explanation.
4. **A private probe advances the PROBER'S OWN attribution ~3x more than a partner's**
   (1.265 vs 0.425). This tightens the standing caveat considerably: private probes are
   largely self-serving on the attribution axis too, so budget share is even further from
   being an altruism measure.

## 7. Open questions, ranked

1. **Why does learning collapse past 3 agents?** Partner-feature saturation was tested and
   REFUTED — blanking the partner features HURTS (0.631 → 0.576), so the policy uses them
   productively. Leading hypothesis now: **credit-assignment noise**. At 8 agents each acts
   in 3 of 24 rounds while its reward depends on a window 7 partners disturb, so its own
   contribution is a small share of the variance in its own return. `a08long` tests it.
   Per-agent-heads (`gnn_hybrid`) did NOT help: 0.008 at 8 agents against sharing's 0.100.
2. **Attribution exists at ONE scale.** Its cost grows with PARTNER COUNT, not window size,
   so the factored belief does nothing for it. **A factored ATTRIBUTION — per-pair owner
   distributions instead of an enumerated grouping — is the missing build**, and was the
   recommendation before the mode discussion intervened. Six agents is already out of reach.
3. **Attribution's truth retention under sampled evidence is UNVALIDATED.** The structure
   channel is at 99.2%; the attribution channel was never checked, and an early 12-episode
   reading came out ABOVE the oracle, which is the wrong direction and suggests the
   estimated "moved" set may over-fire.
4. **The 2.3x headline is single-generator** (Erdős–Rényi only). Queued to the cluster.
5. **Clamp vs vary is decided on 12–40 episodes.** Every previous test used ONE private
   variable per agent on ER graphs — the regime where clamp has least to offer, since a hub
   cannot sit in a one-variable private block. `mode_at_scale.py` is the proper test.

---

## 8. Traps — each of these has already cost a job

- **`make_baselines` used to build every arm eagerly**, including the enumerating greedy
  oracle, which refuses past k=5 and raises on non-exact backends. Now lazy — and the lazy
  class itself had a bug (overrode `__contains__`, so its own guard was always true and it
  never built anything). Both fixed and pinned by tests.
- **Hard-coded backend lists.** `ma/evaluate.py` listed the claim backends literally in two
  places; both were missed when a backend landed and a 25-minute run died in its own report.
  Now uses `CLAIM_BACKENDS`. Grep for literals before assuming a new backend is wired.
- **Baselines that count QUERIES not MOVES.** Under turn-taking every policy is queried every
  round and the inactive agent's move is discarded. `ProbeThenWorkAgent` counted queries,
  ran 3x fast, and scored 0.000 on the thing it exists to do. Read `env.own_counts`.
- **Underpowered comparisons.** SIX claims were withdrawn over three days, four for the same
  reason: a difference smaller than its own noise at 30–40 episodes. Both evaluators already
  run every arm on IDENTICAL episode seeds — compute the per-episode DIFFERENCE, and report
  nothing below 150 episodes.
- **Whole-world changes meant for one agent.** An experiment isolating one agent's
  intervention mode set the mode for everyone and could not answer its own question.
- **Adding a backend needs FOUR edits**: `ma/env.py` constants, the window constructor, the
  argparse `choices` in `scripts/ma_train.py`, and any literal backend lists. Missing the
  third killed a launch silently.

---

## 9. What NOT to touch

- `cb/versionspace.py::equivalence_class` — the enumeration core, behind every exact result.
- `ma/nets.py` — verbatim-frozen by `tests/test_depth.py`.
- Existing `results/` directories — write to a new one. A file-clobbering race between two
  evaluation processes has already cost one result set.

---

## 10. Deliverables that exist

- **Supervisor brief**: `docs/brief_2026_08_27.html` (standalone, opens from disk) and
  published at `https://claude.ai/code/artifact/86b61fa3-c8c5-495f-8ae3-33c2ae954dd9`.
  **It predates the k=15 negative, the ladder, and the metric insight — update before reuse.**
- **Scaling-figure mock**: `https://claude.ai/code/artifact/42e4bc09-6aae-4e24-a96a-88086dd4e3c9`
  Three curves on the same episodes; the argument is that identification collapses with
  window size as an ARTEFACT while connections-resolved stays flat (0.93 at k=15). Use
  connections-resolved as the headline curve and keep the others to show the divergence.

---

## 11. THE CHECK THAT MUST RUN BEFORE ANY LEARNED NUMBER IS REPORTED

Three times on 27 August a number was interpreted before checking whether the arm producing
it was functional: the learner blindfolded to the channel it was scored on (26 Aug), the
BASELINE blindfolded (found by the second agent), and the learner never having trained at
all (found by applying their check to our ladder).

`mi_check.py` in the session scratchpad computes, exactly and in two minutes:

    I(S;A)/H   mutual information between observation and action, normalised
               ~0.03  = a fixed mixture wearing a network. Nothing was measured.
               >0.5   = the policy genuinely conditions on what it sees.

Read it alongside final training entropy against its maximum, ln(n_actions). Entropy within
10% of maximum plus near-zero mutual information is the untrained signature. Both are free.

## 12. Immediate next actions

1. **Collect the four running jobs** (§6) and fold them into `FINDINGS_2026_08_27.md`.
2. **Build factored attribution** (§7.2) — the highest-value remaining build, and the only
   item where the remaining days could change what the thesis claims.
3. **Validate attribution truth retention** under sampled evidence (§7.3) — an unvalidated
   component in the contribution chapter.
4. **Update the brief** with the ladder, the k=15 negative, and the metric insight.
