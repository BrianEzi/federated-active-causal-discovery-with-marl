# Session state — 29 August 2026

**Resume point.** §1 is RESOLVED as of 29 Aug — see
[`FINDINGS_SHD_2026_08_29.md`](FINDINGS_SHD_2026_08_29.md), commit `f276751`. The mechanism
recorded in §1 below is RETRACTED; the banner at the top of §1 says what replaced it. The
next thing to run is the sampled-evidence SHD, for a stronger reason than §1 gives.
Design freeze **31 Aug morning**, experiments to **2 Sep morning**, write-up to **7 Sep 3pm**,
24h buffer to the 8th.

Companion documents: [`OBJECTIVE.md`](OBJECTIVE.md) (what the project is for),
[`ACTION_PLAN_FINAL_2026_08_28.md`](ACTION_PLAN_FINAL_2026_08_28.md),
[`ROADMAP_AGENT_B_2026_08_28.md`](ROADMAP_AGENT_B_2026_08_28.md).
Shareable: candidate slate https://claude.ai/code/artifact/44a05921-2f17-4c5b-af33-ebd726496dda
· evidence ledger https://claude.ai/code/artifact/35dd0c42-0414-42b7-9b25-48651a84ee48

---

## 1. ~~THE LIVE THREAD~~ — RESOLVED 29 Aug, see `FINDINGS_SHD_2026_08_29.md`

> **What this section got wrong.** The mechanism below — "joint success is zero-tolerance" —
> is **RETRACTED**. Under argmax the learner beats greedy on *per-window* solve too, so it is
> not a joint-versus-marginal mismatch. The real answer is that under oracle evidence the
> factored belief is structurally incapable of being confidently wrong (WRONG bucket is
> exactly 0.0000 for every arm at every rung), so soft SHD per pair is identically
> `1 - 1/|surviving marks|` — a count of residual ambiguity, not a distance to truth. And
> `UncertaintyGreedyAgent`'s decision rule is *literally* the count of nonzero-SHD pairs
> incident to each node (6,976 node-scores compared, 0 disagreements), so the baseline
> descends the evaluation metric by construction. Two measured channels: hub-seeking, and
> spending 44–71% of moves on the shared surface where soft SHD's window-average pays n times.
> Also: the table below evaluates the learned arm with `deterministic=False`; argmax roughly
> halves the gap at w08 and w12.
>
> **Do not quote the table below as "greedy's belief is closer to the true MAG."** It is not
> what was measured.

Second agent's commit `cacd4e1`, `scripts/shd.py`, results in `results/cover/shd_ladder.json`.
Soft (mass-weighted) and hard (MAP) structural Hamming distance, sanity-checked on three
hand-built cases first.

| rung | greedy soft SHD | learned soft SHD | learned wins joint success? |
|---|---|---|---|
| w04 | 0.0098 | **0.0095** | yes (tie on SHD) |
| w08 | **0.0107** | 0.0230 | yes |
| w12 | **0.0053** | 0.0136 | yes |
| w20 | **0.0049** | 0.0129 | yes |
| w30 | **0.0154** | 0.0209 | both 0.000 on success |

(one seed per rung in this file; random_vary is 0.07–0.10 throughout, so the metric does
separate competence from noise.)

**Greedy's belief is closer to the true MAG at every rung except k=4, while the learner wins
joint success at every rung.** Same checkpoints, opposite orderings.

**The w30 row kills the easy dismissal.** Both arms score exactly 0.000 on joint success
there, so this is not one arm having given up more than the other — under identical
structural impossibility greedy's belief is still measurably closer.

~~**Mechanism, theirs and I agree:** joint success is ZERO-TOLERANCE, so it rewards
concentrating confidence on whichever claims complete a window rather than minimising
average structural error. Two different objectives; our policy optimises the first.~~
**RETRACTED 29 Aug** — the learner also wins per-window solve under argmax. See the banner.

**Why it is urgent.** The student's result 2 was "under realistic sampled inference the RL
policy beats greedy at fixed budget, shown as SHD against budget". On ORACLE evidence SHD
says the opposite. So either that figure becomes one we lose, or we report both metrics and
say plainly that they answer different questions. **Report both** — a reader who sees only
the success plot and later computes SHD will conclude we hid something.

**THE OPEN QUESTION, and the next thing to run:** does the ordering REVERSE under sampled
evidence? That is where the convergence story actually lives (the noise dial already shows
the learned margin GROWING with data quality: +0.053 / +0.100 / +0.123 at n_int 100 / 1,000
/ 4,000). If greedy leads on SHD under oracle but the learner leads under noise, that is a
coherent and interesting story. If greedy leads in both, the SHD figure cannot be the
headline and the thesis leads on success rate with SHD reported as a limitation.

Secondary: get more than one seed per rung into the SHD table before quoting it.

---

## 2. Established this session

- **Forced-cover characterisation.** Under oracle evidence the belief is a deterministic
  function of the SET of intervened nodes; a repeat does nothing. A directed edge is settled
  by intervening on its TAIL; a confounded pair needs BOTH endpoints. So the required set is
  FORCED, not chosen, and the optimum is closed-form. **Oracle-only** — under sampled
  evidence repeats do add data. Measured: required cover falls from `0.757k` at k=4 to
  `0.542k` at k=30, so budget ∝ k quietly favours large windows. Independently confirmed by
  their `15ff6c0`: w30 climbs exactly where the math predicts.
- **Free-riding.** An agent's return tracks its PARTNERS' causal contribution more closely
  than its own at every agent count (3.2× at eight), and NEGATIVELY at two and three
  (−0.247 ± 0.076, −0.135 ± 0.042). The reward is a function of the STATE of a window, not
  of who moved it.
- **Reward scale is the agent-count collapse.** Per-agent return grows 1.66 → 11.86 from 2
  to 8 agents; MSE value loss squares it; the shared trunk carries it into the policy
  gradient. `reward_scale=0.214` took eight agents from 0.110 to 0.653 (2 seeds). Their
  `normalise_returns` then reproduced it with no hand-picked constant and beat it: a08
  0.665/0.695, a06 0.213 → **0.810** (from losing to greedy by 0.440 to beating it by
  0.200), a03 0.833 → 0.795.
- **The window ladder is clean on that confound** — returns sit flat at 1.6–3.5 across k
  while the agent axis spans 7×.
- **Coordination crossover, retrained.** Learned beats the positional convention at 3 agents
  (+0.160), TIES at 6 (−0.040 ± 0.038), loses at 8 (−0.200 / −0.133). Learned duplicate
  coverage at 6 agents fell 0.491 → 0.133 after the retrain.
- **Transfer to sampled evidence fails.** Learned beats greedy under oracle (0.610 vs 0.470)
  and ties under sampled (0.874 vs 0.868). Caveat: that dense metric is compressed — random
  scores 0.83 on it.
- **Turn order is not the learner's crutch.** Random turn order costs the learned policy and
  greedy the SAME (−0.247 each).
- **Clamp refuted** on hub-heavy graphs: clamp-only 0.233 against vary-only 0.589.
- **JCI read in full** (see §5).

## 3. RETRACTED this session — do not re-quote these

- **"Argmax as primary"** — held on the window ladder, REVERSES on the agent ladder (up to
  −0.153 ± 0.040). Not a general recommendation.
- **Credit assignment as the cause of the agent-count collapse** — withdrawn on a
  pre-registered criterion when `scale21` scored 0.620. The free-riding measurement stands;
  it was not the binding constraint.
- **My Phase 0 verdict ("HEADROOM EXISTS")** — broken three ways: omniscient benchmark so
  the kill criterion could never fire, solve rates 0.07–0.47 against censor 13 so the metric
  measured failure not efficiency, and `attribution_greedy` never registered so it never ran.
- **"`vs_evidence` is silently ignored by the attributed backend"** — FALSE. `ma/env.py:886`
  branches on it and routes to `estimated_moved`. I inferred a data path from a constructor
  signature.
- **"The false-attribution measurement needs a build"** — FALSE. `score_groups` already
  returns right/wrong/unsure.
- **The seed priority I gave the other agent** — w04–w12 already had 3 seeds.
- **My first `PartitionedGreedyAgent` did not coordinate** (dup 0.167 vs plain 0.169, worse
  at k=12). Any head-to-head before commit `7d572a8` is void.
- **The MI gate is a FLOOR, not a quality measure.** Near-zero voids a number; above the
  floor it measures commitment to the TRAINING objective and must never be compared across
  arms with different objectives. I did exactly that quoting "MI 0.035 → 0.546".

## 4. My track (attribution) — BLOCKED, and where

- **`score_groups` fails its own control.** Under ORACLE evidence it reports `wrong` at
  0.075–0.113 where its docstring guarantees wrong cannot occur.
  **Leading hypothesis:** `group_frequency.get(group, 0.0)` returns 0 for a true group absent
  from the belief's enumeration, and at `bar=1.0` that scores `1 − 0 ≥ 1.0` → WRONG. So
  "wrong" may mean a canonicalisation mismatch, not a false attribution. Supporting evidence:
  the rate is IDENTICAL across greedy, probe and random, which a policy-driven error would
  not be. **Everything on this axis is gated on resolving it.**
- Reward magnitude on the attribution config is **3.05** — healthy band. Reward scaling will
  NOT unlock attribution.
- The 20,000-episode attribution learners **did train**: I(S;A)/H = 0.262 / 0.432 / 0.275.
  So the attribution negative is on real policies, not untrained ones.
- Two untested levers: those runs used **`per_agent_reward: False`** (one shared scalar, the
  free-riding pathology in its strongest form) and **`policy_arch: gnn`**, never the portable
  architecture that won the window ladder.
- Second agent's finding that bears on framing: **targeting confounded pairs directly is
  WORSE than resolving structure** (`attribution_greedy` 0.784 vs plain greedy 0.824), and
  plain structural greedy is the best attribution policy in the repo without targeting
  attribution at all.
- The cost wall is **partner count**, not density or window size. `cb/factored.py` makes
  windows cheap and does nothing for partner count.

## 5. JCI (Mooij, Magliacane & Claassen, JMLR 2020) — read in full

Full notes in [`BIBLIOGRAPHY.md`](BIBLIOGRAPHY.md) §19, assumptions now VERBATIM.

- **JCI 3 was recorded inverted** in our bibliography and is now fixed. It asserts every pair
  of context variables IS connected by a bidirected edge — which is why FCI-JCI123 does not
  remove those edges in its adjacency phase.
- **Adaptivity breaks JCI Assumption 1, and the paper says so** (§3.4.2's doctor who
  diagnoses before treating; footnote 15 on protocols fixed beforehand). Active design is
  outside JCI exogeneity by construction. State it first.
- **JCI cannot handle different variables per context** — Table 4 minus for JCI, FCI-JCI and
  ASD-JCI alike; §4.3.7 needs strengthened faithfulness; §6 lists it as future work. Our
  vertical setting is in that named gap.
- **Multiple context variables beat one merged variable** (§4.3.5, Figs 20/23). Our `clean`
  is a SCALAR fraction per row batch — a merged context variable — and `ma/env.py` already
  documents the symptom: the mixture "knows how MANY hidden nodes were clamped, never WHICH".
  That sentence is the attribution problem, written before it had the name.
- **Deterministic relations between context variables violate faithfulness** (§4.1). Under
  round-robin, whose turn it is is a deterministic function of the round. Their remedies:
  group context variables, or omit contexts.
- Worth stealing: ROC **and** PR curves for presence AND absence, bootstrapped confidence
  scores, runtimes reported throughout, and their refusal to treat the Sachs consensus
  network as ground truth.

## 6. Standing rules

- **MI gate before any learned number is quoted.** Floor only, never across objectives.
- **Matched pairs on identical episodes**, and state the seed count in the table.
- **Record the evidence regime in every result file.** The attribution `*_scored.json` files
  do not carry `vs_evidence`, which is why nobody could tell they were all oracle.
- **A control that cannot fail is not a control.** Phase 0 is the worked example.
- No bare `python` — use `.venv/bin/python`.

## 7. What is NOT running

Nothing. No jobs on this machine as of 29 Aug 15:08.
