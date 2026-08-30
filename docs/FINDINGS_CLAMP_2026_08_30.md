# The clamp-only result, re-examined

Investigated 30 Aug 2026 because the recorded number looked too large to believe. It was.
The **direction** survives; the number, the stated mechanism, and the generality do not.

## What was on the books

> "Does clamp earn its keep on hub-heavy graphs? Going badly for the hypothesis: clamp-only
> **0.233** vs vary-only **0.589**, 2 of 4 arms in." — `PLAN_2026_08_28.md`
>
> "Clamp on hub-heavy graphs? **Refuted.** Clamp-only 0.233 against vary-only 0.589."
> — `ROADMAP_AGENT_B_2026_08_28.md`

Three problems with it as evidence, before any measurement:

1. **`mode_at_scale.py` is not in the repository and never was** — `git log --all` finds no
   commit that ever added it. The result is unreproducible by construction.
2. **It was two of four arms.** `PLAN_2026_08_28.md` says so in the same sentence that
   reports the number, and `ROADMAP_AGENT_B` then promotes it to "Refuted" with the caveat
   dropped. The script was cut "after five hours for two lines".
3. **0.233 is also the greedy-bar handicap** (`+0.233` to greedy at four agents, recorded
   in `METRICS.md`, `NOTE_ATTRIBUTION_ALREADY_BUILT` and `SESSION_STATE_2026_08_30`). Two
   unrelated findings carrying the identical three-decimal value is at least worth noticing.

## Finding 1 — mode is PROVABLY INERT under oracle evidence

`cb/factored.py::edge_marginals` on the oracle branch:

```python
fresh = intervened - self._applied
for x in fresh:
    self._apply_ancestry(x, reveal(self.truth, self.k, x))
```

The update is a function of the intervened **set** and the true MAG. The mode is not an
argument, and the data matrix is not read. Measured, 4 agents / 3 private / 3 shared,
identical seeds, greedy:

| evidence | modes | success | steps | soft SHD |
|---|---|---|---|---|
| oracle | `vary` | 0.900 | 4.20 | 0.0010 |
| oracle | `clamp` | **0.900** | **4.20** | **0.0010** |
| oracle | `vary, clamp` | 0.900 | 4.20 | 0.0010 |

Identical to every reported digit.

**Consequence for the run plan: `--vary_only` is a no-op for the entire ORACLE sweep.** All
60 runs would be unchanged by dropping it. Any concern that vary-only removes a move the
thesis needs applies only to the sampled runs — and Finding 2 says what happens there.

## Finding 2 — under sampled evidence clamp really is worse, and it is NOT a power effect

The recorded rationale was that mode is "a FINITE-SAMPLE phenomenon" and clamp's advantage
is "a sharp unambiguous signal (the association vanishes) against vary's graded one, and
that is a power argument". If that were right, the gap would close as samples grow. It does
not:

| n_int | vary success | clamp success | vary soft SHD | clamp soft SHD | ratio |
|---|---|---|---|---|---|
| 20 | 0.020 | 0.000 | 0.0618 | 0.2061 | 3.3x |
| 100 | 0.280 | 0.000 | 0.0256 | 0.1411 | 5.5x |
| 400 | 0.300 | 0.020 | 0.0198 | 0.0969 | 4.9x |
| 1000 | 0.260 | 0.000 | 0.0250 | 0.0937 | 3.7x |

Fifty times the data and the gap is unchanged. It is structural, not statistical.

## The actual mechanism, and why it is more useful than "clamp does not earn its keep"

There are **two different channels**, and clamping helps one and destroys the other:

- **Ancestry / orientation.** `estimated_reveal` detects that x is an ancestor of y by an
  association between them once x is intervened. A clamped node is set to a CONSTANT: zero
  variance, therefore no association with anything, at any sample size. Clamping removes
  exactly the signal this channel consumes.
- **Cutting a confounding path.** Here clamping is the only move that works — a varied node
  still drives its children, so it stays an active variance source. `ma/scm.py` records the
  measurement: at scale 2.0 or 1.0 a do() on the confounder restores 0.0% of a confounded
  agent's identification; at scale 0.1 or 0.0 it restores ~18% and lifts posterior mass on
  the truth from 0.0000 to 0.39.

**The factored backend builds its belief entirely from the first channel.** `edge_marginals`
never receives the mode; clamping reaches it only through the data, and only by removing
variance. So on this backend clamp can only hurt, and no sample size rescues it. The
"association vanishes" argument was sound for the confounding channel and was applied to
the ancestry channel, where it is backwards.

## Finding 3 — attribution survives vary-only, but for an ACCIDENTAL reason

Asked directly: does removing clamp make attribution impossible? Attribution needs a
partner's private intervention to MOVE a pair it confounds, and clamping is the only move
that cuts a confounding path — so the concern is well founded. It turns out not to bite,
and *why* it does not is the interesting part.

`estimated_moved` tests for a **difference of two correlations** (Fisher's z), not for the
association vanishing. Both modes produce a difference:

| | corr(u,v) | detected as moved |
|---|---|---|
| baseline, w free | +0.5014 | — |
| after do(w) by **clamp** | +0.0021 | 90.5% |
| after do(w) by **vary** | **+0.8019** | **92.5%** |

Clamping makes the association vanish. Varying at `intervene_scale=2.0` makes it *stronger*,
because it replaces the confounder's natural variance (~1) with 4 and so raises its share of
the u–v covariance. Both are changes; the detector sees both.

**But vary's signal is a variance CONTRAST, not a structural one, and it disappears when the
contrast does.** Detection as a function of the intervention scale, against a confounder
whose natural noise scale is drawn from U(0.5, 1.5):

| intervene_scale | vary detects |
|---|---|
| 0.5 | 63.0% |
| 0.8 | 35.5% |
| **1.0 — matches the natural scale** | **22.0%** |
| 1.2 | 30.5% |
| 1.5 | 55.5% |
| **2.0 — our setting** | **92.5%** |
| 3.0 | 100.0% |
| **clamp — any scale** | **90.5%** |

A V-shape with its minimum exactly where the intervention scale equals the natural one. Our
attribution signal exists because `intervene_scale=2.0` happens to sit at roughly twice the
SCM's noise scale. Clamping is scale-free by construction.

**What this changes.**

- Attribution under vary-only is **viable**, so the limitation is not "attribution is
  impossible" and the clamp decision does not block that chapter.
- Any attribution result MUST report `intervene_scale` beside it, and should carry the
  sensitivity curve above. A reviewer who asks "why 2.0?" currently has no answer, and the
  honest one is that the signal degrades sharply toward 1.0.
- It strengthens the case for clamp as a **future-work** item rather than weakening it: clamp
  reaches the same signal for a structural reason instead of a numerical coincidence.
- `intervene_scale` is a knob nothing in the sweep varies and no result file emphasises. It
  now has a measured effect on a headline mechanism, so it belongs in the parameters table
  rather than in a default.

## What to say, and what to stop saying

- **Stop quoting 0.233 vs 0.589.** Unreproducible script, two of four arms, and the value
  collides with an unrelated finding. `PLAN_2026_08_28.md` and `ROADMAP_AGENT_B_2026_08_28.md`
  should be corrected rather than cited.
- **Stop saying "clamp is refuted."** It is refuted *for the factored backend*, for a
  structural reason specific to how that backend gets its evidence.
- **Do say:** vary-only is the right choice here, and it is justified by the backend's
  evidence channel rather than by an empirical gap. On the oracle sweep the choice is inert;
  on the sampled sweep it is correct and now has a mechanism behind it.
- **The open question this leaves.** Clamping is the only move that cuts a confounding path,
  and the thesis is about confounding. That the factored backend cannot exploit it is a
  **limitation of the backend**, not a fact about the problem — and it is worth a paragraph
  in the limitations, because a backend that used both channels would have a strictly larger
  action repertoire available to it.
- **And the sharper version, from Finding 3:** attribution does not need clamp, but the
  substitute it relies on is a variance contrast that is strong at our chosen intervention
  scale and weak at the SCM's own. That is a fragility to state, and a sensitivity curve to
  publish, not a gap to hide.

## Finding 4 — "rescue" does not exist on this backend, and that answers the hard question

The hard question is: *you use vary, and you picked an intervention scale that gives vary a
detectable signal in your setup — why not just use clamp?* Without an answer, `--vary_only`
looks like a convenience dressed as a design.

There is an answer, and it is a measurement. Agent 0 sweeps its own window with vary; the
only thing that changes between arms is what the PARTNER does to its private nodes. Scored
on the pairs that are genuinely bidirected in agent 0's true MAG:

| evidence | partner's mode | confounded pairs settled right |
|---|---|---|
| oracle | vary | 100.0% |
| oracle | **clamp** | **100.0%** |
| sampled | vary | 61.8% |
| sampled | **clamp** | **61.8%** |

**A partner clamping the confounder helps exactly as much as varying it: not at all.**
Combined with Finding 2 — clamp-only costs 3.3x to 5.5x on soft SHD for the agent's own
inference, at every sample size — clamp is **dominated** on this backend, not merely
unchosen. That is the answer: not "vary is good enough", but "clamp buys your partner
nothing here and costs you your own experiment".

**Why rescue vanished, and it is structural rather than incidental.** The factored backend
seeds its skeleton from truth (`reset_marks` reads `self.truth`), so adjacency is KNOWN and
only the mark is open. A confounded pair is then settled by intervening on BOTH endpoints
and finding neither is an ancestor of the other — the state of a third node cannot enter.
On an engine that must LEARN the skeleton, a live confounder makes u and v look adjacent,
and clamping it genuinely rescues the partner. That is the engine the rescue measurement
was taken on.

## The document that now contradicts the code

`MA_PROBLEM_STATEMENT.md` says, of the mode split:

> *For your partner*, only CLAMP works... Rescue rate is 0.000 at `intervene_scale` 2.0 and
> 1.0, and rises only as the scale goes to zero. So **clamping is a genuine sacrifice**: it
> removes a confounder for your partner at the cost of a much weaker experiment for
> yourself. **That trade-off is the coordination problem**, and it is measured rather than
> assumed.

Every clause is true of the bootstrap engine it was measured on, and the last one is false
of the engine the thesis now runs. On the factored backend there is no sacrifice to make,
because there is nothing to rescue. **The coordination problem here is ALLOCATION** — not
duplicating effort on the contended surface — which is what `ROADMAP_RUNGS` already says
leads, and what the duplicate-coverage and effort-evenness metrics measure.

This inconsistency should be fixed in the problem statement before submission rather than
left for a reader to find, because the two framings imply different headline claims.

## Where `intervene_scale` actually bites, and how to defend it

Narrower than feared. It has no effect under oracle evidence, and none on rescue, which
does not exist. It affects the **attribution** signal only (Finding 3).

The defence is not realism. It is that **an intervention which reproduces the observational
marginal is uninformative by construction** — so the interventional scale must sit outside
the natural noise range or the experiment carries no information about what the node drives.
`noise_range=(0.5, 1.5)` is itself principled: it spans 3x specifically to avoid the
EQUAL-VARIANCE condition, under which a linear-Gaussian DAG becomes identifiable from
observational data alone (Peters & Buhlmann) — which would hand the agents the answer for
free. `intervene_scale=2.0` sits clearly outside that range, and the V-curve minimum at 1.0
is exactly the degenerate uninformative case.

State the requirement, publish the curve, and report the value in the parameters table. It
is then a design constraint with a measurement behind it rather than a magic number.
