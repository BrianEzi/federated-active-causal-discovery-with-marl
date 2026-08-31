# Resume point — per-pair factored attribution

Written 31 Aug 16:34, immediately before coding, so a compact loses nothing.

---

## 1. THE DESIGN PROBLEM I HIT ONE MINUTE BEFORE STARTING — read this first

I proposed per-pair factored attribution as "the same trick that made the structure belief
scale". **Having now read `cb/attribution.py::consistent_with_partner`, that is wrong, and
naive per-pair factoring would destroy the channel.** Both pruning rules are JOINT across the
pairs of a group; neither decomposes to a single pair.

```python
for group in groups:
    pairs = set(group.pairs())
    hit = pairs & moved
    if hit and hit != pairs:
        return False                       # RULE 2, ATOMICITY -- joint over the group
    if group.owner == owner:
        covered |= pairs
return bool(moved & covered)                # RULE 1, LOCAL DISTURBANCE -- joint over the owner
```

- **Rule 2 (atomicity, sound unconditionally):** one latent moves as a unit, so a candidate
  that assigns a clique to a latent and then sees only PART of that clique move is refuted.
  Refuting requires knowing *which pairs share a latent* — the clique structure — which is
  precisely the joint fact a per-pair belief cannot represent.
- **Rule 1 (local disturbance, an assumption):** at least one moved pair must be covered by a
  group this candidate attributes to `owner`. Also a statement about a group, not a pair.

And the docstring records what happens when the discriminative rule is dropped: **`right`
collapsed from 72 to 0** over 162 groups, because atomicity alone never refutes enough
candidates to reach bar 1.0. So a factoring that loses either rule loses the chapter.

### The fix: factor by CLIQUE, not by pair

Cliques are the natural unit — they are what a latent *is*, and both rules are statements
about cliques. So:

- enumerate **candidate cliques** over the confounded pairs (`maximal_cliques` already does
  this, and it is polynomial in the sparse case), not whole assignments;
- keep, per clique, the set of owners still possible for it;
- apply atomicity per clique (does this clique's pair set move entirely or not at all?) and
  local disturbance across the cliques an owner still holds.

Cost goes from `(2^(n-1) - 1)^P` in the pair count to roughly `n x C` in the clique count.
**This is the design to build. Do not build the per-pair version.**

**Verify before coding:** confirm that atomicity applied per candidate clique refutes the same
candidates as atomicity applied per hypothesis. If a clique appears in several hypotheses, the
per-clique test may be weaker. If it is weaker, say so and measure the cost rather than
shipping it quietly — that is the same trade the factored structure belief makes, and it is
acceptable if it is *conservative* (stays unsure) rather than *unsound* (settles wrongly).

## 2. What already exists and works — do not rebuild

- **`cb/factored_attribution.py`** (251 lines): `FactoredAttributedBackend` +
  `FactoredAttributedBelief`. Factored structure (`FactoredBackend`) + **enumerated**
  ownership. Works to **k=12**. Wired into the env as `--backend factored_attributed`.
- **`tests/crosscheck/test_factored_attribution.py`** — 7/7 passing. Asserts it never
  contradicts the enumerated backend and never settles an attribution wrong.
- **Measured:** k=6 46/76 right (60.5%), k=8 35/72 (48.6%), **k=12 10/39 (25.6%) with 0 wrong
  at `max_attribution_pairs=5`**, 2.97 s/episode. At cap 4 two attributions came out WRONG, so
  5 is the floor for soundness.
- **The wall:** attribution hypotheses are 5 / 35 / 482 / 8.4e10 / 8.9e15 at k = 4 / 8 / 12 /
  20 / 30. k>=20 is what the clique factoring is for.
- **Two defects already found and fixed** in this work, both "an incomplete belief read as a
  confident one": `scope` (16 of 76 groups scored WRONG with no evidence at all — beliefs now
  carry `scope` and `score_groups` treats out-of-scope as UNSURE), and contradiction handling
  (`break` discarded later messages; now `continue`).

## 3. Live state at the moment of writing

| | |
|---|---|
| **Oracle sweep** | **5/60.** Done: `k12s50n10b150` s0/s1, `k30s50n04b150` s0/s1/s2. Running: `k12s25n08b150` x3, `k12s50n10b150` s2. 4 workers, no contention, machine healthy (0 swapins). Expected to finish overnight. |
| Second machine | told to launch the **sampled sweep on Myriad** (66 tasks, `n_int=200` baseline), then profile both machines, then seeds 3-5 on headline cells. Nothing pushed from them yet. |
| Results pipeline | `scripts/sweep_report.py` built and working on partial results. `--figures` works. |
| Killed at 18:15 | the six `k08i*` learned-sampled jobs, which were taking 60% of the machine |

**HEAD is `849cdaa`.** Everything is committed and pushed.

## 4. The deadlines that govern everything

- **EOD 1 Sep — first draft for the supervisor.** Needs the oracle sweep results; attribution
  and Rung 3 are labelled RUNNING.
- **End of 3 Sep — the real freeze.** All compute finished AND analysed. Anything still
  running is abandoned rather than waited for.
- 4th results/discussion, 5th future work/limitations/conclusion, 6th whole-thesis flow,
  **7th submission**. Target grade 80+, so robustness beats novelty where they compete.

## 5. Decisions already taken — do not reopen

- Sweep trains with `--turn_aware_credit --local_epochs 4` (plain FedAvg). **Not** FedYogi:
  best at k=8 (0.993) and collapsed at k=12 (0.332, two of three seeds at zero).
- Sampled baseline `n_int=200`, axis (50, 200, 800). At `n_int=20` neither machine found any
  separation between greedy and random.
- `n_obs=60` is inert under oracle, measured. Unjustified under sampled but `n_int` dominates.
- k=30 is **under-trained, not collapsed** — seed 0's window rate goes 0.27 -> 0.91 -> 1.00
  over the last fifty updates. Re-run at 12,000 episodes is queued for the cluster.
- **Cut:** attribution under sampled evidence; training on the attributed backend; Rung 1
  exact; the ER arm; C3; E4; C1; chasing the random-SHD-worsens-with-data anomaly.

## 6. First actions on resume

1. Re-read `cb/attribution.py::consistent_with_partner` (lines ~374-447) and confirm the
   clique-factoring analysis in section 1.
2. Build `FactoredCliqueAttributedBackend` alongside the existing class — do not replace it;
   the enumerated one is the crosscheck reference.
3. Extend `tests/crosscheck/test_factored_attribution.py` to cover the new backend against the
   enumerated one at k<=8, with the same one-sided assertion: never more decided, never
   differently decided, never wrong.
4. Only if that passes: measure at k=20 and k=30, which is the whole point.
5. Then the eval-only attribution pass over the sweep's finished k<=12 checkpoints (~2 core-h).
