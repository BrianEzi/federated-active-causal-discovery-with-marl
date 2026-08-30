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
