# Decisions taken 31 Aug, and what is still owed

Written 19:30. The decisions below are RULED, not proposed -- do not reopen them without a
reason that is new. The outstanding list is what has to finish before the 3 Sep freeze.

---

## 1. Metric: hard SHD of the pooled global graph is PRIMARY

`success` is the all-agents conjunction and it is the wrong headline for two independent
reasons.

**It saturates.** Every k=12 cell has learned and greedy both between 0.88 and 0.99, so the
comparison lives inside seed noise.

**It amplifies.** At 8-10 agents a per-window rate of 0.98 against 0.90 becomes an episode
rate of 0.94 against 0.61. Seed variance that is modest per window reads as a collapse.

The measured case that settles it: `k12s25n08b150` seed 2 scores **success 0.035** while
recovering the graph to **hard SHD 0.0143**, against random's 0.0537. The conjunction reported
near-total failure for a run that got 98.6% of the graph right.

HARD rather than soft SHD: a pair counts wrong unless the pooled belief settled on exactly the
true mark; soft gives partial credit for an unresolved pair and is the more flattering number.
`scripts/sweep_report.py` now prints the SHD table first and the conjunction below it, labelled.

What this looks like on the runs finished so far:

| cell | k/n | learned | greedy | random | learned vs greedy |
|---|---|---|---|---|---|
| k20s50n04b150 | k=20 | **0.0000** | 0.0005 | 0.0242 | **11x better** |
| k30s50n04b150 s0 | k=30 | **0.0001** | 0.0005 | 0.0177 | **5x better** |
| k12s50n10b150 | n=10 | 0.0009 | 0.0001 | 0.0211 | 6.8x worse |
| k12s50n08b150 | n=8 | 0.0012 | 0.0003 | 0.0252 | 4.2x worse |

**The shape of the result is now legible: greedy wins at k=12, the learned policy wins at k=20
and k=30.** That is the thesis claim -- a myopic rule is enough on a small window and stops
being enough as the window grows -- and SHD shows it where the conjunction did not.

---

## 2. Attribution engine

- **Attribution section: `component_attributed` is primary.** Enumerated stays as the verified
  crosscheck reference. See `FINDINGS_ATTRIBUTION_SCALE_2026_08_31.md`.
- **Rest of the project: `factored` stays primary**, on time grounds. Not revisited.
- **`wrong` is presented as a FINDING about rule 1's local-disturbance assumption**, not as
  engine error. The assumption is named, switchable and its failures are counted
  (`assumption_violations`: 10 / 16 / 41 / 73 / 122 at k = 6 / 8 / 12 / 20 / 30).

**Not doing, deliberately -- FUTURE WORK, do not forget:**

> **Full resolution instead of unit propagation for the cross-component rule-1 clause.**
> Unit propagation is the weakest sound treatment: it applies the clause only when exactly
> one component can still satisfy it. Resolution would recover part of the ~13 decisions
> given up at k=12 (36 right against the enumerated engine's 49). Cut on time, not on
> merit. It is a day of work and it opens a new correctness surface four days from freeze.

---

## 3. The sweep will NOT be re-run on an attribution backend

Cost is ~180 extra core-hours, about 6x the whole current sweep, which does not fit before the
freeze. But cost is not the reason. Under an attribution backend the reward changes -- the
attribution claim REPLACES the bidirected claim -- so it is a different objective and nothing
would be comparable to the 60 runs already in hand. It would be a second sweep of a second
experiment.

Instead: **one cell trained on `component_attributed`** (`k12s50n04b200`, 4 agents, budget 67,
3 seeds, ~4 core-h, running). Paired with the eval-only TRANSFER pass on the same cell, that
gives the comparison the section actually needs -- a policy REWARDED for attribution against
one that merely generates attribution evidence as a by-product.

---

## 4. Two problems found while producing the report

### 4a. The MI gate is excluding cells that are not untrained

The MI floor of 0.15 currently excludes **8 of 16 finished runs**, including all three seeds of
`k12s75n08b150` -- a cell whose learned arm scores **0.885 +- 0.091** with window rates of
0.99 / 1.00 / 0.92. That is not an untrained policy.

The likely cause is that the gate assumes low I(S;A) means "did not learn". At sigma=0.75 the
window is 9 shared nodes against 3 private, and the near-optimal policy there may genuinely be
close to state-independent -- sweep the shared nodes -- so LOW MI IS CORRECT BEHAVIOUR, not
failure. If so the gate is measuring the wrong thing on the high-sigma cells and the headline
("+0.004 +- 0.077 over the 8 runs that clear the gate") rests on a biased half of the data.

**Owed: decide whether the MI floor applies per cell or is replaced by a window-rate floor.**
Not yet actioned. Flagged 31 Aug.

### 4b. An evaluation was run off-distribution and has been redone

The first transfer pass used `--budget 100`, which is `k12s25n08b150`'s budget;
`k12s50n04b200`'s budget is **67**. The policies were trained at 67 and evaluated at 100.
Caught and relaunched at 67. Nothing downstream used the bad numbers.

---

## 5. Cells that still need compute -- the overnight list

### Finished but UNDER-TRAINED (re-run at 12,000 episodes, cluster)

| run | window rate, last 6 checkpoints | reading |
|---|---|---|
| `k12s25n08b150_s2` | 0.12 0.21 0.27 0.29 0.39 0.38 | still climbing; `first_success_episode` 1881 against 3-480 elsewhere. This is the hardest cell (9 private each over 3 shared, 8 agents), not a broken one. |
| `k30s50n04b150_s1` | 0.08 0.12 0.20 0.22 0.16 0.12 | climbing slowly |
| `k30s50n04b150_s2` | 0.06 0.02 0.09 0.05 0.08 0.05 | flat, never reached first success |

Seeds 0 of both cells are fine (0.885 and 0.940), so this is a training-length problem, not a
cell that cannot be learned.

### Modest, worth a longer run if there is room

`k12s50n08b150_s2` (window 0.90), `k12s50n10b150_s2` (0.84), `k12s75n08b150_s2` (0.92). All
three are seed 2. Per-window they are only slightly behind seeds 0 and 1; the conjunction is
what makes them look worse than they are -- see section 1.

### Not yet started -- 14 of 20 cells

```
k04s50n04b150  k08s50n04b150  k12s25n02b150  k12s25n04b150  k12s50n02b150
k12s50n03b150  k12s50n04b100  k12s50n04b120  k12s50n04b150  k12s50n04b200
k12s50n04b500  k12s50n05b150  k12s75n02b150  k12s75n04b150
```

These are the cheap tier and should clear overnight at 4 workers.

---

## 6. Running as of 19:30

| job | detail |
|---|---|
| Oracle sweep | 16/60 runs, 4 workers |
| D7 attribution training | `k12s50n04b200_attr` seeds 0-2, sequential, ~4-6 h |
| Transfer eval | `k12s50n04b200` seeds 0-2 at budget 67, 2 at a time |
| Probes (queued) | D6 mechanism, D3 sound-only sensitivity, D5 k=30 scope trade -- start when the transfer pass frees its slots |
| Myriad | sampled sweep, job array 246859, 66 tasks |
