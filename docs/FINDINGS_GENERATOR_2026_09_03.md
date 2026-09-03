# The generator control: the advantage is not a scale-free artefact, and the myopic rule is

3 Sep 2026, 05:5x. Three Erdos-Renyi seeds at the principal cell's configuration, 12,000
episodes, current constraint engine, measured with `global_shd_paired.py` at the selected
checkpoint (final-update convention still measuring). The scale-free comparator is
`results/sweep12k/shd/k12s50n04b150.json`, measured identically. This replaces the 27 Aug
`results/vs_generator/` runs, whose belief backend the thesis no longer uses -- the gap agent C
found in `sec:meth_ladder`.

## The comparison is fair before it is interesting

* Same flags except `--graph_model`; `prior_p = 0.227` in both configs.
* Mean true edges over 30 draws: **ER 50.0, SF 53.6** -- near-matched density (7%), so the
  family is the operative difference. My own caveat at queue time (unmatched edge counts) was
  wrong for this pair and is withdrawn.
* ER competence: window rates 0.997, 1.000, 0.909 -- all clear the 0.70 floor.

## The numbers

| | learned | myopic | paired learned $-$ myopic |
|---|---|---|---|
| ER seed 0 | 0.00000 | 0.03384 | $-0.03384 \pm 0.00449$ |
| ER seed 1 | 0.00002 | 0.03518 | $-0.03516 \pm 0.00497$ |
| ER seed 2 | 0.00142 | 0.04769 | $-0.04628 \pm 0.00606$ |
| SF (same cell) | 0.00065 | 0.00077 | mixed, one seed significant |

Joint recovery from the runs' own criterion: learned 1.000 / 1.000 / 0.925 against the myopic
rule's 0.400 / 0.400 / 0.395.

## What may be claimed

1. **The learned advantage is not specific to the scale-free generator.** On ER it is present
   on 3 of 3 seeds at 7 to 9 standard errors. The examiner's question the control exists to
   answer is answered in the strong direction.
2. **The arm the generator hurts is the myopic rule.** The learned arm performs at the same
   near-zero level on both families (0.00048 against 0.00065); the myopic rule degrades
   fifty-fold (0.0389 against 0.00077) and its recovery rate falls from 0.918 to 0.400.

## What may not, yet

* Nothing at the final-update convention until that measurement lands (running).
* No mechanism. Why uncertainty targeting collapses on a uniform-edge family while surviving a
  hub-concentrated one is an interpretation question, and it is Brian's. The observation that
  SF concentrates structure on hubs while ER spreads it across the window is the obvious
  starting point, recorded here as a pointer and not a claim.
* No magnitude comparison beyond "present on both": the SF cell is near saturation for both
  arms, so the ER margin cannot be read as "the advantage is 50x larger on ER" -- SF has no
  room to show a margin of that size.
