# Overnight agenda, Sat 5 -> Sun 6 Sep (agent A; Brian asleep; morning report owed)

Standing orders from Brian: work autonomously, sceptically; if the no-skeleton runs finish
early or look wrong, dig into the problem and solve it; if they work as expected, extend to
more cells. Ticks every 30 min (session cron a60c07f2).

## Running jobs (check every tick)
1. No-skeleton TRAINING, k=8, 3 seeds x 12,000 eps, skeleton_source=estimated
   (results/noskel/k08_estskel_s{0,1,2}.json, train_s*.log). On completion: window rates vs
   floor (expect under-floor; report as the answer, never exclude silently), then
   global_shd_paired --override_skeleton estimated at both checkpoint conventions, paired
   against greedy/random under the same skeleton. Compare with the transfer inversion
   (learned-below-random, results/noskel/k08_estskel_transfer.json). If clean -> extend:
   k=12 cell, same recipe. If broken -> diagnose before rerunning anything.
2. Disclosed n_int sweep tails, n=10,000 x 3 seeds (results/nint_disclose/). On completion:
   convergence figure (disclosed vs undisclosed vs oracle reference lines from sweep12k k08),
   rewrite sec:res_nint from the complete grid, extend C8 (or C10) with the disclosure
   mechanism now quotable from measurements, registries, push both repos.

## Offline agenda (between job events, priority order)
3. scripts/check_mustnots.py: grep Ch4/Ch5/Abstract prose against every MUST NOT phrase in
   CLAIMS.md (forbidden: "neglects", "15 of 18", agent-count reversal asserted, ratio of
   means at k30, "exactly unchanged", "centralised" for arm E, mechanism claims on C8, ...).
   Wire into the Monday checklist.
4. Reference/label consistency scan: every \ref/\cref target defined exactly once across
   thesis/*.tex; every figure file referenced exists; report dangling.
5. Abstract refresh: verify all its numbers against CLAIMS.md; fix factual drift only.
6. Ch4 fact-check pass (pre-empting B's Saturday audit): every number in the experiment
   paragraphs recomputed against thesis_results/.
7. Sync agent B each tick that has news; their Saturday audit assignment stands.

## Rules
Recompute before quoting; no partials as final; thesis/ -> Overleaf only; corrections in
writing where the wrong thing was said; one consolidated report for Brian in the morning.
