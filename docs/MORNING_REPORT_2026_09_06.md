# Morning report, Sunday 6 Sep (overnight work, agent A)

## The two results of the night

1. **The disclosed n_int sweep completed, and your prediction was right.** Under the one-bit
   disclosure the U is gone: error is monotone in n for every arm, the learned arm crosses
   the myopic rule between n=100 and 200 and leads at every larger value (3/3 seeds by sign
   at 1,000 and 3,000), both arms converging toward -- not attaining -- their oracle
   references. The two-panel figure (fig:nint), the rewritten sec:res_nint, and CLAIMS C8a
   (with the measured mechanism: false detections 3.1%->17.3%, 0/1,100 full-graph, A/B
   3.25%->0.16%) are in and pushed. The Abstract now carries the finding in two clauses.

2. **The no-skeleton question is answered at two cells.** Trained WITH the supplied skeleton
   and evaluated without it, the policies invert below random (12-20 SE). Trained WITHOUT it,
   the learned edge RECOVERS: ahead of greedy 6/6 seeds across k=8 and k=12 (3-5 SE), ahead
   of random everywhere, in a regime where joint recovery is zero for every arm including
   oracle_cover. The assumption is load-bearing for the POLICY, not the METHOD; the extension
   direction is training against skeleton uncertainty. docs/FINDINGS_NOSKEL_2026_09_06.md.
   Nothing entered Ch4; one candidate limitation sentence is drafted there for your pass.

## Housekeeping done
- check_mustnots.py written (forbidden-phrase gate, arm-E-scoped): CLEAN.
- Ref/label scan: dangling sec:meth_publicgood fixed by restoring the shelved paragraph as a
  \paragraph (VETO NOTE in place -- you shelved it 2 Sep, but two live refs depend on it;
  delete paragraph + both referencing clauses together if you still want it gone).
  fig:coordination and fig:attribution_law now referenced from prose.
- Abstract: all numbers verified against CLAIMS; finite-sample sentence re-scoped.
- Ch4 fact-check: 15 numeric statements recomputed; 14 exact; ONE error fixed (credit k12
  quoted the seed SUM 0.02047 as a mean; per-seed values now stated).
- One incident, caught and fixed: env_from_config silently dropped skeleton_source; the
  first no-skeleton eval was void and quarantined; passthrough + assert added; all other
  overrides audited clean (the disclosure sweep is unaffected).

## Decisions on your desk (unchanged from Friday, plus one)
1. Ceiling: option 1 (report the degeneracy) -- still recommended, still undecided.
2. Figure width 5.40 vs 6.25 in; caption font. Agent C blocked on both for final layout.
3. NEW: the sec:meth_publicgood restore -- veto or keep.
4. Whether the no-skeleton finding gets its one limitation sentence in the Discussion.

## Today per the endgame plan (Sunday)
Your writing day 2: RQ3 markers + Ch5. My day: same-day fact-checks behind you, figure
freeze 18:00, B's audit + bundle-reproduction test.
