# The no-skeleton cell: the assumption is load-bearing for the policy, not the method

Night of 5-6 Sep, agent A. Brian's question: drop the supplied skeleton at one cell, learned
against greedy, learn the gap. Everything below is k=8 principal cell, skeleton_source=
estimated (per-episode estimation from the 60 observational rows), seeded paired evaluation,
200 episodes per seed, oracle evidence.

## Three measurements, one story

1. TRANSFER (policies trained WITH the supplied skeleton, evaluated without it):
   learned 0.263/0.300/0.285 vs greedy 0.230/0.228/0.230 vs random 0.249/0.249/0.250.
   The assumption-trained policy inverts BELOW RANDOM on every seed (12-20 SE paired).
   It acts confidently on belief features whose semantics changed under it.

2. TRAINED WITHOUT the assumption (3 x 12,000 episodes, skeleton_source=estimated;
   ~34 min/run; the flag verified end-to-end in each run JSON):
   learned 0.2278/0.2256/0.2264 vs greedy 0.2301/0.2278/0.2297 vs random 0.2491/0.2485/0.2504.
   The learned arm RECOVERS a small, significant edge: paired learned-greedy
   -0.00234/-0.00223/-0.00335, 3-5 SE, 3 of 3 seeds, and beats random everywhere.

3. The regime itself is mostly unwinnable, as the 65.9%-accuracy probe predicted: joint
   recovery is 0.000 for EVERY arm including oracle_cover, so the claims-based training
   signal is nearly flat and the competence floor fails by construction -- reported here as
   the answer, not excluded. The measurable objective in this regime is committed-marks
   error, where the ordering above lives.

## Reading, and its boundaries

The supplied-skeleton assumption is load-bearing for the TRAINED POLICY (off-assumption it
is anti-calibrated, worse than blind targeting) but not for the METHOD: retrained under the
estimated skeleton, learned selection still buys a small significant edge over the myopic
rule on the errors that remain measurable. The extension direction this points at is
training against skeleton UNCERTAINTY, not merely without the skeleton.

EXTENSION, k=12 (landed later the same night): the finding replicates. Trained under the
estimated skeleton, learned 0.1911/0.1908/0.1872 vs greedy 0.1933/0.1928/0.1892 vs random
0.2055/0.2043/0.1998; paired learned-greedy -0.0021/-0.0020/-0.0020 at 4-5 SE. Two cells,
6 of 6 seeds.

Boundaries: two cells (k=8, k=12), same topology family; the edge is ~1% relative
at enormous absolute error (0.23); joint recovery separates nothing (all zero); the
window-rate floor fails for every arm, so no run here would be admitted to a sweep table --
none is. Exploratory, post-freeze by Brian's explicit order; at most one limitation/future-
work sentence enters the thesis.

Files: results/noskel/ (training runs, transfer eval, shd_s?_best measurements);
scripts/ma_train.py --skeleton_source; global_shd_paired --override_skeleton.
Incident on the way: docs/AGENT_B_INBOX.md 5 Sep, the silent config drop.
