# The agent-count reversal does not survive adequate training

2 Sep 2026, 09:5x. **Supersedes ledger section 1.4 and the agent-count claim in section 4.3 of
the results chapter.** PARTIAL: the converged column below mixes one 12,000-episode seed into
two 4,000-episode ones. Uniform-budget cells are training now; do not quote the converged
ratios until they land.

## The claim being tested

The sweep at 4,000 episodes shows the learned policy leading the myopic rule to four agents and
trailing it from five, with the learned-to-myopic ratio of structural distance rising to 6.75 at
ten agents. That was read as coordination load: more agents contending for the same variables
degrades the learned policy while the myopic rule is unaffected.

## Two of the runs carrying it had not converged

The competence floor of 0.70 admits runs that are far from finished. Both high-agent-count
cells have a seed that cleared it and was nowhere near converged:

| run | window rate at 4,000 ep | learned SHD at 4,000 | at 12,000 | factor |
|---|---|---|---|---|
| k12s50n08b150 s2 | 0.838 (PASSED) | 0.00290 | **0.00005** | 58x |
| k12s50n10b150 s2 | 0.804 (PASSED) | 0.00220 | **0.00001** | 220x |

Joint recovery on the same runs moves from 0.635 to 0.990 and from 0.610 to 0.995.

## The reversal disappears

Learned-to-myopic ratio of structural distance, below 1 meaning the learned policy is better:

| $K$ | as run | excluding seed 2 | with seed 2 converged |
|---|---|---|---|
| 2 | 0.12 | 0.12 | -- |
| 3 | 0.33 | 0.33 | -- |
| 4 | 0.10 | 0.12 | -- |
| 5 | 1.65 | 0.25 | pending |
| 8 | 4.24 | 1.82 | **0.89** |
| 10 | 6.75 | 2.17 | **1.00** |

At eight agents the learned policy becomes better than the myopic rule; at ten it ties. Nothing
in the axis then shows the learned policy losing at any agent count.

## What the honest claim becomes

Not a statement about achievable accuracy. A statement about **sample efficiency under
contention**: at a fixed budget of 4,000 episodes the learned policy degrades as agents are
added, and the degradation does not survive training to convergence. Adding agents makes the
problem take longer to learn rather than making it less learnable.

That is narrower than the original claim and better supported. It also removes the strongest
apparent limitation of the method, which is worth saying plainly rather than burying: the
result the chapter was going to report as the honest boundary of the contribution was an
artefact of how long we trained.

## The methodological point, which is the transferable one

**A run passing a competence threshold is not evidence that it converged.** A window rate of
0.838 cleared our floor while sitting 58x from its converged structural error. Any gate defined
on a saturating quantity has this failure mode, and a fixed episode budget across cells of
different difficulty guarantees that the hardest cells are the ones that hit it.

## Still open

* Uniform-budget cells at $K = 5, 8, 10$, all seeds at 12,000 episodes, are training. Until
  they land the converged column mixes budgets within a cell.
* Whether the contended-fraction reversal at $\sigma = 0.75$ has the same cause. Its seed 2
  sits at window rate 0.758 with joint recovery 0.660, which is the same signature.
