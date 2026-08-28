# Overnight verdict — difference reward vs the plain reward

Matched pairs: same config, same seed, same 4,000 episodes, same evaluation.
The ONLY change is what the agent is paid. Gate before score, always.

| arm | reward | final entropy | I(S;A)/H | learned | greedy | trained? |
|---|---|---|---|---|---|---|
| a06diff_s0 | difference | 0.726 | 0.647 | 0.367 | 0.607 | **YES** |
| a06_s0 | plain | 1.645 | — | 0.213 | 0.607 | (see FINDINGS §15) |
| a08diff_s0 | difference | 0.786 | 0.546 | 0.387 | 0.480 | **YES** |
| a08_s0 | plain | 1.814 | — | 0.100 | 0.480 | (see FINDINGS §15) |
| a03diff_s0 | difference | 0.430 | 0.752 | 0.540 | 0.547 | **YES** |
| a03_s0 | plain | 0.699 | — | 0.833 | 0.547 | (see FINDINGS §15) |
| a08diff_s1 | difference | 1.058 | 0.398 | 0.447 | 0.560 | **YES** |
| a08_s1 | plain | 1.683 | — | 0.120 | 0.560 | (see FINDINGS §15) |

Arms completed: 4 of 4.

READ THIS FIRST: a 'NO' in the trained column voids that row's score entirely --
it is an untrained policy, not a negative result. See SESSION_STATE §11.
