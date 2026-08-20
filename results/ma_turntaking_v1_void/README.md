# VOID -- turn-taking runs made before the forfeit fix (20 August 2026)

These runs used a `step()` in which **a single agent could end the episode by declining its
turn**. The inactive agent's pass is forced by the protocol, so "everyone passed this round"
was true whenever the one agent able to act passed.

Consequence, measured: passing became a free exit from the step cost.

    rr_both    5/10 seeds collapsed, learned mean_steps 1.11
    rr_clamp   6/8  seeds collapsed
    (simultaneous play, same settings: 0/10 collapsed)

Retained only as the evidence for the fix. **No number in this directory describes the
intended protocol.** Superseded by the re-run after the fix, in `results/ma_fixed/`.
