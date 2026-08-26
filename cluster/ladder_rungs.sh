# Shared rung table for the scale ladder. Sourced by the train and eval submit scripts so
# the two cannot drift apart -- an eval job rebuilding a different environment than the one
# trained would be silent, and every number downstream would be wrong.
#
# ONE AXIS AT A TIME, so a rung that fails says which axis broke it:
#   0-2  agents   2 -> 3 -> 5      holding 1 private, 3 shared
#   3-4  shared   3 -> 4 -> 5      holding 5 agents, 1 private
#   5-8  private  1 -> 2 -> 3 -> 4 -> 5   holding 5 agents, 5 shared
#
# BUDGET = 3 * n_agents, so every agent gets three turns under round-robin regardless of
# how many agents there are. A fixed budget would silently starve the larger rungs and the
# ladder would measure turn scarcity rather than scale.
#
# TRAIN_EPISODES falls as the rungs get dearer, sized so every job lands inside its 48h
# limit. Seconds/episode at budget 8, measured after the 2026-08-26 optimisations
# (scripts/ma_rung_timing.py, results/ma_rung_timing_v2.json):
#
#   rung   0     1     2     3      4      5      6      7      8
#   d      5     6     8     9     10     15     20     25     30
#   s/ep  0.85  1.36  2.10  9.91  14.19  21.72  36.13  65.53  146.50
#
# Multiply by BUDGET/8 for the real per-episode cost, which is why rung 8 gets 500 episodes
# and rung 0 gets 3000. Episodes terminate early on identification, so these are upper
# bounds rather than estimates.
#
# The earlier table gave rungs 7-8 only 250 and 150 episodes and flagged them as too few to
# learn. That was BEFORE the batched BGe table and the batched backward scatter, which
# together took rung 8 from 234.9 to 146.5 s/episode and every mid rung by ~2.5x. They are
# no longer a token run, though rung 8 at 500 episodes remains the thinnest on the ladder
# and should be read with that in mind.
rung_config() {
  case $1 in
    0) AGENTS=2; PRIV=1; SHARED=3; TRAIN=3000 ;;
    1) AGENTS=3; PRIV=1; SHARED=3; TRAIN=3000 ;;
    2) AGENTS=5; PRIV=1; SHARED=3; TRAIN=3000 ;;
    3) AGENTS=5; PRIV=1; SHARED=4; TRAIN=2000 ;;
    4) AGENTS=5; PRIV=1; SHARED=5; TRAIN=2000 ;;
    5) AGENTS=5; PRIV=2; SHARED=5; TRAIN=1500 ;;
    6) AGENTS=5; PRIV=3; SHARED=5; TRAIN=1200 ;;
    7) AGENTS=5; PRIV=4; SHARED=5; TRAIN=800  ;;
    8) AGENTS=5; PRIV=5; SHARED=5; TRAIN=500  ;;
    *) echo "unknown rung $1" >&2; exit 1 ;;
  esac
  BUDGET=$((3 * AGENTS))
  D=$((AGENTS * PRIV + SHARED))
  ARM="rung${1}_${AGENTS}a_${PRIV}p_${SHARED}x_d${D}"
}
