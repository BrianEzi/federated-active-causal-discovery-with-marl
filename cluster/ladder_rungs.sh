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
# TRAIN_EPISODES falls as the rungs get dearer, and this is a REAL LIMITATION rather than a
# tuning choice. Measured seconds/episode at budget 8 (scripts/ma_rung_timing.py,
# results/ma_rung_timing.json): 1.8, 3.1, 4.7, 22.8, 37.2, 55.8, 90.9 for rungs 0-6, with
# 7 and 8 extrapolated at ~160 and ~320. Rungs 7-8 therefore get a few hundred episodes
# inside a 48h job, which is very likely too few to learn -- report them as such rather
# than as a fair test.
rung_config() {
  case $1 in
    0) AGENTS=2; PRIV=1; SHARED=3; TRAIN=2000 ;;
    1) AGENTS=3; PRIV=1; SHARED=3; TRAIN=2000 ;;
    2) AGENTS=5; PRIV=1; SHARED=3; TRAIN=2000 ;;
    3) AGENTS=5; PRIV=1; SHARED=4; TRAIN=1200 ;;
    4) AGENTS=5; PRIV=1; SHARED=5; TRAIN=800  ;;
    5) AGENTS=5; PRIV=2; SHARED=5; TRAIN=600  ;;
    6) AGENTS=5; PRIV=3; SHARED=5; TRAIN=400  ;;
    7) AGENTS=5; PRIV=4; SHARED=5; TRAIN=250  ;;
    8) AGENTS=5; PRIV=5; SHARED=5; TRAIN=150  ;;
    *) echo "unknown rung $1" >&2; exit 1 ;;
  esac
  BUDGET=$((3 * AGENTS))
  D=$((AGENTS * PRIV + SHARED))
  ARM="rung${1}_${AGENTS}a_${PRIV}p_${SHARED}x_d${D}"
}
