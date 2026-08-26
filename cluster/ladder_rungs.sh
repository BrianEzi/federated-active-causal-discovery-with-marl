# Shared rung table for the scale ladder. Sourced by the train and eval submit scripts so
# the two cannot drift apart -- an eval job rebuilding a different environment than the one
# trained would be silent, and every number downstream would be wrong.
#
# ONE AXIS AT A TIME, so a rung that fails says which axis broke it:
#   0-2  agents   2 -> 3 -> 5      holding 1 private, 3 shared
#   3-4  shared   3 -> 4 -> 5      holding 5 agents, 1 private
#   5-8  private  1 -> 2 -> 3 -> 4 -> 5   holding 5 agents, 5 shared
#
# ============================================================================
# BUDGET = n_agents * max(5, k+1), and getting this wrong voided a whole grid.
# ============================================================================
#
# The first version used 3 * n_agents -- three turns each -- and MEASURED AT RUNG 0 it
# collapsed the task for every arm at once:
#
#   budget  prior_p   random  greedy   pass
#        6      0.5    0.060   0.060  0.010
#        6   0.6437    0.030   0.030  0.020
#       10      0.5    0.387   0.240  0.007     <- banked, results/ma_fixed/tb_clamp_s0.json
#
# At three turns per agent NOTHING distinguishes the arms: random, greedy and a policy that
# never acts all score within noise of each other. That is a Gate 2 failure -- if choices do
# not matter there is nothing for an agent to beat -- and it would have looked like "the
# learned policy fails to scale" rather than "the budget was too small to act".
#
# The rule now anchors rung 0 at exactly the banked two-agent setting (2 * 5 = 10), so that
# rung is directly comparable to results/ma_fixed/, and scales turns-per-agent with the
# WINDOW size k = private + shared -- which is what each agent actually has to identify --
# rather than with the global d.
#
# TRAIN and EVAL fall up the ladder because budget multiplies episode cost. Seconds/episode
# at budget 8 (results/ma_rung_timing_v2.json): 0.85, 1.36, 2.10, 9.91, 14.19, 21.72, 36.13,
# 65.53, 146.50. Multiply by BUDGET/8.
#
# RUNGS 7 AND 8 CANNOT HAVE BOTH an adequate budget and an adequate episode count inside a
# 48h job -- at rung 8 one episode costs ~1000s, so 120 training episodes is what fits. That
# is a real limit of the exact method at this scale, not a tuning choice, and it is recorded
# here so those two rungs are read as best-effort rather than as a fair test.
rung_config() {
  case $1 in
    0) AGENTS=2; PRIV=1; SHARED=3; TRAIN=3000; EVAL=200 ;;
    1) AGENTS=3; PRIV=1; SHARED=3; TRAIN=3000; EVAL=200 ;;
    2) AGENTS=5; PRIV=1; SHARED=3; TRAIN=2500; EVAL=200 ;;
    3) AGENTS=5; PRIV=1; SHARED=4; TRAIN=1500; EVAL=150 ;;
    4) AGENTS=5; PRIV=1; SHARED=5; TRAIN=1200; EVAL=150 ;;
    5) AGENTS=5; PRIV=2; SHARED=5; TRAIN=800;  EVAL=100 ;;
    6) AGENTS=5; PRIV=3; SHARED=5; TRAIN=500;  EVAL=100 ;;
    7) AGENTS=5; PRIV=4; SHARED=5; TRAIN=250;  EVAL=60  ;;
    8) AGENTS=5; PRIV=5; SHARED=5; TRAIN=120;  EVAL=40  ;;
    *) echo "unknown rung $1" >&2; exit 1 ;;
  esac
  K=$((PRIV + SHARED))
  TURNS=$((K + 1)); [ $TURNS -lt 5 ] && TURNS=5
  BUDGET=$((AGENTS * TURNS))
  D=$((AGENTS * PRIV + SHARED))
  ARM="rung${1}_${AGENTS}a_${PRIV}p_${SHARED}x_d${D}"
}
