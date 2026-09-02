#!/bin/bash
# 15-minute heartbeat over the deterministic grid rebuild.
#
# The stall signal is TOTAL python CPU-seconds per wall-second, not a per-process figure.
# On Windows every python launch shows a near-zero-CPU launcher stub beside the real worker,
# so six workers appear as a dozen processes and any per-process average is diluted by a
# factor that changes with how many jobs are in flight. The total rate read 0.14 through the
# 6.5-hour Modern Standby sleep on 2 Sep and ~4 with four workers running, which is the
# separation that matters.
#
# Log age is REPORTED but is not a trigger. Transfer evaluation writes its log only on
# completion, so 30 minutes of silence is normal in this phase and was the reason two earlier
# detectors cried wolf.
cd "$(dirname "$0")/.."
OUT=results/power/rho/deterministic

cpu() { powershell -NoProfile -Command \
  "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object -Property CPU -Sum).Sum" \
  2>/dev/null | tr -d '\r'; }

while true; do
  a=$(cpu); t1=$(date +%s); sleep 45; b=$(cpu); t2=$(date +%s)
  eff=$(python3 -c "print(f'{max(0.0,($b)-($a))/($t2-$t1):.2f}')" 2>/dev/null || echo "?")
  n=$(ls "$OUT"/xfer_*.json 2>/dev/null | wc -l)
  base=$(ls "$OUT"/xfer_rho1.00_s*.json 2>/dev/null | wc -l)
  live=$(powershell -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*global_shd*'}).Count" \
    2>/dev/null | tr -d '\r')
  phase="1 (baselines ${base}/3)"; [ "${base:-0}" -eq 3 ] && phase="2 (cells ${n}/21)"

  alert=""
  # Terminal states first, so the watch says something on every ending rather than only the
  # happy one -- silence and success must not look the same.
  if grep -q "GRID REBUILD COMPLETE" logs/power/rho/REBUILD.log 2>/dev/null; then
    echo "DONE $(date +%H:%M) grid rebuild complete, ${n}/21 cells -- run compare_deterministic_grid.py"
    exit 0
  fi
  if grep -q "ABORT:" logs/power/rho/REBUILD.log 2>/dev/null; then
    echo "FAILED $(date +%H:%M) phase 1 aborted, a baseline did not land -- see REBUILD.log"
    exit 1
  fi
  stalled=$(python3 -c "print(1 if $eff < 0.5 else 0)" 2>/dev/null || echo 0)
  [ "$stalled" = "1" ] && [ "${live:-0}" -gt 0 ] \
    && alert=" *** STALL: ${live} live but ${eff} cpu/s -- LIKELY ASLEEP, check keep_awake.py ***"
  [ "${live:-0}" -eq 0 ] \
    && alert=" *** NO EVALUATORS RUNNING at ${n}/21 -- the driver died ***"

  echo "CHECK $(date +%H:%M) phase ${phase} live=${live:-0} cpu/s=${eff}${alert}"
  sleep 855
done
