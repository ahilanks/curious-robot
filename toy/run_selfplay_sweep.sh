#!/usr/bin/env bash
# Self-play with the learning-progress reward (run_selfplay.py) on Push-T vs the none arm: every arm x seed runs in
# parallel (each uses 12 simulator processes) and is scored by pusht_eval.py as soon as it finishes.
#   bash toy/run_selfplay_sweep.sh ["SEEDS"] ["CHUNKS"]      e.g.  bash toy/run_selfplay_sweep.sh "0 1 2" "0 20"
# CHUNKS: steps per program, 0 = the whole episode. Runs: toy/runs/pusht/ptf_{none,spep,spch<c>}_s<seed>/;
# logs: toy/runs/pusht/logs/. Then: python toy/pusht_report.py --start fixed
set -uo pipefail
cd "$(dirname "$0")"
SEEDS=${1:-"0 1 2"}
CHUNKS=${2:-"0 20"}
OUT=runs/pusht
mkdir -p $OUT/logs
run() {                                                          # run NAME CMD...: train if needed, then score
  local name=$1
  shift
  if [ ! -f "$OUT/$name/replay.pt" ]; then
    "$@" > "$OUT/logs/$name.log" 2>&1 || { echo "[FAIL] $name"; return; }
  fi
  [ -f "$OUT/$name/eval.json" ] || python pusht_eval.py score "$OUT/$name" > "$OUT/logs/$name.eval.log" 2>&1
  echo "[done] $name"
}
for s in $SEEDS; do
  run "ptf_none_s$s" python run_curiosity.py --env pusht --start fixed --signal none --seed "$s" --save-replay 32 --out $OUT &
  for c in $CHUNKS; do
    if [ "$c" = 0 ]; then arm=spep; else arm="spch$c"; fi
    run "ptf_${arm}_s$s" python run_selfplay.py --chunk "$c" --seed "$s" --out $OUT &
  done
done
wait
echo "[sweep] all done"
