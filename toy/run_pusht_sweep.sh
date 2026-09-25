#!/usr/bin/env bash
# Push-T curiosity sweep: arms x seeds, one PPO run at a time (each run uses 12 simulator processes),
# each finished run scored in the background (pusht_eval.py score: E1 fresh-model fits + E2, GPU).
#   bash toy/run_pusht_sweep.sh START "SEEDS" ["ARMS"]      e.g.  bash toy/run_pusht_sweep.sh fixed "0 1 2"
# Runs: toy/runs/pusht/pt<f|r>_<arm>_s<seed>/; logs: toy/runs/pusht/logs/. Then: python toy/pusht_report.py --start START
set -uo pipefail
cd "$(dirname "$0")"
START=${1:?start mode: fixed|random}
SEEDS=${2:-"0 1 2"}
ARMS=${3:-"none pred lp count ln"}
OUT=runs/pusht
mkdir -p $OUT/logs
for s in $SEEDS; do
  for arm in $ARMS; do
    name="pt${START:0:1}_${arm}_s${s}"
    if [ ! -f "$OUT/$name/replay.pt" ]; then
      python run_curiosity.py --env pusht --start "$START" --signal "$arm" --seed "$s" --save-replay 32 \
        --wm-replay 5000000 --out $OUT > "$OUT/logs/$name.log" 2>&1 \
        && echo "[done] $name" || { echo "[FAIL] $name"; continue; }
    fi
    [ -f "$OUT/$name/eval.json" ] || nohup python pusht_eval.py score "$OUT/$name" > "$OUT/logs/$name.eval.log" 2>&1 &
  done
done
wait
echo "[sweep] all done"
