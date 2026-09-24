#!/usr/bin/env bash
# Curiosity-signal comparison on point-push: {none,pred,lp,ln,count} x {clean, noisy TV} x seeds.
#   bash toy/run_sweep.sh [iters=300] [parallel=10] [seeds="0 1 2"]
# Logs: toy/runs/logs/<run>.log; metrics: toy/runs/<run>/metrics.jsonl. Then: python toy/plot_curiosity.py
set -euo pipefail
cd "$(dirname "$0")/.."
ITERS=${1:-300}
PAR=${2:-10}
SEEDS=${3:-"0 1 2"}
mkdir -p toy/runs/logs
jobs=()
for tv in "" "--tv"; do
  for sig in none pred lp ln count; do
    for s in $SEEDS; do
      jobs+=("$sig|$tv|$s")
    done
  done
done
printf '%s\n' "${jobs[@]}" | OMP_NUM_THREADS=4 xargs -P "$PAR" -I{} bash -c '
  IFS="|" read -r sig tv s <<< "{}"
  name="${sig}$([ -n "$tv" ] && echo _tv)_s${s}"
  python toy/run_curiosity.py --signal "$sig" $tv --seed "$s" --iters '"$ITERS"' > "toy/runs/logs/${name}.log" 2>&1 \
    && echo "[done] $name" || echo "[FAIL] $name"
'
echo "[sweep] all done"
