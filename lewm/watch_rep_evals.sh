#!/usr/bin/env bash
# Run the authors' eval.py (unmodified, via tools/run_capped.py: capped thread pools, pinned cores) on every epoch
# checkpoint of a LeWM training run as it appears. One eval at a time; results in runs/eval_<run>_ep<N>_s<seed>.log
# and the eval's own pusht_results.txt.   Usage: bash watch_rep_evals.sh pusht_rep_s3072 "42" [cores]
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME="$PWD/swm_home" WANDB_SILENT=true SDL_VIDEODRIVER=dummy THREADS=4
RUN=${1:?run}; SEEDS=${2:-42}; CORES=${3:-88-95}
while true; do
  for ck in $(ls swm_home/checkpoints/$RUN/weights_epoch_*.pt 2>/dev/null | sort -V); do
    ep=$(basename "$ck" .pt | sed 's/weights_epoch_//')
    for s in $SEEDS; do
      log="runs/eval_${RUN}_ep${ep}_s${s}.log"
      [ "$RUN" = pusht_rep_s3072 ] && [ "$ep" = 1 ] && [ "$s" = 42 ] && log="runs/eval_rep_ep1_s42.log"
      [ -f "$log" ] && continue
      echo "[watch] eval $RUN epoch $ep seed $s"
      taskset -c "$CORES" .venv/bin/python tools/run_capped.py eval.py --config-name=pusht.yaml \
        "policy=$RUN/weights_epoch_$ep.pt" "seed=$s" > "$log" 2>&1
      tr '\r' '\n' < "$log" | grep -oE "'success_rate': [0-9.]+" | tail -1 | sed "s/^/[watch] $RUN ep$ep s$s /"
    done
  done
  [ -f "swm_home/checkpoints/$RUN/weights_epoch_10.pt" ] && [ -f "runs/eval_${RUN}_ep10_s${SEEDS##* }.log" ] && break
  sleep 300
done
echo "[watch] done"
