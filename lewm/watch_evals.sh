#!/usr/bin/env bash
# One eval queue for the Push-T study, run one at a time on pinned cores with capped thread pools:
#   offline replication epochs  swm_home/checkpoints/<REP>/weights_epoch_N.pt   -> eval.py (authors', unmodified)
#   from-scratch arm checkpoints swm_home/checkpoints/pusht_fs_*_s*/weights_step_N.pt -> pusht_fs_eval.py
# Logs: runs/eval_<tag>_s<seed>.log. Exits when the replication's epoch 10 and every arm's final step are evaluated.
#   Usage: bash watch_evals.sh [REP=pusht_rep_s3072] [SEEDS="42"] [CORES=88-95] [FINAL_STEP=13933]
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME="$PWD/swm_home" WANDB_SILENT=true SDL_VIDEODRIVER=dummy THREADS=4
REP=${REP:-pusht_rep_s3072}; SEEDS=${SEEDS:-42}; CORES=${CORES:-88-95}; FINAL_STEP=${FINAL_STEP:-13933}
run_eval() {  # script policy log
  echo "[watch] $(date -u +%H:%M) $3"
  taskset -c "$CORES" .venv/bin/python tools/run_capped.py "$1" --config-name=pusht.yaml "policy=$2" "seed=$4" > "$3" 2>&1
  tr '\r' '\n' < "$3" | grep -oE "'success_rate': [0-9.]+" | tail -1 | sed "s|^|[watch] $(basename "$3" .log) |"
}
while true; do
  for ck in $(ls swm_home/checkpoints/$REP/weights_epoch_*.pt 2>/dev/null | sort -V); do
    ep=$(basename "$ck" .pt | sed 's/weights_epoch_//')
    for s in $SEEDS; do
      log="runs/eval_rep_ep${ep}_s${s}.log"
      [ -f "$log" ] || run_eval eval.py "$REP/weights_epoch_$ep.pt" "$log" "$s"
    done
  done
  for d in $(ls -d swm_home/checkpoints/pusht_fs_*_s* 2>/dev/null | grep -v killed); do
    arm=$(basename "$d")
    for ck in $(ls $d/weights_step_*.pt 2>/dev/null | sort -V); do
      st=$(basename "$ck" .pt | sed 's/weights_step_//')
      for s in $SEEDS; do
        log="runs/eval_${arm}_step${st}_s${s}.log"
        [ -f "$log" ] || run_eval pusht_fs_eval.py "$arm/weights_step_$st.pt" "$log" "$s"
        # secondary: the random-action eval set (pusht_random_eval.h5), final checkpoint only
        rlog="runs/evalrand_${arm#pusht_}_step${st}_s${s}.log"
        if [ "$st" = "$FINAL_STEP" ] && [ ! -f "$rlog" ]; then
          echo "[watch] $(date -u +%H:%M) $rlog"
          taskset -c "$CORES" .venv/bin/python tools/run_capped.py pusht_fs_eval.py --config-name=pusht.yaml \
            "policy=$arm/weights_step_$st.pt" "seed=$s" eval.dataset_name=pusht_random_eval +fs.search_std=0.2076 > "$rlog" 2>&1
          tr '\r' '\n' < "$rlog" | grep -oE "'success_rate': [0-9.]+" | tail -1 | sed "s|^|[watch] $(basename "$rlog" .log) |"
        fi
      done
    done
  done
  # secondary set for the offline model's final epoch too
  if [ -f "swm_home/checkpoints/$REP/weights_epoch_10.pt" ] && [ ! -f runs/evalrand_rep_ep10_s42.log ]; then
    echo "[watch] $(date -u +%H:%M) runs/evalrand_rep_ep10_s42.log"
    taskset -c "$CORES" .venv/bin/python tools/run_capped.py pusht_fs_eval.py --config-name=pusht.yaml \
      "policy=$REP/weights_epoch_10.pt" seed=42 eval.dataset_name=pusht_random_eval +fs.search_std=0.2076 +fs.action_std=0.2076 \
      > runs/evalrand_rep_ep10_s42.log 2>&1
  fi
  done_all=1
  [ -f "runs/eval_rep_ep10_s${SEEDS##* }.log" ] || done_all=0
  for a in lp1 lp0 pred rnd random; do
    [ -f "runs/eval_pusht_fs_${a}_s0_step${FINAL_STEP}_s${SEEDS##* }.log" ] || done_all=0
  done
  [ "$done_all" = 1 ] && break
  sleep 120
done
echo "[watch] all evaluations done"
