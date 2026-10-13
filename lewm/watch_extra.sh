#!/usr/bin/env bash
# Extra eval seeds (default 43 44), so a from-scratch arm can be told apart from chance, plus the reference models.
#   - each arm's final checkpoint, on each eval set where its seed-42 success >= the random-action baseline's mean
#     there (runs/eval_randompolicy_s*.log, runs/evalrand_randompolicy_s*.log); below that one seed already says
#     "no better than random actions"
#   - references on both sets: released, offline epoch 1, offline epoch 10 (once the main queue has evaluated it)
# One eval at a time on pinned cores, arms before references (one reference per pass). Logs use watch_evals.sh's
# names, so report_pusht_fs.py averages the seeds. Exits when every arm and epoch 10 have been handled.
#   Usage: bash watch_extra.sh [CORES=54-61] [EXTRA="43 44"]
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME="$PWD/swm_home" WANDB_SILENT=true SDL_VIDEODRIVER=dummy THREADS=4
CORES=${CORES:-54-61}; EXTRA=${EXTRA:-43 44}; FINAL_STEP=${FINAL_STEP:-13933}; REP=${REP:-pusht_rep_s3072}
G=/usr/bin/grep
RANDSET="eval.dataset_name=pusht_random_eval +fs.search_std=0.2076"                       # from-scratch models
RANDSET_X="eval.dataset_name=pusht_random_eval +fs.search_std=0.2076 +fs.action_std=0.2076"  # expert-trained models
sr() { [ -f "$1" ] && tr '\r' '\n' < "$1" | $G -oE "'success_rate': [0-9.]+" | tail -1 | $G -oE "[0-9.]+$"; }
mean_sr() { for f in "$@"; do sr "$f"; done | awk '{s += $1; n++} END {if (n) printf "%.2f", s / n; else print 0}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
ev() {  # log script policy seed [overrides...]
  local log=$1 script=$2 policy=$3 seed=$4; shift 4
  echo "[extra] $(date -u +%H:%M) $log"
  taskset -c "$CORES" .venv/bin/python tools/run_capped.py "$script" --config-name=pusht.yaml "policy=$policy" "seed=$seed" "$@" \
    > "$log" 2>&1
  echo "[extra] $(basename "$log" .log) $(sr "$log")"
}
next_ref() {  # run the first missing reference eval; 1 if none is ready
  local s refs=()
  for s in $EXTRA; do refs+=("runs/eval_released_s$s.log|eval.py|released/lewm_pusht|$s|"); done
  for s in 42 $EXTRA; do refs+=("runs/evalrand_released_s$s.log|pusht_fs_eval.py|released/lewm_pusht|$s|$RANDSET_X"); done
  for s in $EXTRA; do refs+=("runs/eval_rep_ep1_s$s.log|eval.py|$REP/weights_epoch_1.pt|$s|"); done
  for s in $EXTRA; do refs+=("runs/evalrand_rep_ep1_s$s.log|pusht_fs_eval.py|$REP/weights_epoch_1.pt|$s|$RANDSET_X"); done
  if [ -n "$(sr runs/eval_rep_ep10_s42.log)" ]; then
    for s in $EXTRA; do refs+=("runs/eval_rep_ep10_s$s.log|eval.py|$REP/weights_epoch_10.pt|$s|"); done
  fi
  if [ -n "$(sr runs/evalrand_rep_ep10_s42.log)" ]; then
    for s in $EXTRA; do refs+=("runs/evalrand_rep_ep10_s$s.log|pusht_fs_eval.py|$REP/weights_epoch_10.pt|$s|$RANDSET_X"); done
  fi
  local r log script policy seed ov
  for r in "${refs[@]}"; do
    IFS='|' read -r log script policy seed ov <<< "$r"
    [ -f "$log" ] && continue
    ev "$log" "$script" "$policy" "$seed" $ov
    return 0
  done
  return 1
}
b_exp=$(mean_sr runs/eval_randompolicy_s*.log); b_rand=$(mean_sr runs/evalrand_randompolicy_s*.log)
echo "[extra] random-action baseline: expert set $b_exp %, random set $b_rand %"
while true; do
  handled=0
  for a in pred random lp0 rnd lp1; do
    arm=pusht_fs_${a}_s0; ck=$arm/weights_step_$FINAL_STEP.pt
    xe=$(sr runs/eval_${arm}_step${FINAL_STEP}_s42.log); xr=$(sr runs/evalrand_fs_${a}_s0_step${FINAL_STEP}_s42.log)
    if [ -n "$xe" ] && ge "$xe" "$b_exp"; then
      for s in $EXTRA; do f=runs/eval_${arm}_step${FINAL_STEP}_s$s.log; [ -f "$f" ] || ev "$f" pusht_fs_eval.py "$ck" "$s"; done
    fi
    if [ -n "$xr" ] && ge "$xr" "$b_rand"; then
      for s in $EXTRA; do f=runs/evalrand_fs_${a}_s0_step${FINAL_STEP}_s$s.log; [ -f "$f" ] || ev "$f" pusht_fs_eval.py "$ck" "$s" $RANDSET; done
    fi
    [ -n "$xe" ] && [ -n "$xr" ] && handled=$((handled + 1))
  done
  next_ref && continue
  [ "$handled" = 5 ] && [ -n "$(sr runs/evalrand_rep_ep10_s42.log)" ] && break
  sleep 120
done
echo "[extra] all done"
