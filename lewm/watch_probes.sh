#!/usr/bin/env bash
# Probe queue (probe_pusht.py, one checkpoint per job, pinned cores): every replication epoch and every from-scratch
# arm's checkpoints matching STEPS (default: the final 13933). Writes runs/probe_<tag>.json; exits when all are done.
#   Usage: bash watch_probes.sh [REP=pusht_rep_s3072] [STEPS="13933"] [CORES=120-127]
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME="$PWD/swm_home" SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
REP=${REP:-pusht_rep_s3072}; STEPS=${STEPS:-13933}; CORES=${CORES:-120-127}
probe() {  # ckpt-spec tag
  [ -f "runs/probe_$2.json" ] && return
  echo "[probe] $(date -u +%H:%M) $1"
  taskset -c "$CORES" .venv/bin/python -u probe_pusht.py --ckpt "$1" --out "runs/probe_$2.json" > "runs/probe_$2.log" 2>&1 \
    || echo "[probe] FAILED $1 (runs/probe_$2.log)"
}
while true; do
  for ck in $(ls swm_home/checkpoints/$REP/weights_epoch_*.pt 2>/dev/null | sort -V); do
    ep=$(basename "$ck" .pt | sed 's/weights_epoch_//')
    probe "$REP/weights_epoch_$ep.pt" "rep_ep$ep"
  done
  for a in lp1 lp0 pred rnd random; do
    for st in $STEPS; do
      ck="swm_home/checkpoints/pusht_fs_${a}_s0/weights_step_$st.pt"
      [ -f "$ck" ] && probe "pusht_fs_${a}_s0/weights_step_$st.pt:uniform" "fs_${a}_step$st"
    done
  done
  n=0; for a in lp1 lp0 pred rnd random; do for st in $STEPS; do [ -f "runs/probe_fs_${a}_step$st.json" ] && n=$((n+1)); done; done
  [ -f "runs/probe_rep_ep10.json" ] && [ "$n" -ge $((5 * $(echo $STEPS | wc -w))) ] && break
  sleep 180
done
echo "[probe] all done"
