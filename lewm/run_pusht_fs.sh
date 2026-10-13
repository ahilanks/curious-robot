#!/usr/bin/env bash
# From-scratch learning-signal arms on Push-T (pusht_explore.py): LeWM trained online on data collected by CEM that
# scores candidate action sequences IN THE SIMULATOR with a learning signal (no policy). One process per arm, each
# pinned to its own cores (simulator worker pool + main), W&B group pusht_fs. Pilot budget: 13,933 gradient steps
# (= epoch 1 of the offline 10-epoch run) and 200k env steps per arm.
#
# Usage: bash run_pusht_fs.sh "lp1 lp0 pred rnd" [seed]      (core block k = 18k .. 18k+17 for the k-th arm listed)
#        CORE0=72 bash run_pusht_fs.sh random                 (start the core blocks elsewhere)
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ../.env; set +a
export STABLEWM_HOME="$PWD/swm_home" WANDB_SILENT=true SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SIGNALS=${1:?signals}
SEED=${2:-0}
k=0
mkdir -p runs
for SIG in $SIGNALS; do
  c0=$(( ${CORE0:-0} + 18 * k ))
  NAME="pusht_fs_${SIG}_s${SEED}"
  echo "[fs] $NAME on cores $c0-$((c0 + 17))"
  setsid nohup taskset -c "$c0-$((c0 + 17))" .venv/bin/python -u pusht_explore.py --signal "$SIG" --seed "$SEED" \
      --workers 16 --wandb ${EXTRA:-} > "runs/$NAME.log" 2>&1 < /dev/null &
  k=$((k + 1))
  sleep 20
done
