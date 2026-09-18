#!/usr/bin/env bash
# Touch-input sweep on the LeWM Cube dataset: JEPATouch (lewm/jepa_touch.py: cls + touch_scale * Embedder(touch)
# before the projector; touch = the dataset's proprio_gripper_contact, z-scored) at LeWM's default lambda 0.09,
# one run per touch_scale, everything else identical to the lambda sweep (seed 3072, 8 x 1000 steps ...).
# Baseline = cube_lam0p09 (touch_scale 0). W&B on. Then: bash finish_sweep.sh 8 with RUNS="cube_lam0p09 cube_touch_a*"
# and compare_sigreg.py --x touch_scale.
# Usage: bash run_touch_sweep.sh <max_epochs> <limit_train_batches>   env: SCALES="0.1 0.3 1 3 10" LAMBDA=0.09
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ../.env; set +a
export STABLEWM_HOME="$PWD/swm_home" WANDB_SILENT=true
EPOCHS=${1:?max_epochs}
LIMIT=${2:?limit_train_batches per epoch}
mkdir -p runs
for SC in ${SCALES:-0.1 0.3 1 3 10}; do
  NAME="cube_touch_a${SC//./p}"
  echo "[sweep] $NAME  touch_scale=$SC  lambda=${LAMBDA:-0.09}  epochs=$EPOCHS x $LIMIT batches"
  setsid nohup .venv/bin/python train.py data=ogb_touch model=lewm_touch "model.touch_scale=$SC" \
      "loss.sigreg.weight=${LAMBDA:-0.09}" "output_model_name=$NAME" "subdir=$NAME" \
      wandb.enabled=True "wandb.config.entity=$WANDB_ENTITY" "wandb.config.project=$WANDB_PROJECT" \
      "wandb.config.name=$NAME" "wandb.config.id=$NAME" \
      "trainer.max_epochs=$EPOCHS" "+trainer.limit_train_batches=$LIMIT" "+trainer.limit_val_batches=${VAL_BATCHES:-50}" \
      "num_workers=${WORKERS:-8}" "seed=${SEED:-3072}" ${EXTRA:-} \
      > "runs/$NAME.log" 2>&1 &
  sleep 2
done
echo "[sweep] launched; tail -f runs/cube_touch_*.log"
