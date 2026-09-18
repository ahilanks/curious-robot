#!/usr/bin/env bash
# SIGReg-lambda sweep on the LeWM Cube dataset with the authors' trainer VERBATIM (lewm/train.py,
# config/train/lewm.yaml + data/ogb.yaml): one run per lambda, from scratch, everything else identical,
# W&B on (entity/project from ../.env). Checkpoints: $STABLEWM_HOME/checkpoints/cube_lam<L>/weights_epoch_N.pt
# (+ config.json), logs: lewm/runs/cube_lam<L>.log. Afterwards, per run:
#   .venv/bin/python train_decoder.py --run cube_lam<L>          (post-hoc decoder, App. D)
# and across runs:
#   .venv/bin/python compare_sigreg.py --runs cube_lam0p0 cube_lam0p01 ...
#
# Usage: bash run_sigreg_sweep.sh <max_epochs> <limit_train_batches>   env: LAMBDAS="0.0 0.01 0.09 0.5 2.0"
#   limit_train_batches = optimizer steps per "epoch" (Lightning re-samples a different random subset each
#   epoch; SaveCkptCallback saves after every epoch, so this is the checkpoint cadence), max_epochs x that = budget.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ../.env; set +a
export STABLEWM_HOME="$PWD/swm_home" WANDB_SILENT=true
EPOCHS=${1:?max_epochs}
LIMIT=${2:?limit_train_batches per epoch}
mkdir -p runs
for LAM in ${LAMBDAS:-0.0 0.01 0.09 0.5 2.0}; do
  NAME="cube_lam${LAM//./p}"
  echo "[sweep] $NAME  lambda=$LAM  epochs=$EPOCHS x $LIMIT batches"
  setsid nohup .venv/bin/python train.py data=ogb "loss.sigreg.weight=$LAM" "output_model_name=$NAME" "subdir=$NAME" \
      wandb.enabled=True "wandb.config.entity=$WANDB_ENTITY" "wandb.config.project=$WANDB_PROJECT" \
      "wandb.config.name=$NAME" "wandb.config.id=$NAME" \
      "trainer.max_epochs=$EPOCHS" "+trainer.limit_train_batches=$LIMIT" "+trainer.limit_val_batches=${VAL_BATCHES:-100}" \
      "num_workers=${WORKERS:-8}" "seed=${SEED:-3072}" ${EXTRA:-} \
      > "runs/$NAME.log" 2>&1 &
  sleep 2
done
echo "[sweep] launched; tail -f runs/cube_lam*.log"
