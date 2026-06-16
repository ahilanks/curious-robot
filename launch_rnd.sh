#!/usr/bin/env bash
# RND sweep: does adding an RND novelty bonus (a real *search* signal) to the curiosity
# reward rescue the alpha=0 exploration/encoder collapse that higher lambda_cur could not?
# alpha=0 (entropy OFF) + curiosity (lambda_cur=20) + RND swept over lambda_rnd.
# Baseline lambda_rnd=0 is the existing a0_lcur20 run (collapsed).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NENV=${NENV:-8}
STEPS=${STEPS:-10000}

# name        lambda_rnd
RUNS=(
  "rnd_l5     5"
  "rnd_l10    10"
  "rnd_l20    20"
  "rnd_l40    40"
)

for row in "${RUNS[@]}"; do
  read -r NAME LRND <<< "$row"
  mkdir -p "runs/$NAME"
  echo "launching $NAME : alpha=0 lambda_cur=20 lambda_rnd=$LRND n_envs=$NENV steps=$STEPS"
  nohup python src/train.py --name "$NAME" --no-hf --wm-grad-checkpoint \
      --n-envs "$NENV" --total-steps "$STEPS" --start-steps 1000 \
      --video-every 1000 --log-every 50 \
      --alpha 0.0 --lambda-cur 20 --lambda-rnd "$LRND" --lambda-safe 0.1 \
      > "runs/$NAME/train.log" 2>&1 &
  echo "  pid=$! log=runs/$NAME/train.log"
done
echo "all launched (detached); follow with: tail -f runs/<name>/train.log"
