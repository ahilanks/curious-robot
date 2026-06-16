#!/usr/bin/env bash
# alpha=0 × high-lambda_cur sweep (+ alpha=0.2 control).
# Question: with the entropy bonus OFF (alpha=0), can MUCH higher curiosity weight keep the
# arm moving (no stall), the WM improving, and the encoder from collapsing?
# Only alpha and lambda_cur vary; lambda_safe=0.1, beta=0.3, seed=0, n_envs fixed across runs.
set -u
cd "$(dirname "$0")"

NENV=${NENV:-8}
STEPS=${STEPS:-10000}
# 5 concurrent WM updates (512-img ViT backward each) would OOM the 80GB A100 (~20GB/run).
# ViT grad checkpointing recomputes activations on backward -> ~4x lower peak, identical grads.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
GRADCKPT=${GRADCKPT:---wm-grad-checkpoint}

# name              alpha  lambda_cur
RUNS=(
  "a0_lcur20        0.0    20"
  "a0_lcur50        0.0    50"
  "a0_lcur100       0.0    100"
  "a0_lcur200       0.0    200"
  "a02_lcur20       0.2    20"     # control: entropy ON (known-good baseline, no collapse)
)

for row in "${RUNS[@]}"; do
  read -r NAME ALPHA LCUR <<< "$row"
  mkdir -p "runs/$NAME"
  echo "launching $NAME : alpha=$ALPHA lambda_cur=$LCUR n_envs=$NENV steps=$STEPS"
  nohup python src/train.py --name "$NAME" --no-hf $GRADCKPT \
      --n-envs "$NENV" --total-steps "$STEPS" --start-steps 1000 \
      --video-every 1000 --log-every 50 \
      --alpha "$ALPHA" --lambda-cur "$LCUR" --lambda-safe 0.1 \
      > "runs/$NAME/train.log" 2>&1 &
  echo "  pid=$! log=runs/$NAME/train.log"
done
echo "all launched (detached); follow with: tail -f runs/<name>/train.log"
