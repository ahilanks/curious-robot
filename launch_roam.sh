#!/usr/bin/env bash
# "Roam from intrinsic reward alone" sweep. Goal: a DETERMINISTIC policy (alpha=0, NO action
# noise) that MOVES THE ARM THROUGH SPACE driven only by curiosity (MSE) + obs-RND -- reward =
# lambda_cur*symlog(r_cur) + lambda_rnd*log1p(scale*r_rnd), NO safety term (lambda_safe=0).
#
# Why this is hard (read the 2026-06-16/17 logistics entries): alpha=0 is a stable collapse
# attractor, and BOTH levers are gameable by in-place jitter -- curiosity is harvested from the
# WM mis-predicting the arm's own bang-bang jitter (no travel needed), and the wrist-cam obs-RND
# is panned by wrist rotation (fake visual novelty). So the hypothesis under test is: does making
# obs-RND DOMINATE while shrinking the jitter-gaming curiosity term let the honest novelty signal
# pull the arm into real roaming before the deterministic policy saturates?
#
# JUDGE ON THE TRAVEL METRICS, NOT arm_speed: explore/prox_step (proximal-joint travel -- the
# ungameable "gross repositioning" signal a wrist wiggle can't fake) and explore/ee_step (gripper
# world-xyz travel). Win = prox_step/ee_step stay well above a dithering baseline with eff_rank_probe
# not collapsing; loss = arm_speed stays high (dither) while prox_step/ee_step decay toward 0.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
set -a; source .env 2>/dev/null; set +a
NENV=${NENV:-8}; STEPS=${STEPS:-10000}
# ViT grad-checkpointing so all 4 fit concurrently on one 80GB A100 (~12GB/run vs ~20GB).
GRADCKPT=${GRADCKPT:---wm-grad-checkpoint}

# name                 lambda_cur  lambda_rnd  rnd_train_every  rnd_lr   (all: alpha=0, explore_noise=0, lambda_safe=0)
RUNS=(
  "roam_c20_rnd20        20    20    1    5e-5"   # literal "MSE + RND, both strong" baseline
  "roam_c5_rnd20          5    20    1    5e-5"   # RND-dominant, weak MSE (shrink the jitter-gaming curiosity)
  "roam_c0_rnd20          0    20    1    5e-5"   # pure obs-RND: isolate whether novelty alone drives roaming
  "roam_c5_rnd40_slow     5    40    4    5e-6"   # strong + persistent RND (slow predictor), weak MSE
)
for row in "${RUNS[@]}"; do
  read -r NAME LCUR LRND TE LR <<< "$row"
  echo "launching $NAME : alpha=0 noise=0 lambda_safe=0 lambda_cur=$LCUR lambda_rnd=$LRND rnd_train_every=$TE rnd_lr=$LR"
  nohup python src/train.py --name "$NAME" --no-hf $GRADCKPT \
      --n-envs "$NENV" --total-steps "$STEPS" --start-steps 1000 \
      --video-every 1000 --log-every 50 \
      --alpha 0.0 --explore-noise 0.0 --lambda-safe 0.0 \
      --lambda-cur "$LCUR" --lambda-rnd "$LRND" \
      --rnd-train-every "$TE" --rnd-lr "$LR" \
      > "runs/${NAME}_stdout.log" 2>&1 &
  echo "  pid=$! log=runs/${NAME}_stdout.log"
done
echo "all launched (detached); watch explore/prox_step + explore/ee_step on W&B (NOT arm_speed)"
