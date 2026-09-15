#!/usr/bin/env bash
# Sim twin of run_hw_wr_sleepret2.sh (verbatim reproduction source per the 07-02 rule): the
# wr_sleepret2 @200k stack FROZEN in MuJoCo -- CEM planner + restored goal archive, NO gradient
# updates -- with the post-hoc pixel decoder attached, and the dashboard's decoder's-eye row
# RECORDED to an mp4 (--live-view-record): one strip per decision step,
#   wrist (real) | decode(z now) | decode(plan -> next z) | decode(z*) | goal photo
# with a header (step, ||z-z*|| vs eps, recon L1, ladder d / pctl, eff amax). Playback only:
# no W&B, no HF, no state snapshot, no ckpt. Diagnostic -- nothing touches a WM gradient.
#
# Inputs (local first, else pulled from $HF_UPLOAD_REPO_ID by train.py / train_decoder.py):
#   runs/wr_sleepret2/ckpt_0200000.pt      the banked sim head (08-15 ledger)
#   runs/wr_sleepret2/decoder_lewm.pt      its decoder (bash run_decoder.sh wr_sleepret2 200000 3000)
# Output: runs/<name>/decoder_eye.mp4 (+ buffer_<N>.npz, metrics.jsonl, train.log)
#
# Usage: bash run_sim_decoder_eye.sh [name] [steps]      (defaults: sim_decoder_eye 1500)
#   env: CKPT=<path>  DECODER=<path>  FPS=15  PORT=8765 (the live dashboard is served too)
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

NAME=${1:-sim_decoder_eye}
STEPS=${2:-1500}
CKPT=${CKPT:-runs/wr_sleepret2/ckpt_0200000.pt}
DECODER=${DECODER:-runs/wr_sleepret2/decoder_lewm.pt}
if [ ! -f "$CKPT" ]; then
  mkdir -p "$(dirname "$CKPT")"
  ln -sfn "$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('${HF_UPLOAD_REPO_ID}', 'wr_sleepret2/ckpt_0200000.pt'))")" "$CKPT"
fi
if [ ! -f "$DECODER" ]; then
  mkdir -p "$(dirname "$DECODER")"
  cp "$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('${HF_UPLOAD_REPO_ID}', 'wr_sleepret2/decoder_lewm.pt'))")" "$DECODER"
fi
mkdir -p "runs/$NAME"
echo "[sim-eye] $NAME steps=$STEPS ckpt=$CKPT decoder=$DECODER -> runs/$NAME/decoder_eye.mp4"

python3 src/train.py \
  --name "$NAME" --n-envs 1 --start-steps 0 --total-steps "$STEPS" \
  --init-ckpt "$CKPT" --frozen-policy \
  --wm-cam wrist --no-proprio --sigreg-pertimestep --freeze-encoder \
  --action-max 5.6 --amax-curric --amax-curric-start 0.4 --amax-curric-floor 0.05 \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 --deterministic-act --alpha 0.0 \
  --goal-explore --goal-select highmse_under_d --goal-curriculum --goal-curric-metric arrival \
  --goal-curric-thresh 0.95 --goal-curric-patience 150 --goal-curric-d-start 10 --goal-curric-d-max 22 \
  --goal-reach-eps 2.8 --goal-update-every 50 --goal-rescore-every 50 \
  --goal-retain --goal-retain-patience 200 --goal-retain-delta 0.005 --goal-retain-maxage 500 \
  --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
  --buffer-frac 4.0 --no-wandb --no-hf --no-save-state --save-every 0 --video-every 0 \
  --live-view "${PORT:-8765}" --decoder "$DECODER" \
  --live-view-record "runs/$NAME/decoder_eye.mp4" \
  2>&1 | tee "runs/$NAME/train.log"
echo "[sim-eye] done: runs/$NAME/decoder_eye.mp4"
