#!/usr/bin/env bash
# Fit the LeWM-style post-hoc pixel decoder (model/decoder.py, arXiv 2603.19312 App. D) on a
# session's real frames with that session's FROZEN encoder, upload it to HF, and render the
# Fig. 7-style imagined-rollout sheet. Diagnostic only — nothing here touches a WM gradient.
#
# Inputs resolve locally first, else from $HF_UPLOAD_REPO_ID (.env):
#   <run>/ckpt_XXXXXXX.pt      the encoder the decoder is tied to (z_dim 192, pixels-only)
#   <run>/state_latest.npz     the frame ring the decoder is fitted on (real wrist photos on hw)
# Outputs:
#   runs/<run>/<out>.pt (+ .sheet.png)      decoder ckpt + real-vs-recon contact sheet
#   runs/<run>/rollout.png (+ rollout.gif)  context | imagined future vs real (viz_decoder_rollout.py)
#   HF: <run>/<out>.pt + <run>/<out>.sheet.png (with UPLOAD=1, default)
# Then: cp runs/<run>/<out>.pt runs/decoder_wrs2.pt on the Mac (run_hw_wr_sleepret2.sh auto-picks
# it up) or DECODER=<path> bash run_hw_wr_sleepret2.sh ...
#
# Usage: bash run_decoder.sh [run] [ckpt_step] [steps]      (defaults: hw_wrs2_c 4000 3000)
#   env: OUT=decoder_lewm  UPLOAD=1  BATCH=128  HIDDEN=256  DEPTH=3  HEADS=4  HORIZON=8
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

RUN=${1:-hw_wrs2_c}
STEP=${2:-4000}
STEPS=${3:-3000}
OUT=${OUT:-decoder_lewm}
CKPT=$(printf "%s/ckpt_%07d.pt" "$RUN" "$STEP")
STATE="$RUN/state_latest.npz"
mkdir -p "runs/$RUN"
UP_ARGS=()
if [ "${UPLOAD:-1}" = "1" ]; then UP_ARGS=(--upload --upload-run "$RUN"); fi

echo "[decoder] run=$RUN ckpt=$CKPT state=$STATE steps=$STEPS -> runs/$RUN/$OUT.pt"
python3 src/train_decoder.py --ckpt "runs/$CKPT" --state "runs/$STATE" --out "runs/$RUN/$OUT.pt" \
  --steps "$STEPS" --batch "${BATCH:-128}" --hidden "${HIDDEN:-256}" --depth "${DEPTH:-3}" \
  --heads "${HEADS:-4}" ${UP_ARGS[@]+"${UP_ARGS[@]}"} 2>&1 | tee "runs/$RUN/${OUT}_train.log"
# train_decoder.py falls back to the HF spec when runs/<...> is missing; reuse whatever it resolved
CKPT_LOCAL=$(python3 -c "import torch;print(torch.load('runs/$RUN/$OUT.pt',map_location='cpu',weights_only=False)['ckpt'])")
STATE_LOCAL=$(python3 -c "import torch;print(torch.load('runs/$RUN/$OUT.pt',map_location='cpu',weights_only=False)['state'])")
python3 src/viz_decoder_rollout.py --ckpt "$CKPT_LOCAL" --state "$STATE_LOCAL" \
  --decoder "runs/$RUN/$OUT.pt" --out "runs/$RUN/rollout.png" --gif "runs/$RUN/rollout.gif" \
  --horizon "${HORIZON:-8}" --n 4 2>&1 | tee "runs/$RUN/${OUT}_rollout.log"
echo "[decoder] done: runs/$RUN/$OUT.pt  runs/$RUN/$OUT.sheet.png  runs/$RUN/rollout.png"
