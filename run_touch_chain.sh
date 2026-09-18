#!/usr/bin/env bash
# The wr_sleepret2 lineage, verbatim, WITH the touch branch (--touch-input effort --touch-scale 1.0; 2026-09-18):
#   stage 0  wt_slpwarm   3k from scratch  = wr_slpwarm  (W&B rb0t6sb0)  encoder trained, action_max 0.4, cem std 1.0, no dwell
#   stage 1  wt_sleepret  20k              = wr_sleepret (W&B 0hvv1z0c)  init stage-0 @3000, encoder frozen, sleeps every 3k, d-start 3
#   stage 2  wt_sleepret2 200k             = wr_sleepret2 (W&B q1dzgjq4) init stage-1 @20000, sleeps every 10k, d-start 10
# Stage recipes reconstructed from the W&B configs (ledger 2026-09-18); the stage-2 block is the 08-15 canonical launch
# command. wr_sleepret was launched with --total-steps 40000 and died at ~20k (disk); its @20000 ckpt seeded the 200k run,
# so stage 1 runs --total-steps 20000 (buffer cap 4.0 x 20k -> clipped to 50k, identical to 4.0 x 40k). Weights-only chaining
# at every stage (the 200k run started with an empty buffer: "no state_latest.npz next to init-ckpt -> starting empty").
# Usage: setsid nohup bash run_touch_chain.sh > runs/chains/touch_chain.log 2>&1 &     env: TOUCH_SCALE=1.0 STAGE_FROM=0
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
export WANDB_SILENT=true MUJOCO_GL="${MUJOCO_GL:-egl}"
TOUCH=(--touch-input effort --touch-scale "${TOUCH_SCALE:-1.0}")
COMMON=(--env-threads 8 --wm-cam wrist --no-proprio --sigreg-pertimestep
  --cem --cem-horizon 1 --cem-replan-every 1 --deterministic-act --alpha 0.0
  --goal-explore --goal-select highmse_under_d --goal-curriculum --goal-curric-metric arrival
  --goal-curric-thresh 0.95 --goal-curric-patience 150 --goal-curric-d-max 22
  --goal-reach-eps 2.8 --goal-update-every 50 --goal-rescore-every 50
  --goal-retain --goal-retain-patience 200 --goal-retain-delta 0.005 --goal-retain-maxage 500
  --consolidate-every 70 --consolidate-epochs 1
  --buffer-frac 4.0 --lambda-safe 0.0 --safety-delta 15.0)
mkdir -p runs/chains
STAGE_FROM=${STAGE_FROM:-0}

if [ "$STAGE_FROM" -le 0 ]; then
  echo "[chain] stage 0: wt_slpwarm (3k from scratch)  $(date -u +%FT%TZ)"
  python src/train.py --name wt_slpwarm --total-steps 3000 --start-steps 1000 "${COMMON[@]}" "${TOUCH[@]}" \
    --action-max 0.4 --amax-curric-start 0.05 --cem-init-std 1.0 \
    --goal-curric-d-start 6 --dwell-hold-mult 0 --dwell-shrink-start 0 --dwell-shrink-min 0.2 \
    --cotrain-every 0 --cotrain-epochs 3 --cotrain-beta 0.02 \
    --keep-local-ckpts --no-save-state 2>&1 | tee runs/chains/wt_slpwarm.log | grep -E "^\[step|\[touch\]|\[done\]|Traceback|Error" | grep -v Warning
fi
if [ "$STAGE_FROM" -le 1 ]; then
  echo "[chain] stage 1: wt_sleepret (20k, init wt_slpwarm@3000)  $(date -u +%FT%TZ)"
  python src/train.py --name wt_sleepret --total-steps 20000 --start-steps 1000 "${COMMON[@]}" "${TOUCH[@]}" \
    --init-ckpt runs/wt_slpwarm/ckpt_0003000.pt --freeze-encoder \
    --action-max 5.6 --amax-curric --amax-curric-start 0.4 --amax-curric-floor 0.05 --cem-init-std 0.3 \
    --goal-curric-d-start 3 --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
    --cotrain-every 3000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 --cotrain-frac-thresh 0.45 \
    --keep-local-ckpts --no-save-state 2>&1 | tee runs/chains/wt_sleepret.log | grep -E "^\[step [0-9]*000\]|\[touch\]|\[done\]|Traceback|Error|sleep" | grep -v Warning
fi
if [ "$STAGE_FROM" -le 2 ]; then
  echo "[chain] stage 2: wt_sleepret2 (200k, init wt_sleepret@20000)  $(date -u +%FT%TZ)"
  python src/train.py --name wt_sleepret2 --total-steps 200000 --start-steps 1000 "${COMMON[@]}" "${TOUCH[@]}" \
    --init-ckpt runs/wt_sleepret/ckpt_0020000.pt --freeze-encoder \
    --action-max 5.6 --amax-curric --amax-curric-start 0.4 --amax-curric-floor 0.05 --cem-init-std 0.3 \
    --goal-curric-d-start 10 --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
    --cotrain-every 10000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 --cotrain-frac-thresh 0.45 \
    2>&1 | tee runs/chains/wt_sleepret2.log | grep -E "^\[step [0-9]*000\]|\[touch\]|\[done\]|Traceback|Error|sleep" | grep -v Warning
fi
echo "[chain] done $(date -u +%FT%TZ)"
