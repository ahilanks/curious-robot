#!/usr/bin/env bash
# Real-arm deployment of the wr_sleepret2 final head (W&B q1dzgjq4, ckpt_0200000, HF-verified)
# with its goal-explore stack, adapted sim->hardware per the hw_zs_hot6b deploy conventions.
#
# Everything below is the canonical wr_sleepret2 recipe (ledger 2026-08-15) EXCEPT:
#   --env-backend hardware          one physical arm (n_envs forced to 1, control_dt 0.03s)
#   --max-episode-steps 100000      no episodic resets on a real arm (hot6b precedent)
#   (no --safety-delta/--lambda-safe) hardware defaults to the arm-calibrated delta=9;
#                                   goal-explore forces lambda_safe=0 regardless
#   (no --start-steps/--env-threads) hardware forces start_steps=0; env-threads is sim-only
#   --no-hf                         session ckpts/metrics stay local under runs/<name>/
#   (state DOES save now)           runs/<name>/state_latest.npz at run end — the real-frame
#                                   corpus for src/train_decoder.py AND the chain state for
#                                   GPU sleep passes / --init-buffer continuation
#   --live-view 8765                live goal dashboard -> http://localhost:8765
#   DECODER=<path>                  optional env var: post-hoc pixel decoder ckpt -> adds the
#                                   "decoder's eye" row (decode z_now / plan-imagined next / z*).
#                                   Train one after a session:
#                                     python src/train_decoder.py --ckpt <session ckpt> \
#                                       --state runs/<name>/state_latest.npz --out runs/decoder_wrs2.pt
#                                   Auto-picked-up from runs/decoder_wrs2.pt if present.
#
# ACTION IS NOT LIMITED (user-directed 2026-08-20): the canonical amplitude stack runs as
# trained — action-max 5.6 with the locality-gated amax curriculum (eff amplitude starts at
# 0.4 rad/step, floor 0.05, raised/lowered by latent locality). What physically bounds motion
# instead, all pre-existing bus/env-level protections, none of them policy limits:
#   * servo Goal_Speed pacing: real travel <= goal_speed 2400 ticks/s (~3.7 rad/s/joint)
#   * FeetechBus max_step_ticks 300 (~0.46 rad): no single command jumps further than that
#     from the present position (runaway backstop if the loop stalls mid-move)
#   * safe envelope (clip_safe): pan held in the cable-safe band, J2-J4 down-caps keep the
#     gripper off the table plane (SOARM_NO_ENVELOPE=1 disables — don't, for autonomous runs)
#   * TorqueGuard (so101_calib.json current_cap_a/cap_trip_steps): sustained over-cap current
#     -> the fighting joint PINS (sticky hold at present position), sustained/duty-cycled
#     pressure -> torque CUT + loud TorqueLimitExceeded. Ground the cap in THIS arm's real
#     draw first: python src/bench_torqueguard.py --port $SOARM_PORT [--fire]
#
# PRE-FLIGHT (ledger hardware ops): clamp the base; route cables out of the sweep volume;
# e-stop within reach; workspace objects placed; LIGHTS ON; wrist cam index verified
# (SOARM_CAM, default 0) — quick check:
#   python3 -c "import cv2; c=cv2.VideoCapture(0); print(c.read()[1].mean())"  # ~0 = wrong cam/dark
# First run ATTENDED: at eff 0.4 rad/step the arm sweeps fast (pacing-bound ~3.7 rad/s).
#
# Usage: bash run_hw_wr_sleepret2.sh [name] [total_steps]     (defaults: hw_wrs2_a 2000)
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

export SOARM_PORT=${SOARM_PORT:-/dev/cu.usbmodem5AA90245791}
export SOARM_CALIB=${SOARM_CALIB:-so101_calib.json}
export SOARM_CAM=${SOARM_CAM:-0}

NAME=${1:-hw_wrs2_a}
STEPS=${2:-2000}
CKPT=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('a5ilank/curious-robot', 'wr_sleepret2/ckpt_0200000.pt'))")
DECODER=${DECODER:-runs/decoder_wrs2.pt}
DEC_ARGS=()
if [ -f "$DECODER" ]; then                 # NOT `[ -f ] && ...`: under set -e a missing file would kill the script
  DEC_ARGS=(--decoder "$DECODER"); echo "[launch] decoder's eye: $DECODER"
fi

# LEARNER=<name>: near-live GPU split. This side goes INFERENCE-ONLY (--frozen-policy;
# the server owns every gradient incl. the sleeps — the local consolidate/cotrain flags
# below are gated out by frozen), streams state to HF every SYNC_EVERY steps (~14 min at
# 2.4 sps; each upload is the full ring, ~150 KB/step collected), and hot-swaps the
# server's wm_live.pt between decisions. Server side: src/wm_sleep_server.py --name $LEARNER.
SPLIT_ARGS=(--no-hf)
if [ -n "${LEARNER:-}" ]; then
  SPLIT_ARGS=(--frozen-policy --pull-wm "$LEARNER" --pull-wm-every "${PULL_EVERY:-60}"
              --save-state-every "${SYNC_EVERY:-2000}")
  echo "[launch] SPLIT mode: frozen collector, state -> HF every ${SYNC_EVERY:-2000} steps, weights <- $LEARNER/wm_live.pt"
fi
echo "[launch] $NAME  steps=$STEPS  ckpt=$CKPT"
echo "[launch] dashboard: http://localhost:8765"

caffeinate -i python3 src/train.py \
  --env-backend hardware --name "$NAME" \
  --init-ckpt "$CKPT" \
  --wm-cam wrist --no-proprio --sigreg-pertimestep --freeze-encoder \
  --action-max 5.6 --amax-curric --amax-curric-start 0.4 --amax-curric-floor 0.05 \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 --deterministic-act --alpha 0.0 \
  --goal-explore --goal-select highmse_under_d --goal-curriculum --goal-curric-metric arrival \
  --goal-curric-thresh 0.95 --goal-curric-patience 150 --goal-curric-d-start 10 --goal-curric-d-max 22 \
  --goal-reach-eps 2.8 --goal-update-every 50 --goal-rescore-every 50 \
  --goal-retain --goal-retain-patience 200 --goal-retain-delta 0.005 --goal-retain-maxage 500 \
  --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
  --consolidate-every 70 --consolidate-epochs 1 \
  --cotrain-every 10000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 --cotrain-frac-thresh 0.45 \
  --buffer-frac 4.0 \
  --max-episode-steps 100000 --total-steps "$STEPS" \
  --live-view 8765 ${DEC_ARGS[@]+"${DEC_ARGS[@]}"} ${SPLIT_ARGS[@]+"${SPLIT_ARGS[@]}"} \
  2>&1 | tee "runs/${NAME}_train.log"   # bash-3.2-safe empty-array expansion (macOS default shell)
