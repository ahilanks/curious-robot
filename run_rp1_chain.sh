#!/usr/bin/env bash
# RP1 FROM SCRATCH, ONLINE (2026-09-20): the wr_sleepret2 lineage with the RP1 planner (critic + refiner fitted on the
# replay buffer only), started from the CANONICAL stage-0 head (wr_slpwarm@3000, HF) so both arms share the exact
# tabula-rasa warm-up of the 08-14/15 campaign.  Two arms, same seed, same chain:
#   ARM=cs   cs_sleepret / cs_sleepret2   canonical recipe + --plan-act-scale                (the control the campaign lacks)
#   ARM=rp   rp_sleepret / rp_sleepret2   + --planner rp1 --goal-budget value --plan-gate-frac 0.35 --rp1-trip-frac 0.5
#            (critic/refiner used only while live frac_rand < 0.35, CEM+L2 otherwise; V-budget goals, L2 arrival;
#             FULL re-fit right after every sleep; vK>v0 tripwire -> CEM until the next full re-fit; NO action noise)
#   stage 1  20k  init wr_slpwarm@3000, encoder frozen, sleeps every 3k, d-start 3      (= wr_sleepret,  W&B 0hvv1z0c)
#   stage 2  200k init stage-1 @20000,   sleeps every 10k, d-start 10                    (= wr_sleepret2, W&B q1dzgjq4)
# Usage: ARM=rp setsid nohup bash run_rp1_chain.sh > runs/chains/rp_chain.log 2>&1 &      env: STAGE_FROM=1
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
export WANDB_SILENT=true MUJOCO_GL="${MUJOCO_GL:-egl}"
ARM=${ARM:?set ARM=cs, rp, lp, rl or rr}
case "$ARM" in
  cs) PLAN=(--plan-act-scale) ;;
  rp) PLAN=(--plan-act-scale --planner rp1 --goal-budget value --plan-gate-frac 0.35 --rp1-trip-frac 0.5
            --vcritic-fit-every 2000) ;;
  lp) PLAN=(--plan-act-scale --goal-score lp) ;;      # learning-progress goals (2026-09-21): cs recipe + LP score
  rl) PLAN=(--plan-act-scale --planner rp1 --goal-budget value --plan-gate-frac 0.35 --rp1-trip-frac 0.5
            --vcritic-fit-every 2000 --goal-score lp) ;;                       # RP1 planner + LP goals
  rr) PLAN=(--plan-act-scale --planner rp1 --goal-budget value --plan-gate-frac 0.35 --rp1-trip-frac 0.5
            --vcritic-fit-every 2000 --goal-score rnd --rnd-train-every 200) ;;  # RP1 planner + RND goals (slow predictor)
  *) echo "ARM must be cs, rp, lp, rl or rr"; exit 1 ;;
esac
COMMON=(--env-threads 8 --wm-cam wrist --no-proprio --sigreg-pertimestep
  --cem --cem-horizon 1 --cem-replan-every 1 --deterministic-act --alpha 0.0
  --goal-explore --goal-select highmse_under_d --goal-curriculum --goal-curric-metric arrival
  --goal-curric-thresh 0.95 --goal-curric-patience 150 --goal-curric-d-max 22
  --goal-reach-eps 2.8 --goal-update-every 50 --goal-rescore-every 50
  --goal-retain --goal-retain-patience 200 --goal-retain-delta 0.005 --goal-retain-maxage 500
  --consolidate-every 70 --consolidate-epochs 1
  --buffer-frac 4.0 --lambda-safe 0.0 --safety-delta 15.0)
mkdir -p runs/chains
STAGE_FROM=${STAGE_FROM:-1}
STAGE_TO=${STAGE_TO:-2}
if [ ! -f runs/wr_slpwarm/ckpt_0003000.pt ]; then
  python -c "import os; from huggingface_hub import hf_hub_download; hf_hub_download(os.environ['HF_UPLOAD_REPO_ID'],'wr_slpwarm/ckpt_0003000.pt',local_dir='runs')"
fi
FILT='^\[step [0-9]*000\]|\[done\]|Traceback|Error|sleep|\[planner|\[vdiag\]|\[plan-|\[goal-curriculum\]|\[amax-curric\]'

if [ "$STAGE_FROM" -le 1 ]; then
  echo "[chain/$ARM] stage 1: ${ARM}_sleepret (20k, init wr_slpwarm@3000)  $(date -u +%FT%TZ)"
  python src/train.py --name ${ARM}_sleepret --total-steps 20000 --start-steps 1000 "${COMMON[@]}" "${PLAN[@]}" \
    --init-ckpt runs/wr_slpwarm/ckpt_0003000.pt --freeze-encoder \
    --action-max 5.6 --amax-curric --amax-curric-start 0.4 --amax-curric-floor 0.05 --cem-init-std 0.3 \
    --goal-curric-d-start 3 --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
    --cotrain-every 3000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 --cotrain-frac-thresh 0.45 \
    --keep-local-ckpts --no-save-state 2>&1 | tee runs/chains/${ARM}_sleepret.log | grep -E "$FILT" | grep -v Warning
fi
if [ "$STAGE_FROM" -le 2 ] && [ "$STAGE_TO" -ge 2 ]; then
  echo "[chain/$ARM] stage 2: ${ARM}_sleepret2 (200k, init ${ARM}_sleepret@20000)  $(date -u +%FT%TZ)"
  python src/train.py --name ${ARM}_sleepret2 --total-steps 200000 --start-steps 1000 "${COMMON[@]}" "${PLAN[@]}" \
    --init-ckpt runs/${ARM}_sleepret/ckpt_0020000.pt --freeze-encoder \
    --action-max 5.6 --amax-curric --amax-curric-start 0.4 --amax-curric-floor 0.05 --cem-init-std 0.3 \
    --goal-curric-d-start 10 --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
    --cotrain-every 10000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 --cotrain-frac-thresh 0.45 \
    2>&1 | tee runs/chains/${ARM}_sleepret2.log | grep -E "$FILT" | grep -v Warning
fi
echo "[chain/$ARM] done $(date -u +%FT%TZ)"
