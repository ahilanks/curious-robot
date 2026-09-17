#!/bin/bash
# Reacher / LeWM pipeline (Table 2/3, LeWM columns, single-frame cost window).
# Usage: scripts/run_reacher.sh [critic|actors|evals|all]
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export STABLEWM_HOME=$PWD/swm_home MUJOCO_GL=egl PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CK=downloads/ckpt/lewm-reacher
CACHE=runs/cache/reacher.npz
OUT=runs/reacher
mkdir -p $OUT/logs
PHASE=${1:-all}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}    # paper: 0..5 planner seeds
EVAL_SEEDS=${EVAL_SEEDS:-"42 43 44 45 46 47"}  # paper: 42..47

if [[ $PHASE == critic || $PHASE == all ]]; then
  echo "[critic] offline value (6000 steps, standardized latents)"
  $PY scripts/train_critic.py --domain reacher --cache $CACHE --out $OUT/critic_offline.pt --seed 0 2>&1 | grep -v Warning | tee $OUT/logs/critic.log
fi

if [[ $PHASE == actors || $PHASE == all ]]; then
  echo "[actors] RP1 refiners: h=25, seeds {$SEEDS} in one process"
  $PY scripts/train_rp1_batched.py --domain reacher --h 25 --cache $CACHE --critic $OUT/critic_offline.pt --ckpt $CK \
      --out_prefix $OUT/rp1 --seeds $SEEDS > $OUT/logs/rp1_actors.log 2>&1
  for f in $OUT/logs/rp1_*.log; do echo "== $f"; grep -v Warning $f | tail -2; done
fi

if [[ $PHASE == evals || $PHASE == all ]]; then
  echo "[evals] first-hit success at tau=0.1 and tau=0.05, 50 episodes x eval seeds {$EVAL_SEEDS}"
  RES=$OUT/results.jsonl
  RP1=""; for s in $SEEDS; do RP1="$RP1 $OUT/rp1_h25_s${s}.pt"; done
  for tau in 0.1 0.05; do
    $PY scripts/eval.py --domain reacher --h 25 --planner noop --ckpt $CK --out $RES --seeds $EVAL_SEEDS --reacher_tau $tau > $OUT/logs/eval_noop_$tau.log 2>&1 &
    $PY scripts/eval.py --domain reacher --h 25 --planner rp1  --ckpt $CK --out $RES --seeds $EVAL_SEEDS --reacher_tau $tau --rp1 $RP1 > $OUT/logs/eval_rp1_$tau.log 2>&1 &
    for obj in latent value; do for pl in cem mppi adam; do
      $PY scripts/eval.py --domain reacher --h 25 --planner $pl --objective $obj --critic $OUT/critic_offline.pt --ckpt $CK --out $RES --seeds $EVAL_SEEDS --reacher_tau $tau > $OUT/logs/eval_${pl}_${obj}_$tau.log 2>&1 &
    done; done
    wait
  done
  $PY scripts/aggregate.py $RES | tee $OUT/table.txt
fi
