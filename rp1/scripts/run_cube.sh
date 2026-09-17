#!/bin/bash
# OGBench Cube / LeWM pipeline (Table 4, LeWM columns; easy + hard scores). Usage: scripts/run_cube.sh [cache|critic|actors|evals|all]
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export STABLEWM_HOME=$PWD/swm_home MUJOCO_GL=egl PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CK=downloads/ckpt/lewm-cube
CACHE=runs/cache/cube.npz
OUT=runs/cube
mkdir -p $OUT/logs
PHASE=${1:-all}

if [[ $PHASE == cache ]]; then
  $PY scripts/cache_latents.py --ckpt $CK --h5 swm_home/datasets/cube_single_expert/cube_single_expert.h5 --out $CACHE --workers 24 2>&1 | grep --line-buffered -v "Warning\|self.pid\|lancedb" | tee runs/cache_cube.log
fi

if [[ $PHASE == critic || $PHASE == all ]]; then
  echo "[critic] offline value (12000 steps)"
  $PY scripts/train_critic.py --domain cube --cache $CACHE --out $OUT/critic_offline.pt --seed 0 2>&1 | grep -v Warning | tee $OUT/logs/critic.log
fi

if [[ $PHASE == actors || $PHASE == all ]]; then
  echo "[actors] RP1 refiners: h in {25,100} x seeds {0,1,2} = 6 runs in one process"
  $PY scripts/train_rp1_batched.py --domain cube --h 25 100 --cache $CACHE --critic $OUT/critic_offline.pt --ckpt $CK \
      --out_prefix $OUT/rp1 --seeds 0 1 2 ${TRAIN_EXTRA:-} > $OUT/logs/rp1_actors.log 2>&1
  for f in $OUT/logs/rp1_*.log; do echo "== $f"; grep -v Warning $f | tail -2; done
fi

if [[ $PHASE == evals || $PHASE == all ]]; then
  echo "[evals] 50 episodes x eval seeds {42,43,44}; no-op floor for the hard score"
  RES=$OUT/results.jsonl
  for h in 25 100; do
    $PY scripts/eval.py --domain cube --h $h --planner noop --ckpt $CK --out $RES > $OUT/logs/eval_h${h}_noop.log 2>&1 &
    $PY scripts/eval.py --domain cube --h $h --planner rp1  --ckpt $CK --out $RES --rp1 $OUT/rp1_h${h}_s0.pt $OUT/rp1_h${h}_s1.pt $OUT/rp1_h${h}_s2.pt > $OUT/logs/eval_h${h}_rp1.log 2>&1 &
    for obj in latent value; do for pl in cem mppi adam; do
      $PY scripts/eval.py --domain cube --h $h --planner $pl --objective $obj --critic $OUT/critic_offline.pt --ckpt $CK --out $RES > $OUT/logs/eval_h${h}_${pl}_${obj}.log 2>&1 &
    done; done
    wait
  done
  $PY scripts/aggregate.py $RES | tee $OUT/table.txt
fi
