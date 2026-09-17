#!/bin/bash
cd /workspace/curious-robot/rp1
export STABLEWM_HOME=$PWD/swm_home MUJOCO_GL=egl PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
until grep -q "CACHE EXIT" runs/cache_tworoom.log; do sleep 15; done
until grep -q "^reacher:" downloads/extract2.log; do sleep 15; done
.venv/bin/python scripts/cache_latents.py --ckpt downloads/ckpt/lewm-reacher --h5 swm_home/datasets/reacher/reacher.h5 --out runs/cache/reacher.npz --workers 24 2>&1 | grep --line-buffered -vE "Warning|warn|ale-py|fontManager|atomic_save" > runs/cache_reacher.log; echo "CACHE EXIT ${PIPESTATUS[0]}" >> runs/cache_reacher.log
until grep -q "^cube_single_expert:" downloads/extract2.log; do sleep 15; done
.venv/bin/python scripts/cache_latents.py --ckpt downloads/ckpt/lewm-cube --h5 swm_home/datasets/cube_single_expert/cube_single_expert.h5 --out runs/cache/cube.npz --workers 24 2>&1 | grep --line-buffered -vE "Warning|warn|ale-py|fontManager|atomic_save|self.pid|lancedb" > runs/cache_cube.log; echo "CACHE EXIT ${PIPESTATUS[0]}" >> runs/cache_cube.log
echo "ALL CACHES DONE $(date)"
