#!/bin/bash
cd /workspace/curious-robot/rp1
until grep -q "CACHE EXIT" runs/cache_cube.log 2>/dev/null; do sleep 20; done
grep -q "CACHE EXIT 0" runs/cache_cube.log || { echo "CUBE CACHE FAILED"; exit 1; }
echo "pipeline start $(date)"
TRAIN_EXTRA=--tf32 bash scripts/run_cube.sh all; echo "PIPELINE EXIT $? $(date)"
