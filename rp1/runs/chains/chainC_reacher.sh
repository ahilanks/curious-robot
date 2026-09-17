#!/bin/bash
cd /workspace/curious-robot/rp1
until grep -q "CACHE EXIT" runs/cache_reacher.log 2>/dev/null; do sleep 20; done
grep -q "CACHE EXIT 0" runs/cache_reacher.log || { echo "REACHER CACHE FAILED"; exit 1; }
echo "pipeline start $(date)"
bash scripts/run_reacher.sh all; echo "PIPELINE EXIT $? $(date)"
