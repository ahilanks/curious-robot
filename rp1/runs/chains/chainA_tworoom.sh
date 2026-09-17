#!/bin/bash
cd /workspace/curious-robot/rp1
until grep -q "CACHE EXIT" runs/cache_tworoom.log; do sleep 15; done
grep -q "CACHE EXIT 0" runs/cache_tworoom.log || { echo "TWOROOM CACHE FAILED"; exit 1; }
echo "pipeline start $(date)"
bash scripts/run_tworoom.sh all; echo "PIPELINE EXIT $? $(date)"
