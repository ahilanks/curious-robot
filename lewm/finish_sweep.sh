#!/usr/bin/env bash
# After run_sigreg_sweep.sh: fit one post-hoc decoder per run (same frames, same seed; BatchNorm running stats
# re-estimated first, see train_decoder.recalibrate_bn) and build the cross-lambda comparison.
# Decoders fit in PARALLEL (one process per run, ~2 GB GPU each; the h5 reads dominate). Re-runnable: decoders
# that already exist for the chosen epoch are skipped.
# Usage: bash finish_sweep.sh [epoch]      env: RUNS="cube_lam0p0 ..."  STEPS=3000  OUT=runs/sigreg_sweep  PARALLEL=1
set -euo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME="$PWD/swm_home"
EPOCH=${1:-}
RUNS=${RUNS:-cube_lam0p0 cube_lam0p01 cube_lam0p09 cube_lam0p5 cube_lam2p0}
EP_ARG=(); [ -n "$EPOCH" ] && EP_ARG=(--epoch "$EPOCH")
FILTER="Warning\|warn\|ale \|atomic_chec\|Created ViT"
pids=()
for R in $RUNS; do
  if [ -n "$EPOCH" ] && [ -f "swm_home/checkpoints/$R/decoder_epoch_$EPOCH.pt" ]; then
    echo "[finish] $R: decoder_epoch_$EPOCH.pt exists, skipping"; continue
  fi
  echo "[finish] decoder for $R ${EPOCH:+(epoch $EPOCH)}"
  if [ "${PARALLEL:-1}" = "1" ]; then
    .venv/bin/python -u train_decoder.py --run "$R" "${EP_ARG[@]}" --steps "${STEPS:-3000}" --seed 0 \
      > "runs/${R}_decoder.log" 2>&1 &
    pids+=($!)
  else
    .venv/bin/python -u train_decoder.py --run "$R" "${EP_ARG[@]}" --steps "${STEPS:-3000}" --seed 0 \
      > "runs/${R}_decoder.log" 2>&1
  fi
done
fail=0
for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || fail=1; }; done
for R in $RUNS; do grep -v "$FILTER" "runs/${R}_decoder.log" | grep "val latent\|val mse" || true; done
[ "$fail" = 1 ] && { echo "[finish] a decoder fit FAILED (see runs/*_decoder.log)"; exit 1; }
.venv/bin/python -u compare_sigreg.py --runs $RUNS "${EP_ARG[@]}" --out "${OUT:-runs/sigreg_sweep}" 2>&1 \
  | grep -v "$FILTER" | tee "runs/compare${EPOCH:+_ep$EPOCH}.log"
