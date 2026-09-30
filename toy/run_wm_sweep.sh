#!/usr/bin/env bash
# Does the online learner generalise? One job at a time (A: online, B: the same data offline, C: online with
# replay-only batches, D-F: fixes for the policy's narrowing, G: the latent encoding); each learner is scored by score_learner.py: the fraction of the latent change it misses on the unseen
# DEMO windows where the block moves / stays still, next to its own training data.
#   bash toy/run_wm_sweep.sh        (log: toy/runs/pusht/wm_sweep.log)
#   python toy/score_learner.py --table toy/runs/pusht/offline/* toy/runs/pusht/ptf_sp*
set -uo pipefail
cd "$(dirname "$0")"
P=runs/pusht
O=$P/offline
mkdir -p $O/logs $P/logs
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
headline() { grep -o 'T moves/T still: h1 [0-9.]*/[0-9.]*' "$1" | tail -n 1; }
off() {                                                          # off NAME ARGS...: offline learner, then its score
  local name=$1
  shift
  if [ ! -f "$O/$name/eval_learner.json" ]; then
    { [ -f "$O/$name/final.pt" ] || python offline_learner.py --name "$name" "$@"; } > "$O/logs/$name.log" 2>&1 \
      && python score_learner.py "$O/$name" >> "$O/logs/$name.log" 2>&1 || { say "FAIL $name"; return; }
  fi
  say "done $name  $(headline $O/logs/$name.log)"
}
random_start_data() {                                            # random walk from gym-pusht's random reset
  local s=$1
  [ -f "$P/ptr_none_s$s/replay.pt" ] || python run_curiosity.py --env pusht --start random --signal none --seed "$s" \
    --save-replay 32 --out $P > "$P/logs/ptr_none_s$s.log" 2>&1 || { say "FAIL ptr_none_s$s"; return; }
  say "done ptr_none_s$s"
}
off() {                                                          # off NAME ARGS...: offline learner, then its score
  local name=$1
  shift
  if [ ! -f "$O/$name/eval_learner.json" ]; then
    { [ -f "$O/$name/final.pt" ] || python offline_learner.py --name "$name" "$@"; } > "$O/logs/$name.log" 2>&1 \
      && python score_learner.py "$O/$name" >> "$O/logs/$name.log" 2>&1 || { say "FAIL $name"; return; }
  fi
  say "done $name  $(headline $O/logs/$name.log)"
}
online() {                                                       # online NAME ARGS...: run_selfplay, then its score
  local name=$1
  shift
  if [ ! -f "$P/$name/eval_learner.json" ]; then
    { [ -f "$P/$name/final.pt" ] || python run_selfplay.py "$@"; } > "$P/logs/$name.log" 2>&1 \
      && python score_learner.py "$P/$name" >> "$P/logs/$name.log" 2>&1 || { say "FAIL $name"; return; }
  fi
  say "done $name  $(headline $P/logs/$name.log)"
}

while pgrep -f "[r]un_selfplay.py|[o]ffline_learner.py|[s]core_learner.py" > /dev/null; do sleep 10; done
# A) online, 20 learner steps per round (half of each batch from the current round): self-play vs the frozen policy
for s in 0 1 2; do
  online ptf_spepk20_s$s --wm-steps 20 --seed $s
  online ptf_spunik20_s$s --wm-steps 20 --frozen-policy --seed $s
done
# B) the same data, learned offline (uniformly, 20k steps): is it the data or the online training?
for s in 0 1 2; do
  off offk20_spep_s$s --data $P/ptf_spepk20_s$s
  off offk20_spuni_s$s --data $P/ptf_spunik20_s$s
done
# C) online with replay-only batches (every step samples uniformly from everything seen so far)
for s in 0 1 2; do online ptf_spepk20p0_s$s --wm-steps 20 --wm-pool-frac 0 --seed $s; done
# G) is it the latent encoding? The same data with one shared frozen encoder (another seed's frozen-policy learner):
#    only the predictor learns. Then SIGReg's weight, offline, on self-play vs frozen-policy data.
for s in 0 1 2; do
  e=$(( (s + 1) % 3 ))
  off swap_spep_s$s --data $P/ptf_spepk20_s$s --encoder-from $P/ptf_spunik20_s$e
  off swap_spuni_s$s --data $P/ptf_spunik20_s$s --encoder-from $P/ptf_spunik20_s$e
done
for d in spep spuni; do
  off offk20lam001_${d}_s0 --data $P/ptf_${d}k20_s0 --sigreg 0.01
  off offk20lam05_${d}_s0 --data $P/ptf_${d}k20_s0 --sigreg 0.5
done
# D-F) fixes for the self-play policy's narrowing, each alone on top of A's self-play arm, a seed of each at a time:
#   D the paper's pool (MAP-Elites mutation + replay bank + expert iteration), E a 10x slower generator, F 5x beta
for s in 0 1 2; do
  online ptf_spepk20pool_s$s --wm-steps 20 --mut-envs 16 --replay-progs 16 --ei 1 --seed $s
  online ptf_spepk20lr4_s$s --wm-steps 20 --lr 1e-4 --seed $s --name ptf_spepk20lr4_s$s
  online ptf_spepk20b05_s$s --wm-steps 20 --beta 0.05 --seed $s --name ptf_spepk20b05_s$s
done
# C's control: the frozen policy with replay-only batches
for s in 0 1 2; do online ptf_spunik20p0_s$s --wm-steps 20 --wm-pool-frac 0 --frozen-policy --seed $s; done
say "all done"
