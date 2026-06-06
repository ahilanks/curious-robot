# Round campaign runbook — hardware → cloud adaptation

One **round** = collect real (Mac, frozen) → upload → fine-tune offline (RunPod) → redeploy.
Round 0 policy = `safe15 @ 100000`.

**Baseline (bench 2026-06-06, the 2×2 probe):** with the measured-τ metric, safe15 at
action_max 0.1 / P=16 scores **r_safe ≈ 0.0** — the old −50…−190 was entirely the
recompute artifact (same motions scored −144 by the old metric and 0 by real current;
|τ_meas| peaked 0.44 N·m). Pacing on vs off was indistinguishable at this config (soft
P self-smooths policy-scale deltas); it stays on as wear insurance.

**Success criterion, reframed:** r_safe can't improve from 0 — adaptation success is now
(a) reward/r_cur + interaction stats (contacts, object_motion once objects are in the
workspace) growing across rounds as the WM/policy de-OOD on real observations, with
(b) r_safe staying ≈ 0 as the now-truthful guardrail (it should only fire when the policy
starts genuinely fighting — stalls/collisions), and (c) q̇-reversal % trending toward
sim's ~34% in the buffer stats. Never compare r_safe against sim's −35 — different τ.

**Open δ item:** the deadband's stall case (deliberate hand-block ≫ δ) is still
unexercised — run it when physically at the arm (hold a link ~2 s mid-motion during any
collect; the dbg line should go clearly negative). Rest + slow-motion cases verified = 0.

## Pinned campaign constants (do not change mid-campaign)

| constant | value | provenance |
|---|---|---|
| `--action-max` | **0.1** | sim probe 2026-06-05: fully paceable (2173 ≤ cap 2400 ticks/s); 0.15+ can't pace its largest moves (≥3259, servo ceiling ~3400 no-load) → permanent plant-sim mismatch. The −52-vs−39 scale-OOD cost at 0.1 is a one-time warm-start cost the rounds absorb. |
| `--lambda-safe / --lambda-cur` | **0.1 / 20** | safe15 ckpt training args (verified from ckpt). The draft round command omitted `--lambda-cur` → would default 1.0 and silently shrink curiosity 20×. |
| `--safety-delta` | **15 → bench #4 value** | δ=15 was calibrated for the recompute τ scale; recheck with measured τ at the bench. |
| `--action-block / --history-size` | **5 / 3** | lineage arch — train.py rebuilds from CLI, not ckpt args; must be passed (or left at defaults, which match). |
| calib json | `goal_speed 2400, pace_dt 0.030, acceleration 0, kt → bench #3` | plant freeze; nothing in env/ or json changes until the campaign ends. |
| offline LRs | `wm 1e-5, actor 1e-4, critic 1e-4` | optimizers restart cold each round (not checkpointed); these are offline_train's defaults. |

## Per-round commands (round N; round 0 = safe15)

```bash
# 1. COLLECT (Mac, physical arm; ~600 transitions ≈ 2 min at ~5 sps)
export SOARM_PORT=/dev/cu.usbmodem5AA90245791 SOARM_CALIB=so101_calib.json
python src/train.py --env-backend hardware --frozen-policy \
  --resume-name round_$((N-1)) \                  # round 1: --resume-name safe15 --resume-step 100000
  --name hw_round_${N} --total-steps 600 --max-episode-steps 10000 \
  --action-max 0.1 --lambda-cur 20 --lambda-safe 0.1 --safety-delta <bench#4> \
  --save-buffer --no-wandb --no-hf
# Ctrl-C once = graceful stop + save. SOARM_DEBUG=25 to watch both r_safe variants live.

# 2. UPLOAD the buffer (adjust the count suffix to what was actually saved)
huggingface-cli upload a5ilank/curious-robot \
  runs/hw_round_${N}/buffer_0000600.npz buffers/hw_round_${N}/buffer.npz

# 3. FINE-TUNE (RunPod; bash setup.sh first)
python src/offline_train.py --resume-name round_$((N-1)) \   # round 1: safe15 --resume-step 100000
  --buffer buffers/hw_round_${N}/buffer.npz \
  --name round_${N} --steps 4000 --save-every 1000 --per-priority td
# --per-priority td matters offline: npz has no priorities -> uniform cold start; under the
# default 'curiosity' mode nothing ever updates them, 'td' self-adapts after the first pass.

# 4. REDEPLOY = next round's collect with --resume-name round_${N}
```

## Anti-forget sim mix (optional, from round 2+)

`buffers/sim_mix_v1/buffer_small.npz` (600 sim transitions, 1:1 with a round's real data) or
`buffer.npz` (3000, sim-dominant 5:1 — dilutes the real adaptation signal; use deliberately).
Collected 2026-06-05: frozen safe15, MuJoCo, **action_max 0.1 + λ 0.1/20** (matches campaign —
mixing buffers with different action_max or λs poisons the reward/dynamics semantics).

- **Round 1: real-only.** Add `--buffer buffers/sim_mix_v1/buffer_small.npz` from round 2 if
  the fine-tune looks unstable (critic_loss exploding, eff_rank collapsing) or the policy
  visibly degenerates.
- NEVER mix: `runs/dryrun_collect/buffer_0000400.npz` (mock-sim, positive rewards),
  `hw_validate`/`hw_dval`/`hw_safe15_match` (old recompute-metric rewards, pre-P0), or the
  `_sim_safe15_*`/`_probe_*` stats buffers (λ_cur=1).

## Guardrails

- **≥ a few hundred real transitions/round** — SAC valid pairs = stream_len−1; under
  `--batch-size` 128 it silently skips (offline_train warns loudly at the end).
- Rewards are stored at collection and replayed as-is offline — λs/δ are baked in at
  collect time, which is why they're pinned above.
- Judge each round by the **next** collection's r_safe (and the `[hw dbg]` means), not by
  offline losses — offline critic/pred losses measure fit, not behavior.
- Buffer-stats comparison script (τ-saturation / |q̇| / q̈ / reversals): see
  `logistics.md` 2026-06-03 entry or ask Claude — run it on each round's buffer; reversal %
  trending toward sim's ~34% is the cleanest behavioral smoothness signal.
- If a round's collect crashes mid-run, the graceful-SIGINT buffer is still valid (shorter);
  upload it with its real count suffix.
