# Bench checklist — P0 plant & metric changes (pacing + current-sourced r_safe)

One sitting with the physical arm. Goal: verify the two changes on hardware, calibrate
`kt`/`delta`, get the **2×2 attribution** (paced × metric) for the old −100…−190 r_safe,
and freeze the campaign config. The code commit before this session is a **checkpoint —
the freeze is the commit at the END of this checklist** (after kt/δ are pinned).

Env prefix used throughout (adjust port if it moved):

```bash
export SOARM_PORT=/dev/cu.usbmodem5AA90245791 SOARM_CALIB=so101_calib.json
```

## 0. Preflight
- [ ] Clear workspace, open start pose, e-stop hand-reachable. Power-cycle the bus once
      (Goal_Speed/Acceleration are SRAM; the new code rewrites them every construction/step,
      so a power-cycle is also a clean test of that self-healing).
- [ ] `python -m env.hardware_env` (mock self-test) green on the deploy checkout.
- [ ] `git log -1` — confirm you're on the P0 commit.

## 1. Pacing smoke — visual A/B (no policy, no metric dependence)
Energize + small scripted sweep, watch the arm:

```bash
python - <<'EOF'
import time, numpy as np
from env.hardware_env import _default_bus
bus = _default_bus()                      # paced by default
q, qd, tau = bus.read()
for k in range(40):                       # ±0.06 rad sine on the elbow (id 3), 30 ms cadence
    goal = q.copy(); goal[2] += 0.06*np.sin(k/6)
    bus.write_goal(goal); time.sleep(0.03)
print("paced speeds last step:", bus._last_speeds)
EOF
```
- [ ] Motion is a continuous glide (no tick-tick-tick stop-slam). Repeat with
      `SOARM_NO_PACE=1` — the old sawtooth should be audibly/visibly back. If there's no
      difference, pacing isn't reaching the servo (check `_last_speeds` prints ≥1 and <2000).
- [ ] **Lag check**: at the end of the paced sweep, read `bus.read()[0][2]` vs the last goal —
      error should be ≲0.02 rad. If it lags badly (≫), the P=16 servo can't track a moving
      ramp: consider P=24/32 (calib json `p_gain`) — decide NOW, it's part of the frozen plant.

## 2. Current sign — GO/NO-GO gate for the metric half
The current-based r_safe needs Present_Current's **sign** to track drive direction. Some
firmwares report magnitude only — verify before any kt work:

```bash
python - <<'EOF'
import time
from env.hardware_env import _default_bus
bus = _default_bus()       # torque ON, holding pose
print("push the shoulder-lift (id 2) DOWN slowly, then UP — watch column 2 flip sign")
for _ in range(200):
    q, qd, tau = bus.read()
    print(" ".join(f"{t:+5.2f}" for t in tau), end="\r"); time.sleep(0.05)
EOF
```
- [ ] Column 2 goes clearly negative one way, positive the other → **GO**.
- [ ] Never changes sign → firmware reports magnitude only → **STOP the metric half**
      (pacing still ships). Fallback to discuss: sign by `(goal − q)` direction — weaker
      fiction, decide deliberately, not at the bench.

## 3. kt calibration (only after #2 GO)
Placeholder is `kt=1.0` (τ = amps × 1 × signs). Two estimates; **when they disagree, trust
the δ-ordinal protocol** (gravity-hold current is stiction-biased — the old "~3× off via
gearbox friction" caveat applies to current too):

- [ ] **Gravity hold**: pose the upper arm horizontal, arm holding still. Read the id-2
      steady `tau` (≈ amps at kt=1). Gravity torque there is roughly 0.3–0.5 N·m (SO-101
      arm masses) → `kt ≈ τ_gravity / amps`. Record the number.
- [ ] **δ-ordinal protocol** (the acceptance test, with the `SOARM_DEBUG` probe in #5):
      - at rest: r_safe ≈ 0 (was already true)
      - slow smooth sweep (#1 script): r_safe ≈ 0 — fights stay under δ
      - deliberate stall (hold a link by hand for a beat while it's mid-move): r_safe fires
        visibly (≪ 0)
      Adjust `kt` (and δ if needed) until those three hold, in that order of trust.
- [ ] Write `"kt": <value>` into `so101_calib.json`.

## 4. δ sanity
δ=15 was calibrated for the recompute-τ scale. With measured τ (typically ≪ 3.35 N·m), the
fight term shrinks ~linearly with kt — δ may now be too high (penalty never fires) or fine.
- [ ] Re-run the three δ-ordinal cases above at the chosen kt. If "deliberate stall" doesn't
      fire, lower `--safety-delta` (collect-command CLI, not code) until it does while slow
      sweep stays ≈ 0. Record the value for the round runbook.

## 5. Decomposition probe — the 2×2 the whole investigation pointed at
Frozen safe15, 10 decisions, BOTH r_safe variants printed every control step:

```bash
SOARM_DEBUG=5 python src/train.py --env-backend hardware --frozen-policy \
  --resume-name safe15 --resume-step 100000 --action-max 0.1 --total-steps 10 \
  --save-buffer --probe-size 0 --video-every 0 --no-hf --no-wandb --name hw_p0_paced
SOARM_DEBUG=5 SOARM_NO_PACE=1 python src/train.py ... --name hw_p0_nopace   # same flags
```
- [ ] Fill in (means over the run, from the `[hw dbg]` lines):

|                       | r_safe recompute (old metric) | r_safe measured (new metric) |
|-----------------------|------------------------------|------------------------------|
| **unpaced** (legacy)  | ≈ −100 expected (the old number) |                          |
| **paced**             |                              | ← the new campaign baseline  |

- [ ] Run the buffer-stats script on both dumps (τ-saturation %, |q̇|, reversals) — Claude
      has it from the 2026-06-03 analysis; compare against sim's 87% / 1.5 rad/s / 34%.
- [ ] Visual: paced run looks like #1's glide even under the policy.

## 6. FREEZE — end of bench
- [ ] `so101_calib.json` final: offsets/signs/vel_scale (untouched), p_gain/d_gain (from #1),
      goal_speed cap, `pace_dt` (0.030 or absent = default), `acceleration` (0 unless #1
      said otherwise), `kt` (#3).
- [ ] Round runbook constants: `--safety-delta` (#4), `--lambda-safe 0.1 --lambda-cur 20`
      (training values — decide consciously if deviating), campaign `--action-max` (P1 sim
      probe), `--action-block 5 --history-size 3`.
- [ ] Commit json + this file's filled numbers (logistics.md) — **this commit is the freeze**.
      Nothing in env/, calib, or reward λs changes until the round campaign ends.
