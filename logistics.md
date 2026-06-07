# Curious Robot — Logistics

JEPA + SIGReg world model co-trained with SAC under an intrinsic curiosity reward, on a
MuJoCo SO-ARM101 6-DOF arm in an object soup. **Built from scratch to the `README.md`
spec** (README = the formulation; this file = the working log). The question every run
answers: *does the arm interact with the blocks and get curious like a baby, and is it
learning dynamics?*

- **Code** — `lewm/` (verbatim JEPA+SIGReg), `env/` (MuJoCo SO101 + safety reward),
  `model/` (ViT-tiny state encoder), `src/` (train · play_policy · eval_predictor).
- **Backends** — W&B project `curious-robot` · HF `a5ilank/curious-robot` · GitHub
  `ahilanks/curious-robot` (private). Creds in `.env` (auto-loaded; gitignored — keep local).
- **W&B — ALWAYS read via the Python API** (`wandb.Api()`), never scrape the web UI. Load
  `.env`, then `Api().run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/<run_id>")`. Entity =
  `ahilan-uc-berkeley-electrical-engineering-computer-sciences` (a *personal* entity, **not**
  `models`); run IDs are in the Sweeps table (e.g. **safe15 = `4zn95btc`**). Use
  `run.history(keys=[...], samples=N)` or `run.scan_history()` for the full per-step time-series
  (the `runs/<name>/metrics.jsonl` files are *hardware* runs only — sim training lives in W&B).
- **Train** — `python src/train.py --name <kw> --n-envs 8 --env-threads 8`. `--name` is
  a short keyword → W&B run name + `runs/<name>/` + HF `<name>/ckpt_*.pt`; every constant
  lives in the W&B config table, not the name. Pod bootstrap: `bash setup.sh`.
- **Inspect** — `play_policy.py --name <kw>` (overhead+wrist rollout videos) and
  `eval_predictor.py --name <kw>` (open-loop pred vs persistence). Both fetch the latest
  checkpoint from HF by default; pass `--ckpt <path>` for a local file or `--step N`.

## Status — last left off (2026-05-27)

Implementation **complete and validated end-to-end** (smoke run, HF up/download
round-trip, deterministic contact check), pushed to GitHub. **No real training run yet** —
the remaining README `?` constants (λ_cur, δ, Kp/Kd) are unswept and stay `?` until then
(β is now pinned at **0.3** — no longer swept; see the 2026-05-28 Log entry).

Recently landed: dual-cam rollout videos every 500 steps; contact-bucketed curiosity MSE
(`wm/mse_block|table|none`); encoder-collapse metrics (`z_std`/`eff_rank`/`feat_corr`);
HF upload-then-clear-disk every 1k steps; `--name` run naming; HF checkpoint fetch in
`play_policy` + `eval_predictor`. **Fixed a latent bug** where unnamed arm geoms left the
arm-contact set empty → all interaction metrics were silently 0; now verified firing.

## To-do

- [ ] **Provide `.env` on new machines** — it's gitignored (kept local), so copy it to
      pods out-of-band (scp / paste), not via the repo. Rotate a token only if it leaks.
- [ ] **Baseline run** to ~200k steps (`--keep-local-ckpts` if you want local eval); fill
      the sweep table.
- [ ] **Sweep the remaining `?` constants** — λ_cur (15 → safety:curiosity ~0.5:1),
      δ (0.05). Watch `reward/safe_cur_ratio`, `interact/contacts_per_step`, pred/persist.
      (β is pinned at 0.3, no longer swept.)
- [ ] **TD-priority ablation** — `--per-priority td` vs `curiosity` (see Ablations).
- [x] **Hardware: adapt WM/policy on real data** — ~~smoothness fix~~ **premise overturned 2026-06-06:
      the "jerk" was ~entirely a measurement artifact** (recompute-τ billed servo arrivals as max-torque
      fights; measured-τ r_safe = 0.0 on the same motion — see the 06-06 log). Adaptation infrastructure
      is BUILT anyway (offline_train + the 24/7 daemons) and now targets the real gap: obs-OOD
      (value-gap 80 vs 136) + interaction, with r_safe as a truthful guardrail.
- [ ] **START the 24/7 loop** — arm: clamp base, route cables out of sweep, objects in workspace,
      screw pass; Mac: `caffeinate -i` + tmux + port check; pod: 1×A100, branch
      `safety/deadband-and-lambda-safe`, `.env`, `bash setup.sh`, tmux, network volume (or
      `--keep-hub-chunks`). First cycle attended. Commands: §Hardware campaign reference below.
- [ ] **Watch across rounds/days** — value-gap `V/(r̄/(1−γ))` → 1 (was 80/136 = 0.59 on real data),
      `reward/r_cur` + interaction stats up, q̇-reversal% → sim's ~34%, r_safe stays ≈ 0,
      probation `rejects` ≈ 0.
- [ ] **Explore-toward-objects is still THE open learning problem** (unchanged from sim — curiosity
      gradient doesn't point at blocks; now it gets real-world data to disprove itself on).
- [x] Pin the swept `?` values into `README.md` — **done 2026-05-31 (maintainer-approved):**
      δ=15, λ_safe=0.1, τ_max=3.35, h_fwd_max=1, r_cur→per-dim-mean. (λ_cur still `?` — at 20, unswept.)

## What to read off the logs (W&B `curious-robot`)

- **Interacts with blocks** — `interact/contacts_per_step`, `interact/object_motion`,
  `interact/frac_touch_block` rising over training.
- **Curious / exploring** — `reward/r_cur` non-trivial; rollout videos show varied reaching.
- **Learning dynamics** — `wm/pred_loss` falling *below* `wm/identity_baseline`; `wm/h_fwd`
  advancing; offline `eval_predictor.py` pred/persist < 1. Bonus signal: `wm/mse_block`
  exceeding `wm/mse_none` means block contacts are the harder-to-predict ("interesting") events.
- **Healthy representation** — `encoder/z_std` not → 0; `encoder/eff_rank` not → 1;
  `encoder/feat_corr` not → 1.

## Sweeps

The remaining README `?` constants (λ_cur, δ) are swept here before being pinned; β is fixed at 0.3.

| run | β | λ_cur | δ | steps | contacts/step | pred/persist | notes |
|-----|---|-------|---|-------|---------------|--------------|-------|
| newarch | 0.3 | 15.0 | 0.05 | 10000 | 0.00 | 0.50 | WM learns (pred/persist 0.50, h_fwd→11, eff_rank 4.4→7.8) but policy never contacts blocks; curiosity harvested from non-contact motion |
| lcur20 | 0.3 | 20.0 | 0.05 | 55250\* | 0.00 | 0.14 | per-dim-MEAN `r_cur` (`5824b7a`) → r_cur≈0.75 (was ≈188); **λ_safe=0** (safety ablated), `h_fwd_max=1`. WM learns *very* well (pred/persist 0.14, eff_rank 30, probe 8.1) but policy still never contacts blocks (contacts/step≈0.002). \*crashed 55250/100k (pod, not code). W&B `199jzlil` |
| safe15 | 0.3 | 20.0 | **15** | 100000 | 0.00 | 0.13 | **first run with corrected safety: δ=15 + λ_safe=0.1.** Safety balanced (actual safe:cur **0.28–0.32:1**, reward/total **+8**, no freeze); r_safe falls **−51.5→−30** (policy smooths). WM excellent (pred/persist 0.13, eff_rank 33.6). Block interaction still collapses post-warmup (contacts 0.34→**0.001**) — explore-toward-objects unsolved, orthogonal to safety. W&B `4zn95btc` |

## Ablations

| ablation | flag | default | hypothesis / what to compare |
|----------|------|---------|------------------------------|
| PER replay priority | `--per-priority {curiosity,td}` | `curiosity` | `td` = \|TD-error\| is sign-agnostic, so it also replays the unsafe (very-negative `r_safe`) transitions the critic mispredicts — which curiosity priority under-samples once the WM has learned them. **Q:** does `td` better suppress motor-fighting states without losing block interaction? Compare `interact/contacts_per_step`, `reward/safe_cur_ratio`, `sac/critic_loss`, `eval_predictor.py`. Curiosity stays the *reward* either way. |

## Log

_(add a dated entry per run)_
- 2026-05-27 — implementation complete + validated end-to-end; no training run yet.
- 2026-05-28 — **aligned the WM with the LeWM reference** after diagnosing universal policy
  freezing across the 26 runs: curiosity reward sits at a ~constant spatially-uniform floor
  (`r_cur`≈220, CV 3–10%; `mse_block` barely > `mse_none`), the WM never learns (`pred_loss`
  flat), so the safety penalty is the only action-discriminating signal → SAC freezes
  (`contacts/step`→0, `r_safe` −55→−3). Reward-scaling/exploration knobs (RND norm, raw
  curiosity, BatchNorm, learnable-α, δ/safety sweeps) were all tried and all still froze.
  Changes: **β SIGReg pinned 0.3** (was 0.9 / swept 1–5, which collapsed the latent —
  eff_rank fell as β rose); **WM loss → plain MSE** `(ẑ−z).pow(2).mean()` (dropped
  symlog-on-summed-sq, which compressed the metric + anti-curriculum-reweighted surprising
  samples); **removed the encoder's final LayerNorm** (it pinned z to a sphere, fighting
  SIGReg). Horizon/`h_fwd` curriculum + online co-training kept. Reward's
  `λ_cur·symlog(r_cur)` unchanged. Open lever: WM is still co-trained online on a freezing
  policy's degenerate data (LeWM is offline) — likely need to pretrain the WM on a random
  buffer until `pred_loss` < persistence before safety dominates.
- 2026-05-29 — **first real run on the LeWM-aligned arch** (`newarch`, 10k steps; β=0.3,
  λ_cur=15, δ=0.05; W&B run `oc6p0un6`). **The WM now learns** — the 2026-05-28 changes worked:
  `pred_loss`/`identity_baseline` ≈ **0.50** (beats persistence), the `h_fwd` curriculum climbed
  **1→11**, and the latent stayed healthy (`z_std` 0.25→0.84, `eff_rank` 4.4→7.8, `feat_corr`≈0.28
  — no collapse). **But the policy never interacts**: `contacts/step`=0 the entire run,
  `object_motion` 0.067→0.0014, `r_safe` −42→−17 (calmer), `safe:cur` 0.67→0.22 (curiosity
  dominates ~4.5:1), `r_cur` 64→188. Curiosity is harvested from non-contact free-space motion
  (`mse_none` 64→189 ≫ `mse_block`≈67 — though the block bucket is starved since contacts≈0, so
  that comparison is chicken-and-egg). **New failure mode**, distinct from the 26 frozen runs
  (where the WM never learned): here the WM learns fine, but the curiosity gradient points *away*
  from blocks → exploration-toward-objects is now the open problem (reward shaping / λ_cur, not WM
  arch). The modest `eff_rank`≈8 likely reflects the low-dim experience (arm-only, blocks static)
  more than encoder health — a fixed diverse probe-set eff_rank would disentangle the two.
- 2026-05-29 (later runs; logged 2026-05-31) — **after `newarch`, two arch changes landed,
  then the last batch of runs went out.** (1) Curiosity `r_cur` switched to the **per-dim MEAN**
  squared pred error (`5824b7a`, was the d_z-summed version) → `r_cur` fell ≈**188 → ≈0.75**, so
  `symlog(r_cur)` now sits in its sensitive region; (2) **safety ablated by default** (`λ_safe=0`,
  `79520f3`, "verified-working config") to isolate pure curiosity, and `h_fwd_max` pinned to 1
  (curriculum off, perf PR). Intervening runs: `nosafe`, `meancur`, `autoalpha` (all λ_safe=0).
  **Latest run `lcur20`** (W&B `199jzlil`; λ_cur=20, λ_safe=0, β=0.3, δ=0.05; **crashed at
  55250/100000** — pod/session death, not a code fault; an earlier 10k `lcur20` `vskepn3w` also
  died at 9150). **Result:** the WM now learns *very* well — `pred_loss/identity` ≈ **0.14**
  (newarch was 0.50), `eff_rank`≈30 (probe 8.1), `z_std`≈0.97, `feat_corr`≈0.14 (no collapse).
  **But the policy still never interacts** — `contacts/step`≈**0.002**, `object_motion`≈0.002,
  `frac_touch_block`≈0.0016 — the same non-interaction failure as `newarch`, now with safety
  entirely off, confirming it's a curiosity-doesn't-point-at-objects problem, not a safety-freeze.
  **Implication for re-enabling safety:** with per-dim-mean `r_cur`, raw `safe_cur_ratio`≈**5**
  (raw |r_safe|≈54 ≫ cur_contrib≈11), so the README weight `λ_safe=1` would now over-weight safety
  ~5:1 (freeze risk); the correctly-weighted value for the documented ~0.5:1 balance is
  **λ_safe≈0.1** (≈0.05 to match newarch's actual 0.22:1).
- 2026-05-31 — **safety-weight bracket** (3× ~10-min test runs, 700 steps, on the `lcur20` config
  = λ_cur=20, β=0.3, δ=0.05, h_fwd_max=1; only λ_safe varied; W&B `curious-robot`). Goal: re-enable
  the safety reward at a *correct* weight after the per-dim-mean rescale (see entry above). Results
  (final-step actual safety:curiosity = λ_safe × raw `safe_cur_ratio`):
  - `safe_p05` (λ_safe=0.05, `wtzyzu8g`) → **0.61:1**, reward/total +1.9
  - `safe_p10` (λ_safe=0.10, `5cwdmp8k`) → **1.37:1**, reward/total −1.7
  - `safe_1p0` (λ_safe=1.00, `d5f76dqm`) → **13.6:1**, reward/total **−55.6** (reward is essentially
    the pure safety penalty — curiosity invisible; this is the freeze-inducing regime).

  All launched/ran clean (no crash); WM already beats persistence by 700 steps (`p/p`≈0.57–0.74).
  **Note on horizon:** raw `safe_cur_ratio`≈**13** at 700 steps vs ≈**5** at lcur20's 55k steps —
  `r_cur` (hence `cur_contrib`) ramps as the WM learns, so the actual ratio *falls* over a run.
  ⇒ **λ_safe=0.1 is the correct weight for the long run** (converges to the documented ~0.5:1 at
  steady-state magnitudes); λ_safe=0.05 only looks on-target early and settles to ~0.25:1. **λ_safe=1.0
  is confirmed mis-scaled** post-per-dim-mean. 700 steps is too short to judge policy freezing
  (newarch/lcur20 needed ≥10k) — the bracket validates the *weighting magnitude*, not long-run
  dynamics. **Next:** a full run at λ_safe=0.1 (rest = lcur20 config) is the recommended baseline.
- 2026-05-31 — **deadband δ pinned via physics analysis.** A multi-lens study of the safety
  deadband on `-τ·q̈` (units N·m·rad/s²; the per-joint quantity δ gates): empirical real-contact
  measurement (δ=10–15), analytical torque-limit derivation (δ=28), STS3215-datasheet/gear-shock
  lens (δ=25), and an adversarial false-positive/false-negative verifier (**δ=12, range [10,15]**).
  Verifier key findings: **0% false-negatives on genuine shock events** (term>30 ∧ |τ|>0.6·τ_max)
  at *every* δ∈[5,28] — so damage detection does **not** bound δ from above; the binding constraint
  is false-positives on **calm free motion** (knee at δ≈8–10: 15.5%@8 → 4.7%@10 → 3.7%@15). Light
  pickup/push are penalty-free for any δ≥10. **Shipped δ=15** (conservative top of [10,15] → lowest
  benign FP, best serving "no penalty for gentle pickup/push", ~15% less damage-gradient resolution
  than δ=10–12 — minor). The old **δ=0.05 penalized essentially all motion** (calm r_safe −2.8 → freeze).
  README + code reconciled (maintainer-approved): δ?→15, add λ_safe=0.1, `r_cur` → per-dim mean
  `(1/d_z)‖ẑ−z‖²`, τ_max 2.94–3.35 → 3.35 (all joints), `H_fwd,max` 20 → 1 (curriculum off by default).
  Branch `safety/deadband-and-lambda-safe`. Still stale, left as-is: README `Δt_safe`=10 timesteps vs
  code 6 (`frame_skip·timestep`=0.030 s); δ=15 is calibrated to the actual 0.030 s.
- 2026-05-31 — **`safe15`: first run with the corrected safety reward** (100k steps, ~10.4 h, 8 envs,
  W&B `4zn95btc`; δ=15, λ_safe=0.1, λ_cur=20, β=0.3, h_fwd_max=1 — the lcur20 config + re-enabled
  safety at the right weight). **Result — safety weighting validated:** actual safety:curiosity
  settles to **~0.28–0.32:1** from 25k on (warmup 1.08:1), reward/total stays **positive** (+5.7→+8.3)
  — **no freeze** (cf. λ_safe=1 → −55 in the bracket). **r_safe falls monotonically −51.5→−30** over
  training = the policy learns smoother, less motor-fighting motion (the safety term's intended effect).
  **WM stays excellent** — pred/persist 4.08(warmup)→**0.13**, eff_rank 3.4→**33.6**, z_std 0.98,
  feat_corr 0.13 (no collapse); the safety reward did **not** hurt WM learning. **But block interaction
  still collapses post-warmup** — contacts/step 0.335 (random warmup) → **~0.001–0.008** (learned policy),
  object_motion ≈0.0006 — the same non-interaction failure as `newarch`/`lcur20`: random actions hit
  blocks, the curiosity-driven policy doesn't. **So the corrected safety reward is balanced /
  non-freezing / WM-safe, but does not by itself solve explore-toward-objects** — that remains THE open
  problem (curiosity gradient points away from blocks; reward-shaping / λ_cur / intrinsic-exploration
  territory, orthogonal to the safety weighting). Final ckpt on HF (`safe15/ckpt_0100000.pt`).
- 2026-06-01/02 — **first hardware deployment: `safe15` onto the physical SO-ARM101** (on the M4 Mac,
  branch `safety/deadband-and-lambda-safe`). Stood up the whole real-arm path and ran safe15 frozen on
  it. Headline: a clean **sim→real gap on motion smoothness**.
  - **Bring-up (new, committed):** `FeetechBus` in `env/hardware_env.py` via Feetech `scservo_sdk`
    (replaced the LeRobot stub — LeRobot wasn't installed and its import path was stale). STS3215 over
    the TTL bus `/dev/cu.usbmodem5AA90245791` (model 777, 1 Mbps, protocol_end 0); wrist cam = OpenCV
    **index 0** (generic UVC). Safe energize: set `Goal_Position`=current **before** `Torque_Enable`
    (boot goal=0 would slam every joint to tick 0); `read()` keeps last-good on a dropped serial read
    (never 0 → no phantom slam); per-command jump clamp. Added **`--frozen-policy`** collection mode to
    `src/train.py` (no grad updates; dumps the replay buffer → `out_dir/buffer_<N>.npz` for offline /
    RunPod training; `--save-buffer`; graceful Ctrl-C save; refuses on hw without a loaded policy).
  - **Calibration → `so101_calib.json`** (checkpoint-independent — the collection still loads safe15):
    `offsets_ticks [2013,1981,2246,1941,1991,2010]`, `signs [1,1,1,1,1,1]` (all +, verified by sim
    render-match **and** under-torque pokes), `vel_scale 0.00153` (empirical; STS3215 Present_Speed is
    sign-magnitude, bit15). Method: hand-pose to the rendered zero for offsets, per-joint render-matched
    pushes for signs. Reference renders in `calib_refs/`.
  - **Matched the safe15 training config** (read from the ckpt `args`, == W&B `4zn95btc`): action_max
    **0.3**, λ_cur **20**, λ_safe **0.1**, δ **15**, action_block 5, history_size 3; `total_steps=100000`
    so **step 100000 is the final/last**. (First mis-deployed at action_max 0.05 + λ_cur 1.)
  - **KEY FINDING — safe15 is JERKY on the real arm; it's the sim→real gap, not the rig.** Real frozen
    runs: r_safe ≈ **−100** at action_max 0.05, ≈ **−190** at 0.3 (vs sim's −30→−51), visually **"very
    jerky"** (user-confirmed). NOT a sensor artifact (q̇=q̈=0 dead-clean at rest). NOT fixable by servo
    PD (a P/D sweep cut a *synthetic* stressor's r_safe 3× but did nothing for real safe15; P=64/D=96 was
    "very jerky" → reverted to shipped-stable **P=16/D=32**, now the default) NOR by action_max (bigger =
    worse). Root cause: safe15 learned smooth, motor-compliant motion *for sim proprio + rendered images*;
    the real arm's proprio/images are mildly OOD so it doesn't reproduce that smoothness (the very effect
    that drives r_safe −51.5→−30 *in sim training* just doesn't transfer) → motor-fighting → large `−τ·q̈`.
    With λ_cur 20 / λ_safe 0.1, curiosity (~+11) only wins when r_safe is small (reward ≈ +2 at 0.05); at
    0.3 the −190 penalty dominates (reward ≈ −8).
  - **Misc:** M4 frozen-collection cadence ~**5 sps** (the M4 is the deploy machine — USB bus + cam live
    on it; RunPod stays for the 8-env sim + offline training). Post-run pan "shaking" = creep toward a
    stale `Goal_Position` (goal=current parks it), not a gain issue. Verified `--frozen-policy` end-to-end
    on mock + real (buffer npz saves; safe:cur balance correct under λ_cur 20).
  - **Conclusion / next:** bring-up + calibration done and validated; the frozen safe15 run collects
    valid (jerky-but-bounded, **safe**) real data. **The smoothness fix is policy/WM ADAPTATION on real
    data (offline), not config tuning** — safe15 only moves smoothly on hardware after it has seen real
    transitions. Gentler-collection levers if needed: soft gains (now default), lower action_max,
    low-pass the action stream. This is the **same class of problem** as the sim explore-toward-objects
    gap — the policy is OOD on real inputs. All on branch `safety/deadband-and-lambda-safe`.
    *(2026-06-06 postscript: the −100/−190 numbers and the "OOD-jerky" framing were ~entirely a
    measurement artifact — see that entry. The bring-up itself stands.)*
- 2026-06-03 — **offline fine-tuning pipeline + the jerk gap decomposed** (`cdda294`).
  - **`src/offline_train.py`** (+ `sac_update` factored out of train.py, behavior-preserving): the
    round loop collect (Mac, frozen) → HF npz → fine-tune (RunPod, env-free WM+SAC on reloaded
    buffers) → redeploy. `load_buffer` = exact inverse of `save_buffer` (each `env_lengths` entry =
    own env row; real+sim mix in one buffer). Verified: bit-equal round-trip; WM+SAC fire on real
    fixtures; ckpt keeps train.py's exact 8 keys so redeploy loads; warm-start carries
    history_size/h_fwd from the ckpt. Gotchas pinned: SAC silently skips when stream pairs <
    batch-size (collect ≥ a few hundred/round); offline prios cold-start uniform → use
    `--per-priority td`; redeploy passes lineage `--action-block 5 --history-size 3` (train.py
    rebuilds from CLI, not ckpt args).
  - **Jerk-gap decomposition** (frozen-safe15 MuJoCo buffers vs real buffers, same stats): sim
    reproduces −35 @ action_max 0.3 and −63 @ 0.05 (the policy is **action-scale-OOD even in sim**;
    the W&B −51.5→−32 plateau is config-specific). At matched 0.3: real r_safe −190 vs sim −35
    (×5.4) **yet the real arm is LESS dynamic at every 150 ms-resolvable scale** (|q̇| 0.83 vs 1.50,
    τ-saturation 90% vs 87%) → the excess penalty lives in the 30 ms window: (a) sim α-ramps targets
    over 6×5 ms substeps, hardware sent one step-goal/30 ms → race-and-stop sawtooth; (b) **metric
    semantics flip**: sim scores `actuator_force` (the torque CAUSING q̈ — its own braking scores
    compliant); hardware *recomputed* τ=kp(goal−q)−kv·q̇ with kp=998, which saturates at **0.19°**
    of tracking error → a P=16 servo pegs |τ|=τ_max whenever moving and every arrival/stall scores
    a max-weight "fight"; (c) quantized Present_Speed inflates q̈ during fast motion.
- 2026-06-05 — **P0 plant+metric fix shipped + campaign prep** (`4878e8e`, `8ad0bff`).
  - **Goal_Speed pacing**: `write_goal` writes per-servo speed = |Δticks|/0.030 s (clamp [1, cap];
    0 = MAX on Feetech) → firmware executes each move as a constant-velocity ramp spanning the
    window — sim's target ramp by construction. `SOARM_NO_PACE=1` = legacy A/B.
  - **Obs/reward torque SPLIT**: r_safe scores **measured** τ (Present_Current-based, EMA matched
    to vel_lowpass, clipped ±τ_max); **proprio/`applied_torque` KEEP the kp-law recompute** (the
    obs distribution the sim-trained encoder expects — sim saturates 87%, recompute 90%). Both
    r_safe variants logged per control step (`SOARM_DEBUG=N`) for attribution.
  - **Campaign prep**: action_max probe (sim r_safe −52.3@0.1 / −39.1@0.15 / −29.7@0.2; reversals
    flat ~34–39% at all scales) → **0.1 pinned**: the largest scale where the FULL action range
    paces in-window (2173 ≤ cap 2400 ticks/s); 0.15+ exceeds the servo ceiling on its biggest
    moves = permanent plant-sim mismatch, vs 0.1's one-time warm-start OOD cost that rounds absorb.
    λ 0.1/20 + δ 15 verified from the safe15 ckpt args. `sim_mix_v1` collected at campaign config →
    HF `buffers/sim_mix_v1/{buffer,buffer_small}.npz` (3000/600 tr; the ONLY mix-safe sim buffers —
    the older `_sim_safe15_*` stats buffers are λ_cur=1, never mix). HF buffer transport round-trip
    sha256-verified.
- 2026-06-06 — **bench session: THE JERK MYSTERY IS CLOSED; campaign frozen** (`8cf390e`→`4145f7a`).
  - **This firmware's Present_Current is MAGNITUDE-ONLY** (pan test: identical +9 mA both drive
    directions, 0/138 negative). **Sign lives in Present_Load reg 60 bit 10** (PWM direction;
    flips 100%/0.8% with direction vs the known-good speed sign) → τ_meas = kt·|I|·sign(load₁₀)·signs,
    verified flipping (+0.010/−0.013) through the production read path.
  - **kt = 10.0 N·m/A** — moving-raise estimate vs exact MuJoCo `qfrc_bias` at recorded poses
    (9.9 raise / 12.4 hold). **Gravity-HOLD calibration is stiction-poisoned** (the elbow holds
    0.22 N·m on 2 mA) — never calibrate torque from hold current. τ_meas clip at 3.35 ≈ the
    servo's physical ~3 N·m ceiling (~0.33 A) — self-consistent.
  - **THE 2×2 VERDICT** (frozen safe15 @ 0.1, P=16, paced AND unpaced): **measured r_safe = 0.0 at
    every control step while the recompute scored the same motions −47/−57 mean (spikes −144)**.
    Peak real effort 0.44 N·m. ⇒ the old −50…−190 was ~entirely the recompute artifact; the arm was
    never fighting its motors at this config. **Pacing is inert at P=16 + action_max 0.1** (weak P
    self-smooths both 6-tick and 65-tick deltas; lag p95 41 mrad benign) — kept as wear insurance.
  - **δ=15 fully verified** — rest + slow sweeps = exactly 0 (exact-EMA replay of bench recordings);
    the stall case fired ORGANICALLY in a 200-decision run (arm pinned itself at joint stops:
    τ_meas pegged 3.35 @ |q̇| 0.09 → measured r_safe −0.003 fired **while the recompute scored that
    exact event −0.0** — the old metric misses real fights AND invents fake ones; anti-correlated).
  - **Reframed success criteria**: r_safe is a truthful guardrail sitting at ≈0 (fires only on real
    stalls/collisions); adaptation success = `reward/r_cur` + interaction stats growing, q̇-reversal%
    → sim's ~34%, and the **value-gap metric** `V_measured/(r̄/(1−γ))` → 1 (safe15's critic reads
    ~80 on real states whose rewards justify ~136 — the critic's to-close gap; re-graph per round).
  - **Behavioral baseline** (100-decision frozen runs): real reversals **15.8%** vs sim 34.9% — the
    P=16+paced plant low-passes the policy's dither; and **|q̇| 0.136 vs sim 1.11 rad/s** — the arm
    moves ~8× slower than sim for the same policy. *The real plant gap is attenuation, not jerk.*
- 2026-06-06 (later) — **24/7 autonomous loop built, integration-tested, drilled on the arm**
  (`0e96e53`→`2cbf0e0`). `src/collect_daemon.py` (Mac) + `src/learner_daemon.py` (always-on 1×A100
  pod) over the HF Hub as mailbox — **no direct connection between machines** (both only make
  outbound HTTPS; chunks Mac→`buffers/auto1/`, ckpts pod→`auto1/`; either side can die/restart
  independently, the hub holds all durable state).
  - **Safety for unattended** (the human's jobs, replaced): **acceptance probation** — every new
    ckpt drives ~30 watched decisions before adoption; press / mean|a|>0.9 / r_safe<−5 / NaN ⇒
    reject + revert to the local `champion.pt` ratchet; rejections persist (a high-numbered bad
    ckpt can't block the lineage or get re-tried). **Temp gate** (reg 63, 2-poll debounce — the
    sensor is FET-adjacent and spikes ~15 °C transiently, 46→33 in 32 s): >50 °C ⇒ park in the
    gravity-stable fold (verified 0.0 mrad drift), torque off, resume <42 °C. **Press watchdog**
    (PER-JOINT pegged-τ + ~0 q̇, 5 consecutive decisions, threshold 2.5 N·m): retreat to joint
    midpoints — static presses score r_safe=0 *by spec*, the reward never fixes them. Disk-bounded
    uploads; replay-ratio governor on the learner (A100 outruns the 2.6 tps arm ~100:1 — keep
    `--replay-ratio` LOW early or it overfits the pool).
  - **Integration-tested on the real hub** (mock arm): chunks→pool→ckpt→probation→**PROMOTE**
    (champion 80, sat 0.66); poisoned ckpt (saturated actor) → **REJECT** at sat 0.998, ratchet
    held. Bugs the test caught: hf-cache `os.replace` dangles relative symlinks (copy via realpath);
    warmstart must boot step −1; rejected-set must persist; boot order = newest non-rejected →
    champion.pt → warmstart.
  - **On-arm drill**: reg-63 confirmed real temperature (pan 33→46 °C under 60 s of drive); temp
    gate tripped/parked/rested/resumed live ×2; gravity-assisted scripted press correctly does NOT
    fire (~0.1 N·m — no fight; real pegs need external loading, per the organic event). Cleared for
    production; cadence ~2.5 sps with the 4-reads/servo loop (block-read regs 56–61 is the shelf
    optimization).

## Hardware campaign reference (frozen 2026-06-06 — change nothing mid-campaign)

**Pinned constants** — `--action-max 0.1`, `--lambda-safe 0.1 --lambda-cur 20 --safety-delta 15`
(always pass λ_cur — the default 1.0 silently shrinks curiosity 20×), `--action-block 5
--history-size 3` (train.py rebuilds from CLI), offline LRs wm 1e-5 / actor·critic 1e-4 (optimizers
restart cold), `--per-priority td` offline. `so101_calib.json`: P16/D32, goal_speed 2400 (pacing
cap), pace_dt 0.030, acceleration 0, **kt 10.0**. Judge progress **real-to-real** — never against
sim's −35 (different τ source).

**The 24/7 loop** (preferred):
```bash
# Mac (tmux; caffeinate stops macOS sleep killing the USB bus)
export SOARM_PORT=/dev/cu.usbmodem5AA90245791 SOARM_CALIB=so101_calib.json
caffeinate -i python src/collect_daemon.py --name auto1 --warmstart-name safe15 --warmstart-step 100000
# RunPod 1×A100 (clone private repo via GH_TOKEN, git checkout safety/deadband-and-lambda-safe,
# .env with HF_TOKEN/HF_UPLOAD_REPO_ID/WANDB_API_KEY/GH_TOKEN, bash setup.sh, then in tmux:)
python src/learner_daemon.py --name auto1 --warmstart-name safe15 --warmstart-step 100000
```
Pre-start: clamp the base; route cables out of the sweep volume (entanglement is the one failure no
watchdog catches); objects in the workspace (light/rigid — curiosity needs material); screw pass.
Pod pool survives pod death via a network volume for `runs/auto1_learner/` or `--keep-hub-chunks`.
First cycle attended: watch `runs/auto1/daemon.jsonl` for heartbeats (healthy: `rejects`=0,
`dropped`=0), the first chunk upload, one `candidate on probation` → `PROBATION PASS`. Sync
cadences: chunk ~6–7 min; ckpt ~5–10 min; ckpt→arm ≤5 min poll + ~12 s probation (~20 min
end-to-end staleness — irrelevant for off-policy SAC). Failure modes: Mac dies → arm HOLDS (or is
parked+limp mid-rest), restart resumes lineage; pod dies → Mac keeps collecting, new pod
warm-resumes; HF unreachable → uploads queue (≤20 chunks) then drop-oldest, polls fail silently.

**Manual round fallback** (the daemons supersede this, but it works):
```bash
python src/train.py --env-backend hardware --frozen-policy --resume-name <prev> \
  --name hw_round_N --total-steps 600 --max-episode-steps 10000 --action-max 0.1 \
  --lambda-cur 20 --lambda-safe 0.1 --safety-delta 15 --save-buffer --no-wandb --no-hf
huggingface-cli upload a5ilank/curious-robot runs/hw_round_N/buffer_*.npz buffers/hw_round_N/buffer.npz
python src/offline_train.py --resume-name <prev> --buffer buffers/hw_round_N/buffer.npz \
  --name round_N --steps 4000 --save-every 1000 --per-priority td   # RunPod
```

**Never mix into fine-tunes**: `runs/dryrun_collect/` (mock-sim, positive rewards),
`hw_validate`/`hw_dval`/`hw_safe15_match` (pre-P0 recompute-metric rewards), `_sim_safe15_*`/probe
buffers (λ_cur=1). Mix-safe sim: `buffers/sim_mix_v1/buffer_small.npz` (1:1) from round 2+ only if
the fine-tune destabilizes. Campaign-valid real buffers collected 06-06: `hw_p0_run100{,b,c}`,
`hw_p0_run200{,b,c}`, `hw_p0_{paced,nopace}` (~720 tr).
