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
- **GitHub — push/fetch with `GH_TOKEN` from `.env`**, NOT the VS Code / `gh` credential
  helper (it fails in this environment with "Repository not found" / credential-socket
  errors). `set -a && . ./.env && set +a`, then `git -c credential.helper= push
  "https://x-access-token:${GH_TOKEN}@github.com/ahilanks/curious-robot.git" <branch>`.
  Reference `$GH_TOKEN` as a **variable** (never paste the literal value); do **not**
  `git push -u` a token URL — it writes the token into `.git/config`. Set the upstream to
  the clean `origin` remote separately (`git branch --set-upstream-to=origin/<branch>`).
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
- [ ] **Deferred review findings (2026-06-06 ultra review; zero runtime impact today, fix when next
      touching the code).** Efficiency: bus `read()` is 4 txn/servo (block-read regs 56–61 + reg 69
      would halve it and recover ~2.5→4+ sps; same idea for `write_goal` via GroupSyncWrite);
      learner rebuilds the whole pool per new chunk (~10 s/cycle — use `ReplayBuffer.add` append);
      ChunkWriter np.stack churn (~450 MB/dump — preallocate). Dedup/altitude: the collector's
      acting loop is hand-copied from train.py (factor it like `sac_update` was — **reward-formula
      drift trap**: editing train.py's reward silently diverges from what chunks bake in); campaign
      constants duplicated as argparse defaults in 3 entry points; `press_tau` (N·m) is silently
      coupled to `kt` (move next to kt in the calib json); warm-start/model-init duplicated
      (learner vs offline_train); `log()` ×2, hub ckpt-listing ×3, chunk-count parse ×3, npz schema
      keys ×3 — each wants one shared helper/constant; `cur_lowpass`≠`vel_lowpass` is a documented
      sign-flip footgun (default-safe); sat/rsafe windows unbounded (deque(maxlen)); `pool_n` dead.
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

- 2026-06-10 — **servo gain-sweep benches on the real arm** (`src/bench_dgain_wide.py`,
  `src/bench_pgain_path.py`): one continuous joint-space path THROUGH 6 waypoints (Catmull-Rom,
  C1 through each point, global min-jerk time warp; pan in the cable-safe +0.05…+0.70, downward
  joints capped for the table). Each gain runs the identical path; restore P16/D32 after.
  **Stiffness (P, reg 21) is a real knob** — brisk path (`--path-time 4`):

  | P-gain | track err (mean/max) | jerk RMS | peak \|τ\| |
  |---|---|---|---|
  | 8 | 7.8° / 41.9° | 3.9 k°/s³ | 4.7 |
  | 16 (default) | 6.1° / 37.3° | 5.8 k°/s³ | 8.3 |
  | 32 | 4.5° / 34.2° | 6.8 k°/s³ | 13.1 |
  | 48 | 3.7° / 20.7° | 6.7 k°/s³ | 13.9 |

  Soft = floaty corner-cutting; stiff = 2× tighter tracking at 3× the peak effort; 32→48 is
  diminishing returns (P=32 the sweet spot for scripted motion; **campaign stays P16** — sim's
  kp=998.22 was derived at P16). **Damping (D, reg 22) is a dead knob** — same path, 6 s
  (spread ≲ run-to-run noise; P16 repeats scored 4.0°/4.0° mean):

  | D-gain | track err (mean/max) | jerk RMS | peak \|τ\| |
  |---|---|---|---|
  | 8 | 4.1° / 23.9° | 3.0 k°/s³ | 4.4 |
  | 32 (default) | 3.9° / 21.4° | 2.9 k°/s³ | 4.0 |
  | 128 | 3.6° / 20.4° | 3.3 k°/s³ | 4.6 |

  Gear friction owns the damping (confirms the 2026-06-09 down-sweep from the high side).
  **D=254 destabilizes**: velocity-noise chatter (~3 A spikes, 9.75° lag) tripped the STS3215
  firmware overload protection, which **silently cut torque on all 6 servos** (reg 40 read 0,
  no bus error) — after any high-current event, verify Torque_Enable before trusting "arm
  holding". `bench_pgain_path.py --reg d` refuses D>128. Gain-write gotcha: a readback
  immediately after an EEPROM write burst drops (reads 0) — wait ~100 ms + comm-check.

- 2026-06-11 — **safe15 jerk attributed** (`src/bench_safe15_jerk.py`, 100 instrumented control
  substeps at campaign config; data `runs/jerk_bench/jerk_safe15_100.npz`). Wall-clock probes on
  every bus/camera call + MPS-synced inference stages. **The "30 ms" substep is really 74 ms
  intra-block / 103 ms at block boundaries** (12.5 substeps/s vs 33.3 design): write 4.2 +
  dt_safe-sleep 30 + bus read 8.3 + **camera read 33.4** (blocks on the next 30 fps frame —
  the single biggest unbudgeted cost) + at boundaries ViT encode 14.2 / curiosity 7.1 /
  actor.sample 3.5. **Jerk is NOT the block-boundary ViT stall**: |accel|/|jerk| are uniform
  across substep positions k=0..4 (k=0 is mildly *smoother*). It's two compounding effects:
  (1) **13 Hz stop-start staccato** — pacing lands every move in exactly 30 ms (planned-arrival
  p50 30.0 ms ✓) but the substep lasts 74–103 ms, so the arm dead-stops ~45–73 ms after every
  move; (2) **policy dither** — safe15 reverses commanded direction on ~70% of substeps per
  joint (mean |Δq| 0.056 rad/substep at action_max 0.1); sim physics + alpha-interp filter
  this, the position staircase executes it literally (= the sim→real gap called on 2026-06-02).
  Fixes by size: free-running camera grab thread (latest-frame buffer; −33 ms → ~42 ms substep),
  then act on tanh(mean) instead of sampling (kills dither, changes exploration), then overlap
  inference (hard: serial by design). **Bonus findings**: true read-to-read dt is 79.5 ms but
  r_safe's qddot divides by 0.030 → accel overestimated ×2.65 on hardware; and **measured
  r_safe was 0.00 for the entire run** (|tau_meas| mean 0.2 N·m — the δ=15 deadband never
  trips; the live safety term is inert in this regime) while r_safe_recompute read −45.9
  (saturated-KP artifact, consistent with the old "real ≈ −45" numbers).

- 2026-06-11 (later) — **jerk fixes #1–#3 landed + verified on-arm** (bench re-runs, mean vs
  sample, `runs/jerk_bench/`): (1) `UsbCamera` free-running grab thread (camera read 33.4 →
  0.3 ms; `SOARM_SYNC_CAM=1` restores blocking grab); (2) `collect_daemon --act-mode mean`
  (default) acts on tanh(mean) — `sample` for A/B; (3) daemon loop defers curiosity/reward/
  chunk.add into the next block's motion window (`flush_pending()` guards every reset/gate/
  shutdown path — verified 108/108 transitions on a mock SIGINT run). **Substep 79.5 → 43.7 ms**
  (intra 41, boundary 55); remaining overage = bus read 8.3 ms (shelf: block-read regs 56–69).
  NOT done by request: pace_dt stretch (mid-move reads) and action pipelining (stale actions).
  **P8/D16 trial** (servos left there; any standard-calib connect restores P16/D32): at equal
  timing, accel 16.7 → 9–10 rad/s², jerk 578 → ~300 rad/s³, peak |tau_meas| 0.65–0.78 N·m —
  smoothest config measured; trade-offs: P8 tracking sag + kp=998.22 obs-recompute was derived
  at P16. **GOTCHA — max_step_ticks clamps vs the last READ**: `write_goal` limits every goal
  to ±300 ticks (0.46 rad) of `_last_pos`, which only `read()` refreshes — any multi-step
  scripted move without interleaved reads silently stops ~0.46 rad short (masqueraded as "P8
  too weak"). Fixed in bench `go_home()` and in `park_and_rest`'s fold (which could have
  dropped torque from a non-park pose). Behavior note: safe15-mean actively pins wrist flex
  against its +1.66 downward clamp — manual raises get undone by the policy.

- 2026-06-11 (evening) — **investigations + λ_cur 20→15**. (1) **Clamp occupancy** (2000-substep
  mean-policy run): 92% of time ≥1 joint within 0.15 rad of a clamp, interior only 8%; pan
  high-clamp 51%, grip open-clamp 92%, lift/elbow folded low 35/37%. Workspace-box tightening
  (DOWN_CAP-style collection limits) is the concrete exploration lever — clamp-riding at a
  table-facing pose beats folded-against-backstops. (2) **P8/D16 obs caveat is DEAD**: the
  kp=998.22 obs recompute is saturated at ±3.35 on **96-97% of joint-samples at BOTH gain
  configs** (the obs torque channel is a near-constant sign bit); effective-stiffness fit from
  logged current-vs-error gives P8/P16 ≈ 0.81 (KP-eq ≈ 813) — irrelevant under saturation.
  Tracking at policy scale is NO worse at P8 (|err| mean 0.036-0.046 vs 0.051-0.057 rad).
  P8/D16 adoption now blocked only by a decision + calib json edit. (3) **Lineage**: auto2
  learner is at ckpt 15909 (hub holds only the latest), all auto2 chunks consumed; today's
  config delta (mean acting, 44 ms loop, λ_cur 15, gains TBD) is a regime boundary → **next
  launch should be auto3**, warm-started from auto2's champion, never appending to auto2.
  (4) **λ_cur switched 20→15** (the long-deferred sweep candidate): collect_daemon default,
  train.py default (was the 1.0 foot-gun), README table — all aligned at 15. (5) **CPU vs MPS**
  on-arm A/B: full-CPU is worse (boundary 57.6 vs 55.1 ms; encode 15.8 vs 11.5) — MPS stays;
  only the actor MLP is faster on CPU (0.2 vs 3.1 ms, kernel-launch overhead), optional ~3 ms
  hybrid. Remaining timing shelf items: bus block-read regs 56-69 (~5-6 ms), fp16 ViT (~5 ms).

- 2026-06-11 (night) — **EMA REMOVED from the hardware env** (user decision; sim has no
  filtering, so raw restores obs parity in phase/lag — the encoder now sees instantaneous
  qd like sim, with Present_Speed quantization noise as the cost). `_read()` returns raw
  qd and clipped raw tau; `vel_lowpass`/`cur_lowpass` params and filter state deleted.
  Verified on-arm: env outputs bit-identical to raw bus reads; loop 43.9 ms unchanged;
  the historical "raw qd blows up r_safe" did NOT recur in the smooth regime (max hinge
  arg 1.4 vs δ=15; raw qddot p99 13.6 rad/s²) — that fear dated from the P64/sampled era.
  **Deadband re-pin numbers therefore use the RAW sweep**: δ≈1.0 (p99), λ_safe≈25-30 for
  the ~0.5:1 jerky-motion balance (jerky −0.18..−0.27 vs smooth −0.006..−0.019 per substep,
  15-30x separation), probation-rsafe −5.0 → ≈−0.1 (currently dead code — no measured run
  in history could trip −5). δ=15 + λ_safe=0.1 stays calibrated-for-sim only; judge
  real-to-real. Bench npz key renamed sub_qd_filt → sub_qd_env. Temps after the day's
  bench series: shoulder/elbow 42/40 °C (gate 50).

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
cadences (near-live ckpt transport, 2026-06-10): chunk Mac→pod ~6–7 min; ckpt pod→hub every
`--save-secs` (45 s) wall clock; hub→arm ≤20 s background poll + download + ~12 s probation —
**~1–2 min weight staleness** (data direction still bounded by chunk cadence). The hub holds
ONLY the latest ckpt (each upload is one atomic add+delete-older commit; every `--squash-every`
=100 uploads the learner squashes repo history and purges stale LFS blobs — without that,
deleted ckpts would keep their storage and grow ~80 GB/day). `champion.pt` on the Mac remains
the rollback ratchet; `--keep-hub-ckpts` restores the old keep-everything behavior. Failure modes: Mac dies → arm HOLDS (or is
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

## 2026-06-12 — deterministic policy (stochasticity removed) + dq_max retired + README constants pass

**Policy is now fully deterministic** (user decision): `Actor` = `tanh(MLP(z))` — Gaussian
head (`log_std`), `actor.sample`, entropy bonus, and `log_alpha`/`--alpha` all deleted.
`sac_update` (name kept for caller/W&B-key continuity) is now TD3-style: target
`y = r + γ(1−d)·min Q̄(z', π(z'))`, actor loss `−min Q(z, π(z))`; twin critics + Polyak target
(ρ=0.005) KEPT — with entropy gone the target net is the only TD stabilizer left, do not remove.
Random-action warmup acting removed too (`--start-steps` now only delays gradient updates).
What still has randomness (data-side, NOT the policy): sim reset/object spawn, PER + minibatch
sampling, WM dropout 0.1, seeds. Exploration now rides entirely on the curiosity reward.
- ckpt schema: new ckpts drop `log_alpha`; actors save without `log_std`. OLD ckpts
  (safe15, auto2, …) load everywhere via `load_actor_state` (filters `log_std.*`); the loaded
  deterministic actor is bit-identical to the old `a_det = tanh(mean)` path, so deployed
  behavior matches the daemon's previous `--act-mode mean` default exactly. `--act-mode`
  removed from collect_daemon + bench (bench npz now `*_det.npz`, `act_mode="det"`).
- touched: train.py (Actor/load_actor_state/sac_update/main/args), collect_daemon,
  learner_daemon, offline_train (arg `--alpha` gone), bench_safe15_jerk, eval_predictor,
  play_policy, README (actor eq, "Actor-critic objective (deterministic)" section, α row out).
- verified: unit smoke (determinism, old-ckpt filter, TD3 step), SOARM_MOCK hardware self-test
  OK, mock bench on real safe15 ckpt OK, 12-step inproc sim train run end-to-end OK, new ckpt
  schema clean. NOTE: a fresh/continued lineage trains the actor toward argmax Q from step 1
  (no entropy regularizer) — watch for premature exploitation; the planned action-rate
  regularizer slots into the new actor_loss in one line.

**dq_max removed everywhere** (env/soarm_adapter, mujoco_env, parallel_env, hardware_env (+self-test),
train.py, play_policy, eval_predictor, bench). It was a 100-rad never-binding duplicate clamp;
README's Δq^max symbol now maps 1:1 to code `action_max` (the actuation equation
Δq=a⊙Δq^max is literally what the code does). Old ckpt `args["dq_max"]` simply ignored.

**README constants table**: every Meaning cell rewritten as a short functional phrase;
Δq^max/action_max rows merged (0.3 sim, 0.1 hardware campaign); ρ row kept + disambiguated
(SAC/TD3 critic-target Polyak, NOT a JEPA EMA teacher — user asked to remove it; kept because
train.py:299 uses it and it's load-bearing, now more than ever). play_policy.py pre-existing
breakage noted: imports `record_rollout` from src.train which no longer exists there.

## 2026-06-12 — safety deadband CALIBRATED ON THE REAL ARM; re-pin package LANDED

Built `src/calib_deadband.py` (hold / reversal / analyze modes) and ran a labeled session at
P8/D16, true-dt hinge args, kt=10. User rated events 0-3. Data (`runs/deadband_calib/*.npz`):
- benign (sev 0: baseline, light touch, ALL scripted reversals up to ±0.25 rad — 2.5x the
  policy's max possible substep — rated "totally fine"): max 3.3; policy-run ceiling 4.3
- sev 1 (firm push): 7.4        — acceptable, must stay penalty-free
- sev 2-3 (grabs, joint blocks, jerky manhandling): 10.7 / 13.0 / 13.9 / 24.4
Clean gap [7.4, 10.7] → **delta = 9** (geometric midpoint). KEY INSIGHT: at these gains the
arm's own motion cannot reach "bad" — delta's job is external events (blocks, snags, collisions).

LANDED (defaults; meant for the NEXT lineage — do not hot-resume auto2 onto these rewards):
- `_control_step` qddot divisor: hardcoded 0.030 → measured read-to-read dt, clamped
  [0.5x, 4x] dt_safe (real loop ~42-44 ms; old divisor inflated args ~1.5x)
- delta 15 → 9 (train.py, collect_daemon, hardware_env, mujoco_env; parallel_env stale 0.05
  defaults synced too); README δ row + §Rewards updated
- lambda_safe 0.1 → 2.2 (one median bad substep ≈ one decision's curiosity ~10; benign
  r_safe is now IDENTICALLY 0, so lambda_safe scales only genuine events)
- probation_rsafe −5.0 (dead code) → −0.05 raw mean over the 30-decision window
  (trips on ~2+ bad decisions; a single external snag passes)
Verified live: 30 gentle substeps at delta=9/true-dt → r_safe 0/30 nonzero, loop 41.7 ms,
mock self-test + py_compile green, arm homed after.

CAVEATS: (1) valid at P8/D16 — recalibrate (5-min reversal ladder) if the lineage deploys at
P16/D32; (2) sim-side arg distribution at delta=9 UNVERIFIED — old sim notes claimed gentle
contact reaches ~15, so measure sim benign args before a long sim pretrain at these defaults;
(3) kt=10 baked into both calibration and runtime — rescaling kt rescales delta with it.

## 2026-06-12 — gains PINNED at P8/D16; sim kp recalculated 499.11 via the RBE501 motor model

User decision after a brief P16/D32 detour ("this is jerky"): P8/D16 is now the campaign
config. `so101_calib.json` p_gain/d_gain -> 8/16 (any standard connect now writes 8/16, the
old gotcha is inverted), servos confirmed 8/16 by read-back.

**kp recalculated for P=8** per the RBE501 DC-motor derivation the XML cites
(kp = Km*Gp*(180/pi)/R — LINEAR in firmware P; kv = Km*Kb/R — back-EMF only, NO firmware
P/D dependence, which matches our "friction owns damping / D cosmetic" bench finding):
kp 998.22@P16 -> **499.11@P8**, kv 2.731 unchanged. Applied to scene.xml +
so101_new_calib.xml (sts3215 class default), hardware_env KP, soarm_adapter docstring,
README K_p row. Saturation band of the obs recompute doubles to 6.7 mrad (~4.4 ticks) —
still mostly saturated during motion; moot if the [q,qd]-only proprio lands.
NOTE: sim plant is softer now — sim-pretrained ckpts from the kp=998 era saw different
dynamics; fine for the auto3 era, do not mix expectations.

**Deadband status**: the P16 detour measured benign policy ceiling 8.0 (500 substeps,
jerk_safe15_500_det.npz) vs 4.3 at P8 — delta=9 was marginal at P16. With gains back at
P8/D16 the 06-12 calibration (benign<=7.4, bad>=10.7, delta=9, lambda_safe=2.2) is VALID
as-is; P8 calib sessions live in runs/deadband_calib/p8d16/, calib_deadband POLICY_CEIL
back to 4.3. No relabel session needed.
## 2026-06-12 — SIM SMOOTHNESS/TRANSFER CAMPAIGN: 10 runs, 6k steps each (branch `sim/transfer-experiments`)

Goal: validate, fully in sim on the new kp=499.11/P8 plant + deterministic actor, the
transferability ideas (action-rate reward, energy reward, torque-free obs, multi-head Q,
deadband verification) and pick the simplest config with smooth, hardware-shaped motion.
All runs: 8 envs, action_max 0.1 (hw campaign scale), λ_cur 15, sim-calibrated safety
(δ=15, λ_safe=0.1 — see below), start_steps 1000, ~75 min/run at 3-way A100 sharing
(4-way OOMs: ~20 GiB/learner). New flags in train.py: `--w-action-rate`,
`--w-action-rate2`, `--w-energy`, `--no-torque-obs`, `--multihead-q`, `--actor-rate-reg`,
`--warmup-random`, `--explore-noise`. New tools: `src/measure_sim_scales.py`,
`src/compare_runs.py`. New metrics in every run: `smooth/{action_rate,action_rate2,
energy,qd_mean,tau_sat_frac,qd_reversal_frac}`, `sac/q_{cur,safe,rate,energy}`.

**Deadband / weights — sim≠real is now MEASURED** (`runs/sim_scales/kp499.json`): sim τ is
the saturated PD actuator force (rails at 3.35; sat 9–100% by regime), so hinge args run
20–50× the real kt-torque args. The real-arm pair (δ=9, λ_safe=2.2) fires on 33% of
joint-samples during SMOOTH sim motion → contrib −71 vs cur +10 → freeze-grade; sim keeps
(δ=15, λ_safe=0.1), which reproduces safe15's measured −30 / ~0.3:1 balance. In sim, δ is
second-order (penalty mass sits at args ≫15; λ_safe is the lever); on the real arm δ is
THE lever (benign ≤7.4 vs bad ≥10.7). Deadband verified firing correctly in both regimes —
they're just different regimes. Hardware keeps (9, 2.2).

**Headline negative result — the 2026-06-12 deterministic actor, from scratch in sim, has
two sticky attractors and r_safe is gameable:**
- No warmup (sbase/srate/senergy/snotorq): post-warmup BANG-BANG (rate 2.1–2.4, τ-sat
  0.9+) — curiosity *rewards* thrash (unpredictable). `sbase` then converged to a
  periodic-windmill loophole: r_safe −1.3 (looks great) at 93% saturation, rate 2.13,
  0 contacts — violent motion scored as "safe" because τ stays aligned with q̈.
  `senergy` found the same class of loophole and ended with WORSE energy (4.08) than
  srate (2.86) — the energy reward failed its own objective. DROP the energy term.
- Warmup alone (`swbase`): the opposite attractor — frozen lull (rate 1.07, sat 0.56,
  contacts 0, curiosity collapsed). Exploration needs a persistent mechanism, not a
  1000-step kick.
- Encoders stayed low-rank everywhere (eff_rank_probe 1.3–3.8 at 6k; rate-family +
  noise runs best) — 6k sprints rank configs relatively; a winner needs a long confirm run.

**What works (final-1k means; rate↓ sat↓ E↓ cont↑):**
| run | config | rate | sat | E | cont/s | cur | effR |
|---|---|---|---|---|---|---|---|
| sbase | control | 2.13 | .93 | 4.62 | .00 | 5.6* | 1.4 |
| senergy | w_energy 1 | 2.11 | .89 | 4.08 | .00 | 3.1 | 2.4 |
| snotorq | no-torque-obs | 1.60 | .74 | 3.80 | .02 | 1.3 | 1.3 |
| srate | w_rate 3 | 1.51 | .75 | 2.86 | **.44** | 2.9 | 3.4 |
| swrate | warmup+w_rate 3 | 1.57 | .73 | 3.38 | .01 | 5.5 | 3.8 |
| smhq | +multihead-q | 1.58 | .79 | 3.85 | .02 | 3.5 | 3.3 |
| swbase | warmup only | 1.07 | .56 | 2.88 | .00 | 2.1 | 1.4 |
| snoise | warmup+w_rate 3+noise .1 | **0.76** | **.48** | **1.12** | .11↗ | 4.0 | 3.4 |
| sareg | warmup+actor-rate-reg 5 | **0.22** | **.13** | 0.90 | .00 | 3.2 | 2.3 |
| scand | warmup+actor-reg 1+noise .1 | 0.50 | .33 | 1.42 | .01 | 2.8 | 1.4 |
(*sbase's cur is thrash-fed, not a win.)

- **Action-rate reward (the user-specced legged_gym term, W=3)** is the only *reward*
  term that bends the policy back from bang-bang AND restores object interaction —
  `srate` had contact BURSTS (rolling mean to 0.76 @ ~5.2k) vs the historic permanent
  ~0.001 collapse (newarch/lcur20/safe15); `snoise` adds steady rising late-run contact.
  Cost (the flagged Q-pollution, now QUANTIFIED via multihead): q_rate ends at −24.7 vs
  q_cur +21.9, q_safe −11.4 — the rate term is the single largest value component.
- **Actor-loss rate regularizer** (`--actor-rate-reg`, the "one line in actor_loss")
  has enormous, Q-clean leverage: W=5 → rate 0.22 / sat 0.13 / reversals at the real
  arm's calm scale — but over-damped (0 contacts). W=1+noise (`scand`) stays smooth
  (rate 0.50/sat 0.33) yet still inert with a weak encoder (effR 1.4) — Q-clean, less alive.
- **Collection noise** (`--explore-noise 0.1`, TD3-style, sim-only; policy stays
  deterministic) breaks both attractors cheaply: snoise = smoothest interactive run.
- **Torque-free obs** (`--no-torque-obs`): no cost detected (snotorq ≈ controls in the
  no-warmup regime); removes the ~96%-saturated sign-bit obs channel = the main sim→real
  proprio mismatch. Adopt; re-verify once under the final recipe before a long pretrain.
- **Multi-head Q** (`--multihead-q`): per-component heads train fine (sum == scalar
  optimum), behavior unchanged (smhq ≈ swrate), and the decomposition is the new
  standard diagnostic for reward-balance questions. Keep as an opt-in.

**kp=998 question — answered + already fixed:** kp came from the RBE501 STS3215 model at
firmware P=16; hardware is pinned P8/D16 since 2026-06-12, sim already recalculated to
499.11 (linear in P; kv=2.731 is back-EMF, P-independent). At action scale the plant still
saturates similarly (measured); the obs-recompute moot point stands if torque-free obs lands.

**RECOMMENDATION — the simplest transferable smooth config (sim pretrain):**
`--warmup-random --w-action-rate 3 --explore-noise 0.1 --no-torque-obs` on the sim-
calibrated safety (δ=15, λ_safe=0.1), λ_cur 15, action_max 0.1 (= the `snoise` recipe
+ torque-free obs). Rationale: snoise is the only run that is simultaneously smooth
(rate 0.76 / sat 0.48 / E 1.12 — 2.6× calmer than srate), interactive (contacts 0.11
and rising — vs the historic post-warmup collapse), curious (4.0), and encoder-healthiest
(effR 3.4). The Q-clean actor-reg variants (sareg W=5, scand W=1+noise) are smoother
still but inert (contacts ≈0) with weaker encoders — keep `--actor-rate-reg 0.5-1` as a
deploy-time smoothness topper, not the primary shaper. Noise/warmup are SIM-ONLY data
levers; the deployed policy stays deterministic.

Caveats: 1 seed per config, 6k sprints; encoder ranks everywhere ≪ safe15's 33@100k;
srate's interaction is bursty, snoise's is small-but-trending. Before trusting the
winner: one 50–100k confirmation run of the recommended config (+ the no-torque-obs
re-verify rides along free). Hardware unchanged: (δ=9, λ_safe=2.2), P8/D16, kp=499.11.

## 2026-06-13 — EXPLORATION DEBUG: why the new deterministic policy parks (sim), and what recovers movement/contact

After the `sconf` 100k smoothness-confirmation run (winner recipe) validated smooth motion but
the arm visibly **parked** (user: "it's not moving at all"), a focused investigation on branch
`sim/transfer-experiments`. Added **joint-travel logging** (`explore/pose_step` = ‖Δq‖ per
decision; `pose_range`/`pose_spread` = config-space coverage; in W&B + step print) and an
**`--explore-noise`** knob (TD3-style collection noise; policy stays deterministic). New default:
`--video-every 1000`.

**Finding 1 — action-rate weight is NOT the exploration lever.** `--w-action-rate` 3 (sconf) and
1 (sexpl1) both park: post-warmup `pose_step` collapses 0.22(warmup)→~0.05, contacts ~0. Lowering
W made it jitter *harder* in place (rate 0.65→1.0) while traveling *less*. The policy parks
regardless; curiosity (1-step pred error) is satisfied by the policy's own in-place micro-jitter.

**Finding 2 — action scale sets warmup movement, NOT learned-policy movement.** Diffed `safe15`
(which roamed) vs these runs: the real differences are **action_max 0.3 (safe15) vs 0.1 (mine)**
and **safe15 = stochastic entropy-SAC (α=0.2) vs the deterministic actor (entropy removed
2026-06-12)**. Tested action_max 0.1 / 0.3 (sexpl3) / 6.0=removed (snomax): all three plateau at
the SAME learned-policy `pose_step` ~0.13 despite a 20× cap difference. The cap only scales the
random-warmup moves; the deterministic policy converges to the same low floor either way. ⇒ the
**deterministic policy is the movement cap**, not action scale or any penalty.

**Finding 3 — strong collection noise recovers block CONTACT (the standing open problem).**
`sexpl4` (W&B `1ab42i8w`: explore_noise **0.6**, w_action_rate **0**, action_max 0.3, λ_cur 20)
sustains **contacts ≈0.105/step** and pose_step ≈0.225 over training — ~10× `sexpl3` and ~**100×
the historic ~0.001 collapse** (newarch/lcur20/safe15 all died here). The deterministic *eval*
policy still parks, but the noisy *collection* roams and hits blocks, feeding real contact
transitions into the buffer (visible in the 1k-step videos). explore_noise 0.3 only *delayed*
parking; 0.6 sustains it. So aggressive collection noise is a genuine, non-architecture-reverting
lever for the "interact with blocks" goal.

**Open decision (maintainer):** to make the LEARNED policy roam like safe15 (not just collection),
re-add the stochastic/entropy actor — but that reverses the deliberate 2026-06-12 deterministic
decision, so it's held for maintainer sign-off. Alternatives that DON'T revert it: (a) ship
explore_noise 0.6 as the collection-exploration default (already gets 0.1 contacts/step); (b) an
RND/state-coverage intrinsic bonus instead of pure 1-step pred error (attacks the "curiosity
rewards in-place jitter" root cause directly). Runs in flight (100k): `sexpl3` (smooth reference,
a_max 0.3/W1) and `sexpl4` (best explorer, noise 0.6/W0). `snomax` (a_max removed) retired after
confirming the cap is irrelevant. Note: even safe15 had contacts collapse post-warmup — entropy
restores *movement*, not necessarily *block contact*; sexpl4's noise is the better contact lever.

## Log — 2026-06-15 — α re-add VALIDATED (partial) at 20k sim: encoder un-parked, but lags safe15's pace

**Resolves the open maintainer decision above** (re-add the stochastic/entropy actor). The α=0.2
re-add shipped as commit `2e3217c` (Option 1: SAC stochastic-train / deterministic-deploy = exactly
safe15's scheme); this 20k sim run tests whether it restores the encoder stability + WM prediction
that the 2026-06-12 fully-deterministic actor lost (it parked → eff_rank 1.3–3.8 everywhere).

**Run** — W&B `alpha20k` = **`6bl63r0y`** (state `finished`, step 19999), ran on a RunPod A100-80GB,
**not pushed** (results recorded here from W&B cloud). Config = safe15 reproduced exactly — λ_cur 20,
λ_safe **0.1**, δ **15**, α 0.2, action_max 0.3, β 0.3, H_fwd,max 1, 8 envs, start_steps 1000,
action_block 5, H_bwd 3, γ 0.9 — with `train.py`'s **hardware** safety defaults (δ=9, λ_safe=2.2)
overridden to these sim-calibrated values (the defaults are freeze-grade in smooth sim). Two earlier
aborts: `ry7rxgs3` stopped @1300 by a session interrupt; relaunched detached (`setsid`) as `6bl63r0y`,
which ran to completion. Video every 1k steps (dual-cam) as requested.

| metric | **alpha20k @20k** | safe15 @20k (bar) | safe15 @100k | det-era (parked) |
|---|---|---|---|---|
| `encoder/eff_rank` | **12.88 ↑** | 22.72 | 33.61 | 1.3–3.8 |
| `encoder/z_std` | **0.914** | 0.953 | 0.984 | →low |
| `encoder/feat_corr` | **0.208** | 0.158 | 0.127 | →1 |
| WM pred/persist (`pred_loss÷identity_baseline`) | **0.338** | 0.149 | 0.130 | — |
| `reward/r_cur` | **0.665** | 0.724 | 0.777 | — |
| `reward/r_safe` | **−30.0** | −36.5 | −31.2 | — |
| `interact/contacts_per_step` | **0.034** | 0.0003 | 0.0009 | ~0.001 |

**Verdict — PARTIAL PASS.** The α re-add does its core job: it **un-parked the encoder**. eff_rank
climbed to 12.9 and was *still rising* at 20k (vs the deterministic era's 1.3–3.8 collapse); z_std 0.91
(not →0), feat_corr 0.21 (not →1); the WM **beats persistence** (0.34 < 1). Squarely in safe15's
*healthy regime*, and it actually makes **~100× more block contact** than safe15 did (0.034 vs
0.0003/step) — a plus for the "interact with blocks" goal. **But it does NOT numerically tie safe15 at
the same step**: eff_rank 12.9 vs 22.7 (~57%) and pred/persist 0.34 vs 0.15 (~2× the loss ratio).
Same direction and regime, quantitatively behind. r_cur / r_safe are comparable.

**Confounds (why the gap, all plausible, none verified):** (1) sim plant is now **kp=499.11** (safe15
ran kp=998.22) — softer plant, baked into the branch XML, can't revert without diverging the branch;
(2) different seed; (3) **eff_rank had not plateaued at 20k** — safe15 needed ~100k to reach 33.6, so
a 20k-vs-20k slice may understate where this lands at 100k (untested). Note `eff_rank_probe` is a
rollout-probe fallback (probe_v1 gone from HF) → not comparable to safe15's probe; the batch
`encoder/eff_rank` above is the apples-to-apples metric. **Next** (if pursued): a 100k run at kp=499 to
see whether eff_rank/pred close on safe15, or accept the healthy-regime result as sufficient for the
α-re-add goal. Read via `wandb.Api().run(".../curious-robot/6bl63r0y")`.

## 2026-06-24 — THE FINAL CULPRIT: CEM goal-reaching fails because the encoder latent has NO temporal locality (it's the DATA, not the planner / sigreg / camera)

Closes the `--cem` goal-explore line (Go-Explore: a CEM controller plans H=5 latent steps to minimize `‖ẑ_H − z*‖²` toward Go-Explore archive goals `z*`). Across ~150 W&B runs + a focused 06-23/24 sweep, **goal-reaching never works**: normalized `goal/dist_to_goal` pins flat at **~20** (≈0.9× the √(2·256)≈22.6 random-pair latent ceiling) and `goal/reach_rate ≈ 0`, regardless of camera, WM accuracy, or goal source.

**Ruled out — one experiment each, none moved dist off ~20:**
- **camera**: wrist ≈ overhead (`cem_wmbs256_iters15_wristcam` vs `_wmcuriosity`).
- **WM accuracy**: `h_fwd_max` 1→5 + `wm_lr` 3e-4 drove `wm/pred_loss` 0.20→0.05 (near-perfect multi-step WM) → no change.
- **goal SELECTION**: `--goal-select near` (nearest archive goal by latent dist) and `--goal-select future` (achieved obs K steps ahead in the env's own episode) both stayed ~20.
- **the PLANNER**: NOT it — verified IDENTICAL from source to the LeWM reference (swm `solver/cem.py:CEMSolver` + `wm/lewm/lewm.py:get_cost`): same Gaussian sampling, force-candidate-0=mean, terminal `‖ẑ_H−z*‖²` summed over dims, top-k elites, refit mean/std, no min-std floor.

**ROOT CAUSE (measured, `scratchpad/probe_geom.py` on trained ckpts): the latent has no temporal locality.** Encode the archive's source `o_t` and outcome `o_{t+1}` (1 decision-step apart): **1-step jump `‖z_t − z_{t+1}‖ ≈ 21` normalized ≈ the random-pair distance, on BOTH wrist (21.3) and overhead (21.8).** One env step decorrelates the latent almost completely → no goal is ever "near" → CEM has no navigable gradient (5-step reachable spread `z_term_spread` ~13 < goal ~20). The archive goals ARE real states the arm visited — but it visited them via jerky exploration, so even a 1-step-ago state sits ~random-pair away. **Seen-ness ≠ navigable structure.**

**NOT instability, NOT sigreg/isotropy.** The latent is healthy (z_std ~1.0, RankMe ~90-100/256, stable low pred_loss). And the regularizer is exonerated by direct comparison: reproduced the official LeWM eval and measured **ITS** cube latent — 1-step jump **1.7 (ratio 0.09)**, growing SMOOTHLY with the gap (1.7 → 6.4 → 10.2 → 14.1 at k=1,5,10,25). LeWM uses the SAME predict-next-embedding + SIGReg objective and is EQUALLY isotropic (random-pair √(2·192)=19.6, z_std 1.0) — yet temporally local. So isotropy never precluded locality.

**The differentiator is the DATA (same objective, opposite outcome):**
- **LeWM**: OFFLINE on smooth **expert** trajectories (cube_single_expert, 2M steps) — small physical change/step → smooth latent path → local.
- **Ours**: ONLINE **curiosity** exploration with `action_max=1.0` (full-radian joint deltas) + a reward that actively SEEKS unpredictable states → large, jerky, novelty-biased moves/step → the encoder faithfully maps that to ~orthogonal latents.

**LeWM reproduction (the comparison anchor; isolated in `/workspace/le-wm`):** official `lucas-maes/le-wm` cloned + run on cube (3D robot-arm) → **74% success (50 ep)**; smoke 80% (5 ep). In the paper's "competitive 3D control" range (DINO-WM has a slight edge on cube per Fig 6 — exact figure is a plot, not text). Needed ~7 unpinned-dep fixes: `transformers==4.49` (ckpt uses pre-v5 HF ViT key naming), `datasets≥3.6` + `hdf5plugin` (pixels are plugin-compressed), the modern `load_dataset` / `checkpoints/<task>/<run>/{weights.pt,config.json}` loader convention, manual dataset fetch from `quentinll/lewm-cube` staged via `/dev/shm` to fit the 150 GB `/workspace` quota, and a 1-line `eval.py` loader patch.

**Fix direction (untested): representation/data, NOT planner, goals, camera, or sigreg.** Either smooth the action regime / train the encoder on smoother (less novelty-biased) trajectories, or add an explicit slowness / temporal-contrastive term to FORCE the locality the data isn't providing.

**Instrumented (`src/train.py`, this commit):** new `encoder/step_jump` = windowed mean `‖z_t − z_{t+1}‖` and `encoder/step_jump_frac_rand` = step_jump / √(2·z_dim) — watch latent temporal locality directly. ~1.0 = no locality (current); LeWM cube was ~0.09.

## 2026-06-24 — slowness term & action_max schedule TESTED → neither fixes planning

Implemented `--lambda-slow` (penalize mean-sq per-step latent jump `‖z_t−z_{t+1}‖²` on real consecutive encoded frames in `wm_update`; SIGReg anchors against the dz→0 collapse) + an action_max schedule (`--action-max-start-frac`/`--action-max-warmup-steps`, ramp effective amplitude small→full) + `--sigreg-pertimestep`.

**Calibration:** SIGReg DOMINATES the WM loss — `wm/sigreg`≈26 → β·sig≈2.35 vs pred_loss≈0.50 (≈5:1). λ must compete with that; λ=0.3 was ~8× too weak (null result).

**Locality sweep (h_fwd_max=1, settle ~3k; corr = step_jump/(√(2D)·z_std); baseline 0.85, LeWM 0.09):** λ=3→0.67 (rank .25), 5→0.65 (.21), 10→0.52 (.185), 20→0.34 (.15). Monotonic; rank declines gracefully (SIGReg holds the floor, no hard collapse); pred_loss improves 0.50→0.11. So the term reliably bends the *locality metric*.

**action_max SCHEDULE = FAILS.** Ramping amplitude 0.1→1.0, step_jump tracks amplitude in REAL TIME (4→17.5 lockstep; corr 0.50@amp0.64 → 0.82@amp1.0 = baseline), no lock-in. Locality ≈ a real-time function of per-step physical change, not a learnable-then-frozen property.

**Goal-reaching payoff sweep (h_fwd_max=5, 12k steps, λ∈{0,10,20}) → NO λ FIXES PLANNING.** `reach_rate`~0 and normalized `dist_to_goal` flat ~21 for ALL λ. Worse, the slowness term SHRINKS CEM steerability: `cem/z_term_spread` fell from 0.48 (λ=0) to 0.17 (λ=20) of the latent diameter (normalized), while goals sit ~0.93 of the diameter away ⇒ the more-local latent is *LESS* reachable.

**KEY INSIGHT:** the slowness PENALTY achieves locality by shrinking ALL latent motion uniformly — including the action-controllable component — so the latent becomes local but UNSTEERABLE. LeWM's locality is DATA-driven (smooth purposeful expert trajectories): consecutive frames are close yet different actions still lead to meaningfully different places, so locality AND steerability coexist. ⇒ the fix is DATA-side (smoother purposeful online behavior / less novelty-biased buffer), NOT a loss penalty. (SIGReg batch composition also RULED OUT: per-timestep vs pooled = 0.85≈0.85.) NEXT: data/behavior levers — action-rate *smoothness* (not amplitude), and code-level encoder diffs vs LeWM.

## 2026-06-24 — the encoder DIFF found: proprio-concat (256-d) was a real culprit, not just the data. Pixels-only LeWM-faithful encoder → TRANSIENT locality + nonzero reach, then rank-collapses

Chased the one architectural divergence from LeWM. Confirmed first that buffer + inter-frame actions + encoding cadence already match LeWM (`history_size=3`; action_encoder gets the whole inter-frame action block flattened — ours `n_dof*action_block=5`, LeWM `frameskip*action_dim=5`; LeWM eval configs literally say `action_block: 5 # frameskip`). **The ONLY divergence left: our state is `z = MLP(ViT_cls) ‖ MLP(proprio)` fused to 256-d, while LeWM is pixels-only `z = projector(ViT_cls)` at embed_dim=192** (`lewm/jepa.py:38` also uses the CLS token — NOT a patch-token field; SIGReg is per-timestep `(T,B,D)`, `lewm/train.py:40`).

**New flags (this commit):** `--no-proprio` (pixels-only encoder, `StateEncoder.use_proprio=False` → `z_dim=192`; proprio still collected for safety/goals but not fed to the WM), `--action-max-end-frac` (curriculum now ramps start→end then HOLDS end, vs always→1.0). Also **β pinned default 0.3→0.09** (LeWM's `lewm.yaml`; the old 0.3 made β·sig dominate pred_loss ~5:1).

**Stricter action-max cap on WRIST cam FAILS (same as all amplitude knobs).** `cem_wrist_a0005_e02` (W&B `36ef0rnj`; start_frac 0.005→end 0.2 over 4k, β=0.09): `step_jump` tracked the capped amplitude in real time (3.7→18.4), z_std-corrected `corr*` pinned ~1.0 the whole run, `dist_goal` pinned ~21 (D=256 ceiling 22.6), `reach`≡0. The amplitude-corrected metric never moved → closes the action-amplitude-schedule lever for good.

**Pixels-only (no-proprio) is CATEGORICALLY different — first run in the investigation to show real locality + nonzero reach.** `cem_lewm_noprop_wrist` (W&B `g068v7m6`; wrist, `--no-proprio --sigreg-pertimestep`, β=0.09, h_fwd_max=1, no curriculum; D=192 so `frac_rand` is apples-to-apples with LeWM's 0.09 cube anchor, and `corr*=frac_rand/z_std`):

| step | frac_rand | corr* | rank_frac | dist_goal | reach |
|---|---|---|---|---|---|
| 1000 | 0.072 | 0.43 | 0.049 | 3.6 | 0.18 |
| **2000** | **0.117** | **0.17** | 0.039 | **7.0** | **0.28** ← peak |
| 3000 | 0.368 | 0.50 | 0.050 | 11.5 | 0.05 |
| 4500 | 0.378 | 0.51 | 0.059 | 14.2 | 0.01 ← end |

vs **every 256-d run** which pinned `corr*≈1.0 / reach=0 / dist_goal≈20` and NEVER budged. So **the proprio-concat / 256-d latent WAS hurting** — pixels-only is roughly halfway to LeWM and produced a real (transient) `reach=0.28`, `corr*` bottoming at 0.17, `dist_goal` never pinning at ceiling.

**BUT it doesn't sustain — the tell is RANK.** `rank_frac` sat at **~0.04–0.06 (≈8–11 of 192 dims) the entire run** (256-d runs ran ~0.25 ≈ 64 dims). The early locality rode on a *small/collapsed* latent (low z_std → all states trivially close → goals reachable). As SIGReg expanded z_std 0.17→0.76 *without* rank recovering, the ~10-D manifold stretched, `step_jump` tripled (2.3→7.5), and `reach` collapsed to ~0.01. The win was a **low-rank transient**, not a stable manifold.

**KEY INSIGHT (refines the prior "it's the data" conclusion → it's BOTH):** proprio was **propping rank UP with junk dims that don't aid navigability** (the smooth proprio block satisfied prediction while vision stayed scrambled). Remove it and the pure-visual encoder **collapses to ~10 dims** in this online-exploration regime. Neither setup is right: proprio = high rank / no locality; pixels-only = low rank / transient locality. The new target is no longer data-vs-architecture — it's **why the pixels-only latent rank-collapses, and how to hold its rank so the step-2000 locality sustains.**

**NEXT (running):** `cem_lewm_noprop_pooled` — no-proprio + the **pooled** SIGReg (drop `--sigreg-pertimestep`; default `B·T=512 > D=192` vs per-timestep's B=128<192). Tests whether under-sampled isotropy let rank collapse. (Caveat: the 2026-06-24 slowness entry already found per-timestep≈pooled for *locality* at D=256 — but this asks a different question, RANK in the low-rank 192-d regime.) If pooling doesn't hold rank, the lever moves to an explicit rank/variance floor or the data/behavior side.

## 2026-06-25 — SIGReg pipeline bug found (LayerNorm vs BatchNorm projector) + fixed → best transient ever, but EXACT-LeWM architecture STILL fails ⇒ it's the DATA, decisively

Audited SIGReg from first principles. The statistic + kwargs (`knots=17, num_proj=1024`) and the per-timestep call matched LeWM, but **the `emb` SIGReg regularizes was produced with the wrong norm.** LeWM's `emb = projector(ViT_cls)` where projector = `MLP(hidden=2048, norm_fn=BatchNorm1d)` (`lewm.yaml`); my `visual_head`/`pred_proj` used `lewm.module.MLP`'s **default `norm_fn=LayerNorm`** (per-SAMPLE, no cross-batch anti-collapse) instead of BatchNorm1d (per-DIM cross-batch whitening = the SSL anti-collapse mechanism). Fixed in the pixels-only path (`use_proprio=False`): projector + pred_proj now `BatchNorm1d`, hidden 2048, matching LeWM exactly.

**Effect of the BatchNorm fix** (`cem_lewm_bn_wrist`, W&B `gwy0u3bl`; vs the LayerNorm twins `g068v7m6`/`2zzk70mo`):

| metric | LayerNorm no-proprio | **BatchNorm no-proprio** |
|---|---|---|
| best early reach (step 1000) | 0.18–0.28 | **0.44** (frac_rand 0.040, < LeWM's 0.09!) |
| rank_frac (sustained) | pinned ~0.05 (~10 dims) | **~0.10–0.11 (~20 dims, ~2.3×)** |
| end-state corr* / reach / dist_goal | 0.5 / 0.01 / ~16–20 | 0.46 / **0.00** / ~19 |

So BatchNorm **was a real bug** — it doubled the rank and produced the strongest transient of the whole investigation. But **the end-state is the same failure.** Decisive evidence in the `z_std` column: it climbs 0.14→1.05 (SIGReg achieving its unit-variance isotropy target) and **locality/reach die in exact lockstep with that climb.** SIGReg succeeding *is* what kills locality.

**VERDICT — it's the DATA, now proven by exhaustion.** Every architectural knob is now *exactly* LeWM — pixels-only 192-d, **BatchNorm projector**, β=0.09, per-timestep SIGReg, 1-step training, action_block=5=frameskip — and it STILL fails the moment SIGReg drives z_std→1. The SIGReg-vs-locality tension is irreducible *for our data*: LeWM's smooth expert demos keep consecutive frames near-identical so isotropy + locality coexist; our jerky online-curiosity frames get shoved apart when isotropized. The only difference left between us and LeWM is the **data smoothness**.

**NEXT (running):** test the data hypothesis directly. SAC-curiosity collection (CEM has no smoothness lever — actor is off) with the BatchNorm pixels-only encoder + **action-rate smoothness penalty `--w-action-rate 3`** (calibrated: dither −4.3 vs cur +10, smooth −0.3) + **low action-max cap (0.15)** → does genuinely smoother/gentler collected data give a *sustained* local latent (corr* staying low even as z_std→1, vs the always-decays pattern)? If yes, the fix is confirmed data-side and CEM goal-reaching gets re-enabled on the smooth latent.

## 2026-06-25 — DATA HYPOTHESIS REFUTED (it's the SIGReg isotropy, not the data) → then multi-epoch consolidation: epochs FIX rank but OVERFIT on small data ⇒ the real gap is data DIVERSITY

**The data hypothesis is dead, killed by its own clean controls.** Built `--collect-smooth` (scripted sub-action OU walk, persisted across decisions; WM-only training, mirrors LeWM's offline-WM-on-data) to remove the curiosity/CEM confound. Also discovered the earlier locality tests were ALL under CEM (goal-explore, SAC off) — the buffer came from a *failing planner*, a chicken-and-egg confound the user flagged. Ran the controls (all no-proprio BatchNorm 192-d, per-timestep SIGReg β=0.09, WM-only):

| run | data | encoder input | end corr* |
|---|---|---|---|
| `ooymluqd` curiosity baseline | jerky (rate~1.0) | wrist (jerky) | ~0.50 |
| `xff36yyd` smooth-collector | **smooth** (rate 0.04) | wrist (still jerky — egocentric viewpoint swings) | ~0.49 |
| `pvjmnyle` smooth + OVERHEAD | smooth (rate 0.04) | **overhead (smooth)** | ~0.41 |

All fail identically. The smoking gun (`pvjmnyle` step 2000): `step_jump=19.7`, `frac_rand=1.006` — consecutive latents a FULL random-pair apart **on near-identical overhead frames** (smooth joints, fixed cam, arm barely moved). **SIGReg satisfies its z_std→1 isotropy target by SCATTERING consecutive frames across the Gaussian — and that scattering is what kills locality, independent of data smoothness OR camera.** Baseline also closes the CEM-confound worry: native curiosity data fails the same as CEM data. The session-long "it's the data" thread is FALSIFIED.

**Then tested the last LeWM difference — SCALE / multi-epoch.** Built `--consolidate-every N` / `--consolidate-epochs E`: every N steps, pause stepping and train E passes over the frozen buffer (pushes our online single-pass regime toward LeWM's 100-epoch offline). First run `rl63c6zt` (every 2 steps, 2 epochs, smooth-overhead) — BUT buffer capped at **1000** (`clip(buffer_frac·total_steps)` floored, total_steps=2000). Result at step 1500 (~500 epochs):

- **pred/persist CRASHED 0.89 → 0.021** (WM predicts ~50× better than persistence — pred_loss WON), and **rank_frac UN-COLLAPSED 0.026 → 0.294** (~56 dims, best ever, toward LeWM's ~0.5). **So epochs FIX the rank collapse that haunted every prior run.**
- **BUT it OVERFIT and went ANTI-LOCAL:** `frac_rand=1.765` (`corr*=2.57`) — consecutive frames now FARTHER than random pairs. ~500 epochs on 1000 transitions = memorize the buffer, scatter frames maximally.

**KEY REFRAME — "scale" = DATA DIVERSITY, not epochs.** The lever (epochs) is powerful — it crushes pred_loss and fixes rank — but with too little data it memorizes and destroys locality. LeWM's real edge is 2M *diverse* frames so 100 epochs generalizes. Epochs/data ratio matters: this run saw each frame ~500–1000× vs LeWM's 100×.

**NEXT (running):** `consol_bigbuf_smooth_overhead` — `--buffer-frac 3.0` (cap 12k, 12×), `--consolidate-every 30 --consolidate-epochs 1` → ~100 passes (LeWM ratio, not 1000). Does a bigger buffer + LeWM-matched epochs hit the sweet spot — high rank AND low corr* (local) — or still overfit (data still too small vs 2M)?

## 2026-06-25 — FIRST REAL PROGRESS: big-buffer + LeWM-ratio epochs breaks the corr*~0.5 wall (no overfit) → it's DATA SCALE. Now testing the CEM payoff on that latent.

`consol_bigbuf_smooth_overhead` (W&B `rxehmdd5`; smooth-overhead data, `--buffer-frac 3.0` cap **12000**, `--consolidate-every 30 --consolidate-epochs 1` = **~100 passes** = LeWM ratio, not the 1000 that overfit). First run where ALL FOUR metrics are good simultaneously:

| step | frac_rand | corr* | rank_frac | pred/persist |
|---|---|---|---|---|
| 1500 | 0.334 | 0.424 | 0.131 | 0.320 |
| 2000 | **0.246** | 0.301 | 0.196 | 0.162 |
| 3000 | 0.277 | 0.306 | 0.189 | 0.088 |
| 3500 | 0.278 | 0.330 | **0.347** | **0.066** |

- **corr* ~0.30** — best non-collapsed locality of the whole investigation (every prior run ~0.5); **~halfway from the failure wall to LeWM's 0.09.**
- **rank_frac 0.13→0.35** (~67 dims) — highest ever, toward LeWM ~0.5. Epochs fix the collapse for real.
- **pred/persist 0.066** — strong, and **healthy generalization** (NOT the 0.021 memorization of the small-buffer run).
- **frac_rand stable ~0.27 — NO anti-local overfit** (small buffer blew to 1.76). The 12× data did its job.

So the **recipe is directionally right**: smooth-overhead data + big buffer + LeWM-ratio epochs → high-rank, well-predicting, meaningfully-more-local latent, no overfit. **BUT it plateaus at frac_rand ~0.27, not 0.09:** frac_rand bottomed at step 2000 and flattened while rank kept rising ⇒ more epochs stopped helping locality ⇒ **DATA-LIMITED.** 12k diverse frames gets ~halfway; closing to 0.09 needs more raw data (toward LeWM's 2M). Next data-scale lever (option a, deferred): cap 30–50k + longer collection.

**PAYOFF TEST (running) — option (b), test CEM on this latent before chasing 0.09.** `cem_on_bigbuf_latent` (W&B `cpybzd8p`): `--cem --resume-name consol_bigbuf_smooth_overhead --freeze-encoder --no-proprio --wm-cam overhead`. Loads the corr*~0.30 latent (ckpt_4000), FREEZES the encoder (latent stationary, the good geometry preserved), predictor keeps training to adapt to CEM's full-range action candidates (smooth collector only trained it on small OU actions). THE QUESTION: does a corr*~0.30 latent finally give **nonzero reach** / dist_goal below the 19.6 random-pair ceiling — i.e. does the halfway locality gain already buy goal-reaching? EVERY prior CEM run on a corr*~0.5 latent had reach=0 / dist~20. Even reach 0.05–0.2 = first sustained nonzero reach = breakthrough. Early signal healthy: contacts/s=0.70 at step 50 (arm interacting), resume + freeze confirmed.

**PAYOFF RESULT (`cpybzd8p`, finished):** the locality gain TRANSFERS to planning but isn't yet enough for reach.

| | prior CEM (corr*~0.5) | this run (corr*~0.30, frozen) |
|---|---|---|
| dist_goal | pinned ~20 (19.6 ceiling) | **~15.5** (first sub-ceiling EVER) |
| reach | 0.00 | **0.00** |
| pred/persist (CEM actions) | — | 0.7-0.8 (predictor adapted, beats persistence) |

Frozen latent stayed GOOD + stationary the whole run (frac_rand ~0.29, rank 0.239, z_std 0.872 constant). **Two clean conclusions:** (1) the locality gain is REAL and transfers — dist_goal dropped from the ~20 ceiling to ~15.5, first CEM run ever below the random-pair ceiling. (2) The predictor confound resolved (pred/persist fell to ~0.7, so it's NOT a predictor-action-mismatch artifact). ⇒ **"halfway-local" (frac_rand 0.29) is not local enough for reach:** CEM closes goals to ~15.5 but can't get within the eps=2.0 reach threshold (per-step latent jump ~5.7 is too coarse). dist_goal scales with locality, so closing the rest of the gap (frac_rand 0.27 → ~0.09) should buy reach. **NEXT = option (a) data scale:** `consol_bigbuf50k_smooth_overhead` — cap 50000 (max, ~4x), ~100-epoch ratio, then re-run this exact CEM test on the result to see if reach finally clears 0.

## 2026-06-25 — BREAKTHROUGH: encoder locality in the CURIOSITY+ONLINE setup via multi-epoch consolidation (it was never the data smoothness)

Goal: locality in the *actual* curiosity+online regime (the smooth-collector wins were artificial). Applied the recipe — pixels-only BatchNorm + overhead cam + big buffer + multi-epoch consolidation — but with the **curiosity SAC actor** generating the data instead of the smooth collector.

**`cur_consol_oh_bn` (curiosity + overhead + buffer 24k + consolidate-every 70): frac_rand 0.160 -> 0.132 -> 0.093 -> 0.081 (steps 1500-3000), oscillating ~0.08-0.16 after.** That IS LeWM's 0.09 — on genuinely JERKY curiosity data (action_rate ~0.75, same as the failed baseline), healthy z_std ~0.8, rank rising to ~0.32. The curiosity single-pass baseline (`ooymluqd`) was corr* 0.5 / frac_rand ~0.4; consolidation drops it to LeWM-level.

**KEY REFRAME (overturns the whole prior arc):** it was NEVER the data smoothness — it's **multi-epoch consolidation** (repeated exposure lets pred_loss carve a local manifold before SIGReg scatters it). Curiosity+consolidation (0.08) even BEATS the smooth-collector 50k (0.148) — curiosity's exploration DIVERSITY helps once consolidation removes the jerkiness penalty. The entire smooth-collector / data-hypothesis thread was a red herring; the lever was epochs all along.

**Camera ablation:** overhead is ESSENTIAL. `cur_consol_wrist_bn` (same recipe, wrist) went 0.247 -> 0.309 (degrading) — egocentric pixel-jerk is too much for consolidation to fix. Overhead (fixed cam, small arm displacement in-frame) is the multiplier; consolidation is the main lever.

**Reach payoff STILL blocked — train/test distribution gap (not predictor):** froze the 0.08 latent, ran CEM (`cem_curiosity_amax03`, `cem_cur_std03_consol`). Predictor consolidation FIXED the predictor (pred/persist -> 0.22, in-distribution), and matching action_max (0.31) + gentle cem_init_std (0.3) helped — BUT reach stayed 0, dist_goal ~15.5. The frozen encoder shows frac_rand ~0.27 ON THE CEM TRAJECTORY (vs 0.08 on curiosity's): the encoder is local for the curiosity *distribution* but NOT for CEM's goal-DIRECTED motion (all 5 sub-actions push one way -> big net latent step). So locality is distribution-specific; goal-reaching needs the latent local for directed trajectories too (chicken-and-egg: needs a working CEM to generate directed data). NEXT: efficiency (cheapest consolidation for 0.08) + the directed-data-distribution problem for reach.

**FOLLOW-UP — drift, efficiency, stability (curiosity+consolidation fully characterized):**
- **Drift:** frac_rand is a TRANSIENT dip, not a stable floor. It bottoms ~0.08 around step 2500-3500 then drifts back UP as SIGReg drives z_std->1 (the same z_std<->locality tension). `cur_consol_oh_bn` 0.081(3000)->0.191(6000); cheap `cur_consol_eff2` (every 140, 12k) drifted to 0.27. Practical fix: CHECKPOINT + FREEZE at the dip (the post-dip drift is irrelevant if you use the best ckpt — which is what the reach-tests loaded).
- **Efficiency:** half-cost consolidation (`--consolidate-every 140` vs 70, buffer 12k) reaches a similar *sustained* ~0.15-0.2 — cheaper but drifts more.
- **STABILITY = bigger buffer DELAYS the drift (wide usable window, not permanent):** `cur_consol_stable` (buffer 32k, every 90) holds frac_rand stable at ~0.11-0.12 from step 2000 to 5000 (a wide window — vs the 12k buffer that drifted immediately), then gets noisy/drifts to 0.27 by step 7500. So a big buffer buys a LONG stable window for checkpointing the ~0.12 latent, but doesn't eliminate the z_std<->locality drift. Practical: train with a big buffer, checkpoint in the stable window (steps ~2500-5000).
- **Directed-data for reach FAILED:** `dir_consol_oh_bn` (`--goal-explore` = goal-directed SAC actor + consolidation, overhead) — frac_rand climbed 0.16->0.27->0.50 (reach 0). Full-amplitude goal-directed motion makes BIG latent jumps -> no locality. **Core unsolved tension: locality needs SMALL per-step moves (small step_jump), but goal-reaching needs BIG DIRECTED moves toward goals -> they conflict.** That's why reach stays 0 despite LeWM-level locality on curiosity (small-move) data: a frozen curiosity-local latent isn't local for CEM's directed trajectories, and training on directed data destroys locality. Reach needs a representation local under BOTH regimes (or a planner that respects the small-move locality) — the genuinely open research problem.

**REACH EXHAUSTED (definitive negative result on the curiosity-local latent):** froze the 0.08 latent and tried, in the loop: CEM at action_max {1.0, 0.31, 0.1, 0.05}, gentle cem_init_std 0.3, predictor consolidation, --goal-select near, step-scaled eps 4.0 — AND a learned goal-conditioned SAC policy (--goal-explore --freeze-encoder). Each lever monotonically REDUCED dist_goal (20 ceiling -> 15 -> 14 -> 11.5 -> **9.4** best, with near-goals+finest-amplitude CEM), but **reach stayed 0 in EVERY case**, even at the lenient eps=4 (~1.7x step_jump). The learned policy was worse than CEM (dist ~16). So: locality clearly HELPS planning (dist halved from the ceiling) but the planner never CLOSES a goal — the directed-motion-breaks-locality tension caps the final approach. Reach with this curiosity-trained latent is a dead end via planner/amplitude tuning; it needs a fundamentally different representation (local under directed motion) or goal/eps reframing. The CURIOSITY+ONLINE LOCALITY GOAL itself is fully achieved (frac_rand 0.08-0.12).

**REACH REFRAME (sharper, two new findings):**
1. **Directed-motion locality IS achievable** — `dir_fine_consol` (`--goal-explore` directed actor + `--action-max 0.1` FINE amplitude + consolidation, encoder TRAINS) hit and SUSTAINED **frac_rand 0.070** (below LeWM's 0.09) ON GOAL-DIRECTED data. The earlier directed-data failure (frac_rand 0.50) was purely the FULL amplitude (1.0); at fine amplitude directed moves stay local. So the "distribution gap" (latent local for curiosity but not directed motion) is SOLVABLE — locality is no longer the reach blocker.
2. **The real reach blocker is the FINAL-APPROACH CLOSING problem.** Even with directed-motion locality (0.07), reach stays 0: every planner/policy reduces dist_goal from the ~20 ceiling to ~9-12 then STALLS ~6 steps out, never closing to within eps. Longer CEM horizon (15 vs 5, with h_fwd_max 15) made it WORSE (dist 12 vs 9.4 — multi-step rollout error compounds, not lookahead-limited). So goal-reaching needs whatever lets a planner finish the last stretch to diverse archive goals — a planner/goal-curriculum problem on top of locality, NOT a locality problem. (Note the `reach=1.0` seen at archive-warmup is trivial: goals==current state, dist~0.)

**SESSION-LOOP SUMMARY:** Goal (curiosity+online encoder locality) ACHIEVED — frac_rand 0.08-0.12, lever = multi-epoch consolidation (NOT data smoothness), recipe = pixels-only BatchNorm + overhead + consolidation + big buffer. Reach extensively explored (~9 experiments): locality helps (dist 20->9.4) and directed-motion locality is solvable (0.07), but reach=0 throughout — blocked by the final-approach closing problem, the genuinely open research direction.

**RECIPE (curiosity+online encoder locality, the goal):** `--no-proprio` (pixels-only BatchNorm 192-d) + `--wm-cam overhead` + curiosity SAC + `--consolidate-every 90 --consolidate-epochs 1` + `--buffer-frac 4`+ (cap >=32k). → sustained frac_rand ~0.12, best ~0.08 = LeWM-level, on jerky curiosity data. The lever is multi-epoch CONSOLIDATION (repeated exposure), NOT data smoothness — the entire prior smooth-data thread was a red herring. Overhead cam is essential (wrist degrades). Big buffer prevents drift. OPEN: reach (CEM goal-reaching) still 0 — the latent is local for the curiosity distribution but not CEM's goal-directed trajectories (train/test distribution gap).

## 2026-06-26 — REACH=0 WALL BROKEN: first nonzero LATENT goal-reaching via a controlled-distance goal probe. It was a RANGE problem, not a dead-end. (Planner-side AND multi-step-WM-side both exhausted first.)

**GOAL CLARIFICATION (read first): the objective is LATENT reachability** — `goal/reach_rate` = fraction of decisions within eps=2.0 z-units of the goal latent z*. The joint-space metrics added this session (`goal/success_rate_qpos`, `goal/qpos_dist`) are **DIAGNOSTIC-ONLY** sanity probes, **NOT the target**. Low qpos_succ alongside nonzero latent reach is a SUCCESS, not a failure — the overhead-cam latent's eps is intentionally looser than physical joint precision. Do not chase qpos.

**New instrumentation (this session, `src/train.py`):**
- `cem/min_cand_to_goal` + `cem/endpoint_disp` (computed at the FINAL CEM iter): the converged plan's BEST candidate→goal distance and mean endpoint displacement from z_now. THE key diagnostic — separates "WM reachable set too small" from "planner exploits WM error."
- `goal/success_rate_qpos` + `goal/qpos_dist` (wired the previously-dead `--goal-success-qpos-eps`): joint-space ground truth, DIAGNOSTIC ONLY (see goal clarification).
- `--cem-gamma` (discounted running/shaped cost `sum_h gamma^(H-1-h)||z_h-z*||^2`; 0=LeWM terminal-only), `--cem-replan-every` (execution stride; 1 = true receding-horizon MPC), `--cem-mppi-temp` (MPPI soft exp(-cost/temp)-weighted update over ALL candidates vs hard top-k elites), `--cem-min-std` (re-activated elite-std FLOOR; was a dead flag), `--h-fwd-override` (force the WM-rollout-training horizon, IGNORING the value a resume pins from the ckpt — see bug below), `--goal-select recent` (controlled-distance goal: the state k decisions ago, reach_gap ~ k*step_jump).

**STEP 1 — the PLANNER side is DEAD (exhaustive; all reach 0, dist floored ~14).** Tested on the frozen LeWM-local latent (cur_consol_oh_bn@3000, frac_rand~0.08): running-cost (`--cem-gamma 0.8`), true-MPC `--cem-replan-every 1`, MPPI soft update (temp 30 & 100), less-aggressive CEM (`--cem-iters 5 --cem-min-std 0.1`), near goals, action_max {1.0,0.31,0.15,0.10,0.05,0.03}. Mechanism nailed by the new diag: aggressive CEM reports `min_cand≈9` (plan BELIEVES it reaches ~9) while realized `dist≈14` — a ~5-unit FANTASY. Cumulative per-block motion is ~1.1–1.85× the gap (plenty), yet net progress ~0: MAGNITUDE is available, DIRECTION is the limiter; `cost_cv` ~0.05–0.21 (weak ranking signal). **The planner was exploiting WM rollout error.**

**MPPI vs CEM (first principles + tested):** they optimize the SAME corrupted objective (the WM rollout) — MPPI changes HOW you search, not WHAT. Confirmed: softer MPPI / less-aggressive CEM make `min_cand` HONEST (9→16, fantasy gone) but realized `dist` does NOT drop — the planner just moves less (`step_jump` 3.3→1.2). Pessimism = correct DIAGNOSIS, not a cure. (`cem/move_vs_gap` = z_term_spread/reach_gap sat at 0.11–0.47 in every prior CEM run, never near 1.)

**STEP 2 — the MODEL side is DEAD too.** Bug caught: a 1-step ckpt's `h_fwd` is restored on resume, silently overriding `--h-fwd-start`; added `--h-fwd-override`. With a genuine 5-step-rollout-trained WM (`cem_v3_hfwd5_fix`, h_fwd=5 confirmed): `min_cand` stayed pinned ~9 (still fantasy), `dist≈14.5`, reach 0. Why: multi-step latent prediction is inherently hard here — `wm/pred_vs_persist` 0.35 (1-step) → **0.61** (5-step). The 5-step rollout is never accurate enough to remove the exploitable error. You cannot make the rollout honest on this latent.

**STEP 3 — BREAKTHROUGH: controlled-distance "recent" goals → first nonzero LATENT reach.** reach=0 was a RANGE problem: archive goals sit ~14–20 away, beyond the planner's closing range. `--goal-select recent` makes the goal the state k decisions ago (gap ~ k*step_jump, controllable). Sweep k=1..15 × action_max {0.05,0.10}, goals refreshed every 25 steps (`cem_v4_recent_*`, `cem_v5_recent_*`, frozen latent):

| action_max | reach (LATENT, the goal) | steady reach_gap | qpos_dist (rad, diag-only) |
|---|---|---|---|
| **0.05 (fine)** | **0.15–0.20** | 5.8–6.0 | 0.63 |
| 0.10 (coarse) | 0.10 | 6.8–7.3 | 0.85 |

**reach 0.15–0.20 — NONZERO for the FIRST TIME in the entire reach investigation** (every prior run — all planner+model variants, incl near-goals@gap≈8 — was exactly 0).

**Findings:**
- It's a RANGE / PRECISION limit, NOT a dead-end. The planner closes nearby goals to a **~5–7-unit steady-state gap** (vs the ~14 floor for far goals) and latent-reaches ~15–20% when it dips inside eps=2.0. The far-goal reach=0 was simply goals out of the ~5–7-unit closing range.
- **Fine amplitude (0.05) clearly wins** — ~2× the reach of 0.10 (smaller per-step latent move lands inside the eps band more often).
- The "k" knob did NOT cleanly map distance — the pursuit equilibrium (~5–7) dominated regardless of k. So the result is a closing-PRECISION floor, not a sharp radius. (qpos_dist ~0.63 rad at fine amplitude — the arm gets roughly back to the k-ago pose; physical precision NOT a target.)

**VERDICT:** LATENT goal-reaching is ACHIEVED in the close-goal regime (reach 0.15–0.20). The blocker was never the planner or the model per se — it was that goals sat beyond the planner's ~5–7-unit closing precision. The whole planner/model search space is now closed; the lever is goal RANGE.

**RECIPE (latent reach, close-goal regime):** frozen LeWM-local latent (cur_consol_oh_bn@3000, `--no-proprio --wm-cam overhead`) + `--cem --goal-select recent --goal-future-k <small> --goal-update-every 25 --action-max 0.05` (fine amplitude essential) → latent reach ~0.15–0.20. Planner tuning (gamma/replan/mppi/min-std/iters) and multi-step WM (h_fwd) do NOT help — all exhausted.

**NEXT:** (a) reachable-radius goal CURRICULUM — goals inside ~5 units, grown outward, to TRAIN goal-reaching on top of this; (b) tighten the ~5–7-unit closing-PRECISION floor (so reach rises and the usable radius grows). qpos precision is explicitly NOT a target — latent reachability is.

**CURRICULUM (a) BUILT + RUN — the reachable radius GROWS via consolidation on directed-reachable data.** `--goal-curriculum`: a pure goal-RANGE schedule (grow `--goal-select recent` offset k by 1 every `--goal-curric-patience` steps once windowed LATENT `reach_rate >= --goal-curric-thresh`; NOTHING in the loss/architecture, qpos never involved; logs `goal/curric_k`). `cem_v6_curriculum` (frozen latent, amax 0.05, 4000 steps, start k=1, thresh 0.12, patience 150): k auto-climbed **1→6 in the first ~1050 steps** (reach ~0.17–0.21 through k=4), then **PLATEAUED at k=6 for ~2100 steps** (reach ~0.086 = the planner's precision limit at that radius), then **BROKE THROUGH to k=7 (step 3300) and k=8 (step 3600)**. So the reachable radius is NOT fixed — sitting in the reachable regime lets the consolidating predictor slowly extend it (k=6→8 over ~2500 steps). This is the directed-data-improves-reach mechanism, finally realized in the CONSTRAINED small-move reachable-goal regime (full-amplitude directed motion still destroys locality, per the prior entry). Caveat: k=7,8 undersampled (run ended ~3850); a longer run is needed to confirm continued growth. Levers for more reach: longer curriculum + tighter closing precision (a per-step finer final approach), both within the latent-reach objective.

**LONGER RUN (12k steps) — radius grows ~3x to k=19 and was STILL CLIMBING at the horizon (corrected from an earlier "plateau at 18" mis-read); driven by a measurably-improving PREDICTOR on a FROZEN encoder.** `cem_v7_curriculum_long` (same recipe, `--total-steps 12000 --goal-curric-max-k 30`): the reachable radius climbed **k=6 → 19** (~3x the early ~k5-6 closing limit) — timeline k=6@1050, 10@6000, 13@7800, 16@9000, 18@10050, **19@11850 (just 100 steps before the run ended)** — while LATENT reach held steady ~**0.09–0.10** across every well-sampled stage (k=6 n=47:0.094; k=9 n=35:0.091; k=12 n=23:0.096; k=17:0.102). The k=18 stretch (steps 10050→11850) was a TRANSIENT pause, NOT a ceiling — it then resumed to k=19, so the radius was **still climbing when the run hit its 12k step budget** (a longer run would very likely continue). **MECHANISM (corrected): the encoder was FROZEN the whole run (latent STATIONARY) — it was the PREDICTOR that improved, NOT the representation: `wm/pred_vs_persist` 1.22→0.096 early→late (worse-than-persistence → ~10x better than copy), `wm/pred_loss` ~13x lower, and every radius bump follows a pred_vs_persist drop. So the bootstrap is the forward MODEL sharpening on accumulating directed-reachable data within a FIXED latent geometry — the geometry that gives reach was set at step 0 and never changed.** Net session result: CEM latent goal-reaching went from reach=0 everywhere → reach ~0.1 with a reachable radius that self-extends k6→19 (still climbing) under the curriculum. To push further: lower the advance threshold, longer run, tighter per-step closing precision, or interleaved encoder thaw (let the representation co-adapt too) — all within the latent-reach objective; qpos remains diagnostic-only.

## 2026-06-26 (overnight autonomous campaign) — WRIST-CAM goal-reaching WORKS (reach ~0.18, radius k=17); interleaved low-duty thaw SURVIVES where 50% duty collapsed; from-scratch wrist locality FAILS

Goal (set by user): self-collecting, stable, **wrist-cam** latent goal-reaching with locality + a well-learned WM (qpos irrelevant). All runs live on W&B (curious-robot, entity ahilan-uc-berkeley...).

**WIN — wrist-cam goal-reaching works via the reachable-radius curriculum.** `cem_v8_wrist` (resume wrist curiosity latent `cur_consol_wrist_bn`@3000, FROZEN, `--wm-cam wrist --goal-select recent` curriculum, action_max 0.05): latent **reach 0.179, reachable radius k=17** (still climbing at step 5900/8000), `wm/pred_vs_persist` 0.056 (WM excellent). Nearly matches the OVERHEAD result (k=19) on the HARDER egocentric wrist cam — **the curriculum + frozen-curiosity-latent recipe TRANSFERS to wrist.** Key insight: goal-reaching does NOT require LeWM-level locality — the wrist latent is only ~0.20–0.24 local (vs overhead 0.13), but the reachable-radius curriculum compensates by keeping goals inside the closing radius. So locality helps but isn't the gate; the curriculum is.

**INTERLEAVED ENCODER ("make thaw work" task) — duty cycle is the lever.**
- **50% duty FAILS** (`cem_v8_thaw`, every 200 / dur 100): DESTROYS locality — frac_rand 0.05→**0.21**, reach collapsed to **0.009**. High-duty co-adaptation on directed motion shoves frames apart → SIGReg isotropizes → locality gone.
- **3.6% duty SURVIVES** (`cem_v9_thaw_lowduty`, every 700 / dur 25): frac_rand STABLE ~**0.20** (does NOT explode), reach 0.049, radius climbing to k=6, but `pred_vs_persist` 0.708 (thaw disrupts the predictor early). So rare/gentle thaw is survivable — interleaving CAN be made to not collapse — but it's **not yet BEATING pure frozen** (0.20 vs 0.13, reach 0.049 vs ~0.1). Next variants to actually HELP: even lower duty, reduced encoder-LR during thaw, or thaw on CURIOSITY (small-move/local) data rather than directed CEM trajectories.

**NEGATIVE — from-scratch wrist locality FAILS.** `wrist_loc_amax03/10` (curiosity + consolidation + big buffer 50k, encoder TRAINS, wrist): frac_rand DEGRADED to **0.46** (amax 1.0) / 0.30→ (amax 0.3) — WORSE than the existing `cur_consol_wrist_bn` (0.24). Rebuilding a local wrist encoder from scratch with the overhead recipe does not work; the existing wrist latent + curriculum is the better path (egocentric pixel-jerk is the obstacle, per prior wrist entries).

**NET:** wrist-cam latent goal-reaching is ACHIEVED (reach ~0.18, radius k=17, self-collected curiosity latent, WM learning well). The reachable-radius curriculum is the key enabler and transfers across cameras. Interleaved encoder co-adaptation is survivable at low duty but needs tuning to beat frozen. New flags this session: `--encoder-thaw-every`/`--encoder-thaw-dur`.

## 2026-06-26 (continued, ~6h autonomous) — wrist radius pushed to k=23; INTERLEAVED resolved (SIGReg-isotropy was the culprit, but the bootstrap needs a STATIONARY latent); EFFICIENCY = consolidation gates the bootstrap

**PERFORMANCE — wrist radius pushed to k=23.** `cem_wrist_long` (frozen wrist latent + reachable-radius curriculum, 16k steps, max-k 40): radius climbed to **k=23** (beats overhead's 19 and earlier wrist's 22), reach ~0.12–0.16 mid-run. Wrist-cam goal-reaching is solidly the achieved goal.

**INTERLEAVED ENCODER — fully resolved (the "make thaw work" task).** Tried, in order: 50% full-β thaw (DESTROYS locality, frac_rand 0.21, reach 0.009); 3.6% full-β and gentle-LR (mildly hurt); then the mechanism-targeted fix. **Diagnosis CONFIRMED: SIGReg's unit-variance isotropy is what scatters co-adapting frames.** With β dropped 0.09→0.02 *only in thaw windows* (`--encoder-thaw-beta`), even **50% duty no longer collapses** (`cem_v12_hiduty_lowbeta`: frac_rand 0.224 stable, z_std healthy — vs the same duty at full β which destroyed it). **BUT reduced-β thaw still does NOT beat frozen** — both `cem_v11`/`cem_v12` stall at **k=4** while frozen rockets to k=14→23 at the same steps. **ROOT CAUSE: the radius bootstrap REQUIRES a STATIONARY latent** — the predictor sharpens against a FIXED geometry (the v7 `pred_vs_persist` 1.22→0.10 mechanism). Thawing the encoder, even non-destructively, keeps the latent moving so the predictor can never sharpen → the radius stalls. So **interleaving fundamentally conflicts with the bootstrap; FROZEN is correct.** Net: the SIGReg-β fix makes thaw *non-destructive* (a real mechanistic result), but co-adaptation can't beat frozen because the bootstrap needs stationarity. New flags: `--encoder-thaw-lr`, `--encoder-thaw-beta`.

**EFFICIENCY — consolidation passes are the engine, not raw steps.** `cem_wrist_eff` (consolidate-every 140 vs 70 = half the WM grad cost): 1.6× faster per step (14 vs 9 sps) but the radius STALLED at k=3 (vs frozen k=17 at the same step). The radius bootstrap is gated by *consolidation passes* (predictor sharpening), so halving consolidation cripples it — cheaper-per-step is NOT cheaper-per-radius. (Testing the opposite — consolidate-every 40, MORE bootstrap — in `cem_wrist_moreconsol`.) **Practical efficiency recipe: keep consolidate-every ~70; the WM-improvement IS the lever, don't cut it.**

**THAW VERDICT (banked negative, with mechanism):** interleaved encoder co-adaptation during CEM goal-reaching cannot beat frozen — not because it breaks locality (the β fix solves that) but because the radius bootstrap needs a stationary latent. The only path that could let the representation improve is OFFLINE/separate re-training on diverse data (LeWM-style), not interleaving during reach.

## 2026-06-26 — CAMPAIGN SUMMARY (wrist-cam goal-reaching) + BEST RECIPE

**Goal achieved:** self-collecting (curiosity), stable, **wrist-cam** latent goal-reaching with locality + a well-learned WM (qpos irrelevant). All runs on W&B.

**Headline numbers (wrist cam, frozen curiosity latent `cur_consol_wrist_bn`@3000 + reachable-radius curriculum):**
- **Max radius:** k=**23** (uncapped curriculum) — beats overhead (k=19); but reach collapses to ~0.05 at that ceiling.
- **Best OPERATING POINT:** cap the radius at **k≈12** → sustained latent **reach ~0.20** (CONFIRMED at the cap, n=39; ~6× the max-radius run's 0.034). Moderate radius >> max radius for usable goal-reaching.
- WM learns well throughout (pred_vs_persist 0.05–0.10 on the frozen latent).

**REACH-VS-RADIUS trade-off:** the curriculum grows the radius but reach drops as goals get harder; capping the radius trades range for a much higher sustained success rate. Pick the cap for the use case.

**Resolved sub-questions:**
- **Camera:** wrist works (k=23 / reach 0.16 capped) despite worse locality (~0.20–0.24 vs overhead 0.13) — the curriculum compensates; locality helps but isn't the gate.
- **Interleaved encoder:** FROZEN is correct. Thaw during CEM breaks locality via SIGReg isotropy (fixed by `--encoder-thaw-beta`), but even non-destructive thaw stalls the radius because the predictor bootstrap needs a STATIONARY latent. Representation improvement must be OFFLINE, not interleaved.
- **Efficiency:** consolidate-every **~70** is the sweet spot — cheaper (140) cripples the radius bootstrap, richer (40) saturates (no gain). The WM-improvement (consolidation passes) IS the engine.
- **From-scratch wrist locality:** fails (frac_rand 0.46) — use the existing curiosity wrist latent.

**BEST RECIPE (wrist goal-reaching):** `--cem --resume-name cur_consol_wrist_bn --resume-step 3000 --freeze-encoder --no-proprio --wm-cam wrist --sigreg-pertimestep --consolidate-every 70 --action-max 0.05 --goal-select recent --goal-update-every 25 --goal-curriculum --goal-curric-start 1 --goal-curric-thresh 0.12 --goal-curric-patience 150 --goal-curric-max-k <12 for high reach | 40 for max range>`. New flags this campaign: `--goal-select recent`, `--goal-curriculum` (+ knobs), `--cem-gamma/replan-every/mppi-temp/min-std`, `--h-fwd-override`, `--encoder-thaw-every/dur/lr/beta`, `cem/min_cand_to_goal`+`endpoint_disp`+qpos diagnostics.

## 2026-06-27 — LeWM-faithful predictor (2.2×) + encoder rebuild → predictor size is a WASH for build locality (corrected a mis-read)

Bumped the WM predictor to LeWM's `lewm.yaml` dims (heads 8→16, dim_head 32→64, mlp_dim 1024→2048; depth 6 unchanged): **predictor 4.89M→10.79M (2.2×), total WM 12.13M→18.03M.** `pred_dims_from_args()` makes loading size-safe — a checkpoint's predictor size is reconstructed from its saved args (old ckpts fall back to 8/32/1024), so all 7 eval/daemon load sites + `train.py` resume load ANY ckpt; resume is tolerant (encoder/pred_proj/action_encoder always match → frozen-encoder resume intact; only a resized predictor re-inits, warned). New flags `--wm-pred-{depth,heads,dim-head,mlp-dim}`.

**Rebuilt the wrist curiosity latent with the big predictor** (`cur_consol_wrist_bn_v2`, W&B `hap8gesl`): `step_jump_frac_rand` (on curiosity data) climbed 0.25→0.78 — looked anti-local, alarming. **The alarm was a mismatched-step MIS-READ** (compared v2@7500 to a misremembered ~0.2 baseline). The ORIGINAL small-predictor build (`astjf6pc`) ALSO climbed 0.247→0.595 by step 3000 then crashed; at MATCHED steps big≈small. ⇒ **predictor size does NOT affect build locality** — both SIGReg-isotropize (`z_std→1`) on jerky `action_max=0.3` curiosity data; the anti-local drift is inherent to the data+regularizer, not the predictor. Rebuilt small-pred as `cur_consol_wrist_bn_v3` (`xk8oyc2o`, byte-identical recipe to the original), froze `@3000` for the reach work. **KEY: the build's curiosity-data frac_rand (~0.6) is NOT the reach-relevant metric** — the frozen encoder is frac_rand ~0.22 on the small-move (`action_max=0.05`) CEM trajectory. The big predictor is used at REACH (re-init via tolerant load on the frozen v3 latent), not the build.

## 2026-06-27 — NEW `--goal-select highmse_under_d`: reach the HIGH-MSE states within a growing latent-distance budget → self-extends to d≈8, then ceilings

The objective reframed (user): reach the **high-MSE / surprising** states (not the easy `recent` self-states), made reachable by a curriculum. New mode `highmse_under_d`: each goal refresh samples `--goal-cand-n` within-episode buffer transitions, **re-scores their CURRENT-WM one-step MSE** (`score_obs_mse`), and per env pursues the goal whose latent sits within the curriculum distance budget `curric_d` of `z_now` — `curric_d` grows (`--goal-curric-d-{start,step,max}`) when windowed reach clears `--goal-curric-thresh` (the latent-distance analog of `curric_k`). Frozen encoder ⇒ `‖z_cand−z_now‖` is a stable reachability metric; reaching a surprising state lowers its MSE so the frontier recedes (Go-Explore-via-reachability).

- **v1 (absolute-MAX MSE, thresh 0.12, d_start 1): STUCK at d=6, reach ~0.03.** The single highest-MSE goal is the hardest-to-predict ⇒ least reachable (deadlock).
- **Softened → broke the stall:** `--goal-highmse-frac 0.25` (sample uniformly from the **top-25%** by MSE, not the max) + thresh 0.05 + **`d_start=3`** (start with the closest, easiest goals; below the ~4.3 step_jump the nearest-fallback just pursues the single nearest state). `cem_highmse_d_v3` (`8xm0alj5`): radius **self-extends d=3→8** (advances at reach ~0.05–0.10), then **PLATEAUS at d=8** — reach falls to ~0.025 (< thresh), predictor fully saturated (`pvp 0.027`). The d≈8 ceiling.

## 2026-06-27 — time-phased CEM-directed encoder co-train (`--cotrain-every`) does NOT break d=8

`--cotrain-every N`: every N steps PAUSE the curriculum, THAW the encoder, consolidate `--cotrain-epochs` over the accumulated **CEM-directed** buffer (low `--cotrain-beta 0.02` so isotropy doesn't scatter directed frames; low `--cotrain-lr` to nudge), then RE-FREEZE — the OFFLINE cycle the campaign concluded was necessary (interleaved thaw stalls the bootstrap). Goal: make the latent local under DIRECTED motion (close the train/test gap, since the frozen latent was local for curiosity but `frac_rand ~0.27` on CEM trajectories).
- **v1 (`q0l7mchc`, every 2000, 3 epochs, lr 2e-5):** phase-1 gave a real TRANSIENT gain (frac_rand 0.258→0.205, dist 12.2→8.0, reach 0.021→0.045) — the mechanism works — but it **did not compound**: oscillated back near baseline, `curric_d` stayed 8.
- **v2 (`02xyp5ca`, every 4000, 1 epoch, lr 1e-5):** gentler/spaced for stability — too weak, the small per-phase gain washed out, `curric_d` never moved off 8 across 2 phases + full re-extend windows.
⇒ **co-train (gentle OR aggressive) does not break d≈8.** Banked negative.

## 2026-06-27 — ★ FAILURE ISOLATED: the CEM planner EXPLOITS the world model (it was NEVER the representation). Horizon sweep. ★

The d≈8 ceiling held across EVERY representation-side lever (no-cotrain, aggressive cotrain, gentle cotrain). So isolated the planner with a controlled **CEM-horizon sweep**: resume the FROZEN warm WM (`cem_highmse_cotrain_v2/ckpt_0010000` — frozen encoder + warm big predictor), consolidation OFF (WM held fixed), fixed high-MSE goals at d=8, vary ONLY `--cem-horizon ∈ {1,2,5}` (`hsweep_h1/2/5`). Averaged (last 10 pts):

| H | reach_gap | min_cand (believes) | realized dist | **fantasy** = dist−min_cand | **move/gap** | **reach** |
|---|---|---|---|---|---|---|
| **1** | 6.98 | 2.83 | 7.60 | **4.76** | 0.48 | **0.130** |
| 2 | 6.63 | 1.48 | 8.29 | 6.81 | 0.67 | 0.058 |
| 5 | 6.58 | 1.47 | 8.09 | 6.63 | 0.96 | 0.034 |

**The failure is PLANNER MODEL-EXPLOITATION, not multi-step rollout error (my hypothesis), not the encoder.** Two decisive signals:
1. **The fantasy gap is already ~4.76 at horizon 1** (not ~0) and barely grows to ~6.7 at H=5 — so it's NOT compounding rollout error. Even the accurate 1-step WM (`pvp 0.022`) gets exploited: the CEM scores 300 candidate actions and the **argmin picks exactly the action where the WM is optimistically wrong** (predicts reaching `z*`), so the plan "believes" it lands ~1.5 away while the arm ends ~8 away.
2. **The inversion `move/gap` ↑ → `reach` ↓.** H=5 has the LARGEST reachable set (move/gap 0.96, can physically span the gap) yet the WORST reach (0.034); H=1 has the SMALLEST reachable set (0.48, one step can't span the gap) yet the BEST reach (0.130). **More planning power → worse reaching** — the unambiguous fingerprint of model exploitation (a longer horizon / bigger candidate set just hands the optimizer more model error to plan into).

This retroactively explains the whole arc: **(a)** why high-MSE goals are worst — high MSE = larger WM error = MORE optimism to exploit; **(b)** why co-train never helped — the WM is already accurate on average, the problem is exploitation not accuracy, so no encoder/data change fixes it; **(c)** why the prior campaign's planner-softening gave "honest `min_cand` but no better reach" — softening removes exploitation AND closing power together. The fix is decisively planner-side: shorter horizon, pessimism (ensemble-disagreement penalty on the CEM cost), or trust-region.

## 2026-06-27 — ★ d=8 CEILING BROKEN by `--cem-horizon 1`: it was a PLANNER ARTIFACT all along ★

Re-ran the `highmse_under_d` curriculum IDENTICALLY to the d=8 run (frozen `cur_consol_wrist_bn_v3@3000` + big predictor, softened top-25% MSE, d_start 3, thresh 0.05) but with **`--cem-horizon 1 --cem-replan-every 1`** (true MPC) — a controlled one-flag test of the sweep's measured 4× lever. `cem_highmse_h1` (`jl7wfyqn`): the radius **broke past the d=8 ceiling and kept climbing — d-timeline 8@2600 → 9@2750 → 10@3200 → 11@3350** (vs H=5's HARD plateau at 8), reach ~0.05–0.13, `pvp 0.038`. **So d≈8 was a planner artifact of the H=5 default (= maximal model-exploitation), not a fundamental limit of the latent or the goal distribution.** The multi-week "it's the representation" thread resolves cleanly: the planner was exploiting the world model the whole time, and constraining its search (horizon 1) recovers honest closing and self-extends the reachable radius. **FINAL: the H=1 radius climbed d=3→12 (8@2550, 10@3150, 12@3450) — a +50% reachable radius vs H=5's d=8 — then ceilinged at d≈12** (reach fell below the 0.05 thresh; usable radius reach≥0.05 is ~d10–11, reach ~0.06–0.13 at the advances; `pvp 0.030`). So H=1 RAISED the planner-set ceiling 8→12 but didn't remove it — model-exploitation still bites at larger d (fantasy gap back to ~8 at d=12). d≈12 ≈ 60% of the random-pair latent diameter (≈qpos 0.85 rad, genuinely distinct arm configs) — a real, useful reachable radius. Run stopped at d=12 (converged).

**OPEN / NEXT:** (a) let `cem_highmse_h1` run out to find the H=1 reachable-radius ceiling (where reach finally drops < thresh) — that's the achievable radius; (b) **sweep `--goal-highmse-frac`** (e.g. 0.1 / 0.25 / 0.5) now that H=1 makes the planner far more capable — can crank the MSE selection MORE aggressive (smaller frac → more surprising goals) while staying reachable, directly serving the reach-high-MSE objective; (c) **pessimism** (predictor ensemble + disagreement penalty subtracted from the CEM cost) to remove the exploitable optimism at any horizon. **Good target distance:** the latent random-pair ceiling is √(2·192)≈19.6, so d≈10–14 (~50–70%, qpos ~0.7–1.0 rad — genuinely different arm configs) at reach ≥ ~0.1 is the meaningful operating band; don't chase d→19.6 (≈ a random/unsafe state). New flags this session: `--wm-pred-*`, `--goal-select highmse_under_d`, `--goal-curric-d-*`, `--goal-cand-n`, `--goal-highmse-frac`, `--cotrain-every/epochs/lr/beta`.

## 2026-06-29 — NESTED MSE-difficulty curriculum WITHIN each d (`--goal-mse-curric`): rising percentile band (low-MSE → high-MSE). Roadmap set: MSE-curriculum → no-frozen co-train → bigger predictor

Roadmap agreed with user, in order: **(1)** an MSE curriculum *within* the d curriculum — within a fixed d, master the easy states first and walk **all the way up** to the highest-MSE (surprising) states before extending d; **(2)** drop the frozen encoder and **co-train** (both online-with-safeguards AND phased-offline — and the offline phase must **fully** consolidate, per the standing "the predictor needs full training, not a couple steps" lesson, unlike the weak 1–3-epoch `--cotrain-every` banked-negatives); **(3)** a **bigger predictor**. Co-train and bigger-predictor are deferred until the MSE curriculum is characterised. Crucial reframing context: the prior "frozen is correct / bootstrap needs a stationary latent" negative was concluded under **H=5 (the exploitable planner)** — the same confound that faked the d=8 ceiling — so re-testing co-train under **H=1** is well-motivated, not a re-run.

**New mechanism — `--goal-mse-curric` (this is experiment 1; frozen encoder, H=1, single-variable change vs the d=12 `cem_highmse_h1` run).** Within a fixed d, the goal is sampled from a percentile BAND (width `--goal-mse-band`, default 0.2) of the **under-d MSE distribution**, sorted ascending, centred on a target percentile `curric_pctl` that **RISES** from `--goal-mse-pctl-start` (0.0 = LOW-MSE, easy, WM-predictable → reliably reachable) to `--goal-mse-pctl-max` (1.0 = HIGH-MSE, surprising = the objective) by `--goal-mse-pctl-step` (0.2) each time windowed reach clears `--goal-curric-thresh`. d grows only once the percentile tops out (the hardest under-d states mastered), then the percentile **resets** for the new d. Smoke-confirmed firing: `pctl 0.0→0.2→0.4→0.6→0.8→1.0 (d held 3) → d=4 (pctl reset 0)`. Logs `goal/curric_mse_pctl`.

**Design correction (important):** the first cut annealed `--goal-highmse-frac` (the *top*-fraction-by-MSE knob) DOWN within d (1.0=any-state → 0.1=highest-MSE-only). That ramps *aggressiveness toward high-MSE* but **never starts at low-MSE** (frac=1.0 = uniform over ALL under-d, a low/high mix) — it can't express "easy first, then harder," and throws some least-reachable hard goals in from step 0 (the deadlock the d=8 v1 hit). Replaced with the **rising-percentile band** above, which is a true low→high difficulty ramp and the faithful reading of the objective. Static top-frac (`--goal-highmse-frac`) is kept as the fallback when `--goal-mse-curric` is off (with a `min(n_under, …)` clamp on k_top). Removed the interim `--goal-frac-*` flags.

**Verification before launch:** adversarial review workflow (3 dims × verify) on the diff → 0 correctness bugs in the intended path; 5 confirmed low/nit findings, all misconfiguration footguns (decay≥1 deadlock, k_top>N topk crash, floor guards) — folded into the percentile rewrite as a fail-fast `SystemExit` (`0 ≤ pctl_start < pctl_max ≤ 1`, `step>0`, `0<band≤1`) and the k_top clamp. Two smokes (frozen H=1, HF resume) confirmed firing + no crash.

**RUN LAUNCHED — `cem_msecur_h1` (W&B `iqkae24b`).** Recipe = the proven frozen-encoder H=1 best recipe + `--goal-mse-curric --goal-mse-pctl-start 0 --goal-mse-pctl-max 1 --goal-mse-pctl-step 0.2 --goal-mse-band 0.2`, d_start 3, d_max 22, thresh 0.05, 24000 steps, n_envs 8. The only changed variable vs `cem_highmse_h1` is the nested MSE curriculum. Watching: does walking low→high MSE within each d (a) raise sustained reach at fixed d, and/or (b) push the d-ceiling past 12? New flags: `--goal-mse-curric`, `--goal-mse-pctl-{start,max,step}`, `--goal-mse-band`.

## 2026-06-30 — ★ RESULT: the nested MSE curriculum BREAKS the d=12 ceiling → d=18 (+6 reachable radius). Verdict: it HELPS the radius (train-time-immune ceiling-break confirmed). ★

`cem_msecur_h1` (W&B `iqkae24b`) climbed the reachable radius to **d=18 (+6 past the static-frac baseline's d=12)** and was still inching up when stopped (~step 20650/24000) to free the GPU for the co-train experiment. Per-d (mean over all logged windows; `all`=all-pctl, `hi`=hardest band pctl≥0.8, `fant`=CEM fantasy gap reach_gap−min_cand, `pvp`=pred_vs_persist):

```
 d :  3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18
all: .03  .07  .06  .11  .13  .11  .14  .19  .20  .22  .14  .21  .19  .18  .23  .24
 hi: .03  .08  .10  .11  .11  .05  .08  .11  .12 .149  .07  .12  .07  .07  .14  .20
fant:-.4  2.7  3.4  3.6  3.9  4.3  4.7  5.3  5.5  5.9  5.9  6.4  6.5  6.8  6.9  8.1
```

**Verdict (calibrated, flipped to FAVORABLE on the decisive test).** Early in the run the read was "too-early / leaning-no on radius": reach rising with d is fatally confounded by training time (corr(d,step)=0.976), and the fantasy gap rises in msecur too — so within-run "wins" prove nothing. The one **train-time-immune** test (adversarial-workflow-defined): does msecur break past the baseline's *converged* d=12 ceiling? **It did — cleanly, to d=18.** Decisive evidence: (1) at MATCHED d=12, on the hardest goals, msecur reaches **0.149 vs the baseline's ~0.06** (the exact point that stalled the baseline); (2) the matched-d **fantasy gap is materially lower** (msecur ~5.9 at d=12 vs baseline ~8) — a per-d planner property independent of training time; (3) the fantasy gap only reaches the baseline's terminal ~8 at **d=18**, i.e. the curriculum pushed the *same* exploitation wall **+6 levels further out**. So the nested low→high MSE curriculum genuinely extends the reachable radius. NOTE the per-d MEAN reach is confound-B-inflated by easy/low-pctl windows — always read the hardest-band (pctl≥0.8) column; absolute hard-goal reach stayed modest ~0.07–0.20 throughout (radius grew, reach% did not).

**CAVEATS (not yet "proven in general"):** single seed; small autocorrelated n per (d,pctl) bin; the train-time-immunity assumes the baseline genuinely converged at d=12 (the lower fantasy gap corroborates). **→ TODO: run 2 MORE SEEDS of `cem_msecur_h1` to convergence and report hard-band reach mean±across seeds at matched d.** Also worth: re-pull baseline `jl7wfyqn` to confirm it truly plateaued (not just ran out of steps) at d=12.

**Mechanism findings this session (all data-verified):**
- **Fantasy gap = DIRECTIONAL optimism, not magnitude.** The CEM argmin over 300 candidates pins its believed `min_cand` at a ~constant ~2 "optimism floor" (the WM's exploitable error scale, independent of goal distance) while realized `reach_gap` grows with d → the gap widens **with distance, NOT with goal hardness** (fantasy is flat ~4.4 across the MSE percentile). And it only switches on once the predictor is ACCURATE (no fantasy at d=3/pvp=0.11; appears when pvp→0.03): **a confident WM ENABLES the fantasy** — "make the WM better" makes it worse, and low pvp is decoupled from the closed-loop exploitation.
- **Reach% is capped ~0.10–0.20 (90% at eps=2.0 is infeasible).** reach = arrive×stay, both walled. STAY: per-step latent jump ~5.4 > 2·eps=4.0, so a *moving* planner overshoots the eps=2.0 ball and cannot dwell. ARRIVE: the fantasy pins steady-state closing at ~5–7. Best reach EVER ~0.20. The deep blocker is representational (directed-motion locality, frac_rand ~0.22) + the dwell geometry — no planner-side fix alone reaches 90%.

**NEXT / OPEN (write-down of the agreed plan):**
1. **Co-train (②) — RUNNING NOW.** `cem_msecur_cotrain`: the same MSE-curriculum + H=1 recipe + PHASED-OFFLINE co-train (`--cotrain-every 2000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02`) — full per-phase consolidation to convergence (the "predictor needs full training, not a couple steps" fix vs the weak banked-negative cotrain runs). Online-safeguarded variant (`cem_msecur_thaw`, continuous gentle thaw) is the staged follow-up. Then ③ bigger predictor.
2. **2 more seeds** of the curriculum run (above) for a defensible radius claim.
3. **Reach-% / fantasy-gap remediation — DEFERRED, to do later (in order):** (a) **geometric probe** — offline, from frozen states at gap~3–4 enumerate tiny real actions, encode actual next states, min‖z_next−z*‖; if <2.0 the blocker is purely the planner, if >2.0 it's a representational floor (decides where to invest; high value-of-information); (b) **WM-free arrival HOLD** (a=0 when ‖z−z*‖<~1.25·eps) + **proximity-gated terminal action-shrink** (drop action_max in the last units so step_jump<eps) — near-free, fixes the STAY wall; (c) **local Jacobian trust region** on the CEM cost (NO ensemble, per user): fit dz≈J·a+b from nearby buffer transitions each replan, hard-reject candidates whose WM move violates it — catches the DIRECTIONAL fantasy (tests the transition edge, not the endpoint), and provably doesn't re-soften (survivors keep full closing cost). Self-validates via an offline unit test (real transition → residual~0; synthetic fantasy move → rejected).
4. **NEW DIRECTION — instill goal-reaching into a learned POLICY, and use the policy to SEED the CEM.** Distill the (curriculum-driven) reaching behavior into a goal-conditioned policy (e.g. the unused `--goal-explore` SAC actor, HER-trained on the reach reward / on the CEM's successful near-goal transitions), then **use that policy as the CEM's proposal / initialization** instead of planning from a zero-mean prior — a learned warm-start should both close faster and, being model-free, sidestep the WM-exploitation on the final approach. (Two birds: a deployable policy + a less-exploitable planner init.)

## 2026-07-02 — NEXT PLAN: staged from-scratch pipeline (A build → B verify → C wake/sleep co-train) — wrist-only, CEM-only (no policy), d=4 fast-iteration ceiling

**Context / decision.** The frozen-vs-scratch contrast is clean: every run that ever reached rode a pre-built frozen local latent (`cur_consol_wrist_bn_v3@3000` → d=12 baseline, d=18 with the MSE curriculum); every from-scratch run failed the same way (frac_rand 0.03→0.30 = de-collapsing anti-local, fake d3 reach that evaporated when SIGReg de-collapsed the latent, stall ~d5). The planner + curriculum machinery is proven GIVEN a local latent; the open problem is building that locality online. The failed scratch runs did everything at once — encoder drifting, curriculum advancing on a moving meter, planner co-adapting. This plan decomposes it into gated stages. **User decisions:** NO policy anywhere (CEM is the only actor, spec-consistent — the 06-30 policy-seeded-CEM direction is OUT for this track); wrist camera for ALL stages; NO EMA target encoder (phased freeze/thaw only); d-ceiling capped at 4 for fast iteration cycles; nested MSE curriculum on by default (code change below).

**The cycle (steady state = Stage C):** move (CEM, encoder stop-grad) → short pause every 70 steps (predictor-only consolidation burst) → every N steps a long SLEEP (encoder+predictor co-train together over the full maintained buffer, low beta) → re-freeze → wake, move again. Stages A/B exist to boot into that cycle from a random encoder: A is the same loop with the stop-grad removed (latent must form before it's worth protecting), B is one frozen lap to prove the latent works before adding C's moving parts.

**Stage A — build the latent (encoder TRAINS, no reach expectations, ~2–3k steps).** CEM-curiosity: goals = highest-current-WM-MSE archive states (`--goal-select mse`), so exploration pressure is novelty-through-the-planner, no actor. Consolidation bursts train encoder+predictor together. The bet (untested cell): wrist-from-scratch at FINE amplitude — prior wrist-scratch failures were amax 0.3/1.0 (frac_rand 0.30–0.6, egocentric pixel-jerk); `dir_fine_consol` proved directed fine-amplitude data sustains frac_rand 0.070 WITH the encoder training, and the v3 latent reads 0.22 on amax-0.05 trajectories.
Flags: `--cem --cem-horizon 1 --cem-replan-every 1 --goal-select mse --wm-cam wrist --no-proprio --action-max 0.05 --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 4 --sigreg-pertimestep`.
**GATE → B:** sustained `encoder/step_jump_frac_rand ≤ ~0.25` (v3-level) + healthy z_std ~1 + rank not collapsed. Fallbacks: frac_rand won't drop → `--goal-select recent` small k (tighter, more repetitive data consolidates easier); collapse instead (z_std→0, motion too tiny to matter) → `--action-max 0.1`; both fail → resume `cur_consol_wrist_bn_v3@3000` (still wrist-built) and proceed.

**Stage B — freeze + d=4 verdict (~1k steps; v3 climbed ~+1 d per ~500 steps).** Resume the Stage A ckpt with `--freeze-encoder` (full stop-grad: params excluded from AdamW + eval(), latent stationary), proven H=1 recipe + the now-default nested MSE curriculum: `--goal-select highmse_under_d --goal-curriculum --goal-curric-d-start 3 --goal-curric-d-max 4 --goal-curric-thresh 0.05` (+ big predictor via tolerant load, as in the v3 reach runs).
**GATE → C:** reach ≥ 0.05 at d=4 **with the qpos ground truth confirming** — `goal/qpos_dist` falling, `goal/success_rate_qpos > 0`. NB "curriculum hit d=4" alone is NOT the gate: the failed scratch runs passed d4 on fake latent reaches and stalled ~d5; qpos can't be faked by a collapsed metric.

**Stage C — wake/sleep co-train at the d=4 ceiling.** Stage B + phased-offline co-train, full per-phase consolidation to convergence (the standing "predictor needs full training, not a couple of epochs" lesson): `--cotrain-every 2000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02` (the `cem_msecur_cotrain` settings). **Success = reach at d=4 HOLDS through thaw cycles** and directed-probe frac_rand doesn't degrade phase-over-phase. Only then raise `--goal-curric-d-max` toward the meaningful d≈10–14 band. Optional hardening if phases regress (not yet built): per-phase rollback gate (snapshot encoder before thaw, revert + halve cotrain-lr if the probe frac_rand degrades) and latent-unit recalibration of curric_d/eps by the pre/post step_jump ratio (goals themselves are drift-immune — z* re-encodes every step — but the scalar thresholds aren't).

**Why this should differ from the failed scratch runs:** A has no reach curriculum to poison (novelty goals; latent distances never gate anything); B runs entirely in the proven frozen regime; C thaws only in consolidated bursts between frozen stretches. Each stage's gate is immune to that stage's known failure mode (frac_rand for A, qpos for B, hold-through-thaw for C). The whole loop at d=4 is fast — roughly one A + one B per iteration before touching C.

**Code change this session:** `--goal-mse-curric` DEFAULT ON (`argparse.BooleanOptionalAction`; disable with `--no-goal-mse-curric`) — the nested low→high MSE curriculum is the d=12→18 ceiling-break lever (06-30) and is now standard in every `highmse_under_d` curriculum run, including Stages B/C above.
