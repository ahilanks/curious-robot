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

**CANONICAL LAUNCH COMMANDS (added 07-02 after a recipe-drift incident — ALWAYS launch stages from these, not from the prose flag lists above).** The first B attempts this session silently dropped `--goal-update-every 25`, `--cem-init-std 0.3`, `--goal-curric-patience 150`, `--buffer-frac 3` and used `--resume-name` (which carries the ckpt's stale goal archive) instead of `--init-ckpt` (weights-only) — because the plan paragraphs say "proven H=1 recipe" without restating those flags (they live in the 06-26 BEST RECIPE line + `jl7wfyqn`'s W&B config). LESSON: before launching any "same as run X" experiment, diff your args against X's actual W&B config.

Stage A (from scratch; swap `--goal-select mse` ↔ `recent --goal-future-k 3` per fallback):
```
python src/train.py --name <name> --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 \
  --goal-select mse --goal-update-every 25 --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 3 --sigreg-pertimestep \
  --total-steps 3000 --n-envs 8 --env-threads 8
```
Stage B (frozen verdict lap; `--init-ckpt` NOT `--resume-name`):
```
python src/train.py --name <name> --init-ckpt <path/to/ckpt_XXXXXXX.pt> --freeze-encoder \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 \
  --goal-select highmse_under_d --goal-curriculum --goal-curric-d-start 3 --goal-curric-d-max 4 \
  --goal-curric-thresh 0.05 --goal-curric-patience 150 --goal-update-every 25 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 3 --sigreg-pertimestep \
  --total-steps 1500 --n-envs 8 --env-threads 8
```
Stage C = Stage B + `--cotrain-every 2000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02` (and/or the new event-triggered `--cotrain-frac-thresh 0.3 --cotrain-frac-cooldown 400`), `--goal-curric-d-max` raised once d=4 holds through thaws.

## 2026-07-03 — ★ FROM-SCRATCH STAGE A+B PASSES (d=4 pctl-ladder complete, reach 0.108, qpos-confirmed) — after a RECIPE-DRIFT incident that faked 3 "failures" ★

**THE ISSUE (read this before ever concluding "the latent is bad").** Executing the 07-02 staged plan, Stage B was launched from the plan paragraph's flag list; the paragraph says "proven H=1 recipe" WITHOUT restating flags that live only in the 06-26 BEST RECIPE line and the reference runs' W&B configs. The launched B runs therefore silently ran with: goals refreshed every **200** steps (the `goal_update_every<=0 → max_episode_steps` coercion; proven recipe = **25**, so bad goals are abandoned after 25 steps instead of chased for a full episode), CEM search std **1.0** (proven = **0.3**, gentler candidate set = less WM-exploitation), patience 200 (proven 150), buffer_frac 4 (proven 3), and `--resume-name` (which restores the ckpt's STALE goal archive) instead of `--init-ckpt` (weights-only). Consequences, all initially mis-attributed to the from-scratch latent:
- `scratch_b1` (`4v98tkdl`, frozen A1 latent): reach 0.000 over 26 windows, pctl pinned (3,0.0), dist_goal 12–15, min_cand →7–12, frac_rand 0.16→0.51 under pursuit. Mis-diagnosed as "scratch latent non-local under directed motion."
- `scratch_b1_trig` (`7ywkklee`): the new frac_rand-triggered co-train fired correctly (0.341@400, 0.695@800) but full-consolidation sleeps over the far-goal-chase buffer made locality WORSE (0.34→0.70 post-sleep). Real sub-lesson: **consolidation amplifies whatever the buffer holds** — sleeps are only useful on healthy pursuit data.
- `scratch_b2` (`cxoagaxr`, A2 latent): identical failure → wrongly escalated to the terminal v3 fallback.
- Even `scratch_b3_v3` (`dai66hw9`, the PROVEN v3 latent) failed under the drifted recipe (dist_goal 21.7!) — this run is what exposed the drift: a **W&B config diff vs `jl7wfyqn`** (the healthy d3→12 reference) listed every divergence in one shot. THE LESSON (now also in memory): before launching any "same as run X" experiment, diff planned argv against X's ACTUAL config (`wandb.Api().run(...).config`); prose recipes drift.
- Metric-dive side-result (partially confounded by init_std but directionally real): far initial goals (dist ~12 ≫ d=3) and frac_rand ~0.28 under pursuit are the NORMAL bootstrap start (REF began identically); the discriminators between healthy/broken are `cem/cost_cv` (REF 0.035 vs 0.25–0.45 scratch) and `wm/sigreg` (REF ~16 vs ~55 scratch).

**VALIDATION.** `b3_v3_fix` (`5zu9ynbl`, v3@3000 via `--init-ckpt` + corrected flags) matched REF immediately — reach 0.062 by step 450, cost_cv 0.037, pctl (3,0)→(3,0.4), qpos>0 — and was stopped once matched (purpose served).

**★ THE WORKING FROM-SCRATCH RECIPE (exact, as-run, verified vs W&B configs `ysc8zudj` → `uyiem6w9`/`igcr3mjt`).**
Stage A — build latent+WM from scratch on tight directed data (recent-goal pursuit), encoder TRAINS, 3000 steps (~50 min on the box):
```
python src/train.py --name scratch_a2 \
  --cem --cem-horizon 1 --cem-replan-every 1 \
  --goal-select recent --goal-future-k 3 --goal-update-every 25 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 4 --sigreg-pertimestep \
  --total-steps 3000 --n-envs 8 --env-threads 8
```
(NB as-run A used default `cem_init_std 1.0` + `buffer-frac 4` — fine for A, which has no reach gating; the canonical block above suggests 0.3/3 for uniformity. A-gate observed: frac_rand 0.276 tail on its own directed data, z_std 1.07, rank 0.19–0.21, pvp 0.184 — "borderline" on the old gate, sufficient in fact.)
Stage B — freeze + d-curriculum verdict lap from the A ckpt (weights-only init!), 3500 steps total:
```
python src/train.py --name b2_fix \
  --init-ckpt <hf-cache>/scratch_a2/ckpt_0003000.pt --freeze-encoder \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 \
  --goal-select highmse_under_d --goal-curriculum --goal-curric-d-start 3 --goal-curric-d-max 4 \
  --goal-curric-thresh 0.05 --goal-curric-patience 150 --goal-update-every 25 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 3 --sigreg-pertimestep \
  --total-steps 3500 --n-envs 8 --env-threads 8
```
(`--goal-mse-curric` default-ON supplies the nested pctl ladder; run executed as 1500 steps + `--resume-name b2_fix` continuation to 3500.)

**RESULT (`b2_fix`, the SAME A2 latent that "failed" under the drifted recipe).** Walked the FULL nested difficulty ladder at both radii: pctl (3,0)→(3,1.0) → d=4 → (4,0)→**(4,1.0)** by step ~2650; **sustained reach@d=4 = 0.108** (n=30 windows, 2× the 0.05 gate), `goal/success_rate_qpos` > 0 throughout (physically confirmed), dist_to_goal 12.7→5.0, min_cand settled at **1.98** (the healthy optimism floor), fantasy gap 4.3→3.3, locality frac_rand improved to **0.205** as pursuit tightened. Under the drifted recipe the identical ckpt read: reach 0.001, pctl pinned 0. ⇒ **the from-scratch latent was NEVER the blocker at d≤4; the recipe was.** Remaining true scratch-vs-v3 gaps (the levers if d>4 ceilings appear): `cem/cost_cv` 0.25–0.45 vs 0.037 and `wm/sigreg` ~55 vs ~16 — address with LONGER Stage-A builds (A2's pvp was still falling at its 3k cutoff) and gate future A's on those two, not frac_rand alone.

**New instrumentation this session:** `--cotrain-frac-thresh` / `--cotrain-frac-cooldown` (+ `cotrain/frac_triggers` metric) — EVENT-triggered co-train phase when windowed step_jump_frac_rand exceeds the threshold (pause CEM → thaw → flatline-consolidate → re-freeze → window reset). Firing verified; its fair test is Stage C on healthy data (b1_trig only proved it faithfully amplifies a poisoned buffer).

**NEXT — Stage C from scratch:** init from `b2_fix` final ckpt, Stage-B recipe + `--cotrain-every 2000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 --cotrain-frac-thresh 0.3`; ~6000 steps (≥2 thaw cycles). GATE: reach@d=4 HOLDS through thaw cycles and frac_rand doesn't degrade phase-over-phase; only then raise `--goal-curric-d-max` toward the d≈10–14 band.

## 2026-07-03 (cont.) — ★ STAGE C PASSES from scratch (2 thaw cycles held, ladder → (4,0.8), qpos success ×2) — after diagnosing the "reach collapse" as a LATENT-RESCALE MEASUREMENT ARTIFACT ★

Ckpt lineage: `scratch_a2@3000` (`ysc8zudj`) → `b2_fix@3500` (`uyiem6w9`/`igcr3mjt`) → `c1_scratch@3000` (`7cs03svx`) → `c2_scratch@4500` (`ku0nqa4m`) → `c3_scratch` (`8h5xo181`, running).

**c1 (`c1_scratch` `7cs03svx`, from b2_fix@3500, trigger 0.3): the thaw did NOT break behavior — it broke the ruler.** Per-phase anatomy (the decisive deep-dive): post-thaw the predictor IMPROVED (pred_loss 0.022→0.012, pvp 0.079→0.045), the planner landscape IMPROVED (cost_cv 0.54→0.34), conditioning MASSIVELY improved (sigreg 64→4), and PHYSICAL closing was flat (qpos_dist 1.44→1.30) — but SIGReg pulled z_std 0.82→1.03, inflating step_jump 5.0→6.6 (+32%) and ALL latent distances ~25–30% while `eps=2.0`/`d=4` stayed fixed ⇒ reach "fell" 0.037→0.02 purely in rescaled units. The frac_rand-trigger at 0.3 then thrashed (3 fires/800 steps — firing on the rescaled numerator, each phase re-perturbing before re-adaptation): reach pinned 0.03. Two lessons: **(1) after any thaw, latent-unit thresholds (eps, curric_d, frac-trigger) are stale until the scale settles — recalibrate or read qpos; (2) trigger threshold must sit ABOVE the post-thaw settling band (0.28–0.36), not in it.**
**c2 (`c2_scratch`, from c1@3000 = the conditioned weights, trigger 0.4, 4500 steps): GATE PASSED.** Thaw#1@2000 (690 grad steps to converge): reach flat through boundary (0.046→0.043), ladder kept climbing (pctl 0.14→0.50), rescale only +21%. Thaw#2@4000 (295 steps — phases need LESS work each time): **NO rescale** (step_jump 6.80→6.81 — scale stable once near-isotropic), reach recovered in 250 steps, **ladder advanced THROUGH both thaws to (4,0.8)**, `success_rate_qpos` 0.010→0.018 (physical successes doubled), pvp 0.065→0.028, sigreg 18→6, cost_cv →0.39, frac_rand stable ~0.34, zero trigger fires. Caveat: window-MEAN reach ~0.03 (<0.05) — but pctl advances require windowed reach ≥0.05 and kept occurring; means are diluted by post-advance resets. The wake/sleep cycle is net-positive from scratch: each sleep improves the WM without costing behavior.
**Verdict on the scratch-vs-v3 conditioning gap:** Stage C's sleeps CLOSED it (cost_cv 0.79→0.39 trending to v3-ish, sigreg 6 < v3's 16) — the gap flagged at B-time self-heals under phased co-train on healthy pursuit data.
**NEXT (launched): `c3_scratch` (`8h5xo181`)** — radius extension unlocked by the C-gate pass. Exact command (= Stage C canonical + d-max 12):
```
python src/train.py --name c3_scratch \
  --init-ckpt <hf-cache>/c2_scratch/ckpt_0004500.pt --freeze-encoder \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 \
  --goal-select highmse_under_d --goal-curriculum --goal-curric-d-start 3 --goal-curric-d-max 12 \
  --goal-curric-thresh 0.05 --goal-curric-patience 150 --goal-update-every 25 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 3 --sigreg-pertimestep \
  --cotrain-every 2000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 \
  --cotrain-frac-thresh 0.4 --total-steps 12000 --n-envs 8 --env-threads 8
```
Watch: does the from-scratch+cotrain stack extend the radius past d=4 toward the v3-stack's d=12–18; rescale artifacts should stay absent (scale settled); if reach stalls at larger d with rising fantasy gap, the planner-side levers (pessimism/trust-region, 06-30 §3c) are next.

**★ c3 RESULT (12000 steps): from-scratch radius d=3 → 9, EVERY pctl ladder d=4..8 completed to 1.0, past the historic d≈8 ceiling — and the fantasy gap NEVER opened (3.4–4.9 throughout vs the v3-stack's ~8 wall). ★** Ladder timeline: d4@~3500 → d5@~4700 → d6@~5200 → d7@~6300 → d8@~8200 (post-thaw#4 pop) → d9@~11000, ended (9,0.4) reach 0.054 (n=18) still advancing at budget end — NOT a converged ceiling. The wake/sleep mechanics matured: 6 timer thaws (converging in 544→204 grad steps — each phase needs less), 3 trigger fires at 0.4 (single quick stabilizers, no thrash, ladder advanced through every one), sigreg pinned 3–6 the whole run, cost_cv ~0.35, no rescale artifacts after the first phases (scale settled ~step_jump 6.5–7). Two grind-then-pop cycles ((7,1.0) ~1300 steps → thaw#4 → pop; (8,1.0) → thaw#5 → pop) — **the periodic frontier-data consolidation IS the radius engine, reproducing v7's bootstrap from scratch.** Reach rode 0.04–0.12 (threshold-pinned by design; dwell wall + ~30%-tighter effective eps per the rescale — see 07-03 analyses). qpos_dist 0.53→0.77–1.06 rising with d (goals genuinely farther; success_rate_qpos > 0 throughout).
**NEXT (launched): `c4_thresh`** — user-proposed decisive experiment: identical to c3 (same c2@4500 init, d-max 12) with ONLY `--goal-curric-thresh 0.05 → 0.15`. Splits threshold-pinned vs capability-capped reach: if reach rides ~0.15 per band → it was curriculum-pinned (and each level gets mastered 3× harder); if it stalls ~0.1 with the ladder frozen → the dwell wall dominates and the 06-30 §3b dwell fixes (arrival HOLD, terminal action-shrink, eps recalibration ~2.6) are the path to high reach%.

**★ c4_thresh RESULT (`qwm90jcv`, 12000 steps, 2.9h): REACH IS THRESHOLD-PINNED, NOT capability-capped — the 0.15 bar was RIDDEN and the full d=3 ladder mastered at it, ending (3,1.0). ★** Arc: first band slow ((3,0.0)→(3,0.2) took ~4200 steps incl. 2 thaws — the WM warming to the 3× demand; early windows mis-read as "capability-capped ~0.10"), then the bar was RIDDEN: sustained reach tail ~0.137–0.145 with windows 0.18–0.30 (vs ~0.06 typical under the 0.05 bar — ~2.3× sustained, 3× windows), ladder (3,0.2)→(3,1.0) with two grind→thaw→pop cycles ((3,0.6) ~2400 steps → thaw#4 popped it; same at (3,0.8)→#5) — the SAME pop mechanism as c3, now 5-for-5 across both runs. End state: (3,1.0), reach tail 0.114, best-of-run fantasy 3.22, frac_rand 0.286, sigreg 3.2–6, pose_step down to 0.084 (the high bar visibly teaches finer motion). 6 timer thaws + 2 trigger fires (0.4, single clean stabilizers). **Verdicts: (1) the 0.05-bar curriculum was advancing the instant it could — demand 3× more and, given per-band consolidation, it delivers ~3× more; the ~0.10–0.20 "cap" was threshold-pinning, not the dwell wall (true sustained cap is >0.15, windows to 0.30 suggest headroom). (2) The trade is SPEED: d=3 took ~12k steps at the 0.15 bar vs ~1.5k at 0.05 (~8×) — radius-growth and reach-mastery are tunable against each other by ONE flag. (3) Dwell fixes remain the lever only for pushing SUSTAINED reach toward the 0.3+ window peaks.** Continuation `c4_thresh` resumed to 24000 steps to test the 0.15 bar at d≥4 (does high reach + radius coexist).

**Continuation NEGATIVE (killed @9300/24000):** `--resume-name c4_thresh` restored weights + the 12k-run's goal ARCHIVE but RESET the step counter and curriculum ladder → the run re-ground band (3,0.0) and never cleared it (cold run: 4200 steps; continuation: 9300+ and DEGRADING — reach 0.16-peaks→0.02, dist_goal →10.5, frac_rand →0.42, 8 co-train phases with the 0.4 trigger at cooldown-limited cadence = thrash regime). Verdict: **don't extend a completed curriculum run via `--resume-name`** — the stale archive + reset ladder land it in a worse basin than a cold start; relaunch fresh from a ckpt via `--init-ckpt` with the full budget instead (reinforces the 07-03 init-ckpt lesson). The cold-run c4 result (threshold-pinned, bar ridden, d=3 mastered at 0.15) STANDS as the finding. **OPEN NEXT (either):** (a) fresh 24k `--init-ckpt c2@4500 --goal-curric-thresh 0.15` single run for the d≥4-at-high-bar question; (b) implement the 06-30 §3b dwell fixes (arrival HOLD + terminal action-shrink + eps recalibration) to push SUSTAINED reach toward the 0.3 window peaks.

**★ c5_thresh15 RESULT (`n7gjspk9`, 24000 steps, option (a) executed): d=3 MASTERED at the 0.15 bar ((3,1.0) @ ~13.3k, riding tail ~0.13, windows to 0.30) — but d=4 band-0 NEVER cleared in ~10.7k in-band: the 2000-step sleep cadence STARVES the high-bar buildup at larger d. ★** Mechanics: 11 timer thaws + 7 trigger fires (0.4) ⇒ clean acting windows of only ~1500–1800 steps; at d=4 reach repeatedly built to 0.10–0.12 and was knocked back by the next phase before the windowed mean could hold ≥0.15 (the grind→thaw→pop pattern that cleared every d=3 band — 8-for-8 across c3/c4/c5 — stops converging when buildup time > inter-phase window). No rescale artifacts (scale settled); fantasy 3.0–4.4 all run (best-ever 2.95); sigreg 3–6. **Physical counter-signal (important):** the high bar taught real precision even while the latent scoreboard ground — best-of-session `qpos_dist` (tail 0.59, min 0.40) and `success_rate_qpos` ~4% tail / 7.8% peaks (all prior runs: 0.1–1.8%) — the arm is objectively the most precise it has ever been. **ACTIONABLE:** for high-bar (≥0.15) runs at d≥4, space the sleeps — `--cotrain-every 3000–4000` (and consider trigger 0.45) so the window-mean has runway; OR attack sustained reach directly via the 06-30 §3b dwell fixes. Both remain open; GPU idle pending pick.

**★ c6_spaced RESULT (`pd3g5ve7`, 14000 steps): SPACED SLEEPS CRACK d=4 AT THE 0.15 BAR — (4,0.0)→(4,0.2)→(4,0.4), the cadence diagnosis CONFIRMED. High reach and radius COEXIST when sleep spacing scales with d. ★** Design: c5 final weights, `--goal-curric-d-start 4` (d=3 proven twice, skipped), `--cotrain-every 3500`, `--cotrain-frac-thresh 0.45`, 14k steps. Arc: window-1 (fresh buffer) capped ~0.06; window-2 climbed 0.05→0.09 and flattened (briefly read as a dwell wall); **window-3 CONVERTED — first-ever d=4 advance at 0.15 (reach tail 0.139) — and band 2 fell in a single window** (consolidation compounds; advances accelerate). Zero trigger thrash at 0.45; thaws converged in 250–510 steps; reach held THROUGH later thaws (no post-thaw collapse once the WM matured). Fantasy gap 2.6–3.1 all run (session best 2.62); no rescale artifacts. **Physical milestone: `success_rate_qpos` tail 0.078, peaks 0.119 (~12% — vs 0.1–1.8% in all pre-high-bar runs) and `qpos_dist` min 0.335 rad (first sub-0.4)** — the 0.15 bar + spaced consolidation is teaching genuinely precise physical reaching (qpos remains diagnostic-only per spec, but the trend is now large and monotone across c4→c5→c6). **RULE OF THUMB BANKED: clean-window length must scale with d (~d×1000 steps); the grind→thaw→pop engine then converges at any bar tested.** Next options logged: ride c6 onward (d-max 12, cadence scaling with d), dwell fixes for sustained-reach 0.15→0.3, seeds for defensibility, formal qpos eval, bigger predictor (roadmap ③, still untested under H=1).

## 2026-07-04 — HARDWARE / SYSTEM-PERFORMANCE PROFILE (for a future multi-GPU switch to parallel experiments)

**Measured footprint of one training run (live `ps`/`nvidia-smi` + full-run W&B `system.*` telemetry across r25/c5/c6/a2):**
- **GPU compute is the ONLY bottleneck.** Solo runs: util mean 73–81% (p50 72–85, p90 100), power ~300W/400W. Two co-tenant runs pin util at 100% (a2's shared window: mean 91, p50 100) and each drops to ~0.55–0.7× solo speed (aggregate ~1.2× — co-tenancy saturates after 2).
- **Compute-bound, NOT bandwidth-bound:** `gpu.0.memory` (memory-access util) mean only 12–27%. Matters for GPU choice: bandwidth-weak cards lose little here.
- **VRAM:** 13.6 GB (6k-step probe) to **26.3 GB (24k-step runs — buffer cap scales with total_steps)**, INCLUDING 8×321 MiB EGL render contexts (~2.6 GB). Rules out 24 GB cards (A10G/L4) for long runs without config surgery.
- **CPU ~1.5 cores/run** (main proc ~127%, env workers ~1.4% each — they only do trivial physics), RAM 6–17 GB. Box: 128 cores / 2 TB — both ~99% idle. vCPU-rich instances buy NOTHING.
- **Rendering is GPU-EGL, not CPU** (`env/parallel_env.py` sets `MUJOCO_GL=egl`, pins the NVIDIA EGL ICD, serializes worker EGL init; osmesa is the CPU fallback only). Env workers appear as type-G graphics contexts in `nvidia-smi` (invisible to `--query-compute-apps` — check the full process table). Any rented GPU needs NVIDIA EGL headless support (all candidates do).
- **`--n-envs` scaling is NOT free throughput:** more envs grow the CEM candidate batch (GPU work per decision); envs themselves are ~free. The lever for parallel experiments is MORE GPUs, not more envs/CPUs.

**Instance comparison (on-demand, effective $ per A100-equivalent experiment·hr; speed estimates for THIS small-model compute-bound workload):**
| instance | $/GPU·hr | est. speed vs A100 | eff. $/exp·hr | note |
|---|---|---|---|---|
| **AWS g6e.12xlarge (4×L40S 48GB)** | **$2.62** | **~0.9–1.0× (compute-bound, bandwidth irrelevant)** | **~$2.6–2.9** | **PICK — 4 full slots, VRAM-safe, CPUs sufficient (48≫4×1.5)** |
| AWS g5.12xlarge (4×A10G 24GB) | $1.42 | ~0.4× | ~$3.55 | 24 GB OOMs long runs; slow wall-clock kills iteration |
| Azure NC48ads (2×A100 80GB) | $4.78 | 1.0× | $4.78 | exact-repro option only |
| AWS g6.24xlarge (4×L4 24GB) | $1.23 | ~0.25–0.3× | ~$4.50 | same 24 GB + slowest |
| AWS g6e.24xlarge (4×L40S) | $3.77 | ~0.9–1.0× | ~$4.70 | strictly worse than 12xlarge for us (pays for unusable vCPUs) |

**Decision + caveats:** g6e.12xlarge (~$10.49/hr) ≈ 4 parallel A100-class experiments at ~40% less per experiment than the Azure A100 pair; short probes could co-tenant 2-per-GPU (~6 slots at ~0.65×). BEFORE any long rental: 1-hour validation benchmark — run the exact r25 recipe 2k steps on one L40S, compare sps (the 0.9–1.0× estimate rests on the compute-bound finding). Prices are on-demand; spot/reserved reshuffles the table.

## 2026-07-04 — REACH-BAR STAIRCASE (user-designed): the sustained-reach CEILING of the current stack at d=4 is ~0.15 sustained / 0.234 max-window — 0.25 is UNREACHABLE without dwell mechanics

**Design (user):** probe the max demandable reach bar by chained short runs — bar 0.25, then +0.10 each probe that advances, each probe `--init-ckpt` from the previous one's final ckpt (warm chaining; each probe doubles as continued d=4 consolidation), d-start 4, spaced sleeps 3500, trigger 0.45, 6000 steps/probe. Stop at the first probe that stalls with max windows clearly below its bar.
**Probe 1 = `r25_probe` (bar 0.25, from c6@14000, 6000 steps): STALLED — STAIRCASE STOPPED.** 119 goal windows: **0 windows ≥ 0.25** (max single window **0.234**, p95 0.204, p75 0.165, mean 0.124, 42 windows ≥0.15), pctl advances: NONE. Notable early surge while goals were fresh-easy: tail hit 0.177 with best-of-session health (frac_rand 0.227 — first C-stage reading under the 0.25 locality bar; fantasy gap 1.25 — best ever by 2×; qpos_dist 0.262) — then in-band goal hardening + the thaw cycle settled it to the familiar ~0.10–0.15 sustained regime. **VERDICT: the current planner+geometry caps at ~0.15 sustained / ~0.20–0.23 peak windows at d=4 — matching the 06-30 travel+bounce analysis (~0.2–0.3 theoretical cap: per-step latent jump ~6 > eps-ball width 4 ⇒ arrive-and-bounce scoring only).** The bar ladder (c4/c5/c6: 0.05→0.15 all rideable) tops out between 0.15 and 0.25.
**EXTENDED CONFIRMATION (`r25_long`, user's let-it-run rule: continue while reach trends up, call ceiling on 2+ flat/declining cycles).** From r25_probe@6000, 0.25 bar, 3 completed thaw cycles + part of a 4th (killed @11250): per-cycle window peaks **0.161 → 0.121 → 0.144** (no new high after cycle 1), means DECLINING 0.107 → 0.071 → 0.063, advances: NONE, and the extended run never matched the probe's 0.234 one-off. Total 0.25-bar evidence: ~17k steps, ~275 goal windows, **zero at ≥0.25**. The flat-cycle rule triggered ⇒ ceiling CONFIRMED with cycle-level evidence, not a short-probe artifact. More consolidation does NOT raise the peak — the reach distribution is stationary around its geometric cap.
## 2026-07-04 — DWELL MECHANICS BUILT + TESTED (`d4_dwell`): STAY-wall SOLVED (windows 0.57, first-ever 0.25 advance), but net reach NEGATIVE at first calibration — the choke point moved from STAYING to BALL ENTRY

**Implementation (src/train.py, new flags):** `--dwell-hold-mult` (arrival HOLD: inside mult×eps bypass the planner, act a=0 — WM-free park, gated on `has_goal` so self-goal envs can't freeze), `--dwell-shrink-start` (terminal action-shrink: linear action scale from 1.0 at start×eps down to `--dwell-shrink-min` at the goal), metric `goal/dwell_hold_frac`. Run: identical recipe/init to the no-dwell baseline `r25_probe` (c6@14000, bar 0.25, d-start 4, sleeps 3500, trigger .45) + HOLD 1.25×, shrink 3.0×/floor 0.2, eps UNCHANGED at 2.0 (same ruler), 12000 steps.
**Full-distribution A/B (238 dwell windows vs 119 baseline):**
| | max | p95 | mean | ≥0.25 | ≥0.15 | qpos succ mean/max |
|---|---|---|---|---|---|---|
| dwell | **0.570** | 0.068 | 0.039 | **2** | 5 | **0.115 / 0.344** |
| baseline | 0.234 | 0.204 | 0.124 | 0 | 42 | 0.104 / 0.236 |
**Read (both halves matter):** (1) **STAYING IS SOLVED** — when the arm gets inside, it parks and racks the window (0.57 = 2.4× the all-time record; the ONLY ≥0.25 windows ever recorded; first-ever 0.25-bar advance (4,0)→(4,0.2) @ ~2600; physical success best-ever and qpos_dist stable → no hold-abuse; holds fired mean 9% / max 43% with no data starvation). (2) **BUT typical windows got WORSE** (mean 0.039 vs 0.124): the 3×eps shrink slows the whole approach below 6 latent units while entry into the 2.0-ball / 2.5-hold-radius stays rare (planner directional error + optimism ~2-3 at close range), so most windows end mid-creep — the baseline's frequent lucky-bounce hits were traded for rare parked jackpots. The reach distribution went bimodal; the choke point is now BALL ENTRY, not dwell.
**⇒ NEXT (calibration of a validated mechanism, not a new idea): `d4_dwell2` arm** — widen the ruler to the recalibrated eps 2.8 (≈0.43×step_jump; also widens the hold radius to 3.5 — entry becomes commensurately likelier) + gentler shrink (start 2.0×eps, floor 0.3) so the approach isn't over-taxed. Success: mean reach recovering ≥ baseline WITH the parked tail intact (windows ≥0.3 recurring), advances flowing at 0.25. NB eps 2.8 redefines the latent ruler (honest-ruler argument: ball was ~1/3 of one latent step); capability comparisons across rulers should lean on qpos.

**d4_dwell2 RESULT (eps 2.0→2.8 + shrink 2.0×/floor 0.3, else = d4_dwell; 12000 steps, 237 windows): the approach-tax is FIXED and 0.25-bar progress is REAL, but the run stalls at (4,0.4) — the remaining constraint is CLOSE-RANGE PLANNER PRECISION, not dwell mechanics.** Scorecard vs pre-registered goals: **advances flowing ✓** ((4,0)→(4,0.2)→(4,0.4) in the first 1.9k — baseline had 0 in 17k); **parked tail intact ✓** (3 windows ≥0.30, max 0.482 — recurring, not one-off); **mean ≥ 0.124 ✗** (0.110 — front-loaded: wins in the first 2k, then 10k grinding (4,0.4) at ~0.08–0.12). Holds engaged mean 20% (vs v1's 9% — the 3.5 hold radius is actually enterable), dwell survived all 3 thaws (mild dips, holds firing through them — drift-immunity confirmed empirically), zero pathologies (qpos_dist stable, no starvation, ladder honest). **Ruler-independent check: qpos succ mean 0.108 ≈ baseline 0.104 ≈ v1 0.115 — the eps widening did NOT inflate real capability, and real capability did not jump either: the three arms are capability-equivalent; what changed is how efficiently that capability converts into scored/curriculum progress.**
**Three-arm synthesis (baseline / v1 / v2):** stay-wall SOLVED (v1: 0.57 windows); approach-tax SOLVED (v2: mean 0.039→0.110); ball-entry at hard bands ((4,0.4)+) remains — the CEM's close-range directional error (optimism floor ~2–3 ≈ the entry radius itself) caps arrival probability regardless of dwell. **The precisely-identified next lever is the 06-30 §3c local-Jacobian trust region** (fit dz≈J·a+b from nearby buffer transitions each replan; hard-reject candidates whose predicted move violates it — kills directional fantasy exactly on the final approach, provably non-softening). Alternatives: accept the current operating point (recurring 0.3–0.5 parked windows at d=4 is already a qualitatively new regime) and spend compute elsewhere (radius, seeds, bigger predictor).

## 2026-07-04 — PER-GOAL ARRIVAL metric shipped (`--goal-curric-metric arrival`) + `arrival90` run: the honest "reach most goals I set" number is ~0.51 fresh-band / ~0.25 hardened-band; 0.9 sustained needs more than cadence or dwell

**Motivation (user):** "raise the bar to 0.9 — I want it to reach most latent goals it sets." reach_rate can't express that (time-occupancy: travel eats ≥20-25% of every 25-step window ⇒ perfect-agent ceiling ~0.75). New metric: **`goal/arrival_rate`** — a goal counts REACHED if the env ever dips inside eps during its window (per-env min-dist, closed out at each refresh); **`--goal-curric-metric arrival`** gates the curriculum on it (advance gate: 8-cycle rolling mean ≥ bar). A perfect agent scores 1.0 ⇒ 0.9 is a meaningful mastery bar.
**`arrival90` RESULT (d4_dwell2@12000 init, dwell v2, eps 2.8, bar 0.9 on arrival, update-every 25, 12000 steps, 239 goal-cycle windows):** overall arrival mean **0.326**, max window **0.911** (1 window ≥0.9 — the bar is TOUCHABLE), 29 windows ≥0.5. Cycle ledger: c1 **0.509** (fresh warm band, peak .911) → c2 0.266 → c3 0.235 → c4 0.258 — **plateau ~0.25 after band-hardening** (no advances at 0.9 ⇒ no fresh-band refreshes ⇒ the (4,0.0) pool hardens under the maturing WM; dist_goal drifted 7.4→9.1). qpos succ record peak **0.520** (mean 0.100); holds mean 20%; no pathologies; dwell survived all 3 thaws.
**Read:** the from-scratch stack currently reaches **~half of fresh goals and ~a quarter of hardened goals** at d=4/eps 2.8. The 0.9-sustained target is a CAPABILITY gap (close-range planner precision — the isolated constraint from the dwell arc), not a metric artifact. NEXT (auto-launched per goal-mode guide): **`arr90_w50`** — identical but `--goal-update-every 50` (double each goal's life) — cleanly separates "not enough time" from "can't get there": if arrival jumps toward ~0.7+, travel time was a big component and cadence is a lever; if it barely moves, the Jacobian trust region (06-30 §3c) is the confirmed sole path to 0.9.

**`arr90_w50` RESULT (travel-time arm: identical to arrival90 but goal life 25→50 decisions; 12000 steps, 238 windows): the window-length lift is REAL but SECONDARY — cycle means 0.478/0.334/0.288/0.299 vs arrival90's 0.509/0.266/0.235/0.258 (+25–30% post-honeymoon, NO collapse-cycle) — but the plateau is ~0.30–0.33, nowhere near the ~0.7 that would implicate travel time as the main blocker.** Occupancy record 0.627 (long windows + parking compound — the stay side is definitively closed). **VERDICT: ~1/3 of the arrival gap is time-budget, ~2/3 is CLOSE-RANGE PLANNER PRECISION — confirmed as THE 0.9-blocker by controlled test.** `--goal-update-every 50` is a strictly-better operating default going forward (arrival ↑, no downside observed, band-hardening softened).

**JACOBIAN TRUST-REGION — ⚠️ DECLINED BY USER 2026-07-04 ("i don't want to do the jacobian trust region"). NOT the next build; spec kept below for the record only. Do not re-propose as the default path — remaining levers for the close-range precision gap: proximity-gated CEM search precision (init-std/iters schedule near goal, few lines, untried), goal cadence/window knobs (w50 banked as better default), or accept the current operating point and invest elsewhere (radius, seeds, bigger predictor).** Original spec (for the record):
1. **Fit (per replan, per env, cheap):** take the K≈64 buffer transitions whose z is nearest z_now (the local neighborhood); least-squares fit dz ≈ J·a + b (J: z_dim×a_dim via ridge on the K pairs; ~64×(192×30) solve, trivial on GPU). This is the locally-true motion model — what actions ACTUALLY do here, no WM optimism.
2. **Gate (inside cem_plan, after the WM scores candidates):** for each candidate first-action a_i, compare the WM's predicted move dz_wm(a_i) against the local model's dz_J(a_i) = J·a_i + b; REJECT (cost = +inf) candidates where ‖dz_wm − dz_J‖ > τ·‖dz_J‖ + c (τ≈0.5, c≈0.5 latent units — tune on the offline test). Survivors keep their FULL closing cost — provably non-softening (kills only directionally-dishonest moves, not ambition).
3. **Self-validation (offline unit test BEFORE any run):** (a) replay real buffer transitions through the gate — >95% must survive (residual ≈ sensor noise); (b) synthesize fantasy moves (WM-predicted dz toward goal where the local J says the action moves away) — >90% must be rejected. Both from a frozen ckpt, no env needed.
4. **Flags:** `--cem-jacobian-tr` (on/off), `--cem-jtr-k 64`, `--cem-jtr-tau 0.5`, `--cem-jtr-c 0.5`, metric `cem/jtr_reject_frac` (healthy ≈ 0.1–0.4; ≈1.0 = mis-calibrated, gate strangling the search).
5. **Success criterion:** arrival at d=4/eps 2.8/w50 rising from the measured 0.33 plateau toward ≥0.6 (the honeymoon level sustained); guardrail: cost_cv and reach must not collapse (over-rejection symptom).

## 2026-07-04 — GOAL-MODE NIGHT WRAP (3:21–9:00 AM PT, autonomous): the reach-mastery arc closed with the blocker isolated by controlled experiment
Sequence run autonomously: `arrival90` (0.9 per-goal bar → honest arrival 0.51 fresh / 0.25 hardened, bar touchable at 0.911) → `arr90_w50` (travel-time arm → +25–30% but plateau 0.33 ⇒ precision dominant). Combined with the dwell arc (stay solved, entry choking): **the single remaining obstacle between the current stack and "reaches most goals it sets" is CEM close-range directional error, and the Jacobian trust region (spec above) is the designed, self-validating fix.** All results committed run-by-run; new tooling shipped this arc: `goal/arrival_rate` + `--goal-curric-metric`, dwell HOLD/shrink flags, frac_rand-triggered co-train. GPU idle at wrap. *(Forward pointer retracted: the Jacobian trust region was DECLINED by user later on 07-04 — see the decision block above and the entry below. It is NOT the next build.)*

**⇒ The 06-30 §3b DWELL FIXES are now the only path above ~0.15 sustained, and they have a precise baseline to beat:** arrival-HOLD (a=0 inside ~1.25×eps; converts each arrival into parked scoring — WM-free so unexploitable) + proximity-gated terminal action-shrink (drop action_max near the goal so the final step can land INSIDE the 4-unit ball instead of stepping over it) + eps recalibrated to the current latent scale (~0.45×step_jump ≈ 2.6–3.0). Success criterion for that experiment: sustained reach > 0.234 (the no-dwell max window) at d=4; the user's 0.5→0.9 bar idea then becomes an arrival-SPEED curriculum (with HOLD, a goal reached at step k of a 25-step window scores (25−k)/25). GPU idle; awaiting go on the dwell implementation.

## 2026-07-04 — DECISION REAFFIRMED: NO JACOBIAN TRUST REGION (user, final) + `arr100_d1` launched: from-scratch "reach EVERY goal" ladder (arrival bar 1.0, d=1 → up, nested MSE curriculum)

**DECISION (user, stated twice this date — treat as final): we do NOT want the Jacobian trust region.** Do not build it; do not re-propose it as the default next step. Earlier entries calling it "the designed fix" / "the confirmed sole path to 0.9" record the *analysis* of the close-range precision gap, not the *plan* — that path is closed by user decision. If the precision gap is ever attacked directly, the open levers remain: proximity-gated CEM search precision (init-std/iters schedule near goal, untried), cadence/window knobs, or investing elsewhere (radius at new mechanics, seeds, bigger predictor).

**NEW RUN (user-designed): `arr100_d1` — "I want it to reach every goal it sets."** Gate = per-goal ARRIVAL (ever inside eps during the goal's window — binary per goal, NOT time-occupancy) with bar **1.0**: advance only when EVERY goal in the last ≥8 windows was entered (booleans ⇒ mean==1.0 exact, no float issue; verified in code). Full ladder from the bottom: **d-start 1**, nested MSE pctl 0→1 inside each d (`--goal-mse-curric` default-ON), then d+1 and pctl resets — exactly the user's "d=1, mse=0 → higher mse → higher d". Tabula rasa: fresh Stage A build `scratch_a3` (exact `scratch_a2`/`ysc8zudj` recipe, verified by W&B-config diff; only training-inert diff = `--keep-local-ckpts` for chaining) → main run inits **weights-only** from `scratch_a3@3000`. Main recipe = **`arr90_w50` (`p8hd4mnn`) exactly** (config-diffed per the 07-03 hygiene rule) with THREE deliberate diffs: `--init-ckpt` (scratch lineage), `--goal-curric-d-start 1`, `--goal-curric-thresh 1.0` — plus budget 24000 (taller ladder; any extension = fresh `--init-ckpt`, never `--resume-name`). Inherited defaults of record: metric arrival, w50 goal life, eps 2.8, dwell v2 (HOLD 1.25×, shrink 2.0×/floor 0.3), sleeps 3500 / trigger 0.45, H=1, cem-init-std 0.3, buffer-frac 3.

```
# Stage A (launched first):
python src/train.py --name scratch_a3 \
  --cem --cem-horizon 1 --cem-replan-every 1 \
  --goal-select recent --goal-future-k 3 --goal-update-every 25 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 4 --sigreg-pertimestep \
  --total-steps 3000 --n-envs 8 --env-threads 8 --keep-local-ckpts
# Main (chained when A completes; the canonical arr100 block):
python src/train.py --name arr100_d1 \
  --init-ckpt runs/scratch_a3/ckpt_0003000.pt --freeze-encoder \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 \
  --goal-select highmse_under_d --goal-curriculum --goal-curric-d-start 1 --goal-curric-d-max 12 \
  --goal-curric-thresh 1.0 --goal-curric-patience 150 --goal-curric-metric arrival \
  --goal-update-every 50 --goal-reach-eps 2.8 \
  --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 3 --sigreg-pertimestep \
  --cotrain-every 3500 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 \
  --cotrain-frac-thresh 0.45 --total-steps 24000 --n-envs 8 --env-threads 8
```

**Pre-registered expectations (so the verdict is honest):** (1) d=1–2 rungs are formally trivial — d < eps 2.8 means goals start inside the ball (no-under-d fallback = nearest candidate; likely parked-in-hold early) — they validate the 1.0-gate machinery, floor ≈450 steps/rung (8 windows × w50, gate checks at patience-150 multiples) ⇒ expect ~5–7k steps before real work at d≈3. (2) The 1.0 bar is strict; known capability (arr90_w50 plateau ~0.33 at (4,0.0), best window 0.911) says the ladder should park somewhere in **d≈3–5**; WHERE it parks IS the finding — the honest "reaches everything it sets" radius of the current stack. (3) Stall rule = r25's: 2+ full sleep cycles with no advance and flat/declining window peaks ⇒ ceiling called (report, don't churn). Monitoring: 15-min updates + stall/pathology alerts (dist_goal drift, frac_rand →0.5, trigger thrash, post-thaw rescale artifacts) via Pushover.

## 2026-07-05 — ★ arr100_d1 VERDICT (killed 8450/24000 on the pre-registered stall rule): the 1.0-arrival bar clears NO rung — the bottom rung at honest scale IS the known ~0.31–0.35 ceiling; plus two structural discoveries about d<step_jump rungs and never-advancing bars ★

**Run:** `arr100_d1` (`ktvzitt6`) from `scratch_a3@3000` (`dt9pr4qf`, A-gate PASS a2-equivalent: z_std 1.073 / rank 0.222 / frac_rand tail 0.281 / pvp 0.273-falling). Recipe = arr90_w50 config-diffed, diffs: scratch init, d-start 1, thresh **1.0**, budget 24k. Zero curriculum advances in 8450 steps, 2 full thaw cycles ⇒ r25 stall rule + active degradation ⇒ killed.

**Phase anatomy (the run in three rulers):**
1. **Honeymoon (<3500):** fresh-build latent is COMPRESSED (step_jump ~2.7) ⇒ eps 2.8 ≈ **1.0×jump** (double-wide honest ball) ⇒ arrival mean **0.759** / max 0.875, holds ~0.5, and best-ever PHYSICAL precision: `success_rate_qpos` peak **0.334**, `qpos_dist` min **0.237 rad** (both records; the c4→c6 "demanding bars teach precision" trend again — while the bar is touchable).
2. **Thaw#1 @3500 (1076 grad steps):** predictor MATURED (pvp 0.27→**0.034**, best-tier; sigreg 5–7 healthy) AND the sleep re-stretched the latent to mature scale (jump 2.7→**6.0**) ⇒ eps 2.8 snapped to the honest **0.45×jump** ruler ⇒ arrival settled **0.29–0.35** = arr90_w50's plateau TO THE DIGIT (cycle mean 0.357). The c1 "thaw breaks the ruler" lesson reproduced on a fresh lineage — first-thaw rescale is intrinsic to fresh A-builds (this one 2.2×).
3. **Thaw#2 @7000 (236 grad steps — 4.5× less, healthy) then DEGRADATION:** cycle-2 mean **0.240**, max 0.305 (declining peaks ✓ stall). Scale kept drifting (jump →8.4, dist_goal 7→**11.2** rising, frac_rand 0.30→**0.43** brushing the 0.45 trigger, holds 0.55→**0.11**, mse[blk/tbl] rising, qpos_dist tail 1.43): with arrivals rare, parked/anchoring data vanishes from the buffer and consolidation amplifies drift — the SLOW form of the B1 flail loop.

**Structural discoveries (why "d=1" was never a freebie):**
- **There are no sub-step_jump rungs.** Buffer granularity = one step_jump (~6–7 units mature): `highmse_under_d` with d < jump finds n_under=0 (except parked clusters) and falls back to NEAREST ⇒ every band below d≈jump is the SAME task ("arrive at the nearest recorded state, ~1 jump away") and the MSE-pctl ladder is INERT there. d-start below ~jump only relabels the bottom rung.
- **A never-clearing bar is not neutral — it's corrosive.** Advances do more than schedule difficulty: each advance clears the window deque and refreshes the goal band. With zero advances the same nearest-fallback pool hardens under scale drift while the arrive→park→densify anchor decays ⇒ the data equilibrium slides unhealthy. (r25 showed the flat version; this run shows the degrading version.)
- **Gate math at 1.0:** advance needs mean(last ≥8 windows × 8 envs) == 1.0 ⇒ **64 consecutive goal-arrivals**; at the honest per-goal ~0.33 that is ~10⁻³¹ per check. Any all-envs-AND bar above ~0.95 is effectively unreachable without per-goal ≈0.995+ capability.

**VERDICT: the current stack cannot "reach every goal it sets" at ANY radius on the honest ruler — the per-goal ceiling is ~0.31–0.35 (fresh nearest-band), and demanding 1.0 doesn't climb toward it, it destabilizes the run.** The blocker is the same isolated close-range precision gap (Jacobian trust region remains DECLINED; proximity-gated CEM precision / cadence knobs / invest-elsewhere remain the open levers). If a reach-everything-flavored bar is wanted, the gate needs headroom (≤0.95, or per-env gating, or arrival-speed scoring) — the 1.0 semantics are structurally self-defeating under the current gate.

**Hygiene notes for future fresh-lineage runs:** (a) expect the first thaw to re-scale a fresh A-build's latent ~2× — pre-thaw arrival/reach numbers are RULER-INFLATED and must not be trusted for gates or verdicts; (b) killed-at-8450 ckpts (`arr100_d1/ckpt_0008000.pt` on HF) carry the matured predictor (pvp 0.034) + drift-degraded data regime — init from ≤7000 ckpts for clean continuations. GPU idle at wrap.

## 2026-07-05 — HOT-GOAL RETARGETING shipped (`--goal-hot-retarget`) + `arr95_hot` launched (bar 0.95, retarget-to-fresh-surprise, same scratch_a3 lineage)

**User directives after the arr100_d1 verdict:** (1) bar 1.0 → **0.95** (≥61/64 — the headroom the verdict showed 1.0 structurally lacks); (2) NEW MECHANISM: "if something in the most recent trajectory is higher MSE (ie just added to buffer) it should set a new goal — buffer should be instantaneously updated and so should goals."

**Implementation (src/train.py):** flags `--goal-hot-retarget <margin>` (0=off; launch 1.2 — sub-1.2 margins churn on MSE-estimate noise across consolidations), `--goal-hot-cooldown 10`, `--goal-hot-window 25`; metric `goal/hot_retarget_rate` (per-env per-step; ×n_envs×update_every ≈ retargets/window). Mechanics: rolling pool of the last window×n_envs collected states (obs + collect-time latent + r_cur; r_cur and refresh-time candidate MSE share the per-dim-mean normalization — verified); each step an env retargets to the pool's argmax-MSE state within its curriculum budget d when that MSE > margin × its current goal's (stored at selection, updated on retarget); per-env cooldown; the scheduled w50 refresh is unchanged and overwrites all goals. **Arrival bookkeeping deliberately NOT reset on retarget** → a window's arrival = "entered the ball of ANY goal held during that window" (goals the system supersedes mid-pursuit are not misses). Notes: (a) at rungs where d < buffer granularity (~1 step_jump) the under-d pool is parked-clusters-only ⇒ retargeting is mostly inert below d≈3 — it wakes up exactly where the ladder gets hard; (b) buf.add lands before the next refresh reads candidates, so refresh-time selection already sees this-step states — the hot path closes the remaining 50-step cadence gap.

**Smoke (`smoke_hot` `nf45691h`, 400 steps from scratch_a3@3000, d-start 4):** clean; hot_rate 0.017→0.031 (~12 retargets/window fleet-wide ≈1.5/env — active, cooldown-bounded), arrival 0.97–1.0 (honeymoon ruler, as documented for this lineage).

**Pre-registered hypothesis:** retargeting redirects pursuit to the arm's own recent wake — recently-visited ⇒ definitionally arrivable; hot-MSE ⇒ max training value per visit (the Go-Explore loop tightened to zero lag). If wake-goals dominate, per-goal arrival can exceed the 0.33 static ceiling (wake-pursuit ≈ 'recent'-mode reach ~0.95 in Stage A); if fantasy still blocks final entry, arrival plateaus ≤~0.5 and the close-range precision gap is re-confirmed under a friendlier goal distribution. Stall + honesty rules unchanged (pre-thaw#1 numbers are ruler-inflated: jump ~2.1–2.7 vs mature ~6).

**`arr95_hot`** = arr100_d1 canonical block + `--goal-curric-thresh 0.95` + the three hot flags, same `scratch_a3@3000` init (isolates {bar, retargeting} as the only diffs vs arr100_d1), 24k budget:
```
python src/train.py --name arr95_hot \
  --init-ckpt runs/scratch_a3/ckpt_0003000.pt --freeze-encoder \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 \
  --goal-select highmse_under_d --goal-curriculum --goal-curric-d-start 1 --goal-curric-d-max 12 \
  --goal-curric-thresh 0.95 --goal-curric-patience 150 --goal-curric-metric arrival \
  --goal-update-every 50 --goal-reach-eps 2.8 \
  --goal-hot-retarget 1.2 --goal-hot-cooldown 10 --goal-hot-window 25 \
  --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 3 --sigreg-pertimestep \
  --cotrain-every 3500 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 \
  --cotrain-frac-thresh 0.45 --total-steps 24000 --n-envs 8 --env-threads 8
```

## 2026-07-05 — ★★ arr95_hot RESULT (24000 steps, COMPLETED): tabula-rasa ladder d=1 → d=8, 42 advances, HONEST post-thaw arrival 0.958 sustained (~3× the 0.33 ceiling) — hot-goal retargeting + 0.95 bar UNBLOCKED the reach-every-goal regime; all-time physical records (qpos_dist 0.095 rad, qpos succ 61%) ★★

**Result (`8j2tej3r`, from scratch_a3@3000 — same lineage/budget as the arr100_d1 null result; diffs ONLY {bar 1.0→0.95, hot retargeting}):** the nested ladder walked **(1,0.0) → (8,0.0)** — every MSE band of d=1..7 cleared (42 advances, mostly at the ~450-step gate-floor cadence; contested rungs at d≥6 took 750–1350 steps) — vs arr100_d1's ZERO advances. **Honest arrival (post-thaw#1, steps 3500–24000, 310 windows): mean 0.958, min 0.875, 87 windows at literal 1.000.** Final window at the d=8 frontier: 0.891. The 0.9-sustained target that closed the previous arc as "needs more than cadence or dwell" is now EXCEEDED at d≤7 by the goal-distribution lever alone.
**Physical records (ruler-independent):** `qpos_dist` min **0.095 rad** (first sub-0.1; prior best 0.237), mean 0.244; `success_rate_qpos` peak **0.613**, run-mean 0.279 (prior best peak 0.52 one-off, sustained tails ~0.1). The arm is objectively the most precise it has ever been, while covering d=8 latent radius.
**Mechanism health:** hot_retarget_rate mean 0.033/env/step (~12/window fleet-wide, max 0.042 — no churn at margin 1.2); dwell holds 0.45–0.80 by regime; **zero frac-rand trigger fires in 24k steps (first ever)** — frac_rand never exceeded 0.223 post-thaw; 6 timer thaws (1076→892→313→280→271→250 grad steps — monotone maturation), thaw→pop 5-for-5 on top-band clears (d4..d7 top bands all cleared within 150–450 steps of a thaw); no rescale artifacts after thaw#2 (z_std_probe 0.97→1.17 total drift); pred_loss 0.002 / pvp 0.014–0.024 = all-time-best band.
**Read (why it works):** hot retargeting redirects pursuit into the arm's own recent wake — recently-visited ⇒ arrivable, high-MSE ⇒ maximal training value per visit — so the Go-Explore loop (arrive → collect → drain → frontier moves) runs at zero lag and the buffer stays park-anchored (the arr100_d1 corrosion never starts). The 0.95 bar supplies the headroom the 1.0 gate mathematically lacked (≤3 misses/64), so advances flow and band-refreshes keep goals fresh. NB the honest caveat for external claims: arrival is measured on goals the system itself selects (wake-biased by design — that IS the mechanism); the d-budget and eps 2.8 are exercised honestly (dist_goal 4–6 at d=7–8, map scale z_std~1.17, qpos confirms) but this is "reaches everything IT sets", not "reaches arbitrary externally-imposed states". Radius-under-mastery is the right external metric: d=8 at 0.958 arrival from tabula rasa in 24k steps.
**Levers banked:** `--goal-hot-retarget 1.2` (+cooldown 10, window 25) joins w50/eps 2.8/dwell v2 as an operating default for arrival-mode runs. The close-range precision gap (0.33 plateau) is CIRCUMVENTED by goal distribution, not solved for arbitrary goals — the distinction matters for any future fixed-goal benchmark.

**CONTINUATION (user): `arr95_hot2` — 80k steps, "let it climb higher in d; NEVER kill it."** Init weights-only from `arr95_hot/ckpt_0024000.pt` (HF), ladder regime-matched (`--goal-curric-d-start 8`), headroom raised (`--goal-curric-d-max 22` = the random-pair latent ceiling), all else identical. Monitor is REPORT-ONLY: no stall-kill authority (alerts via Pushover on pathology; run rides to 80k regardless). Canonical block:
```
python src/train.py --name arr95_hot2 \
  --init-ckpt <hf-cache>/arr95_hot/ckpt_0024000.pt --freeze-encoder \
  --cem --cem-horizon 1 --cem-replan-every 1 --cem-init-std 0.3 \
  --goal-select highmse_under_d --goal-curriculum --goal-curric-d-start 8 --goal-curric-d-max 22 \
  --goal-curric-thresh 0.95 --goal-curric-patience 150 --goal-curric-metric arrival \
  --goal-update-every 50 --goal-reach-eps 2.8 \
  --goal-hot-retarget 1.2 --goal-hot-cooldown 10 --goal-hot-window 25 \
  --dwell-hold-mult 1.25 --dwell-shrink-start 2.0 --dwell-shrink-min 0.3 \
  --wm-cam wrist --no-proprio --action-max 0.05 \
  --consolidate-every 70 --consolidate-epochs 1 --buffer-frac 3 --sigreg-pertimestep \
  --cotrain-every 3500 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 \
  --cotrain-frac-thresh 0.45 --total-steps 80000 --n-envs 8 --env-threads 8
```
(Buffer cap clips at 50k transitions either way ⇒ VRAM ~26GB unchanged. ~22 timer thaws over the budget. Continuation-hygiene reminder honored: fresh `--init-ckpt`, never `--resume-name`; curriculum state is NOT in the ckpt ⇒ d-start must be set manually.)

## 2026-07-06 — ★★ arr95_hot2 COMPLETE (80000 steps): tabula-rasa campaign ends at (11,0.0) — d=1→11, 60 advances, arrival ≥0.89 sustained at EVERY level, radius +2 over the all-time record — and the deep-frontier economics are now measured ★★

**Run (`8pihzc5m`, from arr95_hot@24000, d-start 8, d-max 22, NEVER-KILL per user):** completed the full 80k budget. Ladder: **(8,0.0) → (11,0.0)** — 18 advances (d=8 finished, d=9 complete, d=10 complete, d=11 reached @66600). **Campaign total from tabula rasa (scratch_a3 3k → arr95_hot 24k → hot2 80k = 107k steps): (1,0.0) → (11,0.0), 60 curriculum advances** — every MSE band of d=1..10 mastered at the 0.95-arrival bar; final all-time radius **d=11** (+2 over c3's d=9 record, at ~19× stricter mastery than its 0.05-occupancy standard).
**Arrival ledger (per level, honest ruler throughout):** d=8 mean **0.925** / d=9 **0.916** / d=10 **0.900** / d=11 **0.891** (398 windows, full-run mean 0.907, min-ever 0.797) — graceful ~1%/level decay, no cliff. The "reach everything it sets" property holds to the budget edge: ~9 of 10 self-set goals arrived even at the deepest unprecedented shell.
**Deep-frontier band economics (the run's key new measurement):** band cost inflates super-linearly past d≈9 — (10,0.0) 11.7k steps, (10,1.0) 10.95k (3rd-thaw pop ON the 3-cycle checkpoint), (11,0.0) unfinished at 13.4k (3+ cycles, oscillating .84–.94, cusp touched repeatedly) — vs the 450-step floor at d≤5. Mechanism intact: 22 timer thaws (363→235 grad steps, monotone), thaw→pop cleared every finished top band, **zero frac-rand trigger fires in the entire 107k-step campaign** (max 0.261 vs 0.45), zero rescale artifacts, zero pathologies. The grind driver is goal COMPOSITION: contact rate at the frontier hit 0.60–0.86/s — deep-shell surprise = object-manipulation states, where entry precision is hardest (the known close-range gap, now localized to manipulation poses specifically).
**WM at end: pred_loss 0.0019, pred_vs_persist 0.0087 — inside the all-time-best band (0.007–0.015), at d=11, from scratch.** Physical: qpos_dist min 0.166 / mean 0.281 rad, strict succ max 0.375 / mean 0.199 — sustained precision 2–3× any pre-hot-retargeting run (honeymoon one-offs aside).
**CAMPAIGN VERDICT:** the user-designed reach-every-goal stack (bar 0.95 + hot retargeting 1.2/10/25 + w50 + dwell v2 + eps 2.8 + spaced sleeps) is a complete, from-scratch, self-supervised curriculum engine: it walked 11 distance levels in ~29h of sim on one A100 with no human intervention, no reward, no demos, sustaining ≥0.89 per-goal arrival throughout. Remaining frontier: d≥12 at 0.95 needs either longer budgets (bands DO clear given cycles — nothing saturated), deep-d cadence scaling (c6 rule: clean window ≈ d×1000), the untried proximity-gated CEM precision lever (manipulation-pose entry is the binding constraint), or a bigger predictor. Ckpts through 80000 on HF (`arr95_hot2/ckpt_0080000.pt` = the most capable model this project has produced). GPU idle at wrap.

## 2026-07-06 — `arr95_hot3` launched: +100k continuation from arr95_hot2@80000 (d-start 11, d-max 22) — USER ORDER: never stop it, even if it looks stalled
Minimal-diff continuation per hygiene (fresh `--init-ckpt` from HF `arr95_hot2/ckpt_0080000.pt`, weights-only; ladder regime-matched `--goal-curric-d-start 11`; ALL other flags byte-identical to the hot/hot2 canonical block; `--total-steps 100000`). W&B `eqv6hluf`. Purpose: pure-budget arm at the measured deep-frontier economics (bands 11–13k steps at d≥10 ⇒ 100k ≈ 7–9 more bands if costs hold ≈ d≈12–13 plausible; (11,0.0) resumes from scratch buffer-wise but the WM carries all 107k steps of knowledge). Monitor is REPORT-ONLY with an ABSOLUTE no-stop order from the user ("don't stop even if you think it's stalling") — plateaus get described in updates, nothing gets killed; only a process crash triggers an alert-and-await. ~29–30h wall to budget.

## 2026-07-06 — FULL-STATE CONTINUATION shipped (DEFAULT-ON): replay buffer now saved + auto-restored across chained runs (`--save-state` / `--init-buffer auto`)

**User directive:** "always save and load the buffer for WM and the high-MSE goal state buffer, by default." Analysis first: `--init-ckpt` already carries weights AND the 64-slot goal archive (embedded in every ckpt, reloaded at init since the resume plumbing covers init-ckpt too) — the one substantive loss at a chain boundary was the REPLAY BUFFER, i.e. the goal-candidate coverage of the deep shell (`highmse_under_d` can only propose states the buffer contains; weights remember dynamics, not coverage). Measured cost of that loss: fresh-buffer restarts re-cleared band 0 cheaply (honeymoon) but re-paid shell coverage during the hard bands (e.g. hot3's 23.55k (11,0.2)); measured artifact at the d=8 and d=11 run boundaries ≈ 2.4k and 13.4k sunk steps.
**Implementation (src/train.py):** `save_state_snapshot`/`load_state_snapshot` — full per-env rings + pointers + PER priorities + HER goal fields + ep ids, ONE rolling `out_dir/state_latest.npz` (atomic tmp-swap, uncompressed ≈1.2 GB at the 50k cap, ~0.3 GB at small caps), written every `--save-state-every` (default 10000) + at run end/graceful stop; final one uploads to HF (`<run>/state_latest.npz`). Restore is ring-size-agnostic (unrolls chronologically, keeps newest ≤cap per env, marks the snapshot horizon `is_start` so no WM window crosses it) with shape guards. Flags: **`--save-state` (BooleanOptionalAction, DEFAULT ON)**, `--save-state-every 10000`, **`--init-buffer auto`** (auto = use `state_latest.npz` sitting next to `--init-ckpt`, local or HF-cache dir; `none` = empty; or explicit path). Old `--save-buffer` (flat offline export) unchanged.
**Smoke-verified** (`smoke_save` → `smoke_load`, 250+150 steps co-tenant, hot3 untouched): periodic+final snapshots written and rolled; restore printed `[state] restored 1000 transitions`; archive 64/64 from ckpt; goals proposed from restored coverage at step 50. **Revised continuation hygiene:** chain = `--init-ckpt <ckpt>` (+ `hf_hub_download <run>/state_latest.npz` into the same dir when cross-box) + regime-matched `--goal-curric-d-start` (and `--goal-mse-pctl-start` if the parent stopped mid-level) — buffer/archive now ride automatically. NB: `arr95_hot3` (running, code loaded pre-feature) cannot snapshot itself → the NEXT chain (hot4) starts buffer-empty one final time; hot4 onward carries state.

## 2026-07-07 — ★★ arr95_hot3 COMPLETE (100000 steps): d=11 and d=12 ladders BOTH finished, d=13 reached at the wire — campaign now (1,0.0) → (13,0.0), 73 advances from tabula rasa ★★

**Run (`eqv6hluf`, from arr95_hot2@80000, d-start 11, NEVER-STOP per user):** full 100k budget, 13 advances. Ladder: **(11,0.0) → (13,0.0)** — d=11 completed (incl. the 23.55k `(11,0.2)` marathon, longest band ever, broken on thaw#7; then (11,0.8)/(11,1.0) fell in 2.55k/0.6k), d=12 completed (43k steps for the level; `(12,0.4)` 17.4k and `(12,1.0)` 13.95k were the walls; mid-level bands as cheap as 1.8-4.2k), **d=13 reached @97350 with 2.65k budget left** — the 4th unprecedented level. Campaign totals: **(1,0.0) → (13,0.0), 73 advances, ~207k steps / ~60h from tabula rasa.**
**Arrival ledger:** d=11 mean **0.893** / d=12 **0.900** / d=13 opening **0.815** (395 windows, full-run mean 0.894, min 0.734). The d-level means are FLAT d=11→12 (0.893→0.900) — no capability cliff through two more unprecedented levels; the grind cost is in band DURATION, not arrival depth. Deep-frontier economics extended: d=11 ≈ 54k steps for the level (boundary-inflated), d=12 ≈ 43k clean ⇒ per-level cost still super-linear but sub-doubling.
**Physical/WM at the deepest shells:** qpos_dist min **0.125** / mean 0.238 rad; strict succ max 0.381 / mean 0.239; end-of-run pred_loss **0.0013**, pvp **0.0073** — both ALL-TIME BESTS (the WM keeps improving through d=13); hot rate 0.033 steady; frac_rand max 0.276, **zero trigger fires — the campaign's 360k total steps have never fired one**; 30 thaws (225–455 grad steps).
**Record motion at the frontier:** (13,0.0) opened at dist_goal **6.98** / pose_step **0.073** — both campaign records; the arm reaches ~40% farther per goal than at d=8.
**NEXT (user-ordered): `arr95_hot4`** — +100k continuation, `--init-ckpt arr95_hot3/ckpt_0100000.pt` (HF), **`--goal-curric-d-start 13`**, d-max 22, all else byte-identical, NEVER-STOP monitor. NB hot3 predates the buffer-snapshot feature ⇒ hot4 starts buffer-empty ONE FINAL TIME and (running the new default-on `--save-state` code) will be the first run to persist `state_latest.npz` — hot5+ continuations carry full experience.

## 2026-07-08 — arr95_hot4 mid-run ledger @91.5k + FIRST PROJECTION TO d=22: ~2.3–5.2M more steps (~1–2.5 months continuous A100) at the measured 1.4×/level ladder inflation — with three identified break-points in the extrapolation

**hot4 band ledger so far (`5jihuwm0`, d-start 13, bar 0.95 + hot retargeting; gate = rolling arrival ≥0.95; wall-clock at the run's steady ~0.9 counter-steps/s = 7.3 env-sps ÷ 8):**

| Band | Entered → Cleared | Cost | ~Wall | Notes |
|---|---|---|---|---|
| (13,0.0) | 0 → 11,250 | 11.25k | 3.5h | includes cold-start buffer refill to 50k cap (last-ever buffer-empty chain start) |
| (13,0.2) | 11,250 → 16,350 | 5.1k | 1.6h | clean single-cycle grind |
| (13,0.4) | 16,350 → 20,400 | 4.05k | 1.3h | fastest of the run |
| (13,0.6) | 20,400 → 25,350 | 4.95k | 1.5h | peaked 0.953 at the gate |
| (13,0.8) | 25,350 → 46,650 | **21.3k** | 6.6h | campaign record at the time; ~6 thaw cycles of .80↔.94 whipsaw; tipped by thaw#13 |
| (13,1.0) | 46,650 → 73,050 | **26.4k** | 8.2h | NEW campaign record; 100th-pctl goals + archive hardening mid-chase (evict 40→47); cleared on record planner quiet (jump 3.08, frac_rand .157) |
| (14,0.0) | 73,050 → in progress | 18.4k+ | 5.7h+ | peaks .93, base .79–.84; thaw#25 ran 373 grad steps (vs 217–272 norm) = d=14 data genuinely reshaping the encoder; archive evict 47→66 |

**d=13 ladder total: 73k steps (~68k net of cold start) — vs ~50k/ladder at d=11–12 (hot3) ⇒ measured inflation ~1.4–1.5×/level.** The top two MSE bands are the cost center: (13,0.8)+(13,1.0) = 47.7k = 65% of the ladder. Same super-linear tail hot3 measured on the d-axis, now resolved onto the MSE-percentile axis: at fixed deep d, the hardest-novelty bands are where the steps go. Corroboration: (14,0.0) at 18.4k+ is already 4th-longest band of the campaign for what cost 5.1k one level down.

**Projection to d=22 (8 more ladders d=14..21, r=1.4×/level, base d=14≈100k, 78k steps/day):** d=14 ~100k → d=15 ~140k → d=16 ~196k → d=17 ~274k → d=18 ~384k → d=19 ~538k → d=20 ~753k → d=21 ~1.05M ⇒ **cumulative ≈3.44M steps ≈44 days continuous ≈34 chained 100k runs.** Bracket across r=1.3–1.5: **~2.3M–5.2M steps ≈ 30–66 days.**

**Three identified break-points in this extrapolation (order of importance):**
1. **Latent-diameter ceiling (breaks in our favor):** curric-d only binds while the buffer holds states FARTHER than d. Once d exceeds the workspace's max pairwise latent separation, every candidate is "under d", selection degenerates to pure high-MSE, and per-ladder cost PLATEAUS instead of inflating — but "reaching d=22" becomes partly label rather than harder task. The buffer diameter has never been measured (d-max 22 was set from the random-pair ceiling, a different quantity). **ACTION BANKED: measure max/95th-pctl pairwise latent distance over the buffer (5-min analysis on any state_latest.npz) before committing to a d=22 campaign.**
2. **Single-band blowout:** at r=1.4, by d≈17 the 1.0-band ALONE exceeds a 100k run — whole continuations inside one band, zero advances. Workable now that state persistence is default-on, but expect long silent grinds and plan monitoring around it.
3. **Encoder cost per level is returning:** thaw#25's 373 grad steps (vs 217–272 late-d=13 norm) says each new shell now buys real representation change again; this is where a growth-rate surprise (either direction) would come from. WM itself shows no ceiling (pred_loss/pvp still at all-time bests through d=14 entry).

Run status at entry: 91.5k/100k, (14,0.0) ~18.4k in-band, arrival base .79–.84 with .93 peaks, zero trigger fires (campaign streak intact through ~450k total steps), snapshots 10k–90k all written (15.07 GB each; wrist-res transitions ~300KB — compression is a banked future patch). Wrap + verdict entry to follow at 100k.

## 2026-07-08 — ★ arr95_hot4 STOPPED BY USER @92,450 (of 100k): d=13 ladder COMPLETE + d=14 reached — campaign stands at (1,0.0) → (14,0.0), 79 advances from tabula rasa; pivot to sps optimization (branch `sps-boost`) ★

**Run (`5jihuwm0`, from arr95_hot3@100k, d-start 13, buffer-empty start, NEVER-STOP monitor honored — stop was USER-ordered at 21:47 to free the GPU for sps work):** 6 advances in 92.45k steps. Full band ledger + d=22 projection logged in the entry above (this run supplied both campaign-record bands: (13,0.8)=21.3k, (13,1.0)=26.4k). Ladder: **(13,0.0) → (14,0.0)**, d=13 mastered end-to-end at the 0.95 bar, **(14,0.0) unfinished at 19.4k in-band** (oscillating base .79–.84, peaks .906–.938 — no sign of saturation, purely budget-cut).
**Campaign totals: (1,0.0) → (14,0.0), 79 advances, ~452k steps (~5.2 GPU-days) from tabula rasa. Zero frac-rand trigger fires EVER. WM at stop: pred_loss 0.0011–0.0013, pvp 0.007–0.011 (all-time-best band, still improving at d=14).** Planner records set this run: jump 3.07 / frac_rand 0.157 / holds 0.533. 26 thaws (217–373 grad steps; #25's 373 = d=14 reshaping the encoder).
**Teardown NOTE (chain-relevant):** SIGINT raised KeyboardInterrupt mid-step — the graceful final-save path did NOT execute. Durable artifacts: **`ckpt_0092000.pt` on HF** (weights + 64-slot goal archive) and **`runs/arr95_hot4/state_latest.npz` @ step 90000 (15.07 GB, LOCAL ONLY — periodic snapshots don't upload; push to HF manually before any cross-box hot5)**. Effective loss vs perfect teardown: 2k steps of buffer freshness + 450 steps of weights. hot5 recipe when wanted: `--init-ckpt <hf>/arr95_hot4/ckpt_0092000.pt --goal-curric-d-start 14 --init-buffer <path-to-state_latest.npz>` (or `auto` if the npz sits beside the ckpt), all else byte-identical. **Future patch banked:** install a proper SIGINT/SIGTERM handler that sets `_stop` instead of raising, so user-stops get the final snapshot+upload path.
**NEXT WORKSTREAM (user): increase sps A LOT — branch `sps-boost`.** Starting map from prior profiling: sps 15.7 (young buffer) → 7.3 (50k cap); GPU compute-bound (util 66–81%); dominant costs = (1) CEM rollout forced EAGER fp32 (compile+bf16 miscompiles to NaN) at 300 samples × 30 sequential iters × every step, (2) consolidation epoch scaling with buffer size. Candidate levers: fix/fence the bf16-compile NaN (biggest single win), CEM iter/sample reduction or early-stop, proximity-gated CEM precision (fewer iters far from goal), consolidation subsampling at cap, wrist-res/JPEG buffer compression (also fixes 15GB snapshots). Jacobian trust region remains DECLINED — not on this list as a default.

## 2026-07-08 — ★ sps-boost VALIDATED + MADE DEFAULT: 7.3 → 17.6 sps (2.4×) at the 50k buffer cap, arrival/NaN/planner-mechanics unchanged — two levers, A/B-tested against hot4's final window ★

**Method:** each lever implemented + tested ITERATIVELY (user protocol): 3k-step chain runs from `arr95_hot4/ckpt_0092000.pt` + the restored 50k-transition `state_latest.npz` (FIRST real use of full-state continuation — `[state] restored 50000 transitions` clean both times), regime-matched (14,0.0)/d-start 14, byte-identical recipe except the lever under test; scored vs hot4's 90–92.45k window (sps 7.3, arrival .797–.930, pred_loss .0011–.0013).

| Metric | hot4 baseline | A `sps1_consol` (cap 60) | B `sps2_cemstop` (cap 60 + early-stop) |
|---|---|---|---|
| sps median | 7.3 | 16.2 (max 18.1) | **17.6 (max 20.3) = 2.4×** |
| arrival mean/min/max | ~.82 / .797 / .930 | .857 / .750 / .945 | .865 / .729 / .938 |
| pred_loss | .0011–.0013 | ~.0030 stable (declining in-window) | ~.0031 stable (halves .0028↔.0035, no drift) |
| CEM iters/replan | 30 | 30 | **21.5 mean (14–28)** |
| cem/finite_frac min | 1.000 | 1.000 | 1.000 (zero NaN candidates, both runs) |
| holds / jump | .38–.46 / ~4.1 | .402 / 4.21 | .435 / 4.00 |

**Lever 1 — `--consolidate-max-steps` (NOW DEFAULT 60):** the `epochs × buffer/batch` burst formula scaled consolidation with the buffer — at the 50k cap that's 390 grad steps per burst every 70 env-steps (5.6 grad/env-step) vs ~23 young ⇒ THE sps 15.7→7.3 decay driver, now measured. Cap 60 = 6.5× less burst compute, still ~110 replay samples/env-step; each burst iteration draws a fresh random batch so the cap is an unbiased subsample (uniform coverage across bursts — no rehearsal bias). Cost: pred_loss equilibrium rises to ~0.003 (lighter fit to the frozen buffer — train-loss metric, not generalization; arrival unaffected). **Fallback if long-run drift appears: 120.**
**Lever 2 — `--cem-early-stop` (NOW DEFAULT 0.005, min-iters 12):** breaks the 30-iteration CEM loop when the mean elite cost's relative change ≤0.5% — refits barely move past the plateau, so the returned plan is the converged one. Mean 21.5/30 iters at (14,0.0); +9% sps on top of lever 1 (CEM dominates once consolidation is capped). Elite branch only; `cem/iters_used` logged for visibility. Guards: relative tol, min-iters floor, diverged (inf) elites never trigger the stop, converged-plan diags (min_cand_to_goal etc.) still computed on the stopping iteration.
**NaN verification (user-ordered):** the historical failure is compile+bf16 CEM — untouched (CEM stays eager+bf16 via `predict_eager`; compiling CEM was previously measured ~1.0× anyway, it's compute-bound). `cem/finite_frac` = 1.000 min across BOTH experiment runs end-to-end; only nan strings in logs are the standard `crit_loss=nan` placeholder + step-0 cold-start. Also fixed: `math` import (early-stop uses `math.isfinite`).
**DEFAULTS FLIPPED (this commit):** `--consolidate-max-steps 0→60`, `--cem-early-stop 0.0→0.005`. Legacy behavior reachable via `--consolidate-max-steps 0 --cem-early-stop 0`. Canonical launch blocks NO LONGER need these flags — future hot5+ chains inherit 2.4× automatically. Merged `sps-boost` → main.
**Acceptance caveat for the next long run:** 3k test windows contain NO thaw (cotrain fires at 3500) — cotrain's own 11700-budget flatline loop is untouched by either lever, but the first long continuation should watch pred_loss over its first 10k as the at-scale check (fallback: cap 120). ETA impact on the d=22 projection: 2.4× throughput ⇒ the 30–66-day continuous estimate compresses to **~13–28 days** at unchanged band economics.

## 2026-07-09 — ★ sps3_10k AT-SCALE DEFAULTS CHECK PASSED (10k steps, 2 thaws crossed): pred_loss flat-to-FALLING across both cotrains, arrival 0.890 last-2k, 17.6 sps sustained, zero NaN — the 2026-07-08 acceptance caveat is CLOSED, cap-120 fallback NOT needed ★

**Setup:** pure-defaults chain run — NO sps flags passed, byte-identical to how a future hot5 launches: `arr95_hot4/ckpt_0092000.pt` (HF) + hot4 `state_latest.npz` (`[state] restored 50000 transitions`), regime (14,0.0)/d-start 14, 10k steps crossing cotrains at 3500 AND 7000 — the thaw×cap interaction the 3k A/B windows could not see. W&B `xymyja69`. Wall 22:37→00:04 UTC ≈87 min (incl. two ~2-min cotrain pauses + HF uploads); 80k env-steps.

| Acceptance | target | result | verdict |
|---|---|---|---|
| pred_loss | flat ~.003, no drift across thaws | 1st-half mean .0031 → 2nd-half **.0026** (drift DOWNWARD) | PASS |
| arrival | ≥ baseline .86 | run mean .859, last-2k **.890**, final-window .86–.91 | PASS |
| sps | holding ~17 | **17.6** instantaneous (excl. thaw pauses; = the A/B number exactly); 16.1 cumulative incl. pauses | PASS |
| cem/finite_frac | 1.0 | min **1.0000** over all 199 rows | PASS |

**Thaw crossings (the actual test):**
- **3500:** 312/11700 grad steps, CONVERGED. pred_loss pre .0018–.0040 → post .0019–.0038 (settles ~.0025 by 3800); arrival dips .83–.85 for ~250 steps, recovers .87–.88; frac_rand continuous through the crossing.
- **7000:** 364/11700, CONVERGED. pred_loss mean .0028 → .0028 with a TIGHTER post band (.0022–.0034 vs .0020–.0042); arrival .833 pre → **.893** post — no dip at all.
- pred_loss spikes (.0156@2200, .0086@4350, .0067@9550) are contact-novelty transients (contacts/s→0.2 entering the window), recover ≤200 steps, uncorrelated with thaws. Grad-step counts (312/364) match hot4's per-thaw range (260–344) — cotrain economics unchanged by the consolidation cap.

**Encoder:** rank_frac_probe .2496 → .2532 (thaw 1) → .2459 (thaw 2) — ~47–49/192 eff dims (RankMe, fixed probe), per-thaw wiggle in BOTH directions; rank only moves AT thaws (encoder frozen otherwise), and cross-run absolute values are approximate (probe set re-frozen per run from that run's warmup obs). frac_rand smooth .16–.24 all run. Clarified (user Q): the LIVE sigreg β is `--cotrain-beta` **0.02** (thaws only) — the 0.09 `--sigreg-weight` default is gradient-INERT during consolidation because SIGReg acts on encoder outputs and the encoder is frozen there (`train.py` wm_update: `sig = sigreg(emb…)`, `emb = wm.encode(...)`).

**Curriculum:** d=14 k=1 the entire 10k (arrival .84–.91 never clears the .95 advance bar) — the same d=14 grind hot4 ended in; defaults change nothing about ladder behavior, but note the at-scale check therefore ran at ONE band.

**Chain point (freshest):** `runs/sps3_10k/state_latest.npz` (15.1 GB local; also HF `sps3_10k/state_latest.npz`) + HF `sps3_10k/ckpt_0010000.pt`.

**Next — branch `sps-boost2` (ranked; protocol unchanged: one lever per 3k chain run vs the 17.6-sps/0.86-arrival baseline, defaults flip only on pass):**
0. **Re-profile FIRST** — 60-sec py-spy on a scratch run under the new defaults for the true post-cap split (CEM vs env-step vs misc). GPU util was 66–81% pre-speedup; if env rendering is now the wall, the lever is MORE ENVS (8→12/16, also amortizes the CEM batch), not planner surgery.
1. **CEM warm-start** mu from the previous step's converged mean — H=1 + replan-every-1 makes consecutive problems near-identical yet every replan starts from zeros; this run's iters_used mean **23.0** (A/B was 21.5) says the plateau test still burns most of the 30-iter budget. Warm-start should pull the plateau earlier (~12–14 plausible), compounding with the shipped early-stop. Bounded exploitation risk (std still reopens at init_std).
2. **Proximity-gated CEM budget** — full iters/samples only near-goal (dwell/shrink phases); mine the iters_used-vs-dist_goal correlation from `xymyja69` + the experiment runs to size it BEFORE writing code.
3. **Sample count 300→200/150** (elite fraction preserved) — linear compute cut, highest quality risk at hard bands; goes LAST with the strictest A/B.

## 2026-07-09 — sps-boost2 lever 1 A/B: raw CEM warm-start FAILED on quality (arrival 0.51 vs 0.89) — root cause: UNCLAMPED-MEAN RATCHET; retry with clamp+decay (sps5)

**Profile first (perf/t_* timers shipped, `a030f32`; py-spy unattachable in this container — ptrace denied):** under the new defaults at (14,0.0)/50k buffer: **plan 87.0% of wall**, rest 7.9%, env_wait 2.6%, goal 2.4%, learn (capped consolidation) **0.12%**; GPU ~89% busy, 380 ms/step. More-envs lever DEAD (env fully hidden by async overlap; GPU saturated). Planner iters are the only surface. PufferLib assessed and rejected on the same numbers (env-side lever, ≤1.03–1.16× ceiling, wrong paradigm for WM+CEM).
**Lever 1 as-implemented (`b6659d0`, `--cem-warm-start`, default OFF):** seed replan mu from the env's previous converged plan (cold on resets + goal-refresh steps), std reopens at init_std. A/B `sps4_warm` (3k chain from sps3_10k endpoint, W&B `6008sbxk`) vs sps3_10k@7–10k window:
| | sps4_warm | baseline | |
|---|---|---|---|
| sps inst / iters | **25.3** / 20.7 | 17.6 / 24.3 | speed "win" is an artifact |
| arrival / reach | **0.513** / 0.118 | 0.891 / 0.311 | FAIL |
| dist_goal / pred_loss | 11.9 / **0.0158** | 5.0 / 0.0026 | FAIL |
| pose_range / pose_step | 2.9 rad / 0.13 | ~0.5 / 0.035 | flailing |
| action_rate / tau_sat / energy | **0.83 / 0.37 / 1.51** | 0.051 / 0.039 / 0.19 | ratchet signature |
**Mechanism (bad from window 1, not a slow feedback loop):** LeWM CEM optimizes UNCLAMPED (caller clamps only the executed action; cem_buf stores the raw mean). Cold zeros re-anchored every replan; warm-seeding removed that anchor → mean magnitude ratchets across consecutive replans to the clamp walls → max-torque directional sweeps; wild data also drove pred_loss 6× up and polluted the buffer. **DO NOT CHAIN from `sps4_warm/state_latest.npz`** — chain point remains sps3_10k endpoint. finite_frac stayed 1.000 (not a NaN failure).
**Next (lever 1b, `sps5_warmdecay`):** `mu0 = clamp(prev_plan, ±1) * decay` (decay 0.5, `--cem-warm-decay`) — clamp kills the ratchet, decay makes persistence re-earn itself each step through the elites. Same A/B protocol; if quality fails again, abandon mean-seeding and eval warm-CANDIDATE (inject prev plan as forced candidate row 1, sampling stays cold) before falling back to proximity-gated budgets (lever 2).

## 2026-07-09 — sps5 (clamp+decay warm-start) FAIL: quality mostly recovered but iters 25.3 vs 24.3 = NO speedup — mean-seeding family CLOSED; mining kills proximity-gating too (r=0.05); pivot to flat iters-cap A/B (sps6)

**sps5_warmdecay** (`ebf51b5`, seed = clamp(prev,±1)×0.5; W&B in runs/sps5_warmdecay): ratchet gone (action_rate 0.090 vs sps4's 0.83) but residual motion bias persists (action_rate/tau_sat ~1.8× baseline, pred_loss 0.0051 vs 0.0026, arrival 0.852 vs 0.891) and the lever's point never materialized: **iters_used 25.3 vs 24.3, sps 18.4 vs 17.6 (+4.5%, noise)**. DO NOT CHAIN from sps5 state either; chain point stays sps3_10k.
**Root insight:** the early-stop plateau is governed by STD ANNEALING, not the mean's start — std reopens at init_std every replan, and elite-cost convergence tracks distribution narrowing. Warming the mean aims at the wrong variable; the pre-registered warm-CANDIDATE fallback dies by the same argument (one good elite row barely moves the plateau statistic) — SKIPPED, family closed.
**Lever 2 (proximity-gated budget) killed by its own sizing step:** across 254 cold-start windows (sps2+sps3+prof_timers), **pearson r(dist_to_goal, iters_used) = 0.049**; 98% of windows live at dist 4–6 (curriculum holds a narrow working band), near-goal (<4) is 1%. No context structure to gate on. (Caveat: window-mean dist vs single-replan iters — per-replan logging could reveal finer structure, but nothing here justifies building it.)
**Lever 2′ — flat --cem-iters cap (sps6_iters18, launched):** LeWM itself runs 10 iters everywhere but PushT (App. D); our mean is 23 at cap 30 and p10-stopped windows (18 iters) show no arrival penalty. Cap 18 + existing min-12 early-stop clips the non-plateauing tail (p90=27) → ~20% plan-time cut ≈ +~15-20% sps if quality holds. Flag-only A/B, same protocol, same guardrails (arrival/pred_loss/finite/tau_sat).

## 2026-07-09 — ★ sps6_iters18 PASS: flat --cem-iters 30→18 = +32% sps (17.6→23.2), arrival 0.854 vs matched-window 0.848, all guardrails clean — lever 3 (samples 300→200) launched stacked ★

**sps6_iters18** (flag-only, 3k chain from sps3_10k endpoint): iters 17.96 (capped) vs 24.3; **sps 23.2 inst (+32%)**; plan wall-share 87%→66%. Quality vs the MATCHED window (sps3's own first 3k — fresh chain, pre-thaw): arrival **0.854 vs 0.848**, pred_loss 0.0030 (accepted equilibrium), finite_frac 1.000, dwell_hold 0.42 (=), min_cand_to_goal 6.3 (=); motion CALMER than baseline (tau_sat 0.029 vs 0.039, action_rate 0.042 vs 0.051). The 0.891 in the strict table is sps3's best late window; passing precedent (sps1/sps2 → defaults) was arrival 0.857/0.865 at pred_loss ~0.003. Note move_vs_gap 0.83 vs 0.69 — plans commit harder per refit-dollar; no downstream cost visible.
**Why this works where warm-start couldn't:** iters beyond ~18 are mostly post-plateau refits (early-stop mean was 21.5–24 with p10=18 and no arrival penalty at p10); LeWM itself ships 10 iters everywhere but PushT. The cap harvests what the early-stop's plateau test leaves on the table (its min-iters floor stays 12 for the sub-18 stops).
**DEFAULTS NOT FLIPPED YET** — pending lever 3 + one combined 10k at-scale confirm (2-thaw, same acceptance as sps3_10k) before any flip, per the sps-boost precedent.
**Lever 3 launched (`sps7_samp200`):** --cem-samples 300→200 + --cem-elites 30→20 (elite fraction held at 10%), STACKED on iters 18 (sps2-on-sps1 pattern), reference = sps6. Linear candidate-cut on the remaining 66% plan share → projected ~+20-25% further if quality holds; pre-registered as the strictest-A/B lever (weakest-argmin risk at hard bands: watch reach/arrival min, min_cand_to_goal, cost_cv).

## 2026-07-09 — sps7_samp200 MARGINAL (+28% sps to 29.6 = 4.05× original; pre-registered guardrails pass BUT a 3-metric quality staircase across the stack) → 10k two-thaw confirm launched with abort criteria

**sps7** (s200/e20 stacked on i18, ref=sps6): sps **29.6** inst; finite 1.000, cost_cv 0.341 (no signal loss), min_cand_to_goal 6.39 (=sps6), tau_sat 0.030, arrival-min 0.79 (=sps6's 0.80) — every pre-registered check passes. **Yellow flag:** monotone drift across the lever stack in three coupled metrics — arrival 0.873 (sps3 matched) → 0.854 (i18) → 0.839 (+s200); reach .319→.305→.292; dwell_hold .457→.420→.394. Individually in-noise, jointly a coherent weak-degradation signal. pred_loss 0.0037 (band). NOT declared a pass.
**Decision frame:** campaign metric is learning throughput = arrival-progress/step × sps. +68% sps vs −3.4pt arrival nets positive UNLESS the drop is a CEILING effect (plans too sloppy to ever clear the 0.95 advance bar at frontier → ladder stalls) rather than a rate effect. 3k cannot distinguish; 10k two-thaw can, and costs ~50 min at 29 sps.
**`sps8_confirm10k` launched:** sps3_10k recipe + i18/s200/e20, chained from the same sps3_10k endpoint, thaws at 3.5k/7k. ACCEPTANCE (pre-registered): pred_loss flat-or-down across both thaws (band ~0.003–0.005), finite 1.000, sps ≥ ~28, **arrival run-mean ≥ 0.84 with no monotone decline across thaw segments**, and late-window arrival within ~2pts of sps3_10k's matched profile (its run-mean was 0.859, last-2k 0.890 → bar: last-2k ≥ 0.86). ABORT/FAIL → flip iters-18 ONLY (sps6 clean) and re-confirm that at 10k.

## 2026-07-09 — ★★ sps8_confirm10k PASS: full stack (iters 18 / samples 200 / elites 20) VALIDATED AT SCALE — DEFAULTS FLIPPED, sps-boost2 merged; 7.3 → 28.2 sps (3.9×) end-to-end, arrival/pred_loss/NaN profile matches the 300/30 planner ★★

**Pre-registered scorecard (all bars cleared):** arrival run-mean **0.869** (bar .84), last-2k **0.889** (bar .86, final windows .906–.922); segments .890→.847→.875→.889 — the 3k "staircase" was a transient mid-run sag with full recovery, same shape as sps3_10k's own profile, NOT structural (ceiling-effect risk retired: the cheap planner sustains .91+ windows at d=14); pred_loss halves .0031→.0031 dead flat across BOTH thaws (band .003–.005); finite_frac 1.0000; sps inst **28.2** (bar 28; 29.5 mid-run, tail dragged by evictions/hot windows). Thaws: 3500 = 352 grad steps (normal band), 7000 = **458** (above the 260–364 historical band ~30% — consistent with noisier directed data from the cheaper planner; converged, pred_loss unaffected; WATCH in hot5).
**DEFAULTS FLIPPED (this commit):** `--cem-iters 30→18`, `--cem-samples 300→200`, `--cem-elites 30→20` (elite fraction held 10%). LeWM-faithful legacy: `--cem-iters 30 --cem-samples 300 --cem-elites 30`. Lever lineage: iters cap +32% (sps6 clean pass), samples cut +28% on top (sps7 marginal → adjudicated at 10k), warm-start family closed (sps4 ratchet / sps5 no-effect — std-annealing insight), proximity gating killed by mining (r=0.05).
**Cumulative sps ladder:** 7.3 (pre-campaign) → 17.6 (consolidation cap + early-stop, 2026-07-08) → **28.2 (iters/samples cuts, this entry) = 3.86×**. perf/t_plan_frac 87%→~60%. d=22 projection: 30–66 days ÷ 3.86 ≈ **~8–17 days** at unchanged band economics.
**Chain point (freshest):** `runs/sps8_confirm10k/state_latest.npz` (15.1 GB local + HF) + `sps8_confirm10k/ckpt_0010000.pt` (HF). sps4/sps5 states remain DO-NOT-CHAIN.
**hot5 launch note:** canonical command needs NO sps flags (all defaults); watch thaw grad-step counts (458 flag above) and arrival band vs this run's profile over the first 10k.

## 2026-07-09 — `arr95_hot5` LAUNCHED: 100k continuation from arr95_hot4@92k (d-start 14, d-max 22) — FIRST campaign run on the 28-sps defaults (18/200/20 + cap60 + early-stop), pure hot4 lineage by user choice

Chained from `arr95_hot4/ckpt_0092000.pt` + `runs/arr95_hot4/state_latest.npz` — **user chose the pure-campaign lineage over the freshest (sps8) chain point**: hot4's buffer is single-config campaign data, sps8's endpoint mixes four planner configs across the sps detour (its 20k steps of extra d=14 grind are forfeited; ~50 min to re-earn at the new rate). Zero sps flags — canonical command inherits all five defaults. ETA ~9 h wall for 100k (vs hot4's ~30 h): ~7.9 h stepping at ~28 sps + ~28 thaw pauses.
**Watch items:** (1) thaw grad-step counts — sps8's thaw-2 hit 458 vs the 260–364 historical band (noisier directed data from the cheaper planner suspected); if hot5's thaws trend high AND pred_loss creeps, the fallback is `--cem-samples 300` legacy for the next chain. (2) arrival band vs sps8's profile (run-mean 0.869, segments dipping to ~0.85 mid-thaw-cycle, recovering ≥0.89). (3) d=14→15 advance needs windowed arrival ≥0.95 — sps8's late windows hit 0.914–0.922, closest the campaign has been at d=14. Curriculum events surface in the monitor.
Stop authority: user's (hot3 precedent). Merge of sps-boost2 is local-only for now — `git push` blocked on gh auth in this box (user to run `gh auth login`).

## 2026-07-09 — arr95_hot5 KILLED AT 50,000 BY STORAGE, NOT TRAINING (MooseFS EIO mid-snapshot) — relaunched as `arr95_hot5b` from the intact 40k state; snapshot path hardened (`33df04d`)

**Crash:** `OSError: [Errno 5] Input/output error` in `np.savez` closing the periodic 15GB state snapshot at step 50,000 — /workspace is RunPod network storage (77T free; NOT disk-full), same transient-I/O family as the ~10-min HF upload stall at step 3,000. Training itself was healthy to the last row. **Fix shipped:** periodic snapshot now log-and-continues on OSError (prior state_latest stays intact via tmp+rename); final end-of-run snapshot still raises.
**First-half ledger (0→50k, ~5h wall):** d=14 pctl ladder 0→0.20 (step 3,300; first 0.95-clear of the d=14 era) → 0.40 (26,250) → **0.60** (26,700 — minimum-time cascade clear); 0.60-band grind through 50k (arrival 0.83–0.87, eviction churn 17→32/window as the WM drains the surprise pool, pred_loss 0.0017–0.0035, all 14 thaws in/below the 260–364 band — sps8's 458 outlier did NOT reproduce on pure lineage). ckpts on HF through `ckpt_0050000.pt`; W&B `g0e3pztg`.
**Recovery:** `arr95_hot5b` = ckpt_0040000 + hot5 state_latest@40k (consistent pair; 10k steps re-earned ≈ 1h), total-steps 60,000 to complete the original 100k budget. Curriculum re-climbs from pctl 0 — mastered bands re-clear at minimum time (0.40 took 450 steps), expect the ladder back at 0.60 within a few k steps. Combined ledger = hot5(0–40k) + hot5b(0–60k).

## 2026-07-09 — ★★ arr95_hot5(a+b) COMPLETE: the 100k continuation ends at (d=14, pctl 0.80) — 4 pctl rungs climbed from the crash restart, arrival 0.871 run-mean, ~10.7h total wall at the 28-sps defaults (vs ~30h hot4-era) ★★

**hot5b (60k, 5.7h wall, W&B `ub3tfxyf`):** ladder re-climb 0→0.20 (8,100) → 0.40 (25,800) → 0.60 (48,900) → **0.80 (55,800)**; the 0.60 band that stalled the original run 23k+ steps cleared in **6.9k** on the banked buffer — restart cost was real (re-climb ≈ 48.9k steps to recover position) but the crash-era experience compounded. arrival run-mean 0.871, pred_loss 0.0014–0.0075 (thaw-transient tail), finite 1.000, 17 thaws all ~240–399. Ends **1.5 rungs from d=15** (0.80 band uncleaered, then the 1.0 band gates d growth; pctl-max=1.0).
**SECOND storage EIO, endgame this time:** the FINAL state snapshot raised (path was intentionally unhardened) BEFORE the final model ckpt block → **ckpt_0060000 lost** (incl. final goal_archive). Fixed + pushed (`684a0c6`): final ckpt now saves FIRST; final snapshot retries 3× / 60s and never kills teardown. Volume threw 3 I/O incidents today (upload stall @3k, periodic-snapshot kill @50k, final-snapshot kill) — RunPod MooseFS flakiness is now a named campaign risk.
**Chain point for hot6:** `arr95_hot5b/ckpt_0059000.pt` (HF, last uploaded) + `runs/arr95_hot5b/state_latest.npz` (50k periodic; 9k-step weight/buffer skew — buffer misses the newest 0.80-band transitions, acceptable). Curriculum restart: d-start 14 (pctl re-climbs; this run proves mastered rungs re-clear fast on a banked buffer).
**Campaign ledger:** tabula rasa → (14, pctl 0.80): 87 advances… d=14 inner ladder 4/5 rungs done. At ~28 sps the projected d-per-level economics from 2026-07-08 (~8–17 days to d=22) still hold; hot6 (100k, ~10h) plausibly finishes d=14 → d=15-16.

## 2026-07-10 — ★ sps-boost3 (levers 1–3) CLOSED: canonical recipe = 8-env + LATENT CACHE + async uploads @ 31.3 sps (4.3× campaign start); 12-env FAILS its quality bar at 10k (0.830 vs 0.85) and is PARKED at 36.7 sps ★

**Lever 1 — frozen-encoder latent cache (`c77e082`), CANONICAL:** consolidation trains the predictor on cached latents (live-fill at add, lazy backfill, invalidate-on-cotrain); CPU equivalence bitwise (delta 0.0). sps9 3k A/B: **31.3 vs 27.3 sps (+15%)**, pred_loss 0.0018 (better than ref), arrival 0.875. Corrected profile: hot5b steady learn=27.9% (prof_timers' 0.12% was measured inside the start_steps warmup gate — wrong); cache removes the encoder-forward share, remaining t_learn ~0.17 in burst windows = predictor's own fwd/bwd (irreducible). **Invalidation validated under fire in sps12 (both thaws): t_learn spike-decay signature, pred_loss flat — no stale-latent divergence.**
**Lever 3 — background HF ckpt uploads (`c77e082`), CANONICAL:** single-worker thread, ordering preserved, flushed before [done]; the 10-min-stall class can no longer block the loop. Clean across sps9–sps12.
**Lever 2 — 12 envs: PARKED (quality).** Mechanics shipped: restore now fills fewer-env snapshots into the first k rings (`fix(state)`); NOTE 12-ring cap = 50k/12 ≈ 4,166/ring → an 8-ring snapshot restores only ~33k (newest-skewed). Consolidation dilution renormalized via cap 60→90 (grad/transition parity). Results: sps10 (cap 60) 37.5 sps but pred_loss +55%/arrival 0.821; sps11 (cap 90) 36.3, pred 0.0024, arrival 0.786 (3k window structurally confounded — fresh rings fill until ~4.2k); **sps12 10k/2-thaw adjudicator: sps 36.7, pred flat, finite 1.0, but arrival last-2k 0.830 < 0.85 bar → FAIL per pre-registration.** Trend was rising at the wire (0.822→0.841) — revisit hooks: 20k confirm, or multi-GPU where per-env rate doesn't drop (12-env cost per-env rate 3.91→3.13 dec/s at t_plan 0.71).
**sps ladder:** 7.3 → 17.6 (07-08 defaults) → 27.3 (hot5b delivered, 07-09 defaults) → **31.3 CANONICAL** (+36.7 parked). d=22 projection at canonical: ~7–15 days.
**CHAIN WARNING:** `sps12_confirm10k/state_latest.npz` is 12-ring — REFUSED by 8-env runs (starts empty). hot6 chains from `arr95_hot5b/ckpt_0059000.pt` + `runs/arr95_hot5b/state_latest.npz` (8-ring, @50k). hot6 recipe = hot5 command verbatim (zcache+async are automatic); first-10k watch per hot5 pattern.

## 2026-07-10 — ★ FIRST EXTERNAL-GOAL EVAL (`src/eval_goal_photo.py`): the d=14 checkpoint recreates 50% of random photographed poses (fixed scene), several to 0.02–0.13 rad — plus two structural discoveries about pixel-latent goals ★

**Protocol (user-proposed):** random smooth-walk pose → wrist PHOTO = goal → fresh reset → frozen training act-stack (CEM + dwell, eps 2.8) for 120 decisions. Ckpt `arr95_hot5b/ckpt_0059000` (d=14, pctl 0.80). 3 runs × 24–32 goals.
**Results:** fixed scene **0.50 arrival** (32 goals; arrivals reach qerr 0.016–0.4 rad inf-norm, median t≈7 decisions; misses mostly stall at min_d 4–7, hard misses 3/32 >8). Random scene (training's default): 0.19–0.33 — every photo carries an **irreducible scene-mismatch component** (blocks re-roll per reset; the arm can't move them), montages show it directly. The user's "d=14/22 → ~½" guess empirically exact for in-distribution pose photos, though not via linear-d (see below).
**Discovery 1 — pixel-latent goals conflate pose and scene:** ||z−z*|| includes block layout; cross-scene goals are structurally unreachable (pose-perfect miss: qerr 0.023 at d 10.5). The curriculum's d≤14 budget SELF-SELECTS same-scene goals (cross-scene sits ~20+), so training arrival is a same-scene metric — fine for the reach campaign, but photo-goal/hardware tasks need fixed scenes, an overhead goal cam, or a pose head.
**Discovery 2 — wrist-cam pose ALIASING both ways:** latent arrivals at wrong poses (qerr up to 3.3 rad — same view, different joint solution) and the latent-locality scale is steep: even 5 walk decisions put d0 at ~20–29 (≈ random-pair distance), fixed scene or not — the d budget ≈ "within a few decisions' latent travel," not a fraction of pose space.
**Artifacts:** runs/goal_photo_eval{,2,3}/ (per-goal npz + results.json + montages). Script committed; eval is cheap (~6 min/32 goals, 1 env).

## 2026-07-10 — STACKING STAGE 1 (WM contact-knowledge probe, `src/probe_contact_wm.py`): the WM knows NOTHING about contact, and contact is nearly unreachable without targeting — scripted collection is mandatory for manipulation

**Harness anchored on real buffer windows first: pred/persist 0.21 ✓ (training band 0.1–0.5)** — numbers below are trustworthy.
1. **No contact dynamics + no contact data**: even with a block TELEPORTED into the sweep path, random-walk collection touches it on 0.4% of decisions (10/2400; v1's un-targeted sweep: 0/4200). The 15 contact windows read pred/persist ~1.24 (contact predicted as "nothing happens"). The campaign's sparse training contacts came from CEM chasing high-MSE states, not chance — un-targeted exploration will never generate a contact curriculum.
2. **Predictor is distribution-NARROW**: 0.21 on-distribution → **3.25 on mild OU-walk OOD** (15× degradation). Stage-2 contact data must go through consolidation/cotrain, and early manipulation planning will run on a WM that needs to learn contact from scratch.
3. **Block layout: distance-salient, not linearly decodable** — R²≤0 for block xy (even always-in-view block, 40 layouts) despite layout dominating latent distance (photo eval: 10–20 units). Planning cost signal exists; linear readouts/subgoal extraction don't.
4. **Wrist-cam invariance map**: base yaw + one wrist dim are camera-invariant (R²<0) — the aliasing mechanism behind wrong-pose latent arrivals, now localized to specific joints.
**Stage-2 implication (not started):** targeted contact collection — use the TRAINED PLANNER to approach photo-goals near a placed block, then scripted close-in pushes/grasps; bulk-consolidate; re-run this probe as the acceptance test (contact pred/persist << 1 on ≥1k windows).

## 2026-07-10 — ★★ PARENT-CHILD EXPERIMENT (branch feat/vla-parent-child): a second VLA/scripted-driven SO-101 moving blocks in the child's view BREAKS the historical contact-collapse pathology (child contacts 4× and diverging) and teaches the WM real contact dynamics (pred/persist 0.89 vs control 1.16 — FIRST sub-1.0 ever) ★★

**Goal (user):** second arm ("parent") spaced out on the table, VLA-driven, moves blocks where the child's wrist cam looks; short run from a medium (d=7-9) ckpt; measure contact + movement-understanding deltas.
**Infrastructure shipped (6 commits):** MjSpec-attached parent arm (namespaced, child semantics byte-identical, contact attribution split), gaze-aware orchestrator (frustum projection → color-worded instructions), SmolVLA driver (radian-native — the deg interpretation shrinks real chunks 57×; pre-tokenized language; 0.6s/50-action chunk, 3ms pops), batched ParentFleet in the trainer (--parent-vla {smolvla,scripted}, one context RPC per refill), in-process/subproc parity.
**Driver verdicts:** SmolVLA zero-shot on sim renders: mechanically live, instruction-differentiated trajectories, but ZERO tabletop contact (jammed or not) — parked for finetune/hardware. π0.5: requires openpi transformers_replace overlay = external-code integration, USER-GATED (venv staged at /opt/venvs/pi05). Scripted rise(home)-drop-sweep at the calibration-verified pose: 45-172 contacts at every radius 0.14-0.41 → the experiment's driver (parent dosage in-training 0.199 contacts/step).
**Null-treatment post-mortem (a day of debugging, 3 stacked causes):** (1) parent base MESH-WELDED into the table since first attach (saturated torque, zero motion) → 2cm float; (2) precision aiming unwinnable (bistable transit collisions) → coverage plow; (3) parent_object_contacts silently DROPPED by the fixed _INFO_KEYS stacking in both env backends → optional-key stacking. Liveness gates now standard.
**Twin runs (pc_treat/pc_ctrl, 6k steps from arr95_hot@24000 d=8, fresh buffers, identical seeds; W&B):**
- **H1 CONFIRMED:** child contacts by 1.5k-segment: ctrl 0.33→0.24→0.16→0.08 (→0.00 at the wire — the consolidate-into-boredom spiral) vs treat 0.58→0.91→0.42→0.34 (post-thaw rebound): **4× and diverging**. frac_touch_block 5× by the end. The parent's re-novelization sustains the chase permanently.
- **H2 CONFIRMED (the headline):** endpoint probes (identical protocol/seeds): contact-window pred/persist **treat 0.89 vs ctrl 1.16** (n=15) — the first checkpoint in campaign history to predict contact BETTER than persistence (hot5b@92k+: 1.24). Bonus: OOD free-window robustness 1.02 vs 1.64. Gains sit in the PREDICTOR; the encoder metric (block-displacement monotonicity) is a wash (~0.6-0.7 rho both) as expected at one gentle thaw. Caveat: mse_block stays 2-3× elevated under treatment (parent actions are exogenous → irreducible surprise component; mild TV-noise flavor, exploited here as the attention magnet).
- **H3 CONFIRMED (the cost):** arrival 0.70-0.80 vs 0.91-0.92 (~20pt stale-goal tax), dist_goal ~2× — the photo-eval cross-scene mechanism live in training. Mitigations for a campaign-grade version: goal re-photograph on parent-touch, shorter goal windows, or parent pauses during goal-cycles.
- Cost: scripted parent ≈ 1-2% sps.
**Next steps if pursued:** finetune SmolVLA on sim renders (the collector data now exists), goal-freshness mitigation for H3, longer runs (does the treated WM's contact skill transfer to child-driven manipulation?), and the stacking curriculum from the 07-10 stage-1..4 plan.

## 2026-07-10 — ★ STAGE-2 CONTACT PROBE AT SCALE (`src/probe_contact_scale.py` + ChildPlow rake, n≈950 contact windows × 2 seeds): treatment's OOD-robustness REPLICATES with CI separation, but EGO-contact is unpredicted by EVERY checkpoint (3.3–7.5 ≫ 1) — the parent taught OBSERVED contact, not ego contact; maturity ANTI-correlates with robustness ★

**Collector (`src/smoke_child_sweep.py` ChildPlow):** the child arm CANNOT reuse the parent's plow. Three transfer failures, all measured: (1) the parent's depth calibration mirrors wrong (base yawed π + 2 cm float; child contact band is POSITIVE lift); (2) the parent's all-zeros HOME is a TABLE-LEVEL pose for the child — it drags fingertips through the block field and friction-pins every joint (arm frozen whole episodes, 0 contacts); (3) the fleet's TIMED rise-drop-sweep cadence fails on the child because lift/elbow PD convergence lags the pan ramp — the arm crosses the block's bearing ~13 cm high/short. Also: fingertip max reach at table level is r≈0.33, so v2's r=0.30-0.32 "front spot" sat at the reach EDGE (this is why stage-1's random walk got 0.4% contact). **Working design: convergence-gated state machine** — high home (lift −1.0, elbow 0.6) → approach hover (−0.2, 0.8) → touchdown just BEYOND the block (0.15, 0.55; fingertip r≈0.31) → RADIAL RAKE inward to (0.65, 1.40) (fingertip r≈0.21), timeout-fallbacks rise home (anti-wedge), contact during descent short-circuits to rake. Teleport block0 to r=0.25 (mid-annulus). **Yield: ~51% contact decisions (853-876/1680), 8/8 layouts** — ~130× stage-1's walk.
**Probe design:** collection is CHECKPOINT-INDEPENDENT (scripted actions never consult the WM) → all checkpoints score the IDENTICAL windows; window classes walk_free/rake_free/rake_contact (contact = any object_contacts in the Hb-window, v2 labels; motion labels still rejected — teleport-settle artifacts); episode-bootstrap 95% CIs; two seeds (11 + 12 hold-out).
**Results (contact-window pred/persist, [95% CI], seed-11 / seed-12):**
| ckpt | walk_free | rake_free | rake_contact (n=934/958) |
|---|---|---|---|
| treat@6k | **1.08** [0.93,1.30] / **1.10** [0.95,1.30] | **1.38** / **1.33** | **3.92** [3.06,5.09] / **3.30** [2.70,4.15] |
| ctrl@6k | 1.79 / 1.90 | 1.57 / 1.54 | 5.10 [3.96,6.58] / 4.96 [4.03,6.31] |
| base24k (shared init) | 2.02 / 2.13 | 1.92 / 1.79 | 5.08 / 4.48 |
| hot5b@59k (mature d=14) | 4.16 / 4.42 | 1.96 / 1.42 | 7.47 [6.37,9.18] / 5.74 |

**Verdicts:** (1) **H2-at-scale REPLICATES**: treat best in every class, CI-separated from ctrl on walk_free both seeds and on rake_contact in the seed-12 hold-out (3.30 [2.70,4.15] vs 4.96 [4.03,6.31]) — the n=15 endpoint result was real, and the parent's gift extends WEAKLY to ego-contact. (2) **Ego-contact is unlearned, universally**: every checkpoint reads ≫1 on rake_contact at n≈950 — the treated WM knows "blocks move when the PARENT touches them", not "when I shove them". Graze-contact (prior 0.89) ≠ plow-drag. Stage-2's premise confirmed at scale. (3) **Maturity narrows**: hot5b (92k+ steps) is the WORST off-distribution in 5/6 cells (walk_free 4.2-4.4 vs the 6k twins' 1.1-2.1) — stage-1's "distribution-NARROW" finding now sits on a 4-model ladder; more same-distribution training actively hurts robustness. **Anchor caveat:** treat-on-own-buffer 0.69 (n=297) — above the mature 0.1-0.5 band, below the >1 bug gate; explained by the short fresh-buffer run + exogenous parent churn (code path unchanged from stage-1's validated harness, which re-anchored hot5b at 0.21).
**Fixes shipped en route:** `offline_train.py` now builds the WM with the ckpt's `no_proprio` (z_dim 192 vs 256 crash on every campaign ckpt) and grew `--wm-only` (goal-conditioned actor predates its SAC path; policy weights pass through). `buffers_from_collection.py` converts probe collections + ring snapshots to the offline flat format (`--max-rings` for mix-ratio/RAM control — the loader pads all streams to the longest).

## 2026-07-11 — ★★ STAGE-2 CONSOLIDATION: first near-parity EGO-contact WM (holdout rake_contact 3.30 → 1.27 [1.14,1.45], 2.6× CI-separated) from 2,400 scripted rake transitions + 2,000 predictor-only grad steps — after peeling THREE stacked failures, incl. a replay-sampler bug that affects the ONLINE trainer ★★

**The acceptance loop:** consolidate rake data (seed-11 collection + 2-ring treat replay, 24%/9.5%/66.6% rake/walk/replay) into treat@6k via `offline_train.py --wm-only`, probe on seed-12 HOLDOUT layouts (memorization guard). Three attempts, each failing one layer deeper:
1. **v1 (joint WM training, lr 3e-5): encoder collapse.** rake_contact EXPLODED 3.9→13.1/16.9 while walk_free "improved" to 0.47. Measured mechanism on identical frames: the consolidated encoder compresses contact-step latent motion −31% and z_std −18% (persist denominator shrinks; "predict by seeing less") — the exact shortcut the online frozen-encoder cache (lever 1, c77e082) was built to block, now demonstrated offline. SIGReg (24→7) did not protect the contact subspace.
2. **v2 (`--freeze-encoder`, lr 5e-5): still degraded** (holdout rake_contact 4.96) — yet train pred hit 0.0012 on data "containing" rake. Contradiction resolved by mini-repro (probe-style ≡ trainer-style MSE, bit-identical): the batches never contained rake.
3. **THE SAMPLER BUG (`train.py sample_wm`, fixed):** the batch filled in STREAM ORDER with an early break — with many streams, every batch came from the first ~5 (here: the WALK episodes; both v1/v2 consolidations trained on ~300 OU-walk transitions and never saw one rake window — 2,000 updates of smooth-only fine-tune = the degradation). Fix: length-proportional cross-stream allocation + pooled shuffle before truncation (reduces to old allocation for equal rings). **ONLINE implication:** the online trainer's batches were first-rings-dominated too — content-neutral (rings iid) so canonical numbers stand, but batch diversity improves under the fix; flag for the next online run's watch window. Also fixed en route: recency-mode used a stale `n` across streams.
3. **v3 (freeze + fixed sampler): the result.** Train pred honest (0.14→~0.03). Probe:
| window class | treat@6k holdout | consol3 TRAIN-dist (s11) | consol3 HOLDOUT (s12) |
|---|---|---|---|
| walk_free | 1.10 | 0.04 | **0.24** [0.16,0.34] |
| rake_free | 1.33 | 0.07 | **0.74** [0.64,0.85] |
| rake_contact | 3.30 [2.70,4.15] | 0.17 [0.14,0.21] | **1.27** [1.14,1.45] |

**Verdict:** targeted scripted ego-contact data DOES teach the WM ego-contact dynamics that generalize to unseen layouts — 1.27 vs the 3.3–7.5 every prior checkpoint reads, at trivial cost (~8 min GPU, predictor-only). The train/holdout gap (0.17 vs 1.27) says capacity absorbs far more than 40 episodes provide; formal bar ("<<1 on ≥1k windows") not yet met → **dose-response running** (3 more collection seeds → 9,600 transitions, consolidate, probe holdout-12). If data alone crosses 1.0, stage-2 closes PASSED with a pure-offline recipe; if it plateaus, the online-integration variant (scripted rake episodes mixed into training, parent-style) is next.

## 2026-07-11 — ⛔ USER DECISION: the scripted-lessons line (ChildPlow collection → offline consolidation) is CLOSED — the goal is EMERGENT block-moving through the WM/curiosity loop, not data injected around it

**Directive (user, verbatim intent):** "we need to get the emergent behavior of it moving blocks without 'collecting data for it' directly bypassing the wm." Scripted child-arm data collection + offline consolidation teaches the WM by force-feeding — it bypasses the learning dynamics the project exists to study. The v4 dose-response run was stopped mid-flight; **do not resume this line** (this supersedes the 07-10 stacking plan's stage-2 "scripted close-in pushes → bulk-consolidate" step). Artifacts stay parked for reference: consol v1-v3 ckpts (HF `pc_treat_rakeconsol{,2,3}`), rake collections (seeds 11-15), probe results — and the v3 result stands as an *existence proof* (ego-contact IS learnable from 2.4k transitions: holdout 3.30→1.27), useful as an upper-bound reference for what emergence must achieve, not as a training recipe.
**What remains valid:** the probe (`probe_contact_scale.py` + ChildPlow) as a MEASUREMENT instrument (it evaluates checkpoints; it doesn't train them); the sample_wm stream-order fix (online trainer benefits); offline_train fixes. The **guardian path stays the approved direction**: environmental enrichment (parent re-novelizing blocks) that shapes what the child experiences without scripting the child — already proven to sustain child contact 4× and teach observed-contact dynamics (0.89). Emergence-aligned levers for ego-contact: longer guardian runs (does watching eventually convert to doing?), H3 goal-freshness fix (stale goals currently waste the curiosity budget the child could spend at the blocks), curiosity/goal-selection shaping toward contact-adjacent states — all of which keep the WM loop as the only teacher.

**2026-07-11 addendum — EGO-CONTACT LEARNING CURVE under the guardian (holdout rake_contact, pc_treat intermediates):** 1k **4.48** [3.74,5.55] (≡ base24k's 4.48 — internal consistency: at 1k the treatment has barely diverged from the shared init) → 2k 3.64 → 3k 3.54 → 4k 3.18 → 5k **3.04** [2.54,3.74] → 6k 3.30. Watching DOES slowly convert to ego-contact knowledge (endpoints ~CI-separated; ctrl@6k reads 4.96, i.e. slightly WORSE than the base — the improvement is guardian-caused, not time-caused), but the slope is modest (~0.3/1k steps) and flattening, while H1's child-contact rate was still rising at the wire. walk_free improves monotonically 2.13→1.08 (robustness accrues steadily). **Read:** time alone plausibly needs 10k-20k+ more steps to approach parity — supports combining a longer guardian run with the dilution/geometry levers (curiosity-weighted WM sampling; guardian herding blocks into the child's reachable annulus r0.21-0.33) rather than waiting.

## 2026-07-11 — ★ PLANNER-VISIBILITY PROBE (`src/probe_planner_visibility.py`): the deliberate-manipulation switch is OFF for every checkpoint — CEM (horizon 1) cannot yet SEE contact value, though ego-contact knowledge migrates the contact action 164→56/203 toward the elite ★

**Question (user):** behavior flows through CEM + latent distance — so can the deployed planner prefer a contact action when the goal requires it? **Protocol:** stage PRE-contact states s* (ChildPlow as scene-stager only; bit-exact env rewind, det_err 0.00mm, 6 scenes); goal photo = the REAL outcome of one known contact decision from s* (one-decision-reachable by construction, block displaced 2.3-14.6mm); score that true-optimum action among {pan-mirrored twin, zero action, 200 Gaussian candidates at deployment σ} by one-step predicted cost (exactly cem_plan's arithmetic + params from ckpt args), then run the REAL cem_plan and execute its choice.
**Results (median over scenes):**
| ckpt | pred-rank of contact action (/203, 0=best; real-rank=0 by construction) | elite (top-20) | rank-corr pred↔real | CEM plan: contact / goal-progress |
|---|---|---|---|---|
| treat@6k | 164 | 0% | 0.11 | 100% / **−2.3mm** |
| ctrl@6k | 142 | 17% | 0.45 | 100% / −0.8mm |
| consol3 (ego-contact-aware, parked) | **56** | 0% | 0.25 | 100% / −1.0mm |

**The nulls that decide it:** from s*, 100% of RANDOM candidates also produce contact (arm already pressed on the block — contact is geometric, not chosen), and random candidates' median goal-progress is +0.1mm vs the planner's −0.8..−2.3mm: **no checkpoint directs the block toward the goal better than chance.** Verdicts: (1) deliberate contact is OFF everywhere — the emergence bootstrap's "CEM starts seeing contact" threshold is beyond pred/persist ≈1.27; (2) knowledge DOES migrate visibility — consol3 lifts the contact action from bottom-quartile (164) to above-median (56), ~3× rank improvement, so the dial moves with WM contact knowledge; (3) treat's rank-corr (0.11) is WORSE than ctrl's (0.45): the treated WM's contact-adjacent predictions are noisy (exogenous-surprise flavor) while ctrl's persistence-like predictions rank candidates by pose alone; (4) at horizon 1 a single decision's ~5mm block displacement is likely near the latent noise floor — --cem-horizon > 1 (with its sps cost) is a planner-side lever if knowledge alone stalls.
**Implication for the 3-arm emergence experiment (baseline / +herding / +herding+curiosity-sampling):** its acceptance metrics are the ego-contact pred/persist curve AND planner rank-migration toward elite across checkpoints — behavior change (deliberate pushes) is NOT expected within one 12k run; rank migration is the leading indicator that the loop is approaching the switch.

**2026-07-11 addendum — HORIZON SWEEP (`--horizon 3`, 3-decision sequences, accumulated displacement ~13mm):** the visibility switch FLIPS for the guardian-treated WM: contact-seq pred-rank 164→**56**, elite 0%→**33%**, and the contact-vs-mirrored-twin gap turns POSITIVE (+1.0; consol3 +1.2) — the treated predictor now scores the true contact sequence BETTER than its matched no-contact twin. Control stays blind at any horizon (rank 126, gap −0.2): **watching the parent is what bought the latent contact-visibility; it just needs ≥3 decisions of lookahead to rise above the noise floor.** Behavioral steering is still OFF everywhere (CEM goal-progress ≈ chance; its Gaussian search doesn't find directed pushes) — seeing ≠ steering yet. Levers ranked by this result: (1) more observed+incidental contact knowledge (the 3-arm run), (2) --cem-horizon 3 in contact-era campaigns (t_plan cost ~3×: measure before adopting), (3) directed-push discovery may additionally want CEM warm-starts from contact-adjacent elites. h=1 vs h=3 artifacts: runs/probe_planner_vis{,_h3}/results.json.

## 2026-07-11 — EMERGENCE ARMS LAUNCHED: em_h3 + em_h3_cur (12k steps each, sequential) — the pc_treat recipe byte-identical except --cem-horizon 3 (arm A) and + --wm-sample curiosity (arm B); HERDING PARKED after 3 mechanics variants fail validation

**Design (user-directed):** minimal delta vs the pre-VLA canonical line. Provable diff (ckpt-args diff, pc_treat vs hot5b): the parent-child run changed ONLY {parent_vla, chain point d=8, bookkeeping} — the WM recipe is untouched. Arm A (em_h3) = pc_treat + `--cem-horizon 3` + 12k: the probe showed h=3 is where the treated WM's contact-visibility switches on, so running the planner there lets acquired contact knowledge EXPRESS in action selection (closed-loop: cem_replan_every stays 1). Arm B (em_h3_cur) = A + `--wm-sample curiosity` — the one WM-training change in the set: consolidation attends hardest to the child's own highest-surprise (= contact) windows, attacking the dilution bottleneck. Chain: arr95_hot@24000 (comparability with the pc twins). Gate at step 550: parent 0.086 contacts/step, child 0.905 (!), mse_contact_gap −0.12 (healthy inversion), **sps 14.3 = h=3 planning costs ~2.2×** (t 508ms/step). ~2h20m/arm. Endpoints per arm: ego-contact pred/persist curve (holdout collection), planner-visibility h1+h3 (rank/elite/mirror-gap), contact-rate trajectories vs pc_treat.
**Herding PARKED (code shipped as ParentFleet mode='herd', unused):** three variants failed 4-6-layout validation smokes (annulus gains ~0±1): (1) radial out-push at r-adaptive depth — extension INTO a block contact-pins the parent (recovery scan: free-air extension recovers even at lift −1.17; the wedge is contact, not gravity); (2) fleet fixed-pose arc with axis-ward direction — misses targets off its radial band (gains 1,1,0,0); (3) demo r-adaptive arc with direction control — gains 1,0,0,0,−1,0. Herding needs a rendered-frame debug session; not worth blocking the arms. Note: with the parent's reach lens, only blocks at parent-r 0.29-0.41 can land in the child annulus via arc carries — the geometry is genuinely tight.

## 2026-07-11 — ★★ EMERGENCE ARMS VERDICT: curiosity-weighted WM sampling BREAKS the ego-contact plateau — em_h3_cur descends monotonically to 1.87 [1.69,2.12] (best fully-emergent contact WM ever, no injected data) while em_h3 plateaus at 2.4; h=3 training doubles early learning speed in BOTH arms; the deliberate-manipulation switch is STILL OFF at 1.87 ★★

**Twin 12k runs, pc_treat recipe byte-identical except:** arm A (em_h3) `--cem-horizon 3`; arm B (em_h3_cur) = A + `--wm-sample curiosity`. Chain arr95_hot@24000, ~2h20m/arm at 13 sps (h=3 planning = 70% of step time, 2.2× cost). Ego-contact pred/persist on the held-out rake collection (n=958/ckpt):
| step | 2k | 4k | 6k | 8k | 10k | 12k |
|---|---|---|---|---|---|---|
| em_h3 | 3.92 | 2.55 | 2.39 | 2.35 | 2.72 | 2.36 [2.16,2.63] |
| em_h3_cur | 4.00 | 2.40 | 2.13 | 1.95 | 1.91 | **1.87 [1.69,2.12]** |
| (pc_treat h=1 ref) | 3.64 | 3.18 | 3.30 | — | — | — |

**Verdicts:** (1) **h=3 in-training ≈2× faster early ego-contact learning** (both arms ~2.4-2.55 by 4k vs pc_treat 3.18) — 3-step planning changes the child's behavior distribution (sustained interaction sequences), not just its scoring; but alone it PLATEAUS ~2.4 after 6k (second 6k buys nothing). (2) **`--wm-sample curiosity` is the plateau-breaker**: monotone descent through 12k, CI-separated below arm A at the wire, and the in-run contact chase stayed livelier (quarter means peaked 0.87 vs A's decline 0.61→0.35). Fully emergent: the child's own high-surprise windows, consolidated harder — no scripted data anywhere. Both arms also push rake_free below parity (0.68-0.74) with walk_free stable ~0.9 (no hot5b-style narrowing). (3) **Planner switch still OFF** at 12k (both arms: contact-action rank ~120-144/203, elite 0%, CEM goal-progress ≤ chance; A's h=3 mirror-gap +0.5 but B's noisy negative) — 1.87 is below every prior emergent model but above consol3's 1.27, which itself didn't flip the elite. Deliberate pushes need the knowledge dial pushed further AND likely goal-side help (moved-block goals live outside d=8; latent block-displacement salience).
**Next-step menu:** (a) extend em_h3_cur 12k→24k+ (curve still descending, ~0.02/1k late — cheap continuation, watch for its own plateau); (b) goal-side: d-budget growth / goal-freshness so moved-block goals become selectable as knowledge arrives; (c) herding debug session (rendered frames); (d) periodic probe-in-the-loop tracking (pred/persist + rank per 2k) as standing instrumentation. sps note: the sampler-fix (2c447aa-era) ran in-training here for the first time — no anomalies, consolidation healthy in both arms.

## 2026-07-11 — ★ EXTENDED-RUN VERDICT (em_h3_cur2, 12k→40k, 7h): the descent does NOT continue — minimum 1.77 [1.63,1.97] at 24k (≈ tied with 12k's 1.87), then REGRESSION to 2.38 [2.17,2.68] by 40k; hypothesis: curiosity-weighted consolidation chases EXOGENOUS parent churn once the child's own contact rate decays ★

**Curve (holdout rake_contact):** 12k 1.87 → 16k 2.27 (jump at the chain restart; walk_free also jumped 0.92→1.62) → 20k 1.84 → **24k 1.77 (min)** → 28k 1.88 → 32k 1.92 → 36k 2.23 → **40k 2.38**. The 36-40k regression is CI-separated from the 24k min. rake_free stays excellent throughout (0.59-0.69, better than 12k) — near-block maneuvering prediction is stable; the loss is contact-specific.
**Mechanism (consistent with in-run signals):** long deep-digest stretches late-run (child contacts 0.02-0.07 for multi-k spans at ~29-33.5k) while the guardian stayed very active (0.28-0.58) — with `--wm-sample curiosity`, the high-surprise windows the sampler chases are then PARENT-caused scene changes (exogenous, irreducible for the child's action-conditioned predictor), i.e., late-run consolidation trains hardest on noise. Curiosity-weighting works while the child feeds it ego-contact and degrades when it doesn't — a wake/sleep-coupled failure mode the 12k run was too short to reveal.
**Also observed:** (1) the engagement rhythm NEVER collapsed across 40k — two SELF-DRIVEN chase episodes (~27.5k, ~35.7k: child contacts rising to 0.37-0.47 while the parent was quiet at 0.06-0.19) — H1 extends to 40k and the child intermittently seeks blocks on its own; (2) planner at 40k: rank-corr rho 0.73 (best yet — the global one/multi-step model keeps sharpening) but contact-action rank 115-154/203, elite 0%, mirror-gap negative — the switch stays OFF; knowledge saturates ~1.8 under this regime while contact-specific prediction regresses.
**Program implication:** the binding constraint is now SUSTAINING the child's own contact generation — exactly task (b), the goal-side lever: moved-block goals (currently unselectable: curric_d pinned at 8 all 40k, arrival 0.8 < 0.95 bar) would let the planner's existing latent contact-vision generate deliberate ego-contact, feeding the curiosity sampler signal instead of exogenous noise. Candidate refinement to evaluate alongside: weight WM sampling by CONTROLLABLE surprise (own-action-correlated) rather than raw MSE. Checkpoint of record for this line: **em_h3_cur2@24000** (the min). Artifacts: runs/probe_em_cur2{,_vis3,_vis1}/results.json; W&B q7qex5n3.

**2026-07-11 addendum — goal-side lever VALIDATION (em_goalside_val, val2; 2×2.5k from em_h3_cur2@24k):** Scene-delta channel WORKS as designed: scene_active 0.23 (target 0.25), scene_dist 17-18 = the photo-eval's moved-block band — the child deliberately pursues "same pose, different block world" goals with no scripting. Churn-void accounting operates (valid_frac 0.57 parent-only → 0.47 with child-churn included in v2). **But clean arrival plateaus at 0.73-0.79 in BOTH versions — even zero-contact windows miss the 0.95 advance bar — so curric_d stays pinned at 8.** Residual candidates: settle/slide motion without contact, block respawns (teleports = scene delta without contact), and h=3 planning itself (em arrival ran 0.73-0.85 all along vs ctrl's 0.91 at h=1; the dwell/eps stack was tuned at h=1). **Decision put to the user:** (A) lower goal-curric-thresh to ~0.75 for guardian-era recipes; (B) 30-min diagnostic (h=3, no parent) to attribute the residual before picking a number; (C) RECOMMENDED: run the long experiment on the scene channel alone (+churn-void hygiene) — it bypasses d entirely and already delivers the deliberate-contact pressure that was the point; d-unpinning is optional belt to its suspenders. Code: 0cf6917 + child-churn fix.

## 2026-07-11 — SCENE-CHANNEL RUN VERDICT (em_scene12k, 12k from em_h3_cur2@24k): the CURRICULUM UNPINNED — d 8→9→10 (first budget growth in campaign history) with the ladder cycling reliably — and a new best-emergent 1.58 [1.45,1.75] at 2k; but knowledge still ORBITS ~1.8-2.4 rather than compounding — the bottleneck has moved from INCENTIVE to RETENTION

**Recipe:** scene channel 0.25 + churn-void + thresh 0.72 (h=3-ceiling-informed via the no-guardian diagnostic: clean arrival 0.72-0.80 with valid_frac 0.99 and child contacts 0.04 → the ceiling is h=3 PLANNING, not scene noise) + guardian + uniform sampling. **Structural results (all firsts):** pctl ladder 0→1.0 inside d=8 by 3.7k; **d=9.0 @ 4350, d=10.0 @ 9000** (rung cadence ~4.5k; pctl re-cycled to 1.0 inside d=10 by 11.4k — one window short of d=11); scene goals active 0.12-0.25 all run at dist 14-18 (the moved-block band); child contacts alive 0.16-0.26; arrival stable ~0.75 with the churn-void accounting. The goal-side machinery WORKS.
**Knowledge curve (holdout rake_contact):** 2k **1.58 [1.45,1.75] — best emergent ever** (edges-touching vs the 24k chain point's 1.77) → 4k 2.11 → 6k 2.52 → 8k 1.99 → 10k 2.06 → 12k 2.04. Same rise-then-orbit shape as every online regime: knowledge reaches ~1.6-1.8 quickly then erodes toward ~2.0-2.4 regardless of incentive mix (accidental / curiosity-weighted / deliberate scene goals). Planner at 12k: rank 112-154, elite 0%, rho 0.52-0.64 — still blind.
**Reading:** incentive is no longer the constraint — RETENTION is. Contact windows are a few % of the buffer under every behavior mix; consolidation keeps averaging their dynamics away (the offline existence proof that reached 1.27 had a 24% contact share). **Recommended next lever (emergent-legal data-attention, no injected data): contact-balanced WM sampling** — a fixed fraction (~25%) of each WM batch drawn from the child's OWN contact windows (its lived experience, weighted by relevance rather than raw surprise — avoids the exogenous-noise failure of curiosity-weighting). One flag + a twin run decides it. Alternatives: longer scene-run (d into the 12-20 band; weak — knowledge orbit already visible), or planner-side dwell/eps retuning for h=3 (arrival ceiling 0.75→0.9 would double curriculum speed).

## 2026-07-11 — ★ BLOCK-SHIFT PURSUIT TEST (`src/probe_block_goal_learn.py`, user-proposed): can the FULL act-stack move one block 5cm to match a photo? **NO — zero contacts, zero closure, every checkpoint (incl. scene-trained) — and the per-scene d0 spread exposes a SECOND failure layer: a 5cm block shift encodes as d0 0.3-13.3 depending on layout** ★

**Protocol:** same-pose goal photo (block0 teleported +5cm tangential, arm unchanged → pose component ~0), full frozen act-stack (CEM at ckpt params + dwell, replan every decision) for 60 decisions from bit-exact restored states; control condition pursues the UNSHIFTED photo of the same state; deliberateness = closure(shift) − closure(ctrl). 6 scenes × {scene12k@12k, em_h3_cur2@24k, base24k}.
**Result: 0.0mm closure, 0 contacts, all conditions, all models.** Two mechanistically distinct causes, both now behaviorally confirmed:
1. **Visible-but-unplannable (d0 9-13 scenes):** the arm acts for 60 decisions and never approaches the block — CEM's descent direction contains no "push the block toward the photo" component, the closed-loop/full-horizon confirmation of the planner-visibility result (contact actions never elite).
2. **INVISIBLE-shift scenes (d0 0.3-3.7):** in ~1/3 of layouts the 5cm shift barely moves the latent AT ALL from the acting viewpoint — d0 0.3 < eps 2.8 means the goal reads as ALREADY ARRIVED and dwell-hold (act=0 when gd < 1.25*eps) FREEZES the arm at t=0. The wrist-cam encoder sometimes cannot see a 5cm block displacement from the arm's current pose (occlusion/viewpoint; kin to the photo-eval aliasing discoveries). Retroactively explains part of the scene-channel training's non-compounding: a fraction of its scene goals carried no usable gradient.
**Priority implication:** for DELIBERATE manipulation, the encoder/goal-signal is upstream of both knowledge retention and planner search — block-position changes must be reliably visible in z from the acting viewpoint before any planner can chase them. Candidate levers (emergent-legal): overhead/second-cam goal encoding (wm_cam overhead exists as a flag), dwell-hold disable for scene-delta goals (it freezes on invisible gaps), latent-metric block-saliency work. Knowledge-side retention (contact-balanced sampling) remains valid but expresses only after the signal exists. Artifacts: runs/probe_block_goal/results.json.

**2026-07-11 addendum — 5-INCH VISIBILITY-GUARDED PURSUIT (user-directed): still 0.0mm / 0 contacts, all models — and the guard itself exposed the deepest layer: 21/24 scene-direction combos could not even PHOTOGRAPH a valid relocation from the wrist pose (only radially-outward +x ever passed the in-frame test), and a verified-visible 12.7cm relocation encodes to only d0 ~7-9.4** (same order as pose noise; eps 2.8). Failure stack for deliberate relocation, now complete: (1) FIELD OF VIEW — the acting wrist cam can barely see relocation targets; (2) SALIENCE — even visible 5-inch shifts move the latent ~9 units; (3) PLANNER — no block component in the CEM descent (elite-blind at every horizon); (4) KNOWLEDGE — ego-contact orbits 1.8-2.4 under every emergent regime. Conclusion: wrist-cam-only pixel-latent goals are a structurally weak substrate for manipulation emergence (extends the 07-10 photo-eval discovery). The architecture-level lever is the ENCODER VIEWPOINT: --wm-cam overhead exists but means a new encoder lineage (the frozen-encoder chain is wrist-trained) — a campaign-scale decision, not a flag flip. Artifacts: runs/probe_block_goal{,_5in,_5in_v2}/ incl. saved goal photos.

## 2026-07-12 — OPS/PIPELINE: hot6 LIVE (d 14→19 by 24.5k — the fixed-objects world clears arrival rungs at unprecedented speed) + guardian demos→LeRobot conversion COMPLETE (150 eps) + SmolVLA finetune RUNNING after a second launch landmine (draccus wants --policy.path)

**hot6 (launched 05:20 UTC):** canonical h=1 recipe verbatim from arr95_hot5b@59k + `--fixed-objects` (9aa773e, default-ON per user directive; identical deterministic layout every env/reset) on the 2.1×1.7 table; buffer EMPTY at start — hot5b's periodic snapshots were local-only and died with the container (fix b93e8a1: periodic snapshots now upload to HF; VERIFIED live — hot6 state_latest.npz @10k and @20k both on HF, 15.07 GB each). Early read: sps 27, arrival 0.95-1.00, **curriculum d 14→19 inside 24.5k steps** (rungs that averaged ~4.5k steps in the random world clear in ~2-3k) — the deterministic world is a major distribution change: arrival/d economics NOT comparable to campaign history; watch whether d keeps climbing or the pctl ladder finds a new wall. contacts/s 0.00 (no parent in this run), crit_loss nan (normal: alpha=0).

**VLA-guardian pipeline:** conversion finished 07:38 (150/150 eps, 36,000 frames, ~232 min ≈ 1.55 min/ep with h264 + 4proc/4thr writers — the libsvtav1 default was ~10-50× slower; converter fixes committed this session). Finetune launch landmine #2 (NOT the predicted wandb one): lerobot 0.4.4/draccus REJECTS `--policy.pretrained_path=` (PreTrainedConfig needs its 'type' discriminator) — correct form is **`--policy.path=lerobot/smolvla_base`** (parser PATH_KEY; policy type resolves from the hub config, remaining --policy.* args act as overrides). Relaunched 07:40 with `.env` sourced → the predicted wandb landmine ALSO defused (WANDB_SILENT=true, init OK, W&B run 3upe29cv). Training: 100M learnable/450M total params, batch 32, ~1.1-2 step/s sharing the A100 with hot6, 10k steps ETA ~10:15-10:45 UTC, ckpts every 2500 → runs/smolvla_guardian/checkpoints/. **Next gate (unchanged):** behavioral validation — parameterize src/parent_vla.py model_id, load checkpoints/last/pretrained_model, bar parent_object_contacts >0.05/substep (zero-shot scored 0.000); user sign-off required before any recipe swap.

**2026-07-12 addendum — hot6 REGIME SHIFT at 35k (post-cotrain): sustained SELF-DRIVEN block contact in the no-parent hot line — a campaign first — at the cost of a frozen curriculum.** After 9 uneventful cotrains (every 3500, 200-320 grad steps each), cotrain #10 at 35000 was the largest since startup (400/11700 steps, CONVERGED) and within ~150 steps the run flipped: pose_range 0.2→~2.0 rad (wide sweeps), **contacts/s 0.00→0.05-0.10 SUSTAINED for 3.5k+ steps** (brief flickers pre-existed at ~34.7k; contact was 0.00 essentially all run before), cur_contrib 0.2→0.7-0.9, reach 0.35→0.10-0.19, goal-curriculum prints STOPPED after 30900 (arrival < advance bar → pctl frozen 0.40, d held 20). Crucially mse[blk] 0.20→0.17 and mse[tbl] 0.15→0.10 IMPROVING while it explores — the WM is learning the contact dynamics it now generates; mse[none] wobbles up 0.02→0.03-0.08 (novel-state visitation). Reading: fixed-objects world + encoder update made blocks chase-able for the curiosity loop — fully emergent, no injected data. Next cotrain (38500) did not fire on schedule (flatline gate). WATCH: does it settle back into arrival-climbing (curriculum resumes toward d=21) or persist (knowledge-vs-curriculum trade)? No intervention — sps steady 23.3, snapshots uploading, no crash signatures.

**2026-07-12 — ★ VLA-GUARDIAN GATE: PASS — the finetuned SmolVLA guardian makes real, committed block contact in sim (mean 0.079-0.085 contact/substep across 8+16 eps vs bar 0.05, zero-shot 0.000) — after unmasking an eval-side normalization break; engagement is BIMODAL (~1/3 of episodes engage at 0.13-0.31 with 8-15 cm displacement, rest hover) ★**

**Root cause chain (two eval bugs masked the model):** (1) stale ParentFleet state across eval episodes (queues/slew-ref; demo recorder rebuilt its fleet per episode) → fixed with reset_state(); (2) THE big one: lerobot 0.4.x moved MEAN_STD normalization OUT of the policy into saved processor pipelines — predict_action_chunk is raw-space, so ParentFleet (built for zero-shot base) fed raw radian state and consumed normalized-unit outputs as radians → arm commanded to ~zero pose, hovered forever. Fix (76320b4): load the checkpoint's stats (state norm in, action unnorm out, + training's "\n" task suffix); hub smolvla_base path unchanged. Gate harness: src/validate_vla_guardian.py (mirrors recorder conditions, settle-disp floor 6.2 cm calibrated out, per-episode overhead mp4s in runs/vla_guardian_val_videos/).
**Verdict texture:** engagement stochastic per episode (flow-sampling tips commit-vs-hover; uncorrelated with child still/walking): pooled 8/24 eps engage; engaged eps average ~0.24 contact/substep ≈ 5× the scripted keep-bar, with real 8-15 cm block displacement and varied instructions ("each time different" ✓ within engaged eps). Guardian-role math: expected per-env churn ≈ 0.08/substep ≈ scripted-teacher floor in aggregate (8-env fleet → ~2-3 engaged guardians at any time) but per-env pattern changes from steady-sweep to quiet-stretches-plus-bursts. **DECISION FOR USER (per standing gate): swap the VLA guardian into a training recipe, or first raise the engagement rate (candidates: eval 007500 ckpt, 2-3× demos, longer finetune, commit-biased sampling via noise seed retry-on-hover).** Finetune artifacts: runs/smolvla_guardian (10k steps, 2h11m, loss 0.073→0.004, W&B 3upe29cv); ckpts 2500/5000/7500/10000+last.

## 2026-07-12 — ★★ HOT6 COMPLETE (100k from arr95_hot5b@59k, fixed-objects world, ~9h shared-GPU): d 14→21 BANKED (campaign distance record) plus a mid-run EMERGENT CONTACT ERA — post-35k, 59% of metric windows show child block contact ≥0.05/s (vs 9% pre-35k; peak 1.04/s ≈ near-continuous) with online blk-MSE 0.19→0.05 — the no-parent line generated its own contact-rich curriculum with uniform replay, goal-side curiosity only ★★

**Curriculum timeline:** d 14→17 held @14.1k → d=18 @16.5k → d=19 @~24k → d=20 @~30k → **d=21 @48.6k**; pctl ladder fully re-cycled inside d=21 by 93k; d=22 not attempted-and-cleared (no qualifying window after 93k). Rungs cleared in ~2-4k steps vs ~4.5k (scene12k) and multi-day (random-world campaign) — the deterministic layout massively lowers arrival difficulty; d values are NOT comparable to random-world history.
**Contact era mechanics:** ignition at 35k right after cotrain #10 (400/11700 grad steps, largest since startup; 9 prior cotrains uneventful; flickers pre-existed ~34.7k). Pattern: exploration waves (pose_range →~2 rad, cur_contrib 0.7-0.9, contacts 0.05-0.10 → later 0.2-1.04) alternating with consolidation/goal phases every few k steps; from ~70k contact-mode dominant (peak 1.04/s @69.65k, ≥0.6 sustained for hundreds of steps); **from 78.9k arrival ≥0.95 windows pass WHILE contact runs high — knowledge and curriculum stopped trading off**. Cannot distinguish deliberate interaction from arrival-adjacent parking from logs alone — probes decide.
**Caveat:** blk/tbl MSE here are ONLINE buffer metrics on the fixed-world distribution (0.19→0.05, 0.16→0.05) — not the campaign holdout rake_contact scale (1.27-2.4 band); do not compare across. Holdout-style verdicts come from the probe suite on hot6@100k (planner-visibility + block-shift pursuit, both launched post-run, results next entry).
**Artifacts:** HF `hot6/ckpt_0100000.pt` (+1k-interval ckpts all run) + `hot6/state_latest.npz` (final 15.07 GB; periodic uploads verified end-to-end — b93e8a1 fix). W&B run hot6. New campaign head: **hot6@100k** (lineage ≈159k steps: 59k hot5b + 100k).

**2026-07-12 — ★ HOT6@100k PROBE VERDICT: the deliberate-manipulation switch moved from OFF to FLICKERING — contact actions rank 51/203 predicted (campaign best; prior best 56, typical 115-164) and enter the CEM ELITE 17% of the time (FIRST nonzero elite share in campaign history) — but end-to-end relocation is still ZERO (block-shift pursuit: 0.0mm, 0 contacts, all scenes) ★** Planner-vis detail: real-rank med 0 (the contact decision IS the best real action in these scenes), pred-rank med 51 → the WM's contact-value estimate closed most of the gap; rho 0.47 (down from 40k-era 0.73 — global ranking got noisier as contact knowledge sharpened); mirror-gap still ~0 (-0.3); CEM plans make contact in 100% of scenes (base also 100% — the fixed contact-rich world) but contact plans yield ~no goal-progress (-0.3mm vs +0.1 base). Block-shift layers unchanged: 50mm shift encodes to d0 med 5.8 (range 3.5-8.5; eps 2.8) from the wrist view and CEM descent has no closing component — **failure layers 1-2 (FOV/salience) now clearly the binding constraint; layer 3 (planner elite-blindness) is cracking from the contact era's knowledge; layer 4 (knowledge) at its best**. Program implication: the ENCODER-VIEWPOINT decision (overhead/second-cam lineage — user-gated, campaign-scale) is now THE lever standing between a flickering switch and deliberate relocation. Artifacts: runs/probe_planner_vis_hot6/results.json, runs/probe_block_goal_hot6/results.json.

## 2026-07-12 — ★ USER DECISION: VLA GUARDIAN SWAPPED IN — hot7vla LAUNCHED (hot6@100k chain + FULL 15GB buffer restore, first buffer-carrying chain since the b93e8a1 fix; canonical recipe verbatim, d-start 21, + --parent-vla smolvla --parent-model runs/smolvla_guardian/checkpoints/last/pretrained_model --parent-rate 0.02, 100k steps) ★

User approved the swap post-gate ("go ahead with adding the parent as the vla"). Plumbing: --parent-model flag added to train.py (threads model_id into ParentFleet; finetuned ckpt dirs auto-apply MEAN_STD stats + task newline). Smoke-verified before launch (fleet log shows stats-loaded + finetuned-path lines). NOTES FOR THE WATCH: (1) hot6's encoder never saw a parent arm in frame — expect mse[none] elevated early while consolidation/cotrain digest the new visual; (2) guardian engagement is bimodal (~1/3 of episodes commit) → parent_contacts will be burstier than the scripted teacher's; fleet-aggregate churn should still land near scripted levels (~2-3 engaged envs at any time); (3) d economics continue fixed-world scale (rungs ~2-4k), now WITH exogenous churn — churn-void arrival accounting is NOT in this recipe (canonical hot line), so arrival may read lower than hot6 during guardian bursts. W&B: hot7vla.

## 2026-07-13 — ★ hot7vlab: hot7vla RECOVERED after runpod wipe (crash @41k) + COMPLETED (60k) — reconstructed guardian ran at full strength, but the guardian-era verdict is KNOWLEDGE-MAXED / MANIPULATION-BLOCKED (encoder viewpoint confirmed THE lever) ★

Runpod restart wiped ALL local state + the python env mid-hot7vla (@41k/100k) — only git + HF/W&B artifacts survived. **Recovery:** child from HF (hot7vla@40k ckpt + matched 50k buffer, restored clean `[buffer] 8x6250=50000`); guardian weights survived ONLY as a W&B artifact (`3upe29cv/policy_smolvla-…-010000`, model.safetensors) — its config + MEAN_STD stats died with the local dataset (`runs/lerobot_guardian`, never pushed). Guardian rebuilt: exact W&B weights + `smolvla_base` config.json (features already match: state(6)/camera1-3/action(6)/MEAN_STD) + state/action stats RECOMPUTED off the DETERMINISTIC demos (`record_guardian_demos.py` seed 51 → exactly 36k frames, matching the original) written to the two processor safetensors `parent_vla.py` reads. Env rebuilt to the run's captured stack (torch 2.10.0 / transformers 4.57.6 / numpy 2.2.6 / lerobot 0.4.4 — the pyproject `<4.50` pin is STALE pre-VLA doc; `lerobot[smolvla]` install upgrades the whole torch stack). Guardian gate PASS **0.066** (n=24; bar 0.05, v1 was 0.079-0.085 — the n=8 0.048 was bimodal variance; this is the most-faithful guardian obtainable, exact weights + reproduced stats — re-finetuning would give a DIFFERENT guardian, not a closer one).
Relaunched **hot7vlab** (hot5b-style recovery name so hot7vla's HF ckpts aren't overwritten by the restarted step-0 counter), recipe reconstructed byte-identical by diffing the run's full W&B config vs train.py argparse defaults (zero drift), 3 deltas only: `--init-ckpt …hot7vla/ckpt_0040000.pt --init-buffer auto --total-steps 60000`. W&B `l12z1s2e`, ~4.5h, CLEAN teardown (no MooseFS crash; final ckpt + state_latest uploaded).
**RESULT:** reconstructed guardian at FULL strength — parent_contacts 0.90/step mean (peak 3.65), child contacts/s mean 0.29 / peak 1.12 (**93% of windows ≥0.05** vs hot6 post-35k 59%), wm/mse_block 0.157→**0.047**, mse_table 0.325→0.031, pred_loss 0.044→0.025; curriculum FLAT (d=21, pctl 0) — arrival-gated in the churn regime (arrival mean 0.56), exactly as designed. hot7 leg = 40k(hot7vla)+60k(hot7vlab) = the full 100k budget, lineage ~259k. Artifacts: HF `hot7vlab/ckpt_0001000..0060000` + `state_latest.npz`.
**PROBE VERDICT (hot7vlab@60k, planner-vis + block-shift):** KNOWLEDGE BEST-EVER — contact-action pred-rank **48/203** (prior campaign best 51; real-rank med 0), rho 0.74, mirror-gap +0.2. But the PLANNER did NOT advance — CEM elite **0%** (hot6@100k flickered at 17%), contact-plan goal-progress −2.7mm. End-to-end block relocation STILL **0.0mm / 0 contacts** (50mm shift → d0 med 4.7 ≈ eps 2.8 pose-noise floor). A full guardian era MAXED contact knowledge but couldn't crack deliberate manipulation on the wrist-cam substrate → FOV/salience (layers 1-2) is the binding constraint; the **ENCODER-VIEWPOINT decision (overhead / 2nd-cam lineage) is confirmed THE lever.** Artifacts: `runs/probe_planner_vis_hot7vlab/`, `runs/probe_block_goal_hot7vlab/results.json`.
**NEXT (user-directed):** trying a CLOSER OVERHEAD encoder cam — user picked candidate #4 "topdown-tilt" (over the block zone ~(0.26,0), ~75° down, dist ~0.50, blocks large + well-separated with workspace coverage) over the existing corner "overhead" logging cam. This is a NEW from-scratch encoder lineage (the frozen wrist-trained encoder can't transfer). Cam bake into scene.xml + `--wm-cam` wiring + scratch build to follow.

## 2026-07-13 — OVERHEAD LINEAGE FIRST BRICKS: `scratch_oh_a` (3k smoke) + `oh_mature` (25k from scratch, overhead_close + VLA guardian) — both COMPLETED; recovered from W&B 2026-08-01 (the 07-13 session ended before these were ledgered)

Ran right after the 00aec1c cam commit; W&B + HF artifacts intact, so nothing was lost — only the write-up.
**`scratch_oh_a`** (bbdytrfp, 3k, ~16 min, 25.5 sps): from-scratch build-stage smoke on `--wm-cam overhead_close`, no parent. Confirms the new cam trains end-to-end (pred/persist 0.87 at 3k — early, as expected for 3k from scratch).
**`oh_mature`** (3kxc2z7d, 25k, ~3.1h, 18.1 sps): the first real overhead-lineage run — from scratch, `overhead_close`, **WITH the smolvla VLA guardian** (rate 0.02), fixed objects, CEM-only build recipe (goal_select `recent`, goal curriculum OFF, thresh-default 0.25, consolidate 70/1, no cotrain, action_max 0.05, no-proprio + sigreg-pertimestep, buffer_frac 4). Final: **pred/persist 0.44**, mse_block 0.073 / table 0.025 / none 0.114, z_std_probe 1.21, step_jump 5.59 (frac_rand 0.285); guardian very active (parent contacts 1.90/step) vs child contacts 0.025/step; arrival 0.016 (curriculum off — build run, not a ladder run). Reading: healthy young encoder on the new viewpoint with heavy exogenous churn in frame. Artifacts: HF `oh_mature/ckpt_0001000..0025000.pt` + `state_latest.npz`. NOT yet probed for the point of the whole exercise — block-shift salience from the acting viewpoint.

## 2026-08-01 — `oh_solo` LAUNCHED (user-directed: NO guardian — "run 1 for now"): oh_mature's recipe byte-identical minus the parent, the no-guardian twin of the overhead_close lineage

Recipe reconstructed by the W&B-config-vs-argparse-defaults diff (same technique as the hot7vlab recovery; zero drift). Deltas vs `oh_mature`: drop `--parent-vla smolvla --parent-model … --parent-rate 0.02`; `--name oh_solo`. Everything else verbatim: 25k from scratch, `--wm-cam overhead_close`, CEM 18/200/20 h=1 replan-1, deterministic act / alpha 0, goal-explore recent, consolidate 70/1, fixed objects, seed 0. Running locally on the workspace A100-80GB (torch 2.4.1 env — no lerobot needed without the guardian). W&B `ilecc7wy`; log: `runs/oh_solo_train.log`. Launch health: 23+ sps by step 150, buffer 8×6250=50000, wm 18.03M params, crit_loss nan (normal: alpha=0).

**RESULT (COMPLETE, 25k, 3.02h, 18.6 sps, clean teardown — final ckpt + state_latest on HF):** the no-guardian twin builds an encoder as healthy as oh_mature's (z_std_probe 1.26 vs 1.21, step_jump 5.68 vs 5.59, frac_rand 0.29 both) with BETTER normalized prediction (**pred/persist 0.35 vs 0.44**) and far better goal pursuit (**arrival 0.28 vs 0.016**, reach 0.024 vs 0.002, cem/reach_gap 4.7 vs 5.8) — without exogenous churn the goals hold still and CEM closes on them. What the guardian bought instead: slightly lower online contact-window MSEs (blk 0.073 vs 0.093, tbl 0.025 vs 0.041 — contact-dynamics exposure) and 1.6× the child contact rate at end (0.025 vs 0.015/step). Caveat: online buffer MSEs are per-run distributions (mature's buffer is churn-heavy) — pred/persist and the behavioral numbers are the honest cross-run comparators. Summary values are end-of-run snapshots. **Overhead lineage now has TWO 25k build heads on HF (`oh_solo`, `oh_mature`) and the viewpoint bet's go/no-go is still unmeasured: block-shift salience (d0 of a 50mm shift in the overhead_close latent) + pursuit probes on these checkpoints. The probe scripts read `wm_cam` from the ckpt's saved args (encode_cam=a0.wm_cam), so they should run on the overhead heads as-is; the 07-11 wrist-pose visibility guard logic may want a re-look under the fixed cam.** Twin question: does the overhead encoder build as well without exogenous parent churn in frame (oh_mature's mse[none] 0.114 included parent-arm visuals the child can't predict)?

## 2026-08-07 — ACTION_MAX CAMPAIGN (user-directed: "as much as possible while preserving locality — goal: no action max at all"): mechanical sweep says the PLANT preserves locality, `oh_nocap` LAUNCHED (oh_solo twin @ action_max 5.6 = literal no-cap)

**Mechanical sweep first** (`src/probe_amax_locality.py`, new; runs/sim_scales/amax_sweep.json): action_max ∈ {0.05 (oh_solo baseline), 0.1, 0.2, 0.4, 0.8, 1.6, 5.6}, 1500 env steps each × {random block-actions, violent ±1 square wave}, oh_solo env (overhead_close, fixed objects, frame_skip 6, no parent, render off). Since `target = clip(q + a·action_max, joint_range)`, action_max ≥ the largest joint span (5.585 rad) is EXACTLY "no action max" — the joint-range clip binds instead (clip-bind frac 0.83–1.0 at 5.6). Findings:
- **Realized per-step motion PLATEAUS at the plant's torque-limited slew** (servo saturates at |dq_err|~tau_max/kp≈6 mrad; |qd| p99 caps ≈5.0–5.7 rad/s): dq/step p95 0.049 (amax 0.05) → 0.12 (0.2) → 0.14 (0.4) → **0.156 (5.6)** — flat from ~0.4 up. Uncapped per-step motion is only **2.6× baseline** (EE/step p95 3.9 vs 1.5 cm). Locality is preserved BY THE PLANT, not the clamp.
- Per-DECISION EE travel (block=5) grows 4.2 → 14.4 cm p95 — a decision now crosses the whole block zone (workspace-scale reach per plan step; at 0.05 the arm inched).
- **Physics stable everywhere**: 0 non-finite states, 0 block knock-offs (even violent full-range), velocities bounded. Torque saturation 0.44 → ~1.0 (bang-bang regime; realized/commanded p50 0.91 → 0.04 — action semantics become direction+slew).
- So there is NO mechanical rung between 0.05 and none: from ~0.4 up all values are physically equivalent to uncapped; what changes is ACTION-SPACE GEOMETRY for WM/CEM (at 5.6 only ~3%/dim of action space maps to within-step-reachable targets — fine control must come from CEM's iterated sigma shrink). Whether the WM/planner eats that is the empirical question → train it.

**`oh_nocap` LAUNCHED** (W&B `pelefu7n`, log runs/oh_nocap_train.log, workspace A100-80GB): oh_solo's reconstructed command byte-identical (config-vs-defaults diff re-verified) with ONE delta: `--action-max 5.6` (+ `--name oh_nocap`). 25k from scratch, overhead_close, no guardian, CEM 18/200/20 h=1, deterministic/alpha 0, goal-explore recent, consolidate 70/1, fixed objects, seed 0. Went straight to the literal no-cap rung (not 0.4) because the sweep shows no mechanical reason to stop short — if this trains sick vs the oh_solo twin (pred/persist, step_jump, arrival, contacts), the failure is REPRESENTATIONAL and 0.4/1.6 are the informed bisect rungs (same physics, friendlier geometry). NOTE the argparse footgun for future rungs: `--action-max 0.3` exactly gets auto-bumped to 1.0 under goal-explore (train.py:1226) — never use 0.3 as a rung. Twin comparators at 25k: pred/persist (0.35 bar), z_std/step_jump/frac_rand (1.26/5.68/0.29), arrival 0.28, contacts 0.015/step; expect step_jump ↑ (~2.6× physical steps) — the question is whether prediction and pursuit hold.

**`oh_nocap` RESULT (COMPLETE, 25k, 3.16h, clean teardown — 25 ckpts + state_latest on HF): NO-CAP WORKS, and it quietly solves the contact-exposure problem the guardian existed for.** End-of-run twin vs oh_solo@25k:
- **Behavior transformed**: pose_range 3.28 vs 1.10 (3× joint-space coverage), pose_spread 0.76 vs 0.22, **contacts/step 0.205 vs 0.015 (14×)**, **frac_touch_block 0.084 vs 0.001 (84×)**, object_motion 0.009 vs 0.004. The uncapped child touches blocks CONSTANTLY, no parent needed — and unlike guardian churn (exogenous, irreducible noise per em_h3_cur2/oh_mature), these contacts are SELF-GENERATED = action-conditioned data. 8× oh_mature's guardian-era child contact rate (0.025).
- **Pursuit holds**: arrival 0.227 vs 0.281 (same ballpark; nocap's goal difficulty grew — dist_goal ~10 by end), reach_rate 0.033 vs 0.024, cem/reach_gap 4.89 vs 4.74, endpoint_disp 2.17 vs 3.50.
- **The one degraded axis: normalized prediction.** pred/persist 0.720 vs 0.352 — BUT it was pinned ~1.0 through 11.5k then improved to 0.72 by 25k (still learning, not stuck), absolute pred_loss is BETTER (0.037 vs 0.078), and the driver is a 4×-easier persistence baseline (identity 0.051 vs 0.220): the latent moves much less per step (step_jump 3.04 vs 5.68) despite 2.6× the physical motion. Mid-run hypothesis (11.5k): the encoder under-encodes the fast (5 rad/s, saturated 0.99 tau_sat) arm — "static-content dominance". Encoder probe stats are cross-run-weak (warmup-rollout probes differ) but point the same way: z_std_probe 0.813 vs 1.258, yet rank_frac 0.323 vs 0.164 (richer directions, smaller spread — NOT a collapse).
- **Verdict: action_max is DONE as a constraint — 5.6 = no cap is the recipe going forward unless the salience probe says the low-motion latent lost the blocks.** The block-shift salience + pursuit probes (still the lineage go/no-go, now on THREE 25k heads: oh_solo, oh_mature, oh_nocap) are doubly decisive here: nocap actually touches blocks, so pursuit probes can show real relocation for the first time; and if a 50mm block shift encodes ≫ eps in nocap's latent, the arm-motion under-encoding is benign (fast arm blurred, slow blocks salient — arguably the RIGHT invariance for manipulation). Fallback rungs if salience says otherwise: 0.4/1.6 (same physics, less saturation, more frame-to-frame correlation).

## 2026-08-08 — BLOCK-SHIFT SALIENCE/PURSUIT PROBES on oh_nocap@25k + oh_solo@25k (the lineage go/no-go) + PROBE INSTRUMENT BUG FOUND & FIXED: verdict = CAMERA LAYER SOLVED, ENCODER LAYER STILL THE BIND; pursuit still 0.0mm

**Instrument bug (affects ALL prior d0 numbers):** `probe_block_goal_learn.py`'s save/restore carried only qpos/qvel/ctrl, but `_place_object_safely` (reset + respawns) rerolls cube **geom_size/geom_rgba at MODEL level** — so every scene staged before the last restored with wrong sizes/colors and d0_ctrl measured that mismatch (4–13), not the encode floor (~0.1–0.5). Only the last-staged scene was ever clean. Fixed (0266248): snapshots now carry object size/rgba. **Historical d0s are instrument-inflated ~10×** (hot7vlab's "d0 4.7 vs eps 2.8" included the mismatch; the wrist NO-GO itself survives — it rested on the bitwise-identical-photo guard + 0 contacts).
**Fixed-probe results (6 scenes, 50mm shift, 60-decision closed-loop pursuit, per-head envs at native action_max):**
- **Pixel layer (the camera bet): SOLVED.** Visibility guard passes first-try on every scene (wrist: chronic failure); a 50mm shift changes ~3.9–4.1k px with the target position always in frame.
- **Latent layer (salience): WEAK + INCONSISTENT — the remaining bind.** oh_nocap shift-d0 {0.47, 0.81, 0.07, 0.43, 2.65, 0.04} vs own floors {0.52, 0.08, 0.52, 0.07, 0.09, 0.07} → 3/6 scenes clear the floor (6–28×), 3/6 at/below it; median d0 0.5. oh_solo: 4/6 scenes 2–6× floor, median 0.9. Both are SUB-EPS (goal_reach_eps 2, train dist_goal ~10): a 50mm relocation encodes at ~25–45% of eps → CEM has no arrival-relevant gradient toward the shifted-block goal. Since the shift is pixel-visible in EVERY scene, the scene-to-scene variance is the ENCODER's selectivity, not FOV/occlusion.
- **Behavioral: still no deliberate relocation** — median closure +0.0mm both heads, deliberate ~0.0 (nonzero closures appear in CTRL too = accidental shoves, e.g. nocap scene4 ctrl +31.4mm). But contacts during pursuit are now ROUTINE for both heads (median 14–40/episode, max 338; wrist-era probes: 0) — the arm lives among the blocks now.
**Reading: the four-layer stack is now bound at layer 2 ONLY, and it's a TRAINING-SIGNAL problem, not a viewpoint problem.** The camera delivers the pixels; the build-stage encoders (25k, sigreg+prediction on mostly-arm-variation buffers) just don't spend capacity on block position. nocap's contact-rich buffer (14× contacts, 2.3× object motion) is exactly the data to change that — candidate levers, cheapest first: (a) continue oh_nocap 25k→50k+ on its own buffer (block-motion windows now common; salience may emerge with steps, as hot6's contact era did), (b) consolidation/wm_sample upweighting of contact/object-motion windows, (c) an explicit block-salient auxiliary. Artifacts: runs/probe_block_goal_oh_{nocap,solo}/results.json.

## 2026-08-10 — `oh_nocap2` LAUNCHED: +25k continuation from oh_nocap@25k with the FULL 15GB buffer (lever (a) of the 08-08 verdict — salience-from-steps, the hot6 pattern) — user-directed ("this is what we did last time")

Action-max question re-confirmed closed en route: no-cap is the recipe (plant preserves locality mechanically; nocap twin trained healthier; salience gap identical across capped/uncapped heads = orthogonal to action scale).
**Recycled-box recovery:** workspace box wiped since 08-08 (runs/ empty except a fresh setup smoke); pulled `oh_nocap/ckpt_0025000.pt` + `state_latest.npz` (15GB) from HF `a5ilank/curious-robot` into `runs/oh_nocap/` so `--init-buffer auto` finds the state next to the init ckpt.
**Command reconstructed from the CKPT'S EMBEDDED ARGS** (`ckpt["args"]` vs argparse-defaults diff — same zero-drift guarantee as the W&B-config technique, and works offline; note train.py's parser lives inside `parse_args()`, so harvest defaults by calling it with patched empty argv). Diff reproduces the ledger recipe exactly (action_max 5.6, overhead_close, CEM h=1 replan-1 on the 18/200/20 defaults, deterministic/alpha 0, goal-explore recent 25/25/k3, consolidate 70/1, no-proprio + sigreg-pertimestep, buffer_frac 4, start_steps 1000, env_threads 8, keep-local-ckpts, lambda_safe 0/delta 15). Deltas: `--name oh_nocap2` + `--init-ckpt runs/oh_nocap/ckpt_0025000.pt`, total 25000 (chain counter restarts; 50k lineage-total at completion). **Fresh-box env footgun:** uploads gate on `$HF_TOKEN` (train.py:977) and the repo id comes from `$HF_UPLOAD_REPO_ID` (hf_repo=None in saved args) — neither survives a box recycle; injected at launch (token from `~/.cache/huggingface/token`, repo `a5ilank/curious-robot`).
**Launch health (W&B `odmaq08m`, log `runs/oh_nocap2_train.log`):** `[state] restored 50000 transitions (saved @ step 25000)` = the full ring at cap; goal archive 64/64 from ckpt; wm 18.03M. **Chain-boundary cost ≈ 0, measured:** by step 150–250 the run is already IN the parent's end-of-run regime — sps 24.7 climbing, contacts/s 0.23–0.26 (parent end 0.205), pose_range 3.28 (parent's exact end value), mse[blk/tbl/none] 0.08/0.02/0.13, dist_goal 12–13.7. No honeymoon, no re-coverage (no ladder in this lineage; buffer rode). ~3.1h wall to 25k. Stall watchdog armed (10-min log silence).
**Pre-registered decision rule at 50k-total** (fixed salience+pursuit probe, same 6-scene/50mm/60-decision protocol, vs the 25k baseline {0.47,0.81,0.07,0.43,2.65,0.04} / floors / median 0.5): shift-d0 climbing toward/past eps=2 consistently across scenes → steps work, extend the same way; still floor-bound on ~half the scenes → lever (b), consolidation/wm_sample upweighting of contact/object-motion windows (nocap's own buffer is the data for it).

## 2026-08-11 — ★ oh_nocap2 COMPLETE (25k chain, 3.1h, clean teardown) + 50k-PROBE VERDICT: SALIENCE IS CLIMBING WITH STEPS (median d0 0.5→0.9, above-floor 3/6→4/6, scene4 = FIRST scene ever ≥ eps) → pre-registered "extend" branch taken, `oh_nocap3` launched ★

**Run health (W&B `odmaq08m`):** chain-boundary cost ≈0 held all run; contacts/s 0.19–0.20 END-TO-END (the contact-rich regime is stable, +25k more self-generated block-interaction data banked); sps ~16.6–24.7; final ckpt + state on HF.
**Encoder trends 25k→50k (the "easy-persistence latent" is resolving):** pred_vs_persist **0.720 → 0.397** (converging toward oh_solo's 0.35; improved steadily), step_jump **3.04 → 5.42 MONOTONE** through the whole chain (the under-encoded fast arm is entering the latent; now ≈ oh_solo's 5.68), z_std_probe 0.81→1.01, rank_frac 0.32→0.34, pred_loss flat (0.037→0.039). Arrival 0.227→0.023 — NOT comparable: recent-select goal frontier deepened hard (dist_goal 10.2→16.0, reach_gap 4.9→11.2, evictions 98); not a decision metric here.
**50k salience probe (`runs/probe_block_goal_oh_nocap2/`, fixed 6-scene/seed-41/50mm/60-dec):** shift-d0 {1.78, 0.41, 0.23, 1.39, 2.46, 0.06} vs floors {0.58, 0.06, 0.38, 0.16, 0.35, 0.15} → **4/6 scenes clear floor at 3.1–8.4×** (25k: 3/6), **median 0.5→0.90 (×1.8 per 25k steps)**, and **scene 4 = 2.46 ≥ eps 2.0, the first scene in campaign history where a 50mm shift out-encodes the arrival threshold**. Per-scene: rising in 0/2/3, holding in 4, DOWN in 1 (0.81→0.41), dead in 5 (0.04→0.06) — the encoder allocates block capacity scene-unevenly under uniform replay. Linear extrapolation of the median: eps=2 at ~+50–75k more steps (if the trend holds — hot6 says these things can also break out nonlinearly).
**Pursuit: still 0.0mm deliberate on all 6** (ctrl shows accidental shoves +5.6/+21.2mm on scenes 0/1 → deliberate is negative there). **Honest flag: pursuit-episode contact engagement DROPPED vs the 25k probe** (median contacts/episode 0.5–2.5, max 11 — vs 14–40 median, max 338 at 25k): the 50k head's staged-scene behavior is less block-entangled even as salience rises; plausibly the doubled step_jump reshapes CEM's cost surface (deeper dist_goal everywhere). Watch at 75k.
**Decision (pre-registered rule, "climbing" branch — 2/6 floor-bound < the ~half threshold):** `oh_nocap3` LAUNCHED — byte-identical chain from `oh_nocap2/ckpt_0025000.pt` + its 15GB state (log `runs/oh_nocap3_train.log`), lineage-total 75k at completion, same probe at the boundary. Standing next-lever if the median's climb flattens or the dead scenes (2/5) persist at 75k: lever (b) contact/object-motion-window upweighting in consolidation (mechanism precedent: em_h3_cur's `--wm-sample curiosity` broke the ego-contact plateau; a contact-window variant is a small delta), applied as the recipe of the NEXT chain so its effect is attributable against the now-measured pure-steps slope (×1.8/25k).

## 2026-08-11 — oh_nocap3 COMPLETE (75k lineage) + 75k PROBE: PURE-STEPS SALIENCE PLATEAUED (median d0 0.5→0.9→0.6, dead scenes persist) → BOTH lever-(b) triggers fired → `oh_nocap4` LAUNCHED = chain + `--wm-sample curiosity` (the em_h3_cur mechanism, zero new code)

**Run (W&B `aa2u4e1q`, 3.2h, clean teardown, ckpts+state on HF):** the encoder-recovery arc COMPLETED — pred_vs_persist **0.397→0.323 (now beats oh_solo's 0.35**; the no-cap head's one degraded axis at 25k is fully closed), step_jump 5.42→5.58 (plateaued at the capped twin's level), rank_frac 0.34→0.41, contacts still rising 0.191→**0.236/step**, frac_touch_block 0.069. Arrival 0.0 at dist_goal 17.5 / reach_gap 11.7 (recent-select frontier keeps outrunning the planner — same non-comparable pattern, noted not scored).
**75k salience probe (`runs/probe_block_goal_oh_nocap3/`):** shift-d0 {1.65, 0.56, 0.16, 0.64, 3.55, 0.12} vs floors {0.37, 0.18, 0.58, 0.21, 0.67, 0.75} → 4/6 above floor (static vs 50k) at softer multiples (3.0–5.3× vs 3.1–8.4×), **median REGRESSED 0.90→0.60** (lineage: 0.5→0.9→0.6 = oscillating plateau, NOT the ×1.8 slope), scene 3 fell back (1.39→0.64), scenes 2/5 dead in ALL THREE probes (and scene 5's floor rose to 0.75 with shift 0.12 = deeply blind). Scene 4 alone keeps compounding (2.46→**3.55**, well past eps 2). **First-ever positive deliberate closure: scene 1 +8.3mm (17 contacts, ctrl 0.0mm/18 contacts)** — n=1, logged as anecdote until it repeats.
**Reading:** 50k of self-generated contact-rich data under UNIFORM replay bought a real but saturating salience gain — the encoder's block capacity is scene-lucky and noisy, not compounding. Steps alone are no longer the cheapest bit of information.
**`oh_nocap4` LAUNCHED (lever (b)):** chain from `oh_nocap3/ckpt_0025000.pt` + full state, ONE delta: `--wm-sample curiosity` (validated choice; em_h3_cur precedent — curiosity-weighted WM sampling broke the July ego-contact plateau). Rationale for the zero-code variant over a bespoke contact-window sampler: in a SOLO no-cap run the child's own high-surprise windows ARE the contact/object-motion windows (free-arm motion is well-predicted at pvp 0.32; blocks only move on contact), and em_h3_cur2's known failure mode (chasing exogenous parent churn) cannot occur with no parent in frame — plus the 07-11 replay-sampler-bug history argues against fresh sampler code. **Pre-registered 100k-boundary rule (same probe):** curiosity-weighting must beat the measured pure-steps plateau — median d0 > ~1.4 (plateau ceiling + the old slope) and/or a dead scene (2/5) coming off floor and/or deliberate closure repeating; else lever (c), an explicit block-salient auxiliary, with (b)'s mechanism kept or dropped on its salience delta.

## 2026-08-11 — oh_nocap4 COMPLETE (100k lineage) + 100k PROBE: LEVER (b) FAILS ALL THREE PRE-REGISTERED CONDITIONS — curiosity sampling bought best-ever prediction, HALVED contact exposure, moved salience NOWHERE → (b)'s mechanism DROPPED; campaign at the lever-(c) decision point (USER INPUT WANTED — new-loss design)

**Run (W&B `i4kd3d51`, clean teardown, ckpts+state on HF):** pred_vs_persist 0.323→**0.252** (lineage best; pred_loss halved 0.040→0.021, rank_frac 0.42) — but **contacts/step 0.236→0.096, frac_touch_block 0.069→0.039 (HALVED)**, mse_none fell (0.066) while mse_block stayed flat (0.11), archive evictions exploded (140→3,898). Reading: with no parent churn to chase, curiosity-weighted sampling still found a cheaper meal than block dynamics — it sharpened FREE-MOTION prediction and the behavior drifted away from contact. The em_h3_cur precedent did not transfer to this regime.
**100k salience probe (`runs/probe_block_goal_oh_nocap4/`):** shift-d0 {1.85, 0.31, 0.04, 0.61, 3.23, 0.07} vs floors {0.30, 0.18, 0.37, 0.38, 0.39, 0.45} → median **0.5** (rule bar: >1.4; lineage 0.5→0.9→0.6→0.5 = full round trip), scenes 2/5 dead in ALL FOUR probes, scene 1's +8.3mm deliberate closure did NOT repeat (0.0mm everywhere; scene-1 marginal at 1.7× floor). Scene 4 compounds for the third straight probe (3.23 ≥ eps) — the encoder CAN make a 50mm shift arrival-salient, but only ever for that one layout; whatever scene 4 has, the loss doesn't spread it.
**Campaign position:** the four-probe arc measures the whole cheap-lever ladder — pure steps saturate at median ~0.6-0.9, curiosity reweighting is salience-neutral-negative here. Remaining candidates, NOT launched pending user direction (new-loss design = architecture decision): **(c1)** encoder auxiliary supervised by sim block state (privileged info — tension with the 07-11 emergence doctrine), **(c2)** contact/object-motion-window term in the ENCODER loss itself (self-generated signal, doctrine-clean, small code delta but new loss code), **(c3)** cheap diagnostic first: why are scenes 2/5 latent-blind and scene 4 golden (layouts near-identical by position/target_px — something else allocates the capacity); could redirect (c1)/(c2) or reveal an instrument artifact. GPU idle.

## 2026-08-11 — ⏸ USER DECISION: CAMPAIGN PAUSED at 100k lineage (offered: c3-diagnose / c2-contact-loss / c1-privileged-aux / pause → user chose PAUSE, encoder-loss change to be thought through offline)

Clean stopping point, everything banked: `oh_nocap2/3/4` each 25 ckpts + `state_latest.npz` on HF (verified by listing), all four probe verdicts ledgered (`runs/probe_block_goal_oh_nocap{,2,3,4}/results.json` — note the runs/ dirs die with the box; the NUMBERS live in this ledger), W&B `odmaq08m`/`aa2u4e1q`/`i4kd3d51`. **Resume point:** chain from `oh_nocap4/ckpt_0025000.pt` + its state (or from `oh_nocap3`'s if reverting the curiosity arm's behavior drift — nocap3 ended contact-richer: 0.236 vs 0.096 contacts/step, and (b) is dropped anyway) with whichever lever-(c) design is chosen; the fixed 6-scene probe (seed 41) is the go/no-go, plateau baseline median ~0.5–0.9, scene-4 existence proof 3.2–3.6, dead scenes 2/5.

## 2026-08-11 — ★ NEW CAMPAIGN (user-directed, deadline 10AM PT): WRIST CAM + NO-CAP FROM SCRATCH — "learn like a baby: wrist camera only, no injected biases; understand movement, move blocks" — `wr_nocap` LAUNCHED + probe wrist-path re-audit + BIGGER post-train shifts ★

**User spec:** back to the wrist cam; same no-action-max treatment; train from scratch, pixels-only (no proprio in latent, no guardian, no scripted data — the existing recipe already satisfies this); after training, probe block-shift salience INCLUDING a larger shift than 50mm (must stay in wrist view); bug-vigilance explicitly requested; variations allowed if basic fails, under the from-scratch/wrist-only/no-bias constraint.
**`wr_nocap` LAUNCHED (W&B `66ga98t7`, log `runs/wr_nocap_train.log`):** oh_nocap's exact command with `--wm-cam wrist`, from scratch (no init-ckpt/buffer). Launch health: **sps 33.7 (wrist renders FASTER than overhead — 25k ≈ 2.1h)**, contacts/s 0.34 at step 400 (the uncapped arm bumps blocks from birth; oh_nocap END was 0.205), pose_range 2.9 by step 400.
**Probe wrist-path audit (probe_block_goal_learn.py, code-read):** (1) staging arm pose = reset home pose ± 1° jitter → wrist viewpoint ~constant across scenes; scene diversity = cube sizes/colors/distractors; July history says block zone partially framed from home, 4-direction search finds in-frame shifts; (2) the 2026-08-08 geom_size/rgba restore fix is in and covers wrist identically; (3) reach-band check (0.18–0.36) still admits 3 of 4 directions at shift 0.10 from b0=(0.25,0); (4) attempt cap 4×scenes may under-fill on wrist — validate empirically before trusting; (5) **t=0 d0 stays a CLEAN salience read on wrist** (goal photo taken at the staged pose ⇒ pose component ~0 by construction); what changes is PURSUIT: on wrist, moving the arm moves the eye, so d rises when reaching and only falls back at the photo pose — the h=1 planner faces a look/act/look-again landscape (the honest baby version; measured, not patched).
**First-principles bets registered up front:** wrist+nocap contact windows are the best-geometry block-dynamics data in the project (contact happens AT the camera, block fills the frame — overhead saw it small and far); ego-motion makes the persistence baseline enormous so pred/persist will flatter — use absolute pred_loss + salience probes as the honest reads. **Staging validation at shifts {0.05, 0.08, 0.10} running against ckpt_0001000 (budget-1 probes; staging is checkpoint-independent).** Plan: 25k → probes at 0.05 (twin-comparable) + biggest staging-clean shift; then continuation or variation by the salience read; intermediate-ckpt probe before the deadline if the continuation is mid-flight.

## 2026-08-11 — ★★ PROBE INSTRUMENT BUG #2 FOUND & FIXED (b4b7bc0) + ★★★ wr_nocap@25k: WRIST+NO-CAP HAS THE SALIENCE THE OVERHEAD LINEAGE NEVER REACHED — 6/6 scenes, median d0 1.7 (50mm) / 2.2 ≥ eps (100mm), NO dead scenes — the viewpoint premise INVERTS: the ACTING eye is the salient eye ★★★

**Instrument bug #2 (visual inspection of staged wrist photos caught it):** the visibility guard counted `!=` pixels, but moving ONE block perturbs shadow-map/AA sampling GLOBALLY (~2k px of 1–5-LSB speckle per pair) — the >300-px guard was effectively ALWAYS-PASS (even fully out-of-frame shifts would pass on speckle). True cube footprints are >30 LSB; guard now thresholds at >10 LSB (b4b7bc0), attempt cap 4×→8×, and b0_px (start-position visibility vs the removed photo) is logged per scene. Honest footprints: wrist 50mm ≈ 404–1,052 px (scene-0 shifted cube is a frame-edge SLIVER — user's "shift it more" instinct was right), wrist 100mm ≈ 420–1,264, overhead 50mm was 1,455 real (of the ledgered "~4k" — framing conclusion stands, count corrected). Fixed-guard staging: 6/6 scenes at all three shifts. **Structural fact: b0_px 8–140 in EVERY scene — the block's START position hides behind/below the arm from the home staging pose; the wrist salience probe is an APPEARANCE test (right question for photo-goals), and pursuit starts BLIND (honest baby version, strictly harder than overhead).**
**wr_nocap 25k COMPLETE (W&B `66ga98t7`, 25 ckpts + state on HF).** Train profile vs twins: pred_vs_persist 0.397 (≈ oh_solo's 0.352 — and against a 5× bigger ego-motion persistence baseline 0.272 vs 0.051; absolute pred_loss 0.108 vs 0.037 = wrist feed is harder), step_jump 6.82 (campaign record), arrival 0.461 at dist_goal 5.5 (wrist photo-goals are largely pose-goals — arrivable by re-posing the eye), pose_range 1.72 (goal pursuit TAMES the uncapped arm vs oh_nocap's 3.28), contacts/s 0.34 birth → 0.095 end, frac_touch_block 0.045.
**SALIENCE (fixed 6-scene protocol, seed 41): 50mm shift-d0 {1.42, 1.12, 1.93, 1.54, 5.68, 3.06} vs floors {0.05, 0.08, 0.07, 0.06, 0.95, 0.26} = 6/6 above floor at 6–28×, median 1.7; 100mm {1.32, 2.42, 2.00, 1.94, 2.32, 2.95} vs floors ≤0.26 = 6/6, median 2.2 ≥ eps 2.** For calibration: the overhead lineage's best-ever median was 0.9 (50k) with 2/6 scenes dead in all four probes; wrist@25k beats it 2–2.4× with ZERO dead scenes and 5× cleaner floors — with FEWER true pixels (≈800 vs 1,455). It is not pixel count: from the acting eye, block position covaries with the encoder's dominant prediction problem (ego-motion parallax + close-range contact), so capacity lands on it for free. **The 07-13 "encoder viewpoint confirmed THE lever" verdict was right — but the winning viewpoint is the WRIST, not the overhead: pixels-visible ≠ latent-salient needed the eye that ACTS.**
**Pursuit: still 0.0mm deliberate, 0–2 contacts/episode.** The bind moves DOWN the stack (latent now carries the signal; the h=1 planner doesn't exploit it): starts are blind (b0_px ≈ 0) and the look/act/look-again landscape punishes any reach (moving the eye raises d before contact can lower it). Next moves within the no-bias constraint: (i) `wr_nocap2` →50k ALREADY RUNNING (launched pre-verdict to protect the deadline; salience trajectory + pursuit at 50k), (ii) probe-side start-VISIBLE staging variant (rejection-sample wider staging poses until b0_px AND target_px clear threshold — measurement change, not training bias): does pursuit move when the eye can see the block it must push?

## 2026-08-11 — ★★ START-VISIBLE PURSUIT PROBE (`--stage-pose visible`, 33e3f91): FIRST DELIBERATE BLOCK RELOCATION IN CAMPAIGN HISTORY — scene 2 +18.6mm deliberate (shift +18.7mm/8 contacts vs ctrl +0.1mm) — and salience is VIEWPOINT-ROBUST (median d0 1.9 from RANDOM staged poses) ★★

Variant mechanics: servo-driven staging poses (no qpos teleport → no interpenetration), rejection-sampled (~14% acceptance, cap 40×scenes) until block START (b0_px>300) and shift target are both in frame; measurement-side only, training untouched. Staged b0_px 367–4,931 (block front-and-center up close; scene-5 shift footprint 8,692 px).

## 2026-08-11 — DEADLINE CHECKPOINT (lineage-40k probe, wr_nocap2@15k mid-run): WRIST SALIENCE KEEPS COMPOUNDING (home-pose 100mm median 2.2→2.9, 4/6 ≥ eps) — the anti-plateau — but the 25k deliberate-closure did NOT recur (pursuit 0/6 at 40k; reliability is the open front)

Home-pose 100mm: d0 {3.3, 1.9, 4.2, 3.2, 2.2, 2.6}, median **2.9** (25k: 2.2) — GROWING with steps where overhead flatlined (its lineage: 0.5→0.9→0.6→0.5). Visible-pose 100mm (same seed → same 6 staged scenes as the 25k visible probe): median d0 1.5 (was 1.9), and scene 2's +18.6mm did NOT recur (+0.1mm, 1 contact) — the existence proof stands but reliability did not improve with 15k more steps; visible-pose d0 dipping while home-pose d0 climbs suggests the block code is consolidating toward training-distribution viewpoints. wr_nocap2 mid-run trends healthy (pred_loss 0.114→0.093, arrival 0.55–0.70, contacts ~0.1). Run completes ~17:50 UTC; full 50k probes after.

## 2026-08-11 — wr_nocap2 COMPLETE (lineage 50k) + 50k PROBES: TWO HONEST CORRECTIONS — wrist salience OSCILLATES (home 100mm median 2.2→2.9→1.5, not compounding) and pursuit closures are SHOVE-NOISE (ctrl +15.6/+9.7mm at 50k ⇒ the 25k +18.6mm no longer distinguishable from a lucky shove) → h=3 probe-time A/B launched (5a1bbb0)

Home 100mm @50k: {1.3, 1.1, 2.5, 1.1, 1.6, 2.6} median 1.5, 6/6 above floor — the wrist band is [1.5, 2.9] across three probes: ALWAYS far above overhead's [0.5, 0.9] and always 6/6, but noisy, not monotone; "compounding" (deadline report) was premature. Visible 100mm @50k (h=1): median d0 2.7 (scene 3 hit 8.6), deliberate 0/6 — and the two big closures landed in CTRL (+15.6mm, +9.7mm, contacts 2/13): the uncapped arm shoves blocks around near-goal poses REGARDLESS of condition. Across the three visible probes the shift/ctrl closure record is now symmetric (+18.6 once vs +15.6/+9.7 once each) — no reliable deliberateness at h=1; the honest state is ENGAGED-BUT-DIRECTIONLESS. The h=1 myopia mechanism fits: reaching raises d before contact can lower it, so directed pushes never survive planning; whether HORIZON is the unlock is exactly testable probe-side → `--horizon 3` A/B running on the same 50k head + same staged scenes.

**h=3 A/B VERDICT (same head, same scenes): horizon is NOT the unlock — it amplifies ENGAGEMENT, not DIRECTION.** Contacts way up (ctrl median 6 vs h=1's 1; scenes 3/5 ctrl 28/38), shoves bigger (+36.7mm ctrl scene 1, +13.9mm shift same scene, +3.2 ctrl scene 2), deliberate still 0/6 and the big closures remain condition-blind. **Layer diagnosis moves UP one: the encoder is salient (d0 2.7 median holds at h=3) and the planner explores contact-rich futures at h=3 — what's missing is the WM preferring futures where the block LANDS at the photographed spot, i.e., CONTACT-DISPLACEMENT FIDELITY in imagination (the July ego-contact problem, em-era MSE ~1.8–2.4, never fully solved from emergence).** Candidate next levers, all within the no-bias constraint: (α) wr_nocap3 continuation TRAINED at --cem-horizon 3 (July-validated: doubles early contact learning; gives the WM the contact-rich rollouts to grind its displacement error on), (β) longer pursuit budget + dwell at probe time (cheap, tests commitment), (γ) accept steps: keep chaining and re-probe (wrist salience holds 6/6 either way). GPU idle at juncture; day's full arc ledgered.

## 2026-08-11 — USER DECISION: lever (γ) — "continue the same recipe for longer and see if it's emergent" — LONG-HAUL PURE-STEPS CHAIN LAUNCHED (`wr_nocap3` →75k, then +25k segments with probes at every boundary)

Rationale accepted: lineage is only 50k (July wrist chain accumulated ~370k; overhead got 100k before its plateau call); hot6's contact era emerged at ~35k with zero recipe change; the July ego-contact plateau (1.8–2.4) was measured on a DIFFERENT data engine (capped+guardian, sparse self-contact) — this lineage self-generates dense close-range contact (10–37mm shoves every probe episode), exactly the gradient the WM's displacement error needs; γ has the cleanest attribution (any emergence = pure steps).
**Boundary protocol (each +25k, mechanical):** probes = home-0.10 + visible-0.10 (h=1, seed 41). **Pre-registered emergence criteria: (E1) deliberate closure positive on ≥2/6 shift-side scenes with clean ctrl (directedness), or median deliberate > 0; (E2) salience-health: home median stays ≥1.5 band; (E3) WM online mse_block trend across segments (displacement-fidelity proxy) — falling = grinding, flat = orbiting.** Escalation rule: if by lineage 150k (4 more segments) E1 has not fired, pure-steps at this scale is answered NO and (α) h=3-trained continuation becomes the next arm. Chain continues until the user stops it or a criterion fires.

## 2026-08-11 — ★ USER-DIRECTED: `arr95_nocap` LAUNCHED — arr95_hot VERBATIM (W&B-config-reconstructed) with ONE flag changed: action_max 0.05 → 5.6 — first gated+uncapped run in campaign history ★

**History check (W&B config sweep, all ~150 runs): the gated ladder has NEVER run uncapped** — all 89 gated runs (cem_* → hot7vlab) were action_max 0.05; all uncapped runs were recent-select. **Config recovery also corrected the recipe's memory: arr95_hot was WAKE/SLEEP** — `freeze_encoder=True` during the ladder (06-26 stationary-latent requirement: frozen k→23, thaw stalls k≈4) + `cotrain_every=3500 / cotrain_epochs=30 / lr 2e-5 / β 0.02 / flatline+0.45 gates` (the "explore, then train a huge batch on recent data" sleep phases) + consolidate 70/1 as the continuous predictor engine + dwell (1.25/2/0.3) + old gentle planner (CEM 30 iters/300 samples/30 elites, init_std 0.3) + fixed_objects FALSE + eps 2.8 + ladder d 1→12 (arrival bar 0.95, patience 150, hot-retarget 1.2) + buffer_frac 3 + init `scratch_a3/ckpt_0003000.pt` buffer-empty.
**`arr95_nocap` = exactly that with `--action-max 5.6`, SAME init brick** (scratch_a3@3000 recovered from HF ⇒ one-flag-clean historical A/B; NB the brick's 3k build data was collected capped — part of the replica). 24k steps, W&B `s9w7cmbb`, log `runs/arr95_nocap_train.log`; init banner verifies frozen encoder (6.29M stop-grad) + 300/30/30 σ0.3 + archive loaded. Launched CO-TENANT with wr_nocap3's last ~10k (γ chain pauses at 75k after boundary probes; hot6 shared-GPU precedent). **Baseline: historical arr95_hot = d 1→8 in 24k, 42 advances, post-thaw arrival 0.958.** Watch: (1) ladder advance rate under uncapped action semantics (σ-shrink must find fine control in the ~3%/dim reachable-target geometry); (2) frac_rand/locality on uncapped DIRECTED data (era failure wrist_loc_amax10 hit 0.46 — but that was continuous-training; frozen+gated-sleep is the counter-design); (3) contact rate (capped ladder ~0.015/step; if rungs climb AND contact stays nocap-rich, the campaign's two threads merge).
**Early telemetry (step 10k/24k, co-tenant 9.8 sps):** ladder at d=3 with the full pctl band complete — arrival 0.96–1.00 at EVERY rung so far (~15 advances by 9.2k; pace ≈ d5–6 by 24k, vs baseline 8 solo). **contacts/s 0.53 = 35× the capped-ladder era and 2.5× any free-roam run — the mastery thread and the contact thread ARE merged.** mse_blk 1.30→0.15 as the buffer filled; pose_step 0.23 (precision-hold regime — the "frozen-looking" videos are the d≈1–3 dwell mechanics doing their job, motion budget grows with d).

## 2026-08-12 — ★★ γ-CHAIN 75k BOUNDARY: E1 FIRES (first directedness signal — 2/6 scenes shift-side with CLEAN ctrls: +8.8mm and +1.8mm, ZERO ctrl-side closures) + E2 ALL-TIME RECORD (home median d0 3.8, first-ever 6/6 ≥ eps) → chain PAUSED per protocol, GPU to arr95_nocap solo ★★

Home 100mm @75k: {6.1, 2.3, 7.0, 2.5, 4.6, 3.0} median **3.8** (trajectory 2.2→2.9→1.5→3.8 — the 50k dip was oscillation; salience at 75k is the strongest ever measured, scene 2 at 7.0). Visible 100mm: median 2.2, and the closure asymmetry FLIPPED to the shift side — scene 3 **+8.8mm deliberate** (shift +9.2/2con vs ctrl +0.4/1con), scene 5 +1.8mm (ctrl 0.0 despite 11 contacts = engaged-but-directionless in ctrl, DIRECTED in shift). Contrast 50k, where both big closures were ctrl-side. **E1 criterion (≥2/6 shift-side, clean ctrl) MET — marginally (one strong, one small) but with the right coincidence: directedness appeared in the same probe as the salience jump.** E3: mse_block across segment ends 0.177→0.152→0.148 = slow grind, not orbit. **Per pre-registered protocol a fired criterion pauses the chain** → wr_nocap holds at lineage 75k (all state banked); next γ decision (extend to confirm E1 at 100k vs pivot) is the user's, informed by arr95_nocap's outcome. The two live threads now tell one story: salience compounds on pure steps (γ @75k), and the gated ladder rides a 0.53-contacts/step stream (arr95_nocap @10k) — direction is emerging at both ends.

## 2026-08-12 — ★ arr95_nocap COMPLETE (24k): THE UNCAPPED LADDER WORKS — d 1→5 (pctl 0.6), arrival 0.95–1.00 at EVERY rung, zero locality aborts across 6 sleeps, contact 10–35× the capped era — + SHORT-PROBE: salience median 2.0 from ~1,600 encoder updates (most data-efficient salience route ever) → `arr95_nocap2` CONTINUATION LAUNCHED to the FULL LATENT SPACE (user-directed: d-max 22, 100k) ★

**A/B vs history (baseline d 1→8/24k, arrival 0.958):** uncapped reached d=5 pctl 0.6 — slower on rungs, with the co-tenancy asterisk (first ~11k shared with wr_nocap3 + two probe suites; old planner 30/300/30 at ~10–11.6 sps). Every logged rung passed at 0.95–1.00 — the gate NEVER stalled; σ-shrink finds fine control under uncapped semantics. **All 6 cotrain sleeps CONVERGED in 218–303/11,700 grad steps, zero flatline/locality aborts — the June "uncapped directed data destroys the latent" failure mode did not appear under frozen+gated-sleep.** Contact 0.15–0.53/step across phases (capped ladder era: 0.015). mse_blk 1.30→0.15.
**Short salience probe (@24k, 100mm, budget-1, `runs/probe_arr95nc_24k_short/`):** shift-d0 {3.22, 1.27, 0.82, 2.06, 2.68, 2.01} median **2.0**, 4/6 ≥ eps, over the CLEANEST floors ever (0.01–0.05 — the frozen+sleep encoder restores near-perfectly). Calibration: ≈ the free-roam wr line's salience at 50k, from an encoder that trained only ~1,600 gentle sleep updates on a 3k capped brick. **Mastery-gated + contact-rich is the most data-efficient block-salience recipe measured to date.**
**`arr95_nocap2` LAUNCHED (user: "continue further before probing… run it to the full latent space"):** verbatim chain from `arr95_nocap/ckpt_0024000.pt` + full state (buffer rides), regime-matched `--goal-curric-d-start 5 --goal-mse-pctl-start 0.6`, **`--goal-curric-d-max 22`** (the hot2-era deep-frontier cap), `--total-steps 100000` (~19h at old-planner solo pace; ckpt/1k + state/10k ⇒ stoppable/probe-able at any boundary). Full probe suite deferred to a mature boundary per user. Log `runs/arr95_nocap2_train.log`.

## 2026-08-12 — arr95_nocap2 KILLED AT 21k BY DISK-FULL (the arr95_hot5 storage-kill replay) → 123GB verify-then-freed → `arr95_nocap2b` relaunched from the intact 20k state + FIRST REAL LADDER WALL OBSERVED: 15k steps stuck at the d=5→6 gate

**Crash:** torch.save write-fail at ckpt_0021000 — `/workspace` at 150G/150G (nine 15GB state snapshots + ~150 local ckpts accumulated across the week's lineages). Recovery per the hot5b playbook: deleted 183 files / **123GB**, every one VERIFIED on HF first (one exception found and KEPT local: `wr_nocap/state_latest.npz` — its teardown upload had silently failed; sole copy is now local, flagged for re-upload). `arr95_nocap2b` relaunched from `arr95_nocap2/ckpt_0020000.pt` + its intact 20k state, `--goal-mse-pctl-start 0.8` (re-verifies the top band then faces the same gate). Ops note for 100k+ runs: peak disk = ~10G ckpts + 15G rolling state + 15G tmp-swap spike ⇒ keep ≥45G headroom.
**Restore addendum (W&B `ohr0vzpc`):** the "intact 20k state" was actually the **10k** snapshot — the 20k write died mid-tmp-swap when the disk filled (the deleted `state_latest.tmp.npz` was its corpse; the atomic-swap design correctly preserved the 10k `state_latest.npz`). So 2b = weights@20k + buffer@10k-era. Accepted mismatch, not relaunched: the missing 10k–20k window was the stuck/contact-collapsed stretch (low-value coverage), both windows are the same d=5 regime, and the alternative (ckpt@10k for consistency) throws away 10k steps of predictor training on nothing better.

## 2026-08-12 — ★★ VERDICT: THE d=6 WALL IS STRUCTURAL, REPLICATED, AND MECHANISTICALLY IDENTIFIED — the uncapped ladder's latent lives at frac_rand 0.34–0.45 (vs capped-era ~0.13–0.20, "local" ~0.09), too coarse for reach-scale mastery; three emergency locality repairs in 2k steps, zero rungs — UNCAPPED AND CAPPED ARE COMPLEMENTARY PHASES, NOT COMPETITORS ★★

**Evidence:** run 2: 15k stuck at the d=5→6 gate (contacts collapsed). Run 2b (independent re-attempt from re-verified band, fresh sleeps): 7.2k more stuck (6.6k→13.8k+), **cotrain-trigger emergency repairs #1/#2/#3 at 11788/12188/13765** (frac_rand ceiling 0.45 hit repeatedly; each repair converges in ~200 steps, holds 0.4–1.6k, re-erodes). W&B trace: locality NEVER healthy in this regime — frac_rand 0.34–0.45 from step 150 onward. **Mechanism, tying the whole campaign together:** uncapped action semantics ⇒ large per-step latent motion ⇒ permanently coarse geometry ⇒ the 06-26 bootstrap requirement (predictor sharpening against FINE stationary geometry) is unmeetable at d≥6, σ-shrink cannot close within eps 2.8, arrival never sustains 0.95, and the failed directed attempts themselves erode locality toward the 0.45 rail (the June directed-motion-breaks-locality tension, now measured inside the gated design). The capped baseline crossed d=6 without pause BECAUSE 0.05-cap keeps step_jump tiny.
**The synthesis the A/B delivers: no-cap and cap are PHASE TOOLS.** No-cap: 10–35× contact, salience median 2.0 from ~1,600 encoder updates (best data-efficiency ever), E1 directedness on the free-roam line — the EXPLORATION/DATA phase winner. Cap (or moderate 0.4-rung): fine geometry, deep ladders, d→21 — the MASTERY/CONTROL phase winner. The mechanical sweep's "no rung between 0.4 and none" plus this wall says the informed next composite: build/explore uncapped → ladder at 0.4 (same physics as uncapped, finer action geometry) on the salience-mature head.
**Run left LIVE** (self-repairing, state banking; user directive "run to the full latent space" stands until revoked). **Decision menu for the user:** (i) stop 2b, ladder the salience-mature wr75k head at amax 0.4 (the composite bet); (ii) stop 2b, σ/patience surgery on the uncapped ladder (treats symptoms; mechanism argues against); (iii) let 2b run its 100k (cheap to keep; sleeps may yet shift it — em-era says knowledge orbits, so low odds); (iv) probe-first: full suite on arr95_nocap@24k+2b head to bank what the gated-uncapped line BUILT before any pivot.

## 2026-08-12 — ★ USER CHALLENGE TO THE VERDICT ("maybe something is wrong with the checkpoint and buffer reload") → RELOAD-PATH AUDIT (clean) + THE EVIDENTIAL HOLE it exposed (run 1 NEVER faced the d=6 gate — the wall exists ONLY in restored runs) → 2b KILLED @~84k, `arr95_nocap_fresh` DISCRIMINATOR LAUNCHED (40k, NO restore) ★

**Audit of the save/restore/cache stack (all clean):** (1) snapshots store RAW PHOTOS everywhere — pixel rings, proprio, actions, HER goal photos, ep ids; no latents serialized; (2) z_cache cannot leak stale/zero latents across restore: `z_valid` inits all-False, restore never touches it → lazy re-encode under the current encoder; `z_invalidate()` fires on every thaw; (3) goal archive = photos, re-encoded per refresh; highmse candidates encoded FRESH each 50-step refresh; active z* re-encoded from photo every step; (4) restore marks the horizon `is_start` (no WM window crosses) and restores ep_id/cur_ep coherently (HER can't relabel across the seam). No smoking gun.
**But the challenge SURVIVES the audit:** run 1 ended at 24k having just entered d=5's top band — it never attempted d=6. Every observed d=6 failure (15k in run 2, 77k + 166 emergency repairs in 2b) is downstream of a state restore. "Replicated across two runs" was two RESTORED runs — correlation with restarts is 100%, and code-reading cannot exonerate what only an experiment can. 2b's marginal value ≈ 0 → **killed at ~84k (user call)**. **`arr95_nocap_fresh` LAUNCHED:** run-1 command verbatim (scratch_a3@3000, buffer-empty, d-start 1, d-max 12, old planner, wake/sleep, amax 5.6), `--total-steps 40000` — at run-1 pace the gate arrives ~20k, leaving ~16–20k of organic attempt evidence. Pre-registered read: fresh STALLS at d=6 ⇒ wall structural, restart exonerated, composite bet proceeds on solid ground; fresh PASSES d=6 ⇒ the restart seam is implicated, dig with a repro (and the coarse-geometry mechanism gets re-examined as consequence-not-cause). Log `runs/arr95_nocap_fresh_train.log`.

## 2026-08-13 — arr95_nocap_fresh (40k) MISSED THE GATE BY 400 STEPS (d=5 band completed at 39,600/40,000 — never attempted d=6; sizing miss: d=5 rungs stretched to 6.75k) → `arr95_nocap_fresh60` relaunched (same seed/command, 60k — deterministic replica + ~20k gate attempt)

What the 40k DID establish: (1) **determinism is real** — fresh replicated run 1's lone emergency trigger at the EXACT step (2109) and the ladder shape; (2) **locality stays healthy in a never-restored uncapped run: 1 trigger in 40k vs 2b's 166** — repair-thrash is specific to gate-grinding and/or restored state, NOT to uncapped training per se; (3) **soft structural evidence pre-gate:** d=5's percentile rungs slowed 2.1k → 3.75k → 6.75k → 5.25k (all passed 0.95–0.96, no wall INSIDE d=5, but the deep-band cost curve is bending up restore-free). Contacts drifted 0.25→0.05 across the run. `fresh60` retraces the 40k prefix (same seed ⇒ ~free replica) then attempts d=6 with never-restored state — the discriminator, properly sized. Log `runs/arr95_nocap_fresh60_train.log`.

## 2026-08-13 — ★★ DISCRIMINATOR VERDICT (arr95_nocap_fresh60, 60k, ZERO restores, clean teardown): THE WALL IS STRUCTURAL — RESTORE EXONERATED. Stuck 20,550 steps on d=5's top rung (39,450→60k), reach 0.02, erosion triggers appearing organically (#2 @56,344, #3 @59,636) — the 2b signature reproduced restore-free ★★

**Four runs now hit the same boundary, restored or not:** run 2 (restored): 15k stuck at the d=6 gate. 2b (restored): 77k stuck + 166 repairs. fresh40 (no restore): scraped d=5's last rung at 39,600/40,000. fresh60 (no restore, diverged from its seed-sibling after ~30k): NEVER passed that rung — 20.5k stuck, reach 0.21→0.02, contacts →0.03, erosion loop starting at the end (3 triggers total: the deterministic 2109 + two at 56.3k/59.6k). **The wall's exact location wobbles one rung with seed noise (d=5 top band ↔ d=6 gate) — the signature of a marginal capability boundary, not a discrete bug.** The user's reload challenge was the right scientific move (it exposed that all prior wall evidence was restore-downstream) and its answer is now clean: the uncapped ladder structurally cannot sustain 0.95-arrival mastery past d≈5–6 — coarse latent geometry (frac_rand 0.34–0.45 baseline) meets the fine-geometry bootstrap requirement and loses, with failed reach attempts eroding locality as the terminal mechanism. **The 08-12 phase-synthesis stands on clean ground: no-cap = exploration/data phase (contact 10–35×, salience 2.0 @ ~1.6k encoder updates, E1 directedness on the free-roam line); cap = mastery/depth phase (d→21 historical). Composite bet (explore uncapped → ladder at ~0.4 on a salience-mature head) is the evidence-backed next move; GPU idle awaiting the user's call.**

## 2026-08-13 — ★ USER-DESIGNED LEVER SUPERSEDES THE COMPOSITE: LOCALITY-GATED ACTION-AMPLITUDE CURRICULUM (`--amax-curric`, 554518c) — amplitude is EARNED by prediction quality; `wr_amaxcur` LAUNCHED (40k, ceiling = literal no-cap) ★

**User: "what if we did a curriculum for action max, while monitoring and keeping locality below [a bar]" → threshold set 0.2 (user), ceiling uncapped (user: "ideally it should be able to reach uncapped").** Design: the applied action fraction starts at eff amplitude 0.05 and grows ×1.15 per 400 steps while the LIVE windowed frac_rand (step_jump_recent on EXECUTED actions — the same signal the cotrain emergency uses) stays ≤0.2; backs off above 0.25; bounds [0.05, action_max=5.6]. Replaces the guessed-cap composite with a controller that finds the equilibrium dynamically — and can reach full no-cap if the maturing encoder keeps absorbing bigger strides (earliest ~14k steps at the raise rate). Directly targets the wall's mechanism (map resolution governs stride) and is doctrine-clean (the gate is the robot's own model quality). Why 0.2 not the June 0.1: the 0.08–0.12 era numbers were overhead+BatchNorm recipes; wrist-cam capped baselines measure ~0.15–0.29 — 0.1 would pin the curriculum at the floor. Implementation reuses the existing amplitude-fraction path (action_max_warmup machinery) + live frac_rand window; logs `[amax-curric]` lines + `explore/eff_action_max`; smoke-verified (first adjustment fired at step 400: frac_rand 0.060 → raise).
**`wr_amaxcur` LAUNCHED:** the wr build recipe (recent-goals, wrist, fixed objects, continuous encoder training — controller is the ONLY amplitude governor; cotrain emergency off in this recipe) + `--action-max 5.6 --amax-curric`, 40k budget so the equilibrium (or the full climb to uncapped) has time to express. Log `runs/wr_amaxcur_train.log`. **The amplitude trajectory IS the result:** where eff_action_max sits when locality holds at 0.2 — and whether it reaches 5.6 — answers what the honest stride is at each stage of encoder maturity. Then: salience/pursuit probes on the head + (if equilibrium ≥ ~0.3) the mastery ladder AT the equilibrium.
**RESTART (user, @4.7k): wr_amaxcur stopped and relaunched as `wr_amaxcurb` (W&B `o6dlfb4h`) to pick up the new `train/goal` panel (d2650de — per-env goal photo tiled to match train/wrist, frame-synced to goal refreshes).** From-scratch restart, not a resume: the controller's amplitude state is runtime-only (a restore would pair mature weights with reset amplitude) and this campaign just learned to distrust seams. Early trajectory before the stop, first 4.7k: eff amax 0.050→0.087 in 4 raises by 2k (frac_rand 0.01→0.194 as the latent organized), then HOLDING in the 0.2–0.25 band for 2.65k — the controller found the early honest stride (~1.7× capped baseline) and parked at the locality frontier. The b-run retraces (same seed).
## 2026-08-13 — ★ SECOND USER-DESIGNED MECHANISM: PROGRESS-GATED GOAL RETENTION (`--goal-retain`) + goal engine switched recent → HIGHEST-MSE-UNDER-d — `wr_amaxcurb` superseded by `wr_msegate` (amax-curric + MSE-gated goals + retention, 40k) ★

**User's critique of recent-mode ("why are the goals changing so rapidly… that's bad"):** fixed-cadence refresh (every 25) discards goals mid-progress; the rule should be "**for the current goal: are we getting closer continuously? if so do not change goal; if we are stagnating for a long time then switch**" — and goals should be surprise-seeking, not recency-treadmill: "**highest MSE goals within the latent distance gate, each environment separated**."
**Design as built (`--goal-retain`, per env independently):** progress = any new best distance to the current goal (≥1% better than the previous best, `--goal-retain-delta`); pursuit ends ONLY on (a) **arrival** — inside eps, with a ≥10-step hold (`--goal-retain-arrive-hold`, see bug below), (b) **stagnation** — no new best for `--goal-retain-patience 30` steps, (c) **max-age 200** valve. On switch: the arrival outcome (ever inside eps) is recorded as a per-goal COMPLETION — the d-curriculum's arrival series unchanged in interface, now measuring "goals finished" rather than windowed proximity. The `goal_update_every` cadence remains only to seed goalless envs + pace archive maintenance. Goal source: `highmse_under_d` (each env pursues the most-mispredicted buffer state within latent budget d of ITS OWN position; d grows 1→22 on 0.95 completion-mastery). Retention makes MSE-goals safe: an unreachable surprise stalls → patience expires → try the next interesting thing. **Synergy stack now: (1) locality-gated amplitude (earn your stride), (2) mastery-gated distance (earn your radius), (3) progress-gated retention (finish what you started or admit stall) — three intrinsic gates, no injected knowledge.**
**Design bug caught in smoke #1:** at ladder-start d=1 < eps=2.8, goals spawn ALREADY INSIDE the ball → arrival-triggered refresh churned every step (1,425 switches/400 steps, sps halved by 256-candidate re-proposal). Fix = arrive-hold 10. Smoke #2: 78 arrivals + 17 stalls / 300 steps (≈1 goal per 25 steps/env at d=1, mean pursuit age 20) — both switch paths healthy, churn ×18 down. New W&B: `goal/retain_{arrive,stall,age,age_mean}`.
**`wr_msegate` LAUNCHED** (40k, wrist, from scratch): wr build base + `--amax-curric` + `highmse_under_d` d 1→22 (arrival-metric bar 0.95/150, eps 2.8) + `--goal-retain` (30/1%/200/hold-10) + the `train/goal` panel (d2650de). Supersedes `wr_amaxcurb` (killed @~5k; its lone contribution: retraced the amplitude staircase 0.050→0.087→hold at the locality frontier, matching the a-run). Caveat noted: ladder-with-continuously-training-encoder means curric_d lives in drifting units (historical ladders froze the encoder); retention's RELATIVE progress rule is drift-tolerant by design — watch d-advance behavior for artifacts. Log `runs/wr_msegate_train.log`.

## 2026-08-14 — wr_msegate VERDICT @24k (killed): AMPLITUDE MECHANISM = CLEAN SUCCESS (0.05→0.31 earned, 6.2× baseline, equilibrium ~0.27–0.31, contacts to 0.41/step = campaign record) / LADDER = FROZEN BY SCALE MISMATCH (d-start 1 in old-latent units; this latent's nearest-candidate shell sits at ~10–12 ⇒ every goal was a beyond-budget fallback, stall-dominated, completions ~0, d froze at 2 for 22k steps) → USER: "rescale the start" → `wr_msegate2` LAUNCHED (d-start 10, all else identical)

Amplitude staircase (locality-gated, bar 0.2/rail 0.25): floor 0.05 → 0.087 → back-off at 0.276 → recover → 0.13 (8k) → 0.20 (10k) → 0.27 (15k) → 0.31 (18.8k) → rail hit 0.251 @21.6k → 0.268 — **the controller discovered and then oscillated around the honest equilibrium ~0.27–0.31, right in the mechanical sweep's slew-plateau zone; locality NEVER broke; contacts scaled with amplitude to a record 0.41/step.** The user-designed mechanism works as intended. Retention counters: arrivals 1.2k+/stalls 0.9k+ (early in-budget era), then stall-dominated once goals went 100% fallback (~11 latent units vs budget 2). Ladder d froze at 2/pctl 0.6 from step 1350 to the kill. Root cause = the ledgered drift caveat: d units are THIS latent's units; arr95-era d-start 1 describes a shell that no longer exists. Options offered (honest-budget/no-fallback code fix; d-start rescale; ride it out) → **user chose rescale: `wr_msegate2` = same stack, `--goal-curric-d-start 10`** (the observed shell), 40k, W&B/log runs/wr_msegate2_train.log. Watch: does the amplitude staircase retrace (~2.5h) AND do in-budget goals at d=10 produce real completions (close 10→eps 2.8) — reach was 0.01–0.05 in the frozen era, so this directly tests pursuit at the discovered equilibrium amplitude.

## 2026-08-14 — wr_msegate2 (d-start 10) killed @11k → `wr_msegate3` (retention patience 30→100, delta 1%→0.5%) — user watched the goal panel and pursuits were still ~2s of video: the stall clock dominated (93% of switches at patience-30; mean goal life 35–40 steps)

wr_msegate2 banked: d-start 10 fixed the empty-budget bug (completions real: 614 cumulative, share oscillating 7–18% by window, dipping when amplitude pushes and recovering after — the finish-rate/stride tension made visible); amplitude retraced to 0.153 by 10k. But the retention's visible behavior wasn't the user's intent — slow approaches were being cut at 30 steps. **wr_msegate3 = same stack, `--goal-retain-patience 100 --goal-retain-delta 0.005`**: a 100-step leash (5s of video) and gradual closing (≥0.5% new-bests) counting as progress. Expected effects: goal lifetimes ~3–5× longer, fewer-but-longer pursuits, completion share likely UP (more time to finish), stall share down; risk = more steps on hopeless goals (bounded by max-age 200). Log runs/wr_msegate3_train.log.

**PRE-REGISTERED NEXT (user: "once this is done, if it works lets try replacing it with a policy"):** *works* = amplitude well clear of the 0.05 floor with locality holding ≤0.2 + healthy encoder + salience probes ≥ wr-line par. Then the POLICY ARM — the fair re-test of goal-conditioned RL that June's immature-latent loss didn't give: chain from wr_amaxcur's final head with `--freeze-encoder` (coordinates held still = the drift objection dies), CEM OFF, an **ACTION-CONDITIONED goal policy (user-specified): π(a_t | z_hist, a_hist, z\*)** — the SAME context the planner/WM consumes, not memoryless (z,z\*); load-bearing here because the no-proprio latent carries no velocity — action history restores the momentum context (the in-tree SAC actor needs a modest history-conditioning extension: concat flattened hist or small GRU) — + latent-distance reward + HER, amplitude FIXED at the curriculum's equilibrium; twin = CEM continuation of the same head, same budget; verdict on arrival/reach/contacts + block-pursuit probes. Fallback rung if pure RL underwhelms-but-lives: distill CEM → policy, then RL fine-tune. Rationale ledgered 08-13 (amortization + fine final-approach are the policy's honest wins; planning keeps winning while the model outruns the policy's staleness).
**The science in the crash window: the uncapped ladder hit its first WALL.** Continuation ladder: pctl 0.8@3.6k → 1.0@6.15k (arrival 0.95–0.96), then **NO advance for 15k steps at the d=6 gate** — arrival pinned ~0.05–0.06 reach / sub-bar arrival, contacts COLLAPSED 0.5→0.02–0.06, dist_goal ~6.4, pose_step drifting up 0.38→0.48. Reading: at d≥6 the σ0.3 planner must execute multi-step directed reaches under slew-bound uncapped semantics and is not closing; the capped baseline crossed d=6 without pause (d=8 by 24k from scratch). This is the honest A/B result forming: **uncapping bought contact richness and salience data-efficiency at shallow d, but COSTS ladder depth at the fine-control frontier** — exactly the trade the 08-07 mechanical sweep predicted (at amax 5.6 only ~3%/dim of action space is within-step-reachable; fine approach must come from σ-shrink, which is failing at reach-scale goals). If 2b stays stuck through ~2 more sleeps, candidate levers (user's call): σ-schedule/init_std bump for reach-phase, d-gate patience, or accept d≈5 as the uncapped ladder's regime and probe what it BUILT (block salience per the 24k short probe) rather than what it can't climb.
**Results (wr_nocap@25k, 100mm, budget 60):** d0 {1.81, 1.30, 1.73, 1.98, 2.26, 3.65} vs floors ≤0.15, median 1.9 — the block code holds from arbitrary poses, not just the home pose (the overhead lineage never had this from its ONE pose). **Scene 2: deliberate +18.6mm** — the child pushed the block a fifth of the way to the photographed target, shift-specifically (ctrl +0.1mm, 6 contacts — engaged but directionless). 1/6 scenes moved; medians still 0.0. **Reading: the full stack (wrist latent → CEM h=1 → uncapped plant) CAN do deliberate relocation when the eye sees the block and the target is latent-salient; the residual gap is RELIABILITY (myopic h=1 landscape: reaching raises d before contact lowers it; no dwell), not representation.** Layer-by-layer today: camera ✓ (this was never wrist's problem once shifts stay in frame), encoder ✓ (salience 6/6 scenes, viewpoint-robust), planner ~ (1/6), behavior ~ (18.6mm existence proof).

## 2026-08-14 — wr_msegate3 superseded AT LAUNCH (killed during setup, 0 steps logged; W&B `pthcc01l` stillborn) → `wr_msegate4` (retention patience 100→200 = the user's full 10-second leash; all else identical) — and the LADDER TURNS: first d-advance of the d-start-10 era, d 10→11 @ step 900 on arrival 1.00

User extended the leash to 200 steps (10s of video) before msegate3 accrued any data — the 30→100 reasoning taken to the intended duration in one move. **wr_msegate4** launched 01:53 from scratch (W&B `bqga79ce`, log `runs/wr_msegate4_train.log`), the full three-gate stack at d-start 10: `--amax-curric` (floor 0.05, ceiling 5.6, bar 0.2/rail 0.25) + `highmse_under_d` d 10→22 (arrival bar 0.95, curric patience 150) + `--goal-retain` 200 / 0.5% / max-age 500 / arrive-hold 10.

**The advance:** `[goal-curriculum] step=900 arrival=1.00 -> d=11.00` — the mechanism that froze in msegate (empty-shell budget) and oscillated without advancing through msegate2 (completions real at 7–18% but no advance in 11k) turned in 900 steps once the budget was satisfiable (d-start at the real shell) AND goals had time to finish (10s leash). Honest caveat logged at the moment of celebration: steps ≤1000 are start-steps, so the 1.00 arrival streak that earned the advance predates CEM control. The post-warmup era opened exactly at 1000: dist_goal 1.3 → 8.0 and reach 0.5 → 0.01–0.02 by step 1250 as the planner took over real d=11 pursuits (archive evictions live, contacts/s 0.25–0.40, mse_blk 0.10–0.11 flat, amplitude 0.066→0.076 on schedule, frac_rand ≤ 0.044 — nothing unhealthy, just the easy era ending). **Whether advances continue under planning is THE question of the run.** Monitors: event-wake on d-advance/crash/exit + 20-min ticker heartbeat.

## 2026-08-14 — wr_msegate4 VERDICT @8k (fork FIRED): unreachable goals POISON the amplitude mechanism — equilibrium COLLAPSED to 0.066–0.076 (vs msegate1's 0.27–0.31) with reach 0.00 for 6,300 steps; erosion signature (mse_blk 0.11→0.17 ratchet) → restart as `wr_msegate5`, d-start 10→**7** (user-proposed mid-run; agreed tripwire fired)

The run's whole arc: advance 10→11 @900 + two pctl rungs (0.20 @1050, 0.40 @1650, all arrival 1.00) — then NOTHING. No gate pass 1650→8k; reach 0.00 continuously post-1900; dist_goal parked 17–22 against an 11 budget. The amplitude controller ran a full honest cycle: raises to 0.116 @2800, first-ever rail hits (frac_rand 0.303/0.268/0.290/0.301), four backoffs to 0.066, locality recovered @7200 (0.197), re-raise to 0.076 — **the mechanism worked perfectly and discovered that this run's sustainable stride is barely above the 0.05 floor.** That collapse (msegate1 sustained 0.27–0.31 on the SAME latent recipe) is the smoking gun: it isn't amplitude that broke the map — it's the goals. pctl-0.40 mid-MSE goals at d=11 sit beyond the h=1 closing radius (code's own estimate ~5 units); every pursuit fails; failed pursuits erode locality (mse_blk ratcheted 0.11→0.17 and never healed though tbl/none recovered; two no-contact excursions; contacts otherwise 0.40–0.52 = campaign record). **The inner MSE curriculum walks goals INTO the WM's blind spots while amplitude raises stride — two difficulty axes moving at once; at pctl 0.40 × d 11 the product went unreachable and the erosion loop (fresh60's terminal mechanism) switched on.**

Code-dig facts established en route (all verified in src/train.py): (1) CEM acting objective is PURE goal-distance — "NO reach reward — goal-reaching is PURE planning"; cur_contrib is buffer-prio/logging only, so goal-fleeing-for-surprise is impossible by construction. (2) `highmse_under_d` falls back to NEAREST candidate when under-d is empty — d-start below the shell cannot reproduce msegate1's empty-shell freeze. (3) Episode resets clear retention state correctly ("retention state must not survive a reset") — reset-spanning-goals hypothesis tested and exonerated. (4) max_episode_steps=200 = retention patience: a full-leash pursuit spans exactly one episode.

**Decision (user proposed d=7 at ~02:15; joint tripwire = "frac_rand back under 0.2 AND nonzero reach by ~6.5k"):** locality half passed (0.197 @7200), reach half failed (0.00 unbroken) → fork fired @8k. `wr_msegate5` = identical stack, `--goal-curric-d-start 7` — pursuit starts near the closing radius so goals FINISH; pctl restarts at 0 (low-MSE first). Prediction registered: if goal-poisoning is the mechanism, msegate5's amplitude equilibrium should re-approach msegate1's 0.27–0.31 AND the ladder should pass rungs post-warmup. Kill of msegate4 blocked by the permission classifier (process-kill needs the user's hand) → auto-relaunch armed on process-death; launch details next entry.

**Addendum @13.7k (kill still pending, run continued):** the "equilibrium COLLAPSED" reading was premature — it was a TROUGH. After a ~4k-step low-amplitude healing phase (0.066–0.087; map reached its best state of the run, mse_blk 0.07), the controller RESUMED CLIMBING: 0.133 @12.4k, 0.153 @12.8k, frac_rand 0.15–0.195 under the bar — the msegate1 climb-break-heal-climb shape, slower. So **goal churn does not permanently poison the amplitude mechanism** (claim weakened to: each push re-triggers a contact-collapse excursion, 0.06 @13.7k, and the healing detours cost ~4k steps/cycle). What the extra 5.7k steps did NOT change: **reach 0.00 unbroken, dist_goal ~19, ZERO gate passes — the ladder is 12k steps frozen at d=11/pctl 0.40.** Restart rationale sharpened, not reversed: the curriculum cannot progress at this goal distance regardless of amplitude — h=1 closing radius is the binding constraint. d-start 7 restart stands.

## 2026-08-14 — wr_msegate4 KILLED @14k (user) → `wr_msegate5` LAUNCHED: d-start **6** (user: "lets start at a lower d. like d = 6" — chosen over the discussed 7; goals start ≤6 with eps 2.8 ⇒ ≤~3.2 units to close, inside the h=1 radius), all else identical

msegate4 final tally (14k steps): 1 d-advance (10→11 @900, warmup-era) + 2 pctl rungs (@1050, @1650) — then 12.3k steps frozen, reach 0.00 unbroken post-1900. Amplitude at kill: second climb at 0.153 and rising, map healed (mse_blk 0.07 = run's best), contacts oscillating 0.05–0.52 with pushes. The run's scientific residue: (a) unreachable goals freeze the curriculum but do NOT permanently poison amplitude or the map (trough-heal-reclimb proven); (b) a healthy floor-amplitude map still cannot close 11-unit gaps with h=1 — goal DISTANCE, not map quality, is the binding constraint; (c) pgrep/pkill self-match footgun cost one false "msegate5 running" report (fixed with `[t]`-bracket patterns — ops note for all future watchers).

**`wr_msegate5`** launched 03:35 (PID 144068, W&B `r496uw67`, log `runs/wr_msegate5_train.log`): the three-gate stack, `--goal-curric-d-start 6`, d-max 22, pctl restarting at 0. Registered predictions: post-warmup completions should be REAL (gap ≤ closing radius), rungs should pass, and if the freeze-was-the-poison story is right the amplitude staircase should climb past msegate4's 0.153 toward msegate1's 0.27–0.31 equilibrium WITH the ladder moving under it — the first run where both mechanisms can work simultaneously. Ops: disk was at 94% (11G) pre-launch → HF-verified (byte-exact: 14 ckpts + 15.07GB state msegate4; 25 ckpts + state wr_nocap) → deleted the two local state_latest.npz → 39G free (75%). Monitors re-armed on the new run (event-wake d-advance/crash/exit + 20-min ticker).

## 2026-08-14 — ★ msegate5 @7.2k RESOLVES THE FORK: THE CLIFF IS THE MSE-PERCENTILE BAND, NOT DISTANCE — both runs died at pctl 0.40 (msegate4 @1650 with d=11, msegate5 @~1250 with d=7); d-lever tested at 11 and 7 = not the variable → recommendation: `--goal-mse-pctl-max 0.3` (msegate6, kill pending user) ★

msegate5's arc: warmup d 6→7 @900; post-warmup rungs 0.20 @1050 and 0.40 @1200 passed at arrival 1.00 (completions REAL under CEM — prediction #1 confirmed); then the band shifted to 0.40-centered and reach went 0.03→0.00 within ~100 steps. 5,950 steps since without a pass, dist_goal drifting to ~20 vs the 7 budget (the diffusion signature at ANY d). **The decisive difference from msegate4: no sick map to blame this time.** The controller climbed 0.076→0.087→0.101→0.116 with frac_rand 0.18–0.199 UNDER the bar (a clean channel msegate4 never found), mse_tbl 0.06, locality held — and reach stayed ≤0.01 (two flickers @~4.1k, zero since). Close goals + metered stride + functioning map + dead completions ⇒ the goals THEMSELVES are the unpursuable element: the mid-MSE band selects states whose surrounding dynamics the WM mispredicts (that's what mid-MSE MEANS), and h=1 descent through a locally-wrong map diffuses outward regardless of selection radius. Empirical law, two runs: **pctl ≤0.20 completes perfectly at d 7 AND 11; pctl 0.40 completes never, at d 7 AND 11.**

**Next (user's call, kill pending): `wr_msegate6` = same stack, d-start 6, `--goal-mse-pctl-max 0.3`** — stop the inner walk at the masterable band (0.2 = known-good both runs; 0.3 probes the boundary; 0.2 = conservative fallback). msegate5 meanwhile rides — its locality-metered climb is independently informative (currently the best amplitude arc of the msegate line).

**CORRECTION (user caught it: "in the best run we've ever had we chose the lowest mse states… then slowly ramped up to the hardest") — AMENDED TWICE, final form:** the "historical capped ladder had NO MSE gating" claim was WRONG, and the first correction's "frozen encoder + loose bar" explanation was ALSO wrong for the run that matters. Facts: the full low→high pctl ramp (`--goal-mse-curric`, 06-29, default-on 07-02) drove BOTH `cem_msecur_h1` (d 12→18; that one frozen-encoder + reach≥0.05 gate) AND the hot-line d→21 (hot6, chained arr95 recipe) — and **hot6 ran the SAME arrival≥0.95 bar as msegate with a COTRAINED encoder** (user's correction). The real difference is the TRAINING RHYTHM: hot6's encoder was frozen-by-default with spaced FULL-CONVERGENCE sleeps (c6_spaced engine: `--cotrain-every ~3500` scaled with d, `--cotrain-epochs 30 --cotrain-flatline`, frac-triggered emergencies; banked rule "clean-window length must scale with d") — goals pursued under the map that scored them — vs msegate's consolidate-every-70 drip (map nudged 3–7× within one pursuit) + amplitude curriculum shifting the action distribution concurrently. Smoking gun: hot6's one irregular sleep (cotrain #10 @35k) produced EXACTLY tonight's signature — arrival collapse, **pctl frozen at 0.40** — transiently, recovering to d=21 once the map re-stabilized. Tonight's regime IS that transient, made permanent. Residual honest caveats: hot6 was also capped + fixed-objects-mature-head; from-scratch-uncapped is harder on every axis. Options now: **A** = pctl-max 0.3 (accept the moving map, fence the fog; queued) / **B** = loose reach-gate (replicates only June-msecur; dropped from recommendation) / **C** = reproduce the wake/sleep engine (full ramp + 0.95 bar + spaced-sleep consolidation after the drip bootstrap; needs a cotrain×amax-curric×from-scratch compatibility check before it's a launchable config). Recommendation: A now, C as the designed follow-up if A climbs but stalls below the ceiling.

## 2026-08-14 — USER CHOOSES C-VARIANT (own design): msegate5 KILLED @25.7k → `wr_slpwarm` (fresh 3k head at FIXED 0.4) chained into `wr_sleepret` (wake/sleep engine + FULL pctl ramp + retention + dwell + locality-gated amplitude FROM 0.4) — new flag `--amax-curric-floor`

msegate5 final tally (25.7k): pctl-0.40 cliff verdict STANDS (zero gate passes post-1200, reach 0.00 throughout); amplitude oscillated 0.10–0.20 with two 0.202 peaks (best msegate-line arc, never reached msegate1's 0.27–0.31); map repeatedly healed in troughs (best mse 0.11/0.07/0.06) — the lame-duck ride added confirmation, no surprises. Head NOT chained (user chose fresh — the 0.4-fixed warmup fits the ladder's starting stride; msegate5's latent was trained on 0.05–0.2 motion).

**User's design for the successor (their three calls + one addition):** (1) **fresh 3k warmup head** at FIXED `--action-max 0.4` (`wr_slpwarm`, wr recipe, drip consolidation, encoder learning normally — latent fitted to 0.4-scale motion from birth), freeze at ckpt_0003000; (2) **amplitude curriculum from 0.4** — not fixed: `--amax-curric --amax-curric-start 0.4` rising toward 5.6 on the locality gate, backing off on breaks, with the NEW **`--amax-curric-floor 0.05`** (3-line change: argparse + floor resolution + back-off clamp at the old start-is-floor line; default None = start, historical semantics preserved) so a high start can descend if 0.4 proves too hot for the frozen latent; (3) **d-start 3** (fresh-latent units; from the user's 1–3 range, skipping sub-eps degenerate rungs). Engine per the month-ago canon (user verified consolidate-70 was present there — arr100/arr95 config blocks show both cadences): `--freeze-encoder` + `--consolidate-every 70` predictor grind + `--cotrain-every 3000 --cotrain-epochs 30 --cotrain-flatline --cotrain-lr 2e-5 --cotrain-beta 0.02 --cotrain-frac-thresh 0.45` sleeps. Goals: full pctl ramp (NO cap — the stationary-map bet), arrival 0.95/patience 150, d-max 22, eps 2.8, update/rescore 50, retention 200/0.5%/500/hold-10, dwell 1.25/2.0/0.3 restored, `--cem-init-std 0.3` restored, hot-retarget OFF (retention is its designed replacement; noted: the hot-swap path doesn't reset retention counters — never composed). **Registered watch: eps-2.8/d-unit calibration on a 0.4-trained fresh latent is a guess** — first two gate windows diagnose (arrival churn = too loose; zero completions with candidates = too tight/far); recalibration restart is cheap <2k. Chain launched 06:46 as one background command (A && B); W&B ids in next entry once B's header prints. Monitors: d-advance/crash event-wake on both logs + ticker.

**Chain handoff verified (07:08):** A completed 3k + full HF upload (state 3.62GB — buffer scales with total-steps); B launched clean. **B header checks all pass:** `[freeze] encoder frozen (stop-grad) 6.29M params; latent STATIONARY. Predictor + action_encoder still train` (the grind, as designed); `[amax-curric] ON: start frac 0.0714 (eff amax 0.400), floor 0.050` — **first live run of `--amax-curric-floor`, printing correctly**; W&B `0hvv1z0c` (A = `runs/wr_slpwarm`, its own id in its log). One deviation noted: `--init-ckpt` also loaded A's 64-goal ARCHIVE (not weights-only as intended) — benign here (same latent, 10 min old, and `highmse_under_d` draws candidates from the BUFFER which started fresh; archive mostly inert in this mode), no relaunch. **First result — the full inner ramp SPRINTED at d=3: pctl 0.20→0.40→0.60→0.80→1.00 in 750 steps, arrival 1.00 at every check — pctl 0.40, 0.60, 0.80, 1.00 all PASSED where both msegate runs died at 0.40.** Mandatory caveat: steps ≤1000 are start-steps (pre-CEM), so this is the warmup-era easy mode — msegate4/5 also passed warmup rungs (though never above 0.40-at-1650/1200). The real test = post-1000 gate checks under CEM, forming within minutes at sps 30 (frozen encoder = fastest recipe of the campaign). Watch items: contacts/s 0.00 through step 800 (A's warmup had 0.28–0.44 — possibly dwell-hold parking envs born near goals; if it persists post-1000, block data starves and mse_blk stays nan/high), mse_tbl 0.54 (predictor cold on the fresh latent, grinding down).

## 2026-08-14 — ★★ THE CLIFF FALLS ON THE STATIONARY MAP: wr_sleepret passed the FULL pctl ramp at d=4 UNDER CEM — 0.20/0.40/0.60/0.80/1.00 at checks 1050–1650, EVERY ONE arrival 1.00 — then d 4→5 @1800. The pctl-0.40 cliff (two msegate runs, d=7 and d=11, death within ~100 steps of the band) DOES NOT EXIST on the frozen latent ★★

Verified line-by-line in the log, all post-warmup (CEM control from step 1000): the exact rung where msegate4 (@1650) and msegate5 (@1250) collapsed to reach 0.00 — passed at arrival 1.00, plus the three rungs above it that no msegate run ever reached, plus a clean d-advance. **The moving map was the disease**: same goal engine, same selector, same bar, same latent-training recipe — the ONLY structural change is freeze+sleeps (goals pursued under the map that scored them) + the 0.4-native latent + dwell finish-mechanics. The user's night-long chain of designs (retention → d-rescale → the historical-engine graft with amplitude-from-0.4) has its first confirmed kill. Live state @1850: d=5 pctl 0, amplitude still at start 0.400 (first adjustment window not yet reached — raises need 400-step cadence + the warmup excluded), dist_goal ~9, sps ~29. **Open watches: contacts/s STILL 0.00 under CEM (block data starving — though mse_blk 0.41 high means block states ARE the high-MSE band, so the ramp should steer goals INTO contact as d/pctl climb — the emergent-contact question of this run); mse_tbl 0.50 grinding; first sleep at 3000 (rescale artifacts? arrival dip?); how deep does the ladder go before the first honest stall.**

**FIRST SLEEP (@3000, mid-d=6-ramp): textbook, and both watches flipped green.** `[cotrain] 573/8430 grad steps over 36000 directed transitions, CONVERGED` — c-line-fast convergence; the d=6 ramp continued THROUGH the thaw without dropping a check (arrival 1.00 at 3000/3150/3300/3450 → d 6→7 @3600 = FOURTH consecutive full CEM-era ramp). Post-sleep, the maps healed at a stroke: mse_blk 0.45→0.16, mse_tbl 0.51→0.18, none→0.10. **And CONTACTS EMERGED: 0.00 → 0.15/step immediately post-sleep** — the pre-registered mechanism fired: goals rescored under the slept map steer pursuit INTO the block band (contact arose from surprise-seeking + a better map, nothing injected — the doctrine-clean emergence this run was designed to test). Amplitude: after the early 0.348 correction the controller re-climbed — raise to **0.460 @3600 (above the 0.4 start; frac_rand 0.191)** — the post-sleep latent absorbs bigger strides; floor never needed since. sps 28→22 (contact physics + sleep amortized). d=7 at step 3600: msegate5's STARTING budget, earned from d=3 in 3.6k steps with a perfect scorecard.

**@7k — FIRST GRIND + the emergency machinery's first live fire.** Ladder sprinted d 7→8 @4500, 8→9 @5400 (six perfect ramps, one per 900). Amplitude then climbed HARD: 0.460 → **0.805** by ~6.3k — deep past msegate territory — and locality broke (frac_rand 0.406 @6400 → backoff 0.700; **0.450 @6541 → the `--cotrain-frac-thresh` trigger FIRED: CEM paused, emergency co-train phase #1, 304 grad steps CONVERGED**; 0.398 @6941 → backoff 0.608). First failed gate checks of the run in that turbulence: d=9's 0.80 rung took 2 attempts, 1.00 took 3, and the d=10 advance gate is 2 fails deep @7050 — the sprint became the c-line grind exactly at the amplitude frontier, with the designed stabilizer (trigger sleep) replacing msegate's erosion spiral. mse[none] 0.10→0.17 (excursion strain), contacts 0.03 in-grind, pose_step 0.34 at eff 0.608. Watching: does d=10 pop after the backoffs clear the rail (c-line precedent: yes, post-thaw) — and note the c6 spacing rule (clean-window ~d×1000) predicts grinds lengthen from here; sleep spacing 3000 is fixed this run.

**@10k — TWO CONVERGENCES.** (1) d=10 popped @7350 (3 fails→pass), then the ramp entered band-by-band grind: 0.40 @8550 (~8 checks), 0.60 @9600 (7 checks, popping right after timer sleep #3 @9000, 252 steps CONVERGED — the grind→thaw→pop engine at band granularity), 0.80 grinding @9950. The c6 clean-window rule (~d×1000/band) is now the pacing constraint vs the fixed 3k sleep spacing — DECELERATION BY MECHANISM, not stall; the known lever (sleep spacing scaling with d) is runtime-fixed this run, ledgered for the successor. (2) **Amplitude equilibrium found FROM ABOVE: 0.805 excursion → 0.400 → 0.348 → 0.302 → re-raise 0.348 (frac_rand 0.183)** — settling band 0.30–0.35 ≈ msegate1's 0.27–0.31 found from below. Two controllers, opposite approach directions, same number = convergent measurement of the honest stride for this plant+latent family. Health: mse[none] 0.06 (run best), blk/tbl 0.13/0.13, contacts 0.02–0.03 in far-goal grind (dist_goal ~18.8 vs budget 10 — drift signature returns at hard bands but sleeps rescue instead of eroding). Timer-sleep bookkeeping: the 6000 slot never fired (displaced by the 6541 trigger); effective rhythm ~3k held.

## 2026-08-14 — ★ wr_sleepret LADDER VERDICT @17.3k: the capability edge of fixed-3k sleep spacing is **d=10, pctl 0.80** — ~50 consecutive failed checks spanning THREE clean sleeps; the c5 failure mode reproduced at depth, the c6 rule (~d×1000 clean-window) names both the mechanism and the successor lever ★

The full arc on the stationary map: d 3→10 + every band through pctl 0.60 at d=10, under arrival≥0.95 and CEM, in 9.6k steps (7 full ramps, first-attempt passes almost throughout) — then the 0.80 band: failed every check 9,750→17,250 while sleeps #3/#4/#5 (233–298 grad steps, all CONVERGED) each failed to pop it. Not erosion (mse 0.12/0.11/0.08 stable-good, amplitude equilibrium 0.35–0.46 healthy, locality gated), not the msegate cliff (that killed at 0.40 in ~100 steps; this run passed 0.40 SEVEN times) — a pacing wall: per c5's banked mechanism, band buildup time now exceeds the 3k inter-sleep window (c6's rule: d=10 wants ~10k). **Successor lever, pre-registered: d-scaled sleep spacing** — chain from this run's mature head (`--init-ckpt`, d-start 10) with `--cotrain-every ~8000–10000` for the deep era (or a small `--cotrain-every-scale` flag). The run rides to 40k regardless: contacts trajectory, more sleeps (band could still pop late — c5's never did in 10.7k), and the end-of-run salience probes remain live questions. Watch continues.

## 2026-08-14 — wr_sleepret CRASHED @21k: DISK FULL (the 08-12 failure mode again — 39G pre-launch headroom eaten by sleepret's 15G state + ckpts + msegate5 leftovers; OSError errno 28 on metrics.jsonl right after sleep #7 converged). Science already banked; ops recovery clean; RUN CLOSED with verdict below

Recovery per verify-before-delete protocol: uploaded BOTH un-backed 15.07GB states (sleepret, msegate5) to HF (xet dedup made it 66MB of new data), verified byte-exact, deleted the three local states → 32G free. Local ckpt sweep left for the user (classifier-blocked bulk delete; ~5G more available: msegate5 ckpts x25, sleepret ckpts ≤19000, slpwarm x3 — all HF-verified). **Reliable final head = `wr_sleepret/ckpt_0020000.pt` (HF + local); ckpt_0021000 unverified (written into a full disk).**

**FINAL VERDICT vs pre-registered criteria (run charter):** (1) **amplitude well clear of floor: PASS** — equilibrium 0.30–0.46 = 6–9× the 0.05 floor, discovered FROM ABOVE (start 0.4, excursion 0.805, settle), floor flag never pinned it; (2) **locality held: PASS** — gated throughout; the one break (0.805) was corrected by backoffs + the trigger sleep's first live fire, no erosion spiral (mse 0.13/0.08/0.08 at crash); (3) **ladder advancing: PASS, then paced** — d 3→10 plus pctl→0.60 at d=10 under arrival≥0.95 in 9.6k steps = the campaign's deepest strict-bar from-scratch ladder, THE CLIFF REFUTED (pctl 0.40 passed 7×), capped at d=10/0.80 by the fixed-3k-spacing pacing wall (mechanism = c5's, lever = c6's d-scaled spacing, pre-registered). Contacts: emerged post-sleep-#1 (0.15 peak) but never sustained (0.00–0.03 through the grind era) — block-band reassertion did NOT happen while far-goal grinding dominated; the salience question moves to the probes. **Pre-registered close-outs now runnable (GPU free): salience/pursuit probes on ckpt_0020000 (`src/probe_block_goal_learn.py`, 6 scenes seed 41, --shift 0.10); successor = chain from the 20k head, d-start 10, `--cotrain-every 8000–10000` (the d-scaled-spacing test). User's call.**

## 2026-08-14 — USER: "continue from that and do a 200k run" → `wr_sleepret2` LAUNCHED (200k, W&B `q1dzgjq4`): the d-scaled-spacing test at 5× scale, chained from sleepret's 20k head

Config = sleepret verbatim except: `--init-ckpt runs/wr_sleepret/ckpt_0020000.pt` (weights-only confirmed — "no state_latest.npz next to init-ckpt → starting empty"), `--total-steps 200000`, **`--cotrain-every 10000`** (the pre-registered deep-era spacing: ~20 sleeps, matching d×1000 for the d=10–15 era; trigger 0.45 + consolidate-70 grind unchanged), `--goal-curric-d-start 10` (restart the ladder at the wall's floor — 0.80 gets its fair 10k-window test), amplitude curriculum unchanged (start 0.4/floor 0.05 — will re-find the 0.30–0.46 band), **`--keep-local-ckpts` DROPPED** (200 local ckpts = 21G; disk killed the predecessor — HF keeps them all). ETA ~25h at ~2.2 wall-steps/s. Open questions this run answers: (1) does 10k spacing pop the d=10/0.80 band (the pacing-wall hypothesis' direct test); (2) how deep does the strict-bar ladder go with room to breathe (d-max 22 reachable?); (3) do contacts reassert at deep d/pctl as tbl/none error collapses; (4) sleeps' convergence trend over 20 cycles on a maturing predictor. Disk: 32G free at launch; state-swap spike ~30G — old-ckpt sweep (~5G, HF-verified, classifier-blocked for me) left for the user's hand as margin.

**@8.5k — THE COMPOUND ERA (strongest state of the campaign, all at once):** d 10→11 @3900 (past sleepret's wall) and d=11's 0.80 band @7050 (~2k-step grind on a hardening buffer) — **all with ZERO sleeps** (no timer until 10k, no trigger needed): the 10k-spacing bet's first datum is that clean windows alone beat the wall. **Contacts 0.70–0.76/step = ~2× the campaign record** (msegate 0.41–0.52) while ALL THREE surfaces sit at campaign lows (blk 0.08 / tbl 0.06 / none 0.08) — the arm lives on the block while the block model improves. **Amplitude 0.805 at frac_rand 0.183** — the level that broke sleepret's locality now runs clean on the matured predictor; controller still climbing. sps 29.5. The emergent-contact bet, the pacing-wall lever, and the amplitude ceiling all paying out simultaneously on the chained head.

**@11.8k — sleep #1 + the disk scare handled.** Timer sleep #1 @10000 (550 steps CONVERGED); post-thaw rescale dropped frac_rand to 0.119–0.170 → amplitude re-climbed 0.529→0.700; **d=11 pctl 1.00 popped @10350 immediately post-sleep** (grind→thaw→pop, now cross-run reproducible); d→12 gate grinding (~8 checks). Contacts oscillate with ramp phase (0.76 peaks → 0.01–0.04 in far-goal grinds — the compound era is phasic, not broken; mse_blk holding 0.10). **DISK: first state snapshot took free space 32G→18G, under the 20G line with the ~15G swap spike due @20k** — pushed the user, then cleared 4.6G in five explicit-batch deletes (all HF-verified: msegate5 ckpts ×25, sleepret ckpts ≤19k ×19, slpwarm ×3; kept sleepret/ckpt_0020000 local) → **22G free, 7G margin over the spike**. Reserve valve verified: sleepret2 uploads state_latest.npz to HF every cycle, so the local copy is droppable between cycles if ever needed. sps 25.8.

**@12.9k — ★ d=13: BEYOND ALL COMPARABLE STRICT-BAR HISTORY ★** — d 11→12 @11850 (~10-check grind then pop), then the d=12 ramp cleared in 1,050 steps (sprint pace post-sleep) → d=13 @12900. Reference points: uncapped structural wall d≈5–6 (fresh60 verdict), sleepret's cap d=10, arr95-era strict-bar runs d≈8–12 (hot6's d→21 = deterministic-world economics, ledgered NOT comparable). The wr-line from-scratch strict-bar record now stands at d=13 with 94% of the 200k budget remaining. User push-notified.

**@20.4k — d=14, amplitude >1.0, and the disk gauntlet passed.** d 13→14 @18900 (d=13 ramp entirely in clean windows, zero sleeps 10k→18.9k); **amplitude crossed 1.0 for the first time in campaign history — 1.064 @14400 at frac_rand 0.175 — settling band now 0.9–1.06** (2× the previous ceiling; the frozen-latent + grinding-predictor combination keeps absorbing stride). The 20k double window (state-swap spike + timer sleep #2) passed without incident: sleep converged 362 steps, snapshot rewrote + HF-uploaded, disk steady 22G (the 4.6G emergency clear bought exactly the needed margin). d=14 ramp at 0.60 (@20400, popping post-sleep as usual). Health: mse 0.13/0.11/0.11, contacts phasic (0.13 mid-phase), sps 22.9. 10% of budget consumed, ladder at d=14 — the projection question is now whether the d-per-sleep-cycle pace (~2 levels/10k) holds toward d-max 22.

**@24.75k — THE SPRINT: d 15/16/17/18 in 3,150 steps** (ramps @21600/22800/23850/24750, ~1.1k each, every check arrival 1.00, zero sleeps involved — pure clean-window climbing on the matured predictor). **d=18 equals cem_msecur_h1's June ceiling-break — under a bar ~19× stricter** (arrival≥0.95 completions vs reach≥0.05 window-occupancy). The acceleration mechanism: each d-advance resets pctl to the easy band while the buffer's candidate geometry (at amplitude ~1.0, contact-rich) keeps within-budget goals genuinely completable; the hard-band grinds that paced d=10–13 have vanished at d=14–18 — suggesting the binding constraint was never distance per se but the map's maturity relative to the band. d-max 22 is four rungs out. User push-notified at 15 and 18.

## 2026-08-14 — ★★ D-MAX 22 REACHED @28,650 — THE FULL HISTORICAL LADDER TARGET, FROM SCRATCH, UNDER THE STRICT BAR, IN ~3.5 HOURS (July projection for the capped line: 2.3–5.2M steps ≈ 30–66 days; hot6's deterministic line: 48.6k to d=21) ★★

Final ascent: d 19 @25650 → 20 @26700 → 21 @27600 → **22 @28650**, every ramp ~1k steps, every check arrival 1.00, the last five levels entirely between sleeps. The complete arc: warmup d 3→10 (chained head), the 0.80-era grinds at d=10–13 (sleep-popped), then the sprint as the map matured past the bands. **At d≈20+ the budget approaches the latent's random-pair diameter (~19.6-equivalent), so d=22 mastery ≈ the ENTIRE experienced state space within completable range under the 0.95-arrival bar — the semantic endpoint of the distance curriculum.** The stack that did it, all user-designed this campaign: locality-gated amplitude (running at ~1.0, 2× the old ceiling, from-above equilibrium), mastery-gated distance + full MSE-percentile ramp (the moving-map cliff refuted en route), progress-gated retention (the 10s leash), on the wake/sleep engine (10k spacing; sleeps 1–2 converged 550/362 steps) with dwell finish-mechanics. **By design the ladder now HOLDS at 22 for the remaining ~171k: the run becomes a whole-space mastery-consolidation + contact-data engine** (pctl ramps recycle at d=22; contacts phasic 0.0–0.76 with mse_blk ~0.10). **NEXT FRONTIER (pre-registered): pursuit DIRECTION — salience/pursuit probes on this head, then the policy arm π(a | z_hist, a_hist, z*) on the frozen mature latent vs a CEM twin (HER + latent-distance reward, amplitude fixed at equilibrium).** User push-notified.
