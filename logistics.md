# Curious Robot — Logistics

JEPA + SIGReg world model co-trained with SAC under an intrinsic curiosity reward, on a
MuJoCo SO-ARM101 6-DOF arm in an object soup. **Built from scratch to the `README.md`
spec** (README = the formulation; this file = the working log). The question every run
answers: *does the arm move around and act curious like a baby, and is it learning
world dynamics?*

- **Code** — `lewm/` (verbatim JEPA+SIGReg), `env/` (MuJoCo SO101 + safety reward),
  `model/` (ViT-tiny state encoder), `src/` (train · play_policy · eval_predictor).
- **Backends** — W&B project `curious-robot` · HF `a5ilank/curious-robot` · GitHub
  `ahilanks/curious-robot` (private). Creds in `.env` (auto-loaded; gitignored — keep local).
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
- [x] Pin the swept `?` values into `README.md` — **done 2026-05-31 (maintainer-approved):**
      δ=15, λ_safe=0.1, τ_max=3.35, h_fwd_max=1, r_cur→per-dim-mean. (λ_cur still `?` — at 20, unswept.)

## What to read off the logs (W&B `curious-robot`)

- **Interacts with blocks** — `interact/contacts_per_step`, `interact/object_motion`,
  `interact/frac_touch_block` rising over training.
- **Curious / exploring** — `reward/r_cur` non-trivial; rollout videos show varied reaching.
- **Roaming vs dithering in place** — `explore/prox_step` (proximal-joint travel; a wrist wiggle
  CAN'T fake it) and `explore/ee_step` (gripper world-xyz travel) > 0; `explore/pose_range`/`ee_range`
  covering space. **`interact/arm_speed` (|qvel|) is a TRAP** — it stays HIGH for a saturated in-place
  limit cycle (measured: collapsed policy `arm_speed`≈1.3 but `pose_step` 0.23 < random's 0.89). Use the
  `explore/*` travel metrics, never velocity, to tell roaming from dithering.
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
- 2026-06-16 — **entropy (search) is the missing ingredient, not intrinsic-reward magnitude or shape:
  α=0 collapses regardless of curiosity weight OR an added RND novelty bonus.** Tested the hypothesis
  that removing the SAC entropy term (α=0) and compensating with a stronger / better-shaped intrinsic
  reward keeps the arm exploring and the encoder healthy. It does not.
  - **Infra / perf** (`src/train.py`, `env/`): (1) **render-skip** — `step_block` keeps only the final
    substep's obs but rendered all `action_block=5` wrist frames; skipping the 4 discarded renders is a
    *pure* speedup (render is read-only → trajectory bit-identical) → **dec-steps/s 2.7→4.7 (~1.8×)** single-run.
    (2) New collapse/stall diagnostics: `interact/arm_speed` (mean |q̇|, direct stall signal),
    `policy/entropy` + `policy/action_absmean` (policy-collapse signal), `wm/pred_over_identity`.
    (3) α=0 supported cleanly (`log_alpha=-inf`, entropy terms vanish, no NaN). (4) New composable intrinsic
    rewards: `--lambda-rnd` (RND) and `--lambda-knn` (k-NN state-entropy coverage), each added iff weight≠0.
    (5) 5-way concurrency OOMs the 80 GB A100 at the step-1000 WM-engagement (5×~20 GB ViT-backward peaks)
    → run with `--wm-grad-checkpoint` (~4× lower peak, numerically identical grads).
  - **Sweep 1 — α=0 × λ_cur ∈ {20,50,100,200}** (+ α=0.2 control; λ_safe=0.1, β=0.3, seed=0; W&B names
    `a0_lcur{20,50,100,200}`, `a02_lcur20`). Identical seeded warmup ⇒ identical buffer/encoder at step 1000,
    diverging *only* by α/λ_cur. **Monotonically null in λ_cur.** Every α=0 run collapses identically: policy
    → saturated bang-bang (`a|·|`=1.00, entropy ≈ −500), `eff_rank_probe` pinned **1.5–3.6**, contacts → 0 — at
    λ_cur=20 *and* 200. Only the α=0.2 control stays healthy (eff_rank ≈ 6, entropy ≈ +5, `a|·|`=0.62, contacts
    ≈0.035). **Smoking gun:** under α=0 `r_cur` *falls* (0.26→0.10) while under α=0.2 it *rises* (0.26→0.58) — the
    deterministic policy starves its own curiosity (the WM learns its narrow behaviour), the stochastic one keeps
    generating surprise. ⇒ λ_cur is a reward *scale* (scale-invariant on the optimal policy once curiosity
    dominates safety), not an exploration knob.
  - **Sweep 2 — α=0 + curiosity(λ_cur=20) + RND**, λ_rnd ∈ {5,10,20,40} (W&B `rnd_l{5,10,20,40}`; baseline
    `a0_lcur20`=λ_rnd 0). RND = frozen random target + chasing predictor on the latent z, predictor trained in the
    SAC loop, reward mean-normalised (raw RND err ~0.005 is ~1000× below cur_contrib → normalise to ~O(λ_rnd)).
    **RND does not rescue α=0.** *Transient* effect only: at step ~1200 the policy is markedly less saturated
    (entropy −67/−95, `a|·|` 0.82/0.85 for λ5/λ20 vs baseline −450/→1.0), but by step ~3000–3500 **all** RND runs
    fall into the same basin — eff_rank 2.1–3.5, `a|·|` 0.94–1.00, entropy −301…−515, contacts ≤0.03 — far below the
    α=0.2 control. λ_rnd=20 is the *least* collapsed (entropy −301, `a|·|` 0.94, contacts 0.028) but still collapses.
    **Two reinforcing causes:** (a) RND on the *co-trained* latent z is coupled to the very encoder that's collapsing
    — as eff_rank(z)→2 the RND distances vanish, so RND can't prevent the collapse it depends on; (b) α=0 removes the
    *stochastic search* — a deterministic policy commits to one trajectory, the RND predictor learns it, the bonus
    drops, and with no entropy the policy can't escape.

  | run | α | λ_cur | λ_rnd | eff_rank_p | entropy | act_abs | contacts/s | verdict |
  |-----|---|-------|-------|-----------|---------|---------|-----------|---------|
  | a02_lcur20 | 0.2 | 20 | 0 | ~6 | +5 | 0.62 | 0.035 | **healthy** |
  | a0_lcur20 | 0 | 20 | 0 | 2.9 | −512 | 1.00 | 0.000 | collapsed |
  | a0_lcur200 | 0 | 200 | 0 | 3.6 | −521 | 1.00 | 0.000 | collapsed |
  | rnd_l5 | 0 | 20 | 5 | 2.1 | −515 | 1.00 | 0.002 | collapsed |
  | rnd_l10 | 0 | 20 | 10 | 2.3 | −435 | 1.00 | 0.008 | collapsed |
  | rnd_l20 | 0 | 20 | 20 | 2.4 | −301 | 0.94 | 0.028 | collapsed (least) |
  | rnd_l40 | 0 | 20 | 40 | 3.5 | −492 | 1.00 | 0.000 | collapsed |

  **Synthesis:** the missing ingredient for exploration-without-stalling is *search* (policy entropy), not the
  *magnitude* (λ_cur) or *shape* (RND) of the intrinsic reward. Curiosity and RND are *rewards*; SAC's exploration
  is the entropy term. Open follow-ups: (i) RND on a *fixed* encoding (raw obs / frozen random encoder) to decouple
  novelty from the collapsing z and isolate cause (a) from (b); (ii) RND/coverage as a bonus in the *working* α>0
  regime to attack the standing explore-toward-blocks problem; (iii) a small target-entropy floor instead of α=0.
- 2026-06-16 (cont.) — **isolating "search, not the entropy reward" + the collapse is a stable attractor.**
  Three follow-ups confirmed the synthesis. New code: `--actor-logstd-min` (policy-std floor), `--explore-noise`
  (post-tanh DDPG/TD3 action noise), both at α=0.
  - **std-floor (α=0, log_std≥−1/0):** *defeated by tanh mean-saturation* — flooring the std bounds the std but
    the greedy α=0 mean still drives to the ±1 rails, where a floored std produces ~no action variation. Re-collapsed
    by ~2.5k (entropy −370…−400, `a|·|`→1.0, eff_rank 2–4). Don't floor std; inject in action space.
  - **action-noise (α=0 + curiosity, ε ∈ {0.3,0.5,0.8}):** *saturation-proof and it works at the right level —
    NON-MONOTONIC.* ε=0.5 held eff_rank ≈5 (≈ the α=0.2 control) through 3.4k with the arm moving — **the closest
    anything came to meeting the 3 criteria WITHOUT the entropy term** (policy still saturated, but the noise keeps
    the *data*/encoder diverse). ε=0.3 too weak (collapsed, arm_speed→0.76 = stalling); ε=0.8 dipped (eff_rank 2.4,
    over-randomised). ⇒ a *sufficient* external search source substitutes for the entropy term — confirming search,
    not the entropy reward, is the necessary ingredient — but it's finicky (a tuned band, unlike the self-tuning α).
  - **`rnd_l20_nosafe` (pure curiosity+RND, λ_safe=0, α=0) → 20k:** *the α=0 collapse is a STABLE ATTRACTOR, not a
    transient.* Run ~6× past first collapse, no recovery: entropy pinned −510 the entire run, `a|·|`=1.0, eff_rank
    wobbled 2.0–4.3 and **declined to 2.0 by 19.5k** (lowest of the run). Once the policy saturates there is no
    un-saturating force in the reward, so recovery is mechanically impossible. (`logstd_min`/`explore-noise`/`lambda-rnd`
    are the new knobs; std-floor noted as a trap in `--help`.)
  - **Anatomy of prediction error under collapse (read off `rnd_l20_nosafe`):** (1) absolute WM error `r_cur` and the
    persistence baseline *both rise* in lockstep (latent-scale artifact — collapsed encoder inflates the magnitude of
    its few active dims, `z_std` 0.29→0.43), so the scale-invariant **`pred/per` ratio is FLAT ≈0.55–0.6 the whole
    run** — the WM nails the degenerate behaviour to a fixed relative accuracy and plateaus. ⇒ **"prediction improving"
    is a TRAP health metric**: low `pred/per` means the behaviour is trivially predictable (collapsed), not that the
    model is healthy. (2) The **RND predictor error → ~0 almost immediately** and stays there: the predictor learns the
    narrow collapsed state-distribution perfectly, so its novelty signal vanishes — *the coverage signal dies with the
    coverage*, the mechanistic reason RND can't self-rescue on the co-trained latent. The honest health signals remain
    `encoder/eff_rank_probe` (+`z_std`/`feat_corr`) and `policy/entropy`/`interact/arm_speed` — NOT `pred/per`.

- 2026-06-17 — **obs-RND + the freeze is in-place DITHERING, not stillness (measured); curiosity AND arm_speed are both
  gameable.** Moved RND off the co-trained latent z onto the RAW obs (84×84 grayscale wrist + per-dim-normed proprio;
  obs-norm `RunningMeanStd`, Atari conv + proprio MLP — `RNDObsNet`). New knobs: `--rnd-reward-scale` (×200 lands the
  ~5e-3 obs-RND error in log1p's active range so λ_rnd≈20 gives an O(10) bonus, **honest — no EMA divide**), err/MSE-weighted
  predictor loss (`--rnd-loss-clip`, per-sample novelty weighting that survives Adam), and `--rnd-train-every` + lower
  `--rnd-lr` to slow the predictor so novelty persists.
  - **obs-RND is honest but still collapses at α=0.** Unlike latent-RND (error→0 as z collapses, yet the old EMA-divide
    faked a steady `rnd_contrib`≈13.6 — a *phantom*; see `a0_sf_rnd`), obs-RND's `rnd_contrib` decays *honestly* to ~0.5
    (raw obs can't collapse; the predictor just learns the narrow visited set). `a0_obsrnd20` (fast) collapsed by ~1250;
    `a0_obsrnd_slow` (lr 5e-6, train-every-4) **sustained** novelty (`rnd_contrib`≈14 through 1250, eff_rank_probe 6.7 vs
    the fast run's 1.9) yet the policy still saturated — **entropy −477 while novelty was still 13.2.** A 3× novelty bonus
    cannot stop the saturation: confirms the α=0 stable-attractor finding with the encoder-collapse confound removed.
  - **The freeze is a saturated in-place LIMIT CYCLE — measured by rolling out the collapsed ckpt** (random vs ckpt_1000
    vs ckpt_3000, 100 decisions): collapsed policy `act|·|`=1.00 (bang-bang), `arm_speed`(|qvel|)=1.27, **yet `pose_step`
    (joint travel/decision)=0.23 vs random's 0.89.** The LEARNED policy travels ~4× LESS than random noise while commanding
    2× the action — it thrashes in place. ⇒ **`arm_speed` is a TRAP metric** (high |qvel|, ~0 net travel).
  - **Why it parks (NOT choosing stillness):** delta-target PD (`target=clip(q+a·0.3)`) turns a constant saturated action
    into oscillation; **curiosity (`cur_contrib`≈4 even when frozen) is harvested from the WM failing to predict its own
    bang-bang jitter — prediction-error needs no travel**, so dither maximizes it. obs-RND would punish a static scene, but
    the wrist cam is ON the arm so wrist rotation pans it (fake visual novelty), and RND gets out-earned by
    curiosity-on-jitter then collapses before it can redirect; the collapsed encoder (eff_rank→1) then removes the actor's
    ability to represent state-dependent travel, locking the cycle.
  - **New DEFAULT metrics** (`explore/*`, distinguish roaming from dithering — `arm_speed` can't): `pose_step` (joint
    travel/decision; ~0=parked), **`prox_step`** (proximal joints 0-2 = gross repositioning a wrist wiggle CAN'T fake — the
    ungameable roam signal), `ee_step` (gripper WORLD-xyz travel/decision), `pose_range`/`pose_spread`/`ee_range`
    (config/world-space coverage of the recent 200-decision window). Env now emits `ee_pos` (gripper world xyz); logged every step.
