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
- **Train** — `python src/train.py --name <kw> --n-envs 8 --env-threads 8`. `--name` is
  a short keyword → W&B run name + `runs/<name>/` + HF `<name>/ckpt_*.pt`; every constant
  lives in the W&B config table, not the name. Pod bootstrap: `bash setup.sh`.
- **Inspect** — `play_policy.py --name <kw>` (overhead+wrist rollout videos) and
  `eval_predictor.py --name <kw>` (open-loop pred vs persistence). Both fetch the latest
  checkpoint from HF by default; pass `--ckpt <path>` for a local file or `--step N`.

## Status — last left off (2026-05-27)

Implementation **complete and validated end-to-end** (smoke run, HF up/download
round-trip, deterministic contact check), pushed to GitHub. **No real training run yet** —
the README `?` constants (β, λ_cur, δ, Kp/Kd) are unswept and stay `?` until then.

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
- [ ] **Sweep the `?` constants** — β (0.9), λ_cur (15 → safety:curiosity ~0.5:1),
      δ (0.05). Watch `reward/safe_cur_ratio`, `interact/contacts_per_step`, pred/persist.
- [ ] **TD-priority ablation** — `--per-priority td` vs `curiosity` (see Ablations).
- [ ] Pin the swept `?` values into `README.md` — **ask the maintainer before editing it.**

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

The README `?` constants are swept here before being pinned.

`pred/persist` = wm/pred_loss ÷ wm/identity_baseline (<1 ⇒ WM beats the persistence baseline).
`norm` column: ✗ = README reward λ_cur·symlog(r_cur); ✓ = `--normalize-curiosity` (running mean/std).

| run | β | λ_cur | norm | δ | steps | contacts/step | pred/persist | feat_corr | notes |
|-----|---|-------|------|---|-------|---------------|--------------|-----------|-------|
| osmesa_2k | 0.9 | 15 | ✗ | 0.05 | 2k | 0.007 | 1.02 | 0.38 | README default reward → collapse + freeze |
| beta5_10k | 5 | 15 | ✗ | 0.05 | 10k | 0.000 | 0.95 | 0.30 | β=5 + longer does **not** help: permanent freeze (contacts→0), WM only learns static obs; eff_rank later limps back on degenerate data |
| norm_b1 | 1 | 15 | ✓ | 0.05 | 5.4k† | 0.124 | 0.96 | 0.29 | de-sat helps but λ=15 marginal — interaction fluctuates near-freeze |
| norm_b5 | 5 | 15 | ✓ | 0.05 | 5.4k† | 0.060 | 1.06 | 0.35 | β=5 no better than β=1 under norm (worse feat_corr/WM) → **β is not the lever** |
| **norm_b1_lc40** | 1 | 40 | ✓ | 0.05 | 10k | **0.10** | **0.90** | **0.22** | **best**: sustained block contact, healthiest rep (eff_rank ~12), WM beats persistence (mse_block 295→225) |
| norm_b1_lc60 | 1 | 60 | ✓ | 0.05 | 10k | 0.00 | 0.92 | 0.24 | healthy rep + WM learning, but ended low-contact — λ=60 gives no extra vigor over λ=40 |
| norm_b1_lc40_d10 | 1 | 40 | ✓ | 0.10 | 10k | 0.04 | 0.95 | 0.28 | wider safety deadband: similar — interaction vigor capped by reward design, not δ |
| norm_lc40_s1 | 1 | 40 | ✓ | 0.05 | →10k‡ | ~0.02–0.10 | 0.98 | 0.27 | **seed=1 reproduces** λ=40: no collapse (eff_rank ~9), active (r_safe −15), WM beats persistence → not seed luck |

‡ seed=1 (all other runs seed=0) — robustness check that the λ=40 result isn't seed luck.

`pred/persist` above is the *in-training* symlog ratio. **Offline** `eval_predictor.py` (raw open-loop
MSE, the logistics success metric) is more decisive: **norm_b1_lc40 = 0.45–0.56 across h=1..10**
(WM error ~half persistence, on a *dynamic* trajectory, persist baseline ~460) vs the frozen control
**beta5_10k = 0.74→0.56** (beats persistence only on its own near-static trajectory, persist baseline ~290).

Interaction is gentle/intermittent (contacts ~0.0–0.1, fluctuating; object_motion stays low), **not**
vigorous play, and is **not reliably increased by λ_cur or δ** — the reward design makes "safe curious
touching" the equilibrium (curiosity is satisfied by any contact surprise; the safety reward penalizes
forceful/jerky pushing). The primary, robust win is **no permanent freeze + no collapse + real
multi-step dynamics learning**, not motion volume.

† stopped ~5.4k once the λ=15 trend was clear, to free the (single) GPU for the λ-sweep.

eff_rank is a noisy single-batch estimate that partially recovers even in the frozen control (SIGReg
re-isotropizes whatever data it gets), so read collapse off **feat_corr + contacts + pred/persist**, not eff_rank alone.

### BatchNorm + UTD + safety-weight (2026-05-28)

All below add **BatchNorm1d** to the encoder projector/fuse/pred_proj (matching le-wm; the `module.MLP`
default is LayerNorm). All use `--normalize-curiosity`, λ_cur=40, β=1 unless noted. eff_rank shown for
the *active* window (it's the metric that gets gamed — see the hollow-eff_rank note).

| run | config | steps | contacts | obj_motion | eff_rank | feat_corr | takeaway |
|-----|--------|-------|----------|-----------|----------|-----------|----------|
| bn_2k | BN, no-norm (symlog), default reward | 2k | ~0 | ~0.002 | ~13 active / ~4 frozen | 0.31–0.38 | BN lifts rank *while exploring* but can't stop collapse once the agent freezes (same data-degeneracy lesson) |
| bn_norm_2k | BN + norm λ40 | 2k | **0.20** | 0.003 | 12 mean / 18 peak | **0.21** | BN + exploration = best 2k rep (vs LayerNorm+norm 10/13) — modest lift |
| bn_norm_10k | BN + norm λ40 + **UTD 4/2** | ~6k* | →**0.002** | →0.001 | 14→**29** | 0.14 | **hollow eff_rank**: rank rose toward le-wm's 50 *as interaction collapsed* — BN+SIGReg+high-UTD inflate the metric without behavior |
| sw_1p0 | BN + norm λ40, safety×**1.0** | ~2.5k* | 0.007 | 0.0017 | 10.1 | 0.24 | A/B control |
| sw_0p3 | BN + norm λ40, safety×**0.3** | ~2.5k* | 0.015 | 0.0019 | 12.5 | 0.22 | lowering r_safe 3.3× → **no interaction gain** (within noise); safety magnitude isn't the lever |

*stopped early once the trend was clear. **Across all of these, vigorous block interaction never
emerged** — it's an exploration/discovery problem, not a reward-balance one (see the 2026-05-28 log).

## Ablations

| ablation | flag | default | hypothesis / what to compare |
|----------|------|---------|------------------------------|
| PER replay priority | `--per-priority {curiosity,td}` | `curiosity` | `td` = \|TD-error\| is sign-agnostic, so it also replays the unsafe (very-negative `r_safe`) transitions the critic mispredicts — which curiosity priority under-samples once the WM has learned them. **Q:** does `td` better suppress motor-fighting states without losing block interaction? Compare `interact/contacts_per_step`, `reward/safe_cur_ratio`, `sac/critic_loss`, `eval_predictor.py`. Curiosity stays the *reward* either way. |

## Log

_(add a dated entry per run)_
- 2026-05-27 — implementation complete + validated end-to-end; no training run yet.
- 2026-05-27 — **collapse diagnosis + curiosity-normalization fix.** First real runs (β-sweep,
  2k) all collapsed: encoder eff_rank → ~2–3, feat_corr ↑, and the agent *froze* (object_motion
  → 0, contacts → 0, r_safe → ~0). Root cause: **curiosity saturation.** r_cur ≈ 250–295 (the
  z_dim=256-driven "WM predicts nothing" floor) sits on symlog's flat tail, so λ_cur·symlog(r_cur)
  is a near-constant ~85 (per-state spread crushed ~30→1.8). SAC then sees no curiosity gradient
  and optimizes the only signal that varies — r_safe — by *not moving* (still arm ⇒ no jerk
  penalty ⇒ r_safe→0). Static obs ⇒ degenerate data ⇒ SIGReg can't hold the representation up.
  β is **not** the lever: a longer β=5 run to 10k (`beta5_10k`) froze permanently and its WM
  learned only static free-space (mse_block frozen/stale, mse_none↓); β=5 collapses fastest of all.
  **Fix:** `--normalize-curiosity` — standardize r_cur by a bias-corrected EMA running mean/std
  instead of symlog (de-saturates: restores per-state spread ~30×; unit-tested). Stays SIGReg-only
  (no EMA target, no IDM); default off = exact README reward. Result: de-saturated curiosity gives
  SAC a real exploration gradient that out-competes the freeze attractor **iff λ_cur is large
  enough**. λ_cur=15 (tuned for the symlog scale) is too weak under normalization (interaction
  fluctuates near-freeze); **λ_cur≈40 (`norm_b1_lc40`, β=1) is the clean win** — full 10k with
  sustained block contact (~0.13 vs control 0), healthiest rep (feat_corr 0.24), WM beating
  persistence (pred/persist 0.92, mse_block 295→225). Curiosity weight λ_cur, not β, is the lever;
  this is also *why it worked for push-t* (no safety-reward freeze attractor / curiosity not
  saturated there). Interaction is gentle/intermittent, not vigorous play — the primary win is
  no-collapse + dynamics-learning. λ-sweep continues (`norm_b1_lc60`). README `?` constants unchanged
  (ask maintainer before pinning).
- 2026-05-28 — **BatchNorm, UTD, δ, safety-weight — chasing eff_rank and interaction.** le-wm's
  encoder projector/pred_proj use **BatchNorm1d** (set in its Hydra config; the shared `module.MLP`
  defaults to LayerNorm, which curious-robot had been using). Added BatchNorm1d to
  visual_head/fuse/pred_proj (README updated, with approval). Findings:
  • **BatchNorm lifts rank only *with* exploration**: under the freeze (no norm) it can't stop
    collapse; with normalization it modestly raises eff_rank (`bn_norm_2k` 12 mean/18 peak vs
    LayerNorm+norm 10/13).
  • **High UTD inflates eff_rank *hollowly*** (`bn_norm_10k`, `--updates-per-step 4 --wm-update-every 2`):
    eff_rank climbed 14→**29** (toward le-wm's ~50) *while block interaction collapsed* (contacts
    0.10→0.002, r_safe −32→−12). BatchNorm+SIGReg+high-UTD spread even low-diversity inputs to satisfy
    SIGReg, so **eff_rank ≠ good behavior** — the 50-vs-12 gap is largely a regularizer artifact. Read
    behavior off contacts/object_motion, not eff_rank.
  • **δ is a dead knob**: measured the `−τ·q̈` distribution (sanity-checked against env r_safe) — the
    hinge fires ~13%/joint-step on *large* events (p90 +25, p95 +79, max 464) and firing is **flat
    from δ=0.05→5**; it'd need δ≈25–100 to move. So δ=0.05 isn't "firing on noise" — the penalty is
    large because the motion genuinely fights its own acceleration.
  • **Safety-weight A/B** (`sw_1p0` vs `sw_0p3`, new `--safety-weight`): lowering r_safe ×0.3 gave
    **no interaction gain** (contacts 0.015 vs 0.007, both ~0; object_motion identical). Safety
    magnitude isn't the lever either.
  **Meta-conclusion:** across β · normalization · λ_cur · BatchNorm · UTD · δ · safety-weight, the
  **representation is reliably made healthy and the WM learns dynamics, but vigorous block interaction
  never emerges** (contacts ~0–0.1, object_motion ~0.002 in every config). Interaction is an
  **exploration/discovery** problem, not reward-balance: free-space arm-jitter yields "enough"
  curiosity and hitting small blocks from scratch is sparse. The one untried lever is **exploration**
  (adaptive-α / pink-noise — le-wm had both; we use fixed α=0.2).
  **Infra:** workload is CPU-render-bound (GPU ~14%, model tiny) — more `--n-envs` ≠ faster single run
  (per-step cost grows); concurrency fills the GPU. **Code:** added `--normalize-curiosity` /
  `--cur-norm-momentum` / `--safety-weight`; defaults n_envs 8→32, start_steps 1000→200, h_fwd_max
  20→**1** (curriculum off by default — runs had climbed to h_fwd 11–12 because a *flatlined* WM loss
  was misread as "mastered"); BatchNorm1d in the encoder; new `src/hw_config.py` (GPU/CPU max-util
  recommender) and `src/analyze_run.py` (collapse/freeze read-out).
