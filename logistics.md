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

The remaining README `?` constants (λ_cur, δ) are swept here before being pinned; β is fixed at 0.3.

| run | β | λ_cur | δ | steps | contacts/step | pred/persist | notes |
|-----|---|-------|---|-------|---------------|--------------|-------|
| newarch | 0.3 | 15.0 | 0.05 | 10000 | 0.00 | 0.50 | WM learns (pred/persist 0.50, h_fwd→11, eff_rank 4.4→7.8) but policy never contacts blocks; curiosity harvested from non-contact motion |
| lcur20 | 0.3 | 20.0 | 0.05 | 55250\* | 0.00 | 0.14 | per-dim-MEAN `r_cur` (`5824b7a`) → r_cur≈0.75 (was ≈188); **λ_safe=0** (safety ablated), `h_fwd_max=1`. WM learns *very* well (pred/persist 0.14, eff_rank 30, probe 8.1) but policy still never contacts blocks (contacts/step≈0.002). \*crashed 55250/100k (pod, not code). W&B `199jzlil` |

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
