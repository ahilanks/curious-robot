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
| _tbd_ | 0.3 | 15.0 | 0.05 | | | | baseline (β pinned 0.3; λ_cur≈15 → safety:curiosity ~0.5:1) |

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
