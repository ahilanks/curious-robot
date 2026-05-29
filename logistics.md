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
| nosafe | 0.3 | 15.0 | 0.05 | 5000 | 0.175 | 0.58 | **λ_safe=0** (safety ablated) + sum-curiosity, fixed α=0.2 → **interacts** (frac_block 0.066); stopped at 5k. W&B `5n2ir1vl` |
| meancur | 0.3 | 1.0 | 0.05 | 10000 | ~0.04 | 0.32 | λ_safe=0, **r_cur=mean** (λ_cur=1), fixed α=0.2 → bursty/undirected (entropy dominates curiosity **9.5:1** measured); encoder healthy on clean probe (eff_rank_probe 5.6→8.8). W&B `v0k0mvxz` |
| autoalpha | 0.3 | 1.0 | 0.05 | 10000 | 0.005 | 0.39 | λ_safe=0, mean-curiosity, **learnable α** (target −\|A\|=−30) → α decayed 0.20→~0.015, entropy 20→6, **re-froze**; standard target miscalibrated for the 30-dim tanh action. W&B `kowlqmq1` |

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
- 2026-05-29 — **λ_safe ablation + curiosity-normalization + auto-α arc** (3 runs; branch
  `perf/faster-runs`, PR #2 — also landed async actor-learner, train videos, 1-step WM, and a
  fixed-probe `eff_rank_probe` on an HF-cached uniform-pose probe `probe_v1`).
  • **`nosafe`** (λ_safe=0, sum-curiosity λ_cur=15, α=0.2; 5k, W&B `5n2ir1vl`) — **removing the
    safety penalty unfroze interaction**: contacts/step 0→**0.175** (newarch was 0), frac_block
    0.066. Confirms the safety term was suppressing block contact. (Its `eff_rank_probe` 11.7→3.0
    "collapse" was a weak warmup-probe artifact — see meancur.)
  • **`meancur`** (λ_safe=0, **r_cur=per-dim mean**, λ_cur=1, α=0.2; 10k, W&B `v0k0mvxz`) — switched
    r_cur to mean so symlog stays in its discriminative region (the d_z-summed version saturated
    symlog's flat tail). Encoder **healthy** on the clean probe (eff_rank_probe 5.6→8.8, feat_corr
    0.33→0.24) — so nosafe's "collapse" was the probe, not the encoder. But interaction went
    **bursty/undirected**: shrinking the reward ~150× left the fixed-α entropy bonus dominating
    **9.5:1** (measured: −α·logπ≈3.6 vs cur_contrib≈0.38).
  • **`autoalpha`** (λ_safe=0, mean-curiosity, **learnable α**, target −\|A\|=−30; 10k, W&B
    `kowlqmq1`) — standard SAC auto-entropy **fails here**: α decayed monotonically 0.20→~0.015,
    entropy collapsed 20→6, interaction **re-froze** (contacts 0.005). The −30 target sits far below
    the 30-dim tanh policy's natural entropy (+6..+20), so the tuner just kills α; with exploration
    dead the greedy policy can't discover contact. WM/encoder healthy throughout (pred/identity
    0.39, eff_rank_probe 7.0).
  **Cross-run lesson:** directed interaction needs curiosity to **dominate the entropy bonus while
  exploration stays alive**. `nosafe` had it (curiosity ~74 ≫ entropy ~3.6, fixed α keeping entropy
  ~18); mean+λ_cur=1 and auto-α both broke it. **Next:** mean-curiosity with **λ_cur≈15–25, fixed
  α=0.2** (restore nosafe-style curiosity dominance + keep entropy alive for exploration). Code
  defaults now: λ_safe=0, r_cur=mean, λ_cur=1, h_fwd pinned 1 (see README).
