# HANDOFF — curious-robot campaign state (2026-08-14)

**Read this + the tail of `logistics.md` (the ledger, chronological) to resume. Every claim below has a ledger entry with numbers.**

## The goal
Emergent block manipulation: a from-scratch agent (SO-101 sim, wrist-cam pixels only, no proprio in the latent, no injected data/biases — "learn like a baby") that understands movement and can move blocks to match photographed goals. Stack: JEPA-style world model (encoder + predictor), CEM planning in latent space, curiosity + self-proposed goals.

## Live right now
`wr_msegate4` (40k steps, W&B `pthcc01l`-successor — see ledger; log `runs/wr_msegate4_train.log`): the **three-gate stack**, all user-designed:
1. **Locality-gated amplitude curriculum** (`--amax-curric`): action amplitude starts 0.05, grows ×1.15/400 steps while live frac_rand ≤ 0.2, backs off > 0.25, ceiling = uncapped (5.6). Proven: finds equilibrium ~0.27–0.31 without breaking the latent.
2. **Mastery-gated goal distance** (`highmse_under_d` + curriculum): goals = highest-WM-surprise buffer states within latent budget d of each env; d grows 1-at-a-time on 0.95 completion rate. d-start **10** (this latent's unit scale — NOT 1; see scale-mismatch finding).
3. **Progress-gated goal retention** (`--goal-retain`): per env, keep the current goal while new best distances arrive (≥0.5% improvement, `--goal-retain-delta 0.005`); switch only on arrival (≥10-step hold), stall (**patience 200** = 10s of video, user-specified), or max-age 500.

Monitors: 20-min ticker + d-advance/crash watch + stall watchdog (re-arm after any restart).

## Campaign findings (each ★-ledgered)
- **Wrist + no-cap from scratch has block salience overhead never achieved**: 6/6 scenes, median shift-d0 1.7–3.8 (100mm probes), viewpoint-robust; overhead lineage plateaued at 0.5–0.9 with dead scenes. The acting eye is the salient eye.
- **No-cap vs cap are complementary phases**: uncapped = contact/salience/data-efficiency winner (contacts to 0.41/step; salience median 2.0 from ~1.6k encoder updates on the gated head); capped/fine geometry = mastery-depth winner (historical ladder d→21). The d≈5–6 wall for uncapped ladders is **structural, restore-exonerated** (replicated with zero restores; mechanism = coarse latent geometry, frac_rand 0.35–0.45, can't support the fine-geometry bootstrap; erosion repair-loop under gate-grinding).
- **The amplitude curriculum resolves the phase tension dynamically** — amplitude is earned by prediction quality. Clean success in wr_msegate (0.05→0.31, locality never broke).
- **Latent units drift**: d/eps are measured in the current latent's units. d-start 1 (arr95-era) described an empty shell for the continuously-trained latent (nearest candidates ~10–12) → fallback-far goals → ladder froze. Fix: d-start 10.
- **Health vs baseline** (msegate runs vs wr_nocap): contacts 3.5×, mse_block better (0.116 vs 0.175), locality better; pred_vs_persist ~0.76 is a low-amplitude ratio ARTIFACT (absolute pred_loss fine); watch-item = rank_frac 0.15–0.20 (possible capacity narrowing — salience probes decide).
- **Pursuit** (moving blocks to match photos): salience solved, direction NOT yet — closures are condition-blind shoves (probe-verified); h=3 amplifies engagement not direction; the bind is WM contact-displacement fidelity + finish-precision. First-ever deliberate closure +18.6mm (25k head, start-visible staging) did not repeat reliably.

## Pre-registered next (user-ordered)
When the three-gate run "works" (amplitude well clear of floor + locality held + salience probes ≥ par): **the policy arm** — replace CEM with an **action-conditioned goal policy π(a | z_hist, a_hist, z\*)** on the FROZEN mature latent, HER + latent-distance reward, amplitude fixed at equilibrium; twin vs a CEM continuation; distill-then-finetune fallback. Rationale ledgered 08-13.

## Probes (the instruments)
`src/probe_block_goal_learn.py` — block-shift salience/pursuit: 6 fixed scenes (seed 41), `--shift 0.10` standard, d0 vs ctrl floors = salience; deliberate closure (shift minus ctrl) = pursuit. `--stage-pose visible` = start-visible staging variant; `--horizon N` = probe-time planner override; `--budget 1` = salience-only short probe. Two instrument bugs found+fixed this campaign (geom restore, speckle threshold) — d0s before 2026-08-08 are inflated ~10×.

## Ops essentials
- **HF**: repo `a5ilank/curious-robot` (all ckpts + 15GB state snapshots per run). Uploads need `HF_TOKEN` (from `~/.cache/huggingface/token`) + `HF_UPLOAD_REPO_ID` injected at launch — not in the box profile.
- **git push**: repo-local credential helper sources `GH_TOKEN` from `/workspace/curious-robot/.env` (never print it). After a box recycle re-run the helper config (see memory / ledger 08-11).
- **W&B**: entity `ahilan-uc-berkeley-electrical-engineering-computer-sciences`, project `curious-robot`, auth via `~/.netrc`. Run reconstruction: diff ckpt["args"] (or W&B config) vs argparse defaults (parser lives inside `parse_args()`).
- **Chains**: `--init-ckpt <ckpt>` + `state_latest.npz` beside it (`--init-buffer auto`); regime-match curriculum flags. Curriculum/controller state (amax_frac, retention) is RUNTIME-ONLY — never resume a curriculum run mid-flight; restart from scratch (runs are seed-deterministic and retrace).
- **Disk**: 100k runs need ≥45G headroom (ckpts + rolling state + tmp-swap spike). Verify-on-HF before deleting local artifacts (the 08-12 crash).
- **Time**: 25k steps ≈ 2.5–4h depending on planner (modern 18/200/20 fast; arr95-era 30/300/30 slow) and co-tenancy.

## Parked-but-alive threads
- γ pure-steps chain: `wr_nocap3` @ lineage 75k, E1 fired (first directedness), banked, resumable.
- Gated-uncapped heads: `arr95_nocap`/`2`/`2b` + fresh/fresh60 (the wall evidence), all on HF.
- Overhead lineage (oh_*): closed at 100k — encoder-loss levers (c1/c2/c3) ledgered if ever revisited.
