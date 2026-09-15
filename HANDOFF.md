# HANDOFF — curious-robot campaign state (2026-09-15)

**Read this + the tail of `logistics.md` (the ledger, chronological) to resume. Every claim below has a ledger entry with numbers.**

## The goal
Emergent block manipulation: a from-scratch agent (SO-101 sim, wrist-cam pixels only, no proprio in the latent, no injected data/biases — "learn like a baby") that understands movement and can move blocks to match photographed goals. Stack: JEPA-style world model (encoder + predictor), CEM planning in latent space, curiosity + self-proposed goals.

## Where things stand
- **Sim campaign: DONE and banked (08-15 ★★★).** `wr_sleepret2` (200k, W&B `q1dzgjq4`): every pre-registered criterion passed — d 3→22 (the full historical ladder target) in 28,650 steps under arrival ≥0.95, then ~170k steps of whole-space consolidation; amplitude equilibrium 0.5–1.4 (floor never touched); 19/19 sleeps converged; contacts ~0.13/step with zero decay. All 200 ckpts + the 15.07 GB final state on HF. **Head = `wr_sleepret2/ckpt_0200000.pt`** (canonical launch block in the 08-15 entry).
- **Hardware line: PAUSED — user directive 09-15: no hardware deploy.** Five real-arm sessions, ~11k steps (08-20/21 `hw_wrs2_a`/`c`): the stack transfers; the 0.40 cliff refuted on hw; the ladder stalls at pctl 0.20 with a real budget; empty-difficulty-budget warm-up found. Near-live GPU split (`src/wm_sleep_server.py` + `--pull-wm`) built and loop-closed. Resumable via `run_hw_wr_sleepret2.sh`.
- **Nothing is running.** 09-14/15 were spent on the decoder instrument (below) on a RunPod A100.

## The decoder — the campaign's new instrument (09-14/15; diagnostic only, never in a gradient path)
- `model/decoder.py` = LeWM App. D post-hoc pixel decoder (z → 224² RGB: 196 patch queries cross-attend to the latent, 2.67M). `bash run_decoder.sh <run> <ckpt_step> <steps>` = fit on a run's frames with its FROZEN encoder → HF upload → Fig.-7 rollout sheet. Fitted: `hw_wrs2_c/decoder_lewm.pt` (real frames, val 0.0138) and `wr_sleepret2/decoder_lewm.pt` (sim, val 0.0026); val = ALL val frames since the 09-15 `evaluate()` fix.
- **What z keeps (the blur finding, real frames):** the latent ≈ a 7×7 colour thumbnail (σ≈16 px blur). Detail survives to the ViT patch tokens (linear readout 0.0008) and dies at the single-[CLS] pooling (0.016); JEPA keeps only what predicts; pixel-MSE turns the gap into blur. Decoder capacity is not the limit (10× params: no gain; 3× steps: −20% then memorising).
- **In sim:** wall/table geometry decodes sharply; the magenta cube decodes as a blob in the right place (the block IS in z); small / blue / red / black cubes are lost. Block pixels are ~75× harder than background (MSE 0.15 vs 0.002).
- **Goal archive @200k** (`src/viz_goal_archive.py`): 48/64 archived goals contain no block — viewpoint goals (wall/table edges). The 16 block goals are the highest-surprise ones; the top-4 are block close-ups (10–22k block px) where the decode marks the object but not its colour. d=22 mastery is largely camera-pose mastery.
- **Checkpoint sweep** (`src/decoder_ckpt_sweep.py`, 1k/10k/30k/100k/200k): decodability FLAT across the 19 sleeps — val 0.0027→0.0026, block-pixel MSE 0.156→0.150, block detected 45%→50%. The sleeps consolidated the predictor without changing what the latent keeps of the pixels.
- **Open-loop imagination horizon** (`src/viz_decoder_rollout.py --horizon T`): scale note — z_mse is per-dim, so 0.04 ≈ L2 2.8 = reach eps and 2.0 ≈ L2 19.6 = the random-pair diameter. Imagination stays within eps ~2–3 steps, coherent (z_mse < 0.3) ~8–12 steps, at random-pair level by T=16–32 on every segment. CEM at horizon 1 + replan-every-step is the right regime for this predictor.
- **Decoder's eye in sim:** `bash run_sim_decoder_eye.sh [name] [steps]` = wr_sleepret2@200k FROZEN in MuJoCo (no gradients) + `--live-view-record` mp4: wrist | decode(z now) | decode(plan → next z) | decode(z*) | goal photo. Shows the plan steering toward the goal view. (`--live-view` dashboard works on any box now — PIL fallback for the JPEG path.)

## Campaign findings (each ★-ledgered)
- **Wrist + no-cap from scratch has block salience overhead never achieved**: 6/6 scenes, median shift-d0 1.7–3.8 (100mm probes), viewpoint-robust. The acting eye is the salient eye.
- **No-cap vs cap are complementary phases**: uncapped = contact/salience/data-efficiency winner; capped/fine geometry = mastery-depth winner. The d≈5–6 wall for uncapped ladders is structural, restore-exonerated.
- **The amplitude curriculum resolves the phase tension dynamically** — amplitude is earned by prediction quality (`--amax-curric`, floor via `--amax-curric-floor`).
- **The cliff falls on the stationary map** (08-14): the pctl-0.40 cliff that killed msegate at d=7 and d=11 does not exist on the frozen latent; the wake/sleep engine with d-scaled sleep spacing (10k) took the ladder to d-max 22 in 3.5 h (July projection: 30–66 days).
- **Latent units drift**: d/eps are in the current latent's units (d-start 10 for this lineage, not 1).
- **Pursuit** (moving blocks to match photos): salience solved, direction NOT yet — closures are condition-blind shoves; the bind is WM contact-displacement fidelity + finish-precision. The decoder now adds: the archive's hardest goals are exactly the contact views, and those decode worst.

## Pre-registered next (user-ordered) — all SIM
1. **Close-out A**: salience/pursuit probes on `ckpt_0200000` (`src/probe_block_goal_learn.py`); the decoder gives a pixel-side second opinion.
2. **Close-out B — the policy arm**: π(a | z_hist, a_hist, z\*) on the FROZEN mature latent, HER + latent-distance reward, amplitude fixed at equilibrium; twin vs a CEM continuation (`--her-frac` exists).
3. Recover `--cem-hier` (two-level latent CEM; the 09-13 pod's code was lost, ledger description only).

## Probes (the instruments)
- `src/probe_block_goal_learn.py` — block-shift salience/pursuit: 6 fixed scenes (seed 41), `--shift 0.10`, d0 vs ctrl floors = salience; deliberate closure = pursuit; `--stage-pose visible`, `--horizon N`, `--budget 1`. d0s before 2026-08-08 are inflated ~10× (two instrument bugs, fixed).
- Decoder tools (above): `run_decoder.sh`, `run_sim_decoder_eye.sh`, `src/viz_decoder_rollout.py`, `src/viz_goal_archive.py`, `src/decoder_ckpt_sweep.py`; `tests/test_decoder.py` (13 CPU checks).

## Ops essentials
- **Pod bootstrap**: `.env` per `.env.example` (GH/W&B/HF/Pushover keys; gitignored, chmod 600), then `bash setup.sh` (deps, CLIs, hooks, GPU/MuJoCo checks, 60-step smoke). Run everything long in `tmux`.
- **HF**: `a5ilank/curious-robot` — every run's ckpts + state snapshots (`<run>/ckpt_XXXXXXX.pt`, `<run>/state_latest.npz`), decoders (`<run>/decoder_lewm.pt`). `train_decoder.py` / the viz tools resolve `runs/<run>/...` locally first, else pull from HF.
- **git push**: `gh` reads `GH_TOKEN` from the environment (`set -a; source .env; set +a`). The token that sat in the public history was auto-revoked by GitHub secret scanning (09-15). **History purge (09-15): `main` rewritten with `git filter-repo --invert-paths --path .env` and force-pushed (`c769033` → `779cdc2`); backup `/workspace/curious-robot-pre-purge.bundle` on the 09-15 pod. Eight feature branches on origin still carry `.env` in their history — user decision 09-15: leave them and the old W&B/Pushover keys as is (closed). Every other clone of `main` must re-clone.** Never print `.env`.
- **W&B**: entity `ahilan-uc-berkeley-electrical-engineering-computer-sciences`, project `curious-robot`. Run reconstruction: diff ckpt["args"] (or W&B config) vs argparse defaults.
- **Chains**: `--init-ckpt <ckpt>` (+ `state_latest.npz` beside it for the buffer); `--frozen-policy` = act with the loaded stack, no gradients. Curriculum/controller state is RUNTIME-ONLY — never resume a curriculum run mid-flight.
- **Disk**: a 200k run's state is 15 GB; ≥45G headroom for 100k runs. Verify-on-HF before deleting local artifacts.

## Parked-but-alive threads
- Hardware line (above) — five sessions banked on HF (`hw_wrs2_c/` ckpts + 2.08 GB real-frame state); `hw_dry_split` transport test.
- γ pure-steps chain: `wr_nocap3` @ lineage 75k, E1 fired (first directedness), banked, resumable.
- Gated-uncapped heads: `arr95_nocap`/`2`/`2b` + fresh/fresh60 (the wall evidence), all on HF.
- Overhead lineage (oh_*): closed at 100k.
