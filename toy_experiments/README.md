# Toy experiments: which learning signal explores best?

A 1-D toy of the curiosity-signal comparison in the main project (`--goal-score {mse,rnd,lp}`),
with everything else stripped away.

## Setup

- **Target f**: a fixed random MLP on [-1, 1] (1 → 64 → 64 → 1, sine activations, first-layer
  scale 8). Random tanh MLPs average out to a few smooth bumps that the small learner fits to
  under 2% of the variance, so the target uses sine units, which keep fine structure. Output is
  standardised to zero mean, unit variance. Seed picks the target; each seed's target is shared
  by every method.
- **Learner g**: a smaller tanh MLP (1 → 8 → 8 → 1). It cannot fit f: a full-grid fit with
  unlimited data bottoms out around 0.5–4% of the variance, with the residual concentrated on the
  sharpest wiggles. g takes **one Adam step on each newly visited point** (no replay, by design).
- **Explorer**: a SAC policy π(δ | x), |δ| ≤ 0.1. The next point is x' = clip(x + δ), f(x') is
  observed, g takes its step on (x', f(x')). The reward is the chosen signal evaluated at x'
  **before** g's step, stored in SAC's replay at collection time (as in the main trainer), and
  divided by its running standard deviation so the three signals share a scale.
- **Signals**
  - `mse`: g's squared error at x'.
  - `rnd`: predictor-vs-fixed-random-target error at x' (the predictor takes one Adam step per visit).
  - `lp`: learning progress in the main trainer's convention. Every visited point is a sticky
    pool entry; each step the pool is re-scored under the current g, LP = |φ − EMA φ| (EMA rate
    0.3, absolute value), and the reward is the kernel-weighted mean LP of pool points near x'
    (Gaussian, width 0.05). A brand-new region has no history and scores 0.
- **Baselines**: `uniform` (x' ~ U[-1, 1], ignores locality) and `walk` (δ ~ U[-0.1, 0.1]).
- **Metric**: g's MSE against f on a 512-point grid, every 50 samples, plus where the samples went.
- Defaults: 5000 samples, 5 seeds, SAC with γ 0.9, τ 0.005, lr 3e-4, batch 128, 200 random warm-up steps.

## Run

```bash
python run_all.py                      # 5 methods x 5 seeds -> runs/*.npz + figures + summary.md
python run_all.py --out runs_replay32 --g-replay 32   # control: g also replays 32 past points
python toy_signals.py --method lp --seed 0 --lp-alpha 0.1     # one run, any flag
python plot.py --runs runs             # re-plot
python diagnose.py --runs runs --seed 0   # exact reward maps over time + what SAC stored -> why.png
# video: dense snapshots of seed 0, then animate the three signals side by side
for m in mse rnd lp; do python toy_signals.py --method $m --seed 0 --eval-every 20 --no-oracle --out runs/video_data; done
python animate.py --runs runs/video_data --out runs/video.mp4
```

Every flag of `toy_signals.py` passes through `run_all.py`. Figures: `mse_vs_steps.png`,
`visitation.png` (x visited over time, one seed, and the pooled density), `final_fit.png`
(final g against f with the visit histogram), `policy.png` (reward traces and the final mean
action over x: a zero crossing from + to − is an attractor).

## Results (2026-09-21, defaults, 5 seeds; the target has unit variance, so 1.0 = predicting the mean)

### Default: one step on the new point only (`runs/`)

| method | final MSE | best MSE | mean over run | coverage |
|---|---|---|---|---|
| mse | 2.30 ± 1.83 | 0.76 | 1.55 | 0.97 |
| rnd | 0.92 ± 0.24 | 0.72 | 1.09 | 1.00 |
| lp | 1.98 ± 1.14 | 0.80 | 1.72 | 0.93 |
| uniform | 0.62 ± 0.31 | 0.58 | 0.69 | 1.00 |
| walk | 1.12 ± 0.32 | 0.79 | 1.22 | 1.00 |

- **Nothing local beats the constant predictor.** The random walk already lands at 1.1. A learner
  that takes one step on the latest point only remembers its last few hundred samples, so
  whatever a local explorer does, the rest of the interval is forgotten. The signal is second-order
  to this.
- **mse and lp fixate on the same attractor.** In 4 of 5 seeds the mode of their last 1000 visits
  agrees to two decimals (e.g. seed 0: both +0.32), with 50–80% of those visits within ±0.1 of it.
  rnd keeps wandering (16–38% within ±0.1 of its mode), walk 9–28%. The `policy.png` action
  curves show the mechanism: mse and lp learn a single stable zero crossing, rnd learns none.
- **Why (from `diagnose.py`, which replays the visit sequence to rebuild the exact reward map each
  policy faced; `runs/why.png`):** the mse explorer's final resting place is a *low*-error trough.
  In the last 2000 steps of seed 0 it collects 0.04 (normalised) at the attractor against 0.24
  elsewhere, and the same holds on seeds 2 and 3. It went there while the region was the brightest
  part of the map (the big peak the learner could not fit), the learner then fitted the centre under
  it, and the policy stayed: its mean action points inward at 0.05–0.07 per step while its
  exploration noise is about 0.03, so the bands of high error 0.2 away are effectively out of
  reach. The reward collapsed under the policy and the policy has no mechanism to follow it.
- lp lands on the same attractor because |φ − EMA φ| scales with φ: a change of a large error is a
  large LP, so early on the LP map is a noisy copy of the error map. After that its map is spatially
  flat: the reward at the attractor equals the reward elsewhere on every seed (e.g. 0.40 vs 0.54,
  0.60 vs 0.59). The one-step learner's update moves g everywhere at once, so "error is changing"
  is true everywhere at once (vertical stripes in `why.png`). LP here is a signal about *when* g is
  moving, not *where* to go.
- rnd's map is nearly flat too (0.21 vs 0.21), for a different reason: the predictor generalises
  over a 1-D input, so a visit erases novelty in a halo about 0.3 wide, and two thirds of the
  running standard deviation that normalises the reward comes from the first 200 steps. Late
  novelty differences are small in the units SAC sees, the untouched edges are 5–8 steps away
  (γ 0.9), and the buffer's stored rewards are stale, so it drifts as a slow biased walk rather
  than sweeping to uniform coverage.
- Reward does not rise with local visit density for any signal (corr −0.13 mse, −0.25 rnd,
  −0.11 lp), so none of this is a "more visits → more reward" loop.
- mse's final error blows up on two seeds (3.1, 5.6): concentrated one-step training on one
  region extrapolates wildly elsewhere.

### Control: the same step also replays 32 random past points (`runs_replay32/`)

| method | final MSE | best MSE | mean over run | coverage |
|---|---|---|---|---|
| mse | 0.23 ± 0.15 | 0.18 | 0.44 | 1.00 |
| rnd | 0.16 ± 0.11 | 0.16 | 0.41 | 1.00 |
| lp | 0.44 ± 0.45 | 0.28 | 0.55 | 0.90 |
| uniform | 0.07 ± 0.05 | 0.07 | 0.30 | 1.00 |
| walk | 0.22 ± 0.19 | 0.22 | 0.47 | 1.00 |

- Everything learns once forgetting is removed. Uniform is still 2–3× better than any explorer.
- rnd is the best signal and the only one that beats the random walk on the mean. mse ties the
  walk. lp is worst and bimodal: one seed reaches the capacity floor (0.013), one ends at 1.27.
- lp's failure mode with replay is under-exploration: on seed 0 it parks at x ≈ −0.75 from step
  2000 on, fits the wiggles there very well, and never visits x > 0 (coverage 0.90 vs 1.00).
  "Stay where the error is changing" holds it where the learner is still improving.

### Caveats

1-D, five seeds, one hyper-parameter setting. Rewards are stored at collection time, so SAC lags a
nonstationary reward (as in the main trainer). LP re-scores every step with EMA rate 0.3, so it is
close to a per-step |Δφ|; slower settings (`--lp-alpha 0.05 --lp-rescore-every 10`) are untested.
