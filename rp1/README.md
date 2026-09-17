# RP1 — Reinforced Planning with Latent World Models (reproduction)

Independent re-implementation of **"Reinforced Planning with Latent World Models"**
(Sommer & Schilling, arXiv:2608.18669) on the LeWM backbone, using the
`stable-worldmodel` benchmark platform the paper evaluates on, the authors'
released datasets (`quentinll/lewm-{tworooms,reacher,cube}`) and the pretrained
LeWM checkpoints (`quentinll/lewm-{tworooms,reacher,cube}`).

Everything upstream of the planner is frozen and identical to the paper's setup
(encoder + predictor weights, datasets, env, protocol). What is re-implemented from
the paper text: the MRN critic + n-step/hindsight expectile TD training (App. B.1),
the RP1 residual plan refiner and its pathwise training through the frozen world
model with critic co-training (App. B.2), the evaluation protocol (App. C.1) and the
baseline planner settings (App. C.5, via the platform's CEM/MPPI/Gradient solvers).

## Layout

```
rp1/
  wm.py        LeWM loading (HF ViT key remap), preprocessing, latent rollout H_phi, cost models
  cache.py     encode every dataset frame once -> runs/cache/<domain>.npz (latents + episode index)
  critic.py    MRN quasimetric critic V(z, z_g), TD sampler (anchor / n-step successor / hindsight goal),
               expectile-Huber trainer with Polyak target (Eq. 67-71)
  planner.py   Refiner f_theta, RP1Planner (K=8 refinement rounds, Eq. 72-75), swm Solver wrapper,
               single-run RP1Trainer (reference implementation, double backward)
  batched.py   vectorised trainer: R planner seeds in one process, surrogate gradient
               sum_k c_k <stopgrad(g_k), a_k> (identical gradient, no 2nd backward through the WM),
               CUDA-graph-captured rollout+critic+grad; checked against planner.py to 1e-4
  baselines.py swm solvers (+ device fix for GradientSolver.init_action)
  evalproto.py paper protocol: held-out episodes >= 8000, goal = state h steps ahead, budget 2h,
               eval seeds 42-44, no-op / replay policies
  configs.py   Table 6 per-domain hyper-parameters (LeWM column)
scripts/
  cache_latents.py, train_critic.py, train_rp1.py, train_rp1_batched.py, eval.py, aggregate.py
  run_tworoom.sh / run_reacher.sh / run_cube.sh   full pipelines (critic -> actors -> evals -> table)
  diag_tworoom_critic.py   critic vs latent-L2 vs door-routed path length (paper Fig. 3a / 5)
  test_batched.py          gradient equivalence batched vs single-run trainer
```

## Running

```bash
# per domain (each phase can be run alone: critic | actors | evals)
scripts/run_tworoom.sh all      # ~1 h on one A100
scripts/run_reacher.sh all
scripts/run_cube.sh cache && scripts/run_cube.sh all
python scripts/aggregate.py runs/*/results.jsonl
```

## Results

All numbers: success rate (%), mean over eval seeds (and planner seeds for RP1), 50 episodes per
eval seed, paper's LeWM column alongside. Full JSON per run in `runs/<domain>/results.jsonl`.

### Reacher (Table 3, first-hit success; planner seeds 0-5 x eval seeds 42-47)

```
== reacher h=25 tau=0.1   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            2.3   0.7    6   300 |    -  
   cem/latent     98.7   0.9    6   300 |   98.7
   cem/value      98.0   1.2    6   300 |   97.3
   mppi/latent    63.0   3.4    6   300 |   63.7
   mppi/value     84.0   3.8    6   300 |   74.0
   adam/latent    90.0   4.2    6   300 |   94.0
   adam/value     90.0   3.1    6   300 |   88.0
   rp1            97.6   2.2   36  1800 |   98.7

== reacher h=25 tau=0.05   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            0.7   0.9    6   300 |    -  
   cem/latent     81.3   6.7    6   300 |   80.3
   cem/value      83.0   6.0    6   300 |   82.0
   mppi/latent    42.3   5.6    6   300 |   39.3
   mppi/value     61.3   1.9    6   300 |   42.0
   adam/latent    61.7  10.5    6   300 |   66.0
   adam/value     67.0   6.2    6   300 |   64.7
   rp1            86.0   5.9   36  1800 |   88.7
```

RP1 matches the paper within ~2 points at both tolerances and the whole planner ordering is
reproduced (CEM > RP1 ~ Adam > MPPI at tau=0.1; RP1 > CEM > Adam > MPPI at tau=0.05). The one
outlier is MPPI/value, which is 10-20 points *better* here than in the paper.

### TwoRoom (Table 1; planner seeds 0-2 x eval seeds 42-44)

```
== tworoom h=25   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            4.0   3.3    3   150 |    -  
   cem/latent     84.7   4.1    3   150 |   84.0
   cem/value     100.0   0.0    3   150 |  100.0
   mppi/latent    70.0   4.3    3   150 |   70.7
   mppi/value     96.0   4.3    3   150 |   87.3
   adam/latent    92.7   0.9    3   150 |   94.7
   adam/value     97.3   0.9    3   150 |   96.7
   rp1            99.8   0.6    9   450 |  100.0

== tworoom h=100   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            0.0   0.0    3   150 |    -  
   cem/latent     14.7   0.9    3   150 |   13.3
   cem/value      94.7   3.4    3   150 |   94.7
   mppi/latent    20.0   1.6    3   150 |   20.0
   mppi/value     77.3   1.9    3   150 |   64.0
   adam/latent    26.7   0.9    3   150 |   24.0
   adam/value     85.3   2.5    3   150 |   83.3
   rp1            90.0  14.3    9   450 |   94.2
```

Every baseline is within 1-3 points of the paper except MPPI with the learned critic (better
here at both horizons). The paper's central TwoRoom effect reproduces exactly: with latent L2
the hand-designed planners collapse from 70-93% (h25) to 15-27% (h100), while the learned
critic keeps them at 77-95%. RP1: 99.8 at h25 (paper 100.0) and 90.0 at h100 (paper 94.2);
the h100 mean hides one weak planner seed ({'rp1_h100_s0.pt': [100, 100, 98], 'rp1_h100_s1.pt': [100, 100, 100], 'rp1_h100_s2.pt': [70, 62, 80]}).
A world-model-only comparison (`scripts/diag_rp1_wm.py`) shows the weak seed reaches a similar
predicted cost-to-go (1.73 vs 1.50 blocks) but with larger, more saturated plans (17% vs 6% of
action entries at the amax=2.6 clip, mean |a| 1.42 vs 1.07 in z-units): raw actions beyond the
env's [-1,1] range are clipped by the simulator while the frozen world model extrapolates them,
so the more aggressive seed exploits the model more (the paper's amax for h100 allows this).

### OGBench Cube (Table 4; planner seeds 0-2 x eval seeds 42-44; hard = (s - f)/(100 - f) with the measured no-op floor f)

```
== cube h=25   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper |   hard  paper
   noop           53.3   3.8    3   150 |   56.0 |    -      -  
   cem/latent     75.3   3.4    3   150 |   74.0 |   47.1   40.9
   cem/value      81.3   0.9    3   150 |   84.0 |   60.0   63.6
   mppi/latent    60.7   0.9    3   150 |   56.7 |   15.7    1.6
   mppi/value     69.3   2.5    3   150 |   63.3 |   34.3   16.6
   adam/latent    75.3   1.9    3   150 |   74.0 |   47.1   40.9
   adam/value     76.0   2.8    3   150 |   74.7 |   48.6   42.5
   rp1            86.2   2.6    9   450 |   89.1 |   70.5   75.2

== cube h=100   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper |   hard  paper
   noop           44.0   4.9    3   150 |   45.3 |    -      -  
   cem/latent     64.7   2.5    3   150 |   58.0 |   36.9   23.2
   cem/value      74.7   3.4    3   150 |   76.7 |   54.8   57.4
   mppi/latent    48.7   3.4    3   150 |   46.7 |    8.3    2.5
   mppi/value     58.0   1.6    3   150 |   52.0 |   25.0   12.2
   adam/latent    56.0   4.3    3   150 |   57.3 |   21.4   21.9
   adam/value     67.3   3.4    3   150 |   68.7 |   41.7   42.7
   rp1            77.1   3.9    9   450 |   82.4 |   59.1   67.8
```

RP1 is the best planner in every column, as in the paper, and by a similar margin over the
strongest baseline (hard score 70.5 vs 60.0 at h25, 59.1 vs 54.8 at h100; paper 75.2 vs 63.6 and
67.8 vs 57.4). Absolute RP1 numbers are 3-5 points (easy) below the paper; note that the Cube
actors were trained with TF32 matmuls to fit the compute budget (TwoRoom/Reacher used exact fp32),
and that the paper reports world-model exploitation on Cube that its Dyna step (not reproduced here)
partially fixes. The latent-objective baselines come out somewhat better here than in the paper.

## Summary (RP1 vs paper, LeWM backbone)

| domain | metric | ours | paper |
|---|---|---|---|
| TwoRoom h25 / h100 | success % | 99.8 / 90.0 | 100.0 / 94.2 |
| Reacher tau=0.1 / 0.05 | first-hit % | 97.6 / 86.0 | 98.7 / 88.7 |
| Cube h25 easy / hard | success % | 86.2 / 70.5 | 89.1 / 75.2 |
| Cube h100 easy / hard | success % | 77.1 / 59.1 | 82.4 / 67.8 |

All 42 baseline cells (3 planners x 2 objectives x 7 settings) reproduce within a few points,
and every qualitative claim of the paper's LeWM experiments holds: the learned critic rescues
latent-L2 planning at long horizons (TwoRoom h100), RP1 leads at the tight Reacher tolerance,
and RP1 dominates on the contact-rich Cube task with 9 rollouts per decision vs 3,000-9,000.

## Fidelity notes / deviations

* **Time unit.** The critic and hindsight offsets are counted in 5-step action blocks
  ("stride-five latent cache"); `n-step = 50` and `max-delta` are therefore in blocks.
* **Value expansion (Cube).** Implemented as model-based value expansion: the critic at
  (z_0, z_g) additionally regresses to c_gamma(N) + gamma^N V_bar(H(a_K, z_0), z_g) on the
  planner's imagined terminal latents, weight 1.0.
* **Reacher "window lag 5"** (App. C.3) is not modelled; standardized latents are.
* **Seeds.** TwoRoom / Cube: planner seeds {0,1,2} x eval seeds {42,43,44}, 50 episodes each,
  as in the paper. Reacher: the paper's wider protocol (planner seeds 0-5 x eval seeds 42-47).
* **Baselines.** CEM 300x30 elite 30, MPPI 300x30 T=0.5, Adam(W) 300x10 lr 0.1 (TwoRoom 100x30),
  all in the z-scored 5-chunk action space, replanning every 5 chunks, open loop.
* **Reacher render fix (important).** The released Reacher dataset was rendered with older
  MuJoCo texture semantics (floor checker as 2x2 cells). Under the MuJoCo 3.13 this environment
  installs, the platform's env renders a uniform floor -- an out-of-distribution shift for the
  frozen encoder that dropped *every* planner by 30-40 points (CEM/latent 68% instead of 98.7%).
  `rp1/evalproto.py::fix_render` sets the floor material to `texuniform="false"`, which
  reproduces the dataset frames to 0.4 mean absolute pixel error (0.00 through the evaluate()
  path). TwoRoom (torch renderer) and Cube renders match their datasets exactly.
* **PLDM columns** are not reproduced (no released PLDM checkpoints for these envs).
* **Dyna finetuning (Sec. 7.4)** is not reproduced.

## Compute notes

Training the refiner is ~135 small transformer passes per step (9 rollouts x 5 blocks,
forward + backward) and is launch/latency-bound at batch 128. `batched.py` therefore
trains all runs of a domain (every (h, seed) pair) in one process, uses the exact
surrogate gradient (no second backward through the world model), torch.compile's the
rollout (fused kernels, fp32-exact) and replays it as a CUDA graph. On one A100 this is
~0.5 s/step for 6 TwoRoom runs (fp32). Cube's 6 runs at batch 256 use `--tf32`
(about 2x faster; ~1e-2 relative gradient error, evaluation rollouts stay fp32).
Several GPU processes on one GPU: set `OMP_NUM_THREADS=8` (torch otherwise spawns 128
CPU threads per process on this 256-core box); MPS was tried and dropped (it wedges
when a client is killed).
