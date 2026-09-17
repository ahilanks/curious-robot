# RP1 reproduction: independent rerun, 2026-09-17

Fresh pod, fresh venv (torch 2.14+cu130, transformers 5.17, stable-worldmodel 0.1.1, ogbench 1.2.1, mujoco 3.13), datasets and LeWM checkpoints re-downloaded, latent caches, critics and RP1 actors retrained from scratch with the committed scripts (Cube actors with --tf32 as documented). Baselines are seeded and the eval tasks are fixed, so baseline cells are a near-deterministic replay of the README; RP1 cells use newly trained refiners.

## RP1 vs README (previous run) vs paper

| domain | metric | rerun | README | paper |
|---|---|---|---|---|
| TwoRoom h25 / h100 | success % | 99.3 / 97.6 | 99.8 / 90.0 | 100.0 / 94.2 |
| Reacher tau=0.1 / 0.05 | first-hit % | 97.9 / 86.6 | 97.6 / 86.0 | 98.7 / 88.7 |
| Cube h25 easy / hard | success % | 86.0 / 70.0 | 86.2 / 70.5 | 89.1 / 75.2 |
| Cube h100 easy / hard | success % | 77.8 / 60.3 | 77.1 / 59.1 | 82.4 / 67.8 |

TwoRoom h100 RP1 per planner seed x eval seed: s0 [100,100,100], s1 [90,98,90], s2 [100,100,100] (README run had one weak seed at [70,62,80]).

## Full tables (scripts/aggregate.py on runs/*/results.jsonl; 'paper' = LeWM column)

```

== cube h=25   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper |   hard  paper
   noop           53.3   3.8    3   150 |   56.0 |    -      -  
   cem/latent     75.3   3.4    3   150 |   74.0 |   47.1   40.9
   cem/value      81.3   0.9    3   150 |   84.0 |   60.0   63.6
   mppi/latent    61.3   0.9    3   150 |   56.7 |   17.1    1.6
   mppi/value     69.3   2.5    3   150 |   63.3 |   34.3   16.6
   adam/latent    75.3   1.9    3   150 |   74.0 |   47.1   40.9
   adam/value     76.0   2.8    3   150 |   74.7 |   48.6   42.5
   rp1            86.0   2.8    9   450 |   89.1 |   70.0   75.2

== cube h=100   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper |   hard  paper
   noop           44.0   4.9    3   150 |   45.3 |    -      -  
   cem/latent     64.7   2.5    3   150 |   58.0 |   36.9   23.2
   cem/value      74.7   3.4    3   150 |   76.7 |   54.8   57.4
   mppi/latent    49.3   3.8    3   150 |   46.7 |    9.5    2.5
   mppi/value     58.7   1.9    3   150 |   52.0 |   26.2   12.2
   adam/latent    56.0   4.3    3   150 |   57.3 |   21.4   21.9
   adam/value     67.3   3.4    3   150 |   68.7 |   41.7   42.7
   rp1            77.8   4.0    9   450 |   82.4 |   60.3   67.8

== reacher h=25 tau=0.1   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            2.3   0.7    6   300 |    -  
   cem/latent     98.7   0.9    6   300 |   98.7
   cem/value      98.0   1.2    6   300 |   97.3
   mppi/latent    63.7   3.9    6   300 |   63.7
   mppi/value     84.0   3.8    6   300 |   74.0
   adam/latent    90.0   4.2    6   300 |   94.0
   adam/value     89.7   3.5    6   300 |   88.0
   rp1            97.9   2.1   36  1800 |   98.7

== reacher h=25 tau=0.05   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            0.7   0.9    6   300 |    -  
   cem/latent     81.3   6.7    6   300 |   80.3
   cem/value      83.0   6.0    6   300 |   82.0
   mppi/latent    43.3   4.7    6   300 |   39.3
   mppi/value     61.3   1.9    6   300 |   42.0
   adam/latent    61.7  10.5    6   300 |   66.0
   adam/value     67.0   6.2    6   300 |   64.7
   rp1            86.6   5.5   36  1800 |   88.7

== tworoom h=25   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            4.0   3.3    3   150 |    -  
   cem/latent     84.7   4.1    3   150 |   84.0
   cem/value     100.0   0.0    3   150 |  100.0
   mppi/latent    70.0   4.3    3   150 |   70.7
   mppi/value     96.0   4.3    3   150 |   87.3
   adam/latent    92.7   0.9    3   150 |   94.7
   adam/value     97.3   0.9    3   150 |   96.7
   rp1            99.3   0.9    9   450 |  100.0

== tworoom h=100   (runs = eval seeds x planner seeds; eps = episodes)
   planner        ours    sd runs   eps |  paper
   noop            0.0   0.0    3   150 |    -  
   cem/latent     14.7   0.9    3   150 |   13.3
   cem/value      94.7   3.4    3   150 |   94.7
   mppi/latent    20.0   1.6    3   150 |   20.0
   mppi/value     77.3   1.9    3   150 |   64.0
   adam/latent    26.7   0.9    3   150 |   24.0
   adam/value     85.3   2.5    3   150 |   83.3
   rp1            97.6   4.1    9   450 |   94.2
```
