# GPU-resident world-model RL: target architecture (MJWarp + torch WM/CEM)

Design target for moving the SO-ARM101 curiosity/goal-explore stack from a CPU-subproc,
OSMesa-rendered, IPC-bound loop to a **single-process, GPU-resident** loop. Grounded in the
measurements from the 2026-06-18 analysis session.

## Decisions locked
- **Sim-only fast is acceptable** (no hardware path required for the speed work). MJWarp's
  low-fidelity raycast renderer is fine; the WM retrains on it.
- **MJWarp is the sim accelerator. PufferLib is a parts bin** (Protein sweeps + the MinGRU
  planning-head idea), not the core engine — its 20M-sps model needs a C/Ocean env and PPO,
  neither of which we have.

## Why this shape (the measurements)
- Env is **render-bound**: per decision (1 env) physics=3.3 ms vs **OSMesa render=35.5 ms** (91%).
- **EGL render=3.0 ms** (12× over OSMesa) but aborts against the trainer's CUDA context → today's
  OSMesa+subproc workaround. GPU-native rendering sidesteps the whole fight.
- **CEM is compute-bound**, not launch-bound: CUDA-graph capture = **1.0×**, more envs = **+11%**
  (8→64), **bf16 = 2.1×**, hard CUDA crash at batch (N·K) ≳ 76,800.
- Therefore: the env win is **GPU physics + GPU batch render + no host⇄device copies**; the CEM
  win is **precision + fewer sequential steps + a lighter planning head**, NOT parallelism.

## The core principle
One process. N (10²–10³) worlds resident on the GPU. Pixels never leave the GPU. Zero-copy
Warp⇄torch via dlpack. The 224² images that dominate cost today are produced and consumed on-device.

## Per-decision data flow (target)
```
            ┌─────────────────────────── all on GPU, one process ───────────────────────────┐
 CEM/actor (torch, bf16) ─actions(torch)→ d.ctrl (Warp, zero-copy)
      │                                        │
      │                          PD + action-block substep kernel ─×frame_skip→ mjw.step  (graph-captured)
      │                                        │
      │                          mjw.render (batch raycaster) ─rgb(Warp)→ torch (dlpack)
      │                                        │
      └────────── ViT encode → z ── WM update / reward / goal archive / replay (all torch on GPU)
```
No IPC, no OSMesa, no numpy reward math, no pixel transfers.

## Component port map (current → target)
| current (CPU/numpy) | file:fn | target (GPU) | notes |
|---|---|---|---|
| `SubprocVectorMujocoEnv` 8 procs + pipes | `env/parallel_env.py` | one MJWarp env, `nworld=N` | vectorization is now the GPU batch |
| OSMesa wrist/overhead render | `mujoco_env._get_obs` | `mjw.render` batch raycaster → torch | low-fi; WM retrains |
| PD actuation | `env/soarm_adapter.py` | Warp kernel → `d.ctrl`, or `m.callback.control` | delta-target PD per substep |
| action-block substep interp | `mujoco_env.step` (`substep_interp`) | loop over `frame_skip` writing `d.ctrl` between `mjw.step` | keep the interp semantics |
| `safety_reward` (−τ·q̈ hinge) | `env/safety_reward.py` | torch on batched `d.actuator_force`,`d.qacc` | pure elementwise → easy |
| object/table contact counts | `mujoco_env.step` | from `d.contact` (or a count kernel) | mind MJWarp contact-sensor semantics |
| `object_motion`, `ee_pos` | `mujoco_env` | torch on batched `d.xpos` | trivial gather |
| `_maybe_respawn_fallen` | `mujoco_env` | torch mask `xpos.z<thr` → rewrite `d.qpos` rows | per-world reset |
| `GoalArchive` (numpy raw obs) | `src/goal_explore.py` | keep (K=64, cheap) or GPU ring | re-encode stays drift-immune |
| `ReplayBuffer` (pixels) | buffers | **see memory decision below** | the crux at scale |

## Key design decision: what the replay buffer stores
`wm_update` trains the encoder **through raw pixels** (`encode … reshape … predict`, grad on),
so the buffer can't just hold latents. At N·cap scale, 224²×3 uint8 is the memory wall:
1024 worlds × 150 cap × 150 KB ≈ **23 GB**. Options:
- **(a) Smaller/fresher buffer.** MJWarp collects so fast you can run nearer on-policy — short
  ring, high turnover. Probably the right answer; cuts memory and matches the fast-sim regime.
- **(b) Store latents + periodic raw snapshots** for the encoder-training minibatches only.
- **(c) Lower WM-input resolution** (e.g. 112²) — 4× memory + 4× render savings; test WM quality.
Decide this with the render/throughput spike, not upfront.

## CEM / WM on GPU (the compute levers, in priority order)
1. **bf16 autocast for CEM rollouts** — measured 2.1×, safe (inference, rank-based elite select).
   Keep **WM *training* in fp32** (PufferLib's "torch unstable in bf16" applies to training).
2. **MPPI / fewer iters + warm-start.** Trade the 10 sequential refit iters for one wide sampling
   pass (K≫) and/or seed `mu` from the previous decision's shifted plan → 50 → ~5–15 sequential
   predicts. This is the real "use the GPU" move (width, not depth).
3. **Lighter planning head (MinGRU).** A small RNN latent-dynamics head — cheap to roll 50×, parallel
   over time, and (unlike the ViT) cudagraph-friendly. Distill from / co-train with the full WM.
4. CEM batch must **chunk** to stay < ~76,800 rows (the SDPA crash ceiling).

## What stays unchanged
WM architecture (`model/state_encoder.py`, `lewm/`), the JEPA+SIGReg loss, the goal-explore
objective and HER, the curiosity reward — all just become batched-on-GPU. No algorithm change.

## Protein (PufferLib) for sweeps
Wrap the train entrypoint as a Protein-tunable target (cost = wall-clock, score = a coverage/
reach metric). Orthogonal to the sim; the one PufferLib piece that helps directly.

## Staged plan (each stage independently shippable)
0. **EGL re-test** (decoupled quick win): try EGL in the CUDA-free workers on this driver/MuJoCo
   3.9. If it survives next to the trainer → ~12× render today, no new deps.
1. **mjwarp feasibility spike**: `mjw.put_model(scene.xml)` (parity gate), batch-step, parity-check
   qpos vs CPU, `mjwarp-testspeed`. Answers: does our scene run, how fast (state-only).
2. **batch-render spike**: render wrist+overhead for N worlds, eyeball fidelity vs current,
   `testspeed --function=render` for pixels/sec. Answers the memory/res decision.
3. **GPU env**: physics + render + ported env logic (table above), validated numerically vs the
   CPU env on a fixed action sequence.
4. **Wire into the loop**: swap `VecEnv` for the GPU env; keep WM/CEM/SAC. Latents/pixels stay on GPU.
5. **CEM speed**: bf16 + MPPI/warm-start (+ MinGRU head later).
6. **Protein sweeps.**

## Risks / open questions
- **66 nv** (10 free objects + 6 arm) sits on MJWarp's ">60 DoF degrades" edge → maybe ~8 objects.
- **Mesh collision geoms on the arm** → CCD pipeline: memory + compile time + box-mesh/mesh-mesh
  need `margin=0` (we're at 0). Consider convex collision proxies.
- **Render fidelity gap** (raycast vs OpenGL) — fine sim-only; would matter if hardware re-enters.
- **GPU determinism**: MJWarp is non-deterministic on GPU (atomics). Fine for RL; note for repro.
- **The env-logic port is the bulk of the work**, not `mjw.step`. mjlab is the reference.
- Throughput stays **render-bound even on GPU** (scales with pixels×cameras×worlds) — Stage 2
  gives the real number; don't assume the state-only 2M/18M figures.

## Rough throughput expectation
- now: ~8 sps (CEM) / collection ~200 dec/s (render-bound, 8 CPU workers)
- + EGL only: collection ~1k+ dec/s
- + MJWarp phys+render, N≈1k: collection **render-bound on GPU** — target 10³–10⁵ dec/s (spike-confirmed)
- CEM mode stays learner-capped → bf16+MPPI+MinGRU head ≈ 5–10× current, independent of sim sps
