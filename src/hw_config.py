"""Hardware probe + max-utilization recommender for Curious Robot training.

WHY THIS EXISTS
---------------
Curious Robot training is **CPU / environment-bound, not GPU-bound**. The world model
is tiny (~15M params); a single run uses only ~6 GB of GPU and ~14% GPU utilization
(measured on an H100). The real bottleneck is CPU-side MuJoCo/OSMesa rendering — each
parallel env runs in its own worker process and saturates a few cores. A bigger GPU
won't help; a cheaper one would run a single job just as fast.

IMPORTANT — what each knob actually buys (measured):
  * `--n-envs` raises *env-throughput* (env-steps/sec) and parallel data diversity, but does
    NOT speed up a fixed-`--total-steps` run: per-decision-step wall time *grows* with n_envs
    (8->48 envs ≈ 2x slower per step), so a 10k-step run is slower, not faster, at 48 envs.
    (It also interacts with the small PER buffer, which evicts most of the extra transitions.)
  * **Concurrent runs are the real way to fill the GPU** — ~6 tiny jobs saturate compute and
    a 80 GB card fits ~13 by memory. For sweeps, launch several jobs at once.

So: pick n_envs for the data regime you want (not for speed); use concurrency to fill the GPU.

USAGE
-----
    python src/hw_config.py            # human-readable recommendation for THIS machine
    python src/hw_config.py --json     # machine-readable {recommended_n_envs, max_concurrent_runs, ...}
    python src/hw_config.py --concurrent 4   # n_envs if you intend to run 4 jobs at once

All heuristics live in the PROFILE block below; they were measured on this project and
are the only thing to retune if the model or env-rendering cost changes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

# --------------------------------------------------------------- measured profile
# (one curious-robot run, n_envs=8, on an H100 + 224-core box — see logistics.md)
PER_RUN_GPU_GB = 6.0          # GPU memory one run occupies (~5.6 GB observed)
CORES_PER_ENV = 4.5           # CPU cores one env-subprocess effectively draws (OSMesa render)
RESERVED_CORES = 8            # leave for the trainer's main process + OS
SINGLE_RUN_GPU_UTIL = 0.14    # mean GPU util of one run (bursty); ~1/this runs saturate compute
N_ENVS_MIN, N_ENVS_MAX = 8, 256


def detect_gpus() -> list[dict]:
    """[{name, mem_total_gb, mem_free_gb, util_pct}] via nvidia-smi; [] if no GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, mtot, mfree, util = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "mem_total_gb": float(mtot) / 1024,
                     "mem_free_gb": float(mfree) / 1024, "util_pct": float(util)})
    return gpus


def detect_cpu() -> dict:
    n = os.cpu_count() or 1
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0
    return {"cores": n, "load1": load1, "load5": load5, "load15": load15}


def recommend(gpus: list[dict], cpu: dict, concurrent: int | None = None) -> dict:
    """Compute max-utilization settings for the detected hardware.

    - max_concurrent_runs: how many jobs fit at once (min of GPU-memory and GPU-compute caps).
    - recommended_n_envs:  envs for a *single* run (uses ~all free cores).
    - n_envs_per_run_if_full: envs per run if you launch `max_concurrent_runs` jobs together.
    """
    free_cores = max(cpu["cores"] - RESERVED_CORES - cpu["load1"], 1.0)

    if gpus:
        mem_free = sum(g["mem_free_gb"] for g in gpus)
        mem_cap = max(1, int(mem_free // PER_RUN_GPU_GB))
        compute_cap = max(1, round(0.9 / SINGLE_RUN_GPU_UTIL))  # ~6: how many bursty runs fill compute
        max_concurrent = min(mem_cap, compute_cap)
    else:
        mem_cap = compute_cap = max_concurrent = 1

    def envs_for(n_jobs: int) -> int:
        return int(max(N_ENVS_MIN, min(free_cores / (CORES_PER_ENV * max(n_jobs, 1)), N_ENVS_MAX)))

    return {
        "recommended_n_envs": envs_for(1),
        "max_concurrent_runs": max_concurrent,
        "n_envs_per_run_if_full": envs_for(max_concurrent),
        "n_envs_if_concurrent": envs_for(concurrent) if concurrent else None,
        "free_cores": int(free_cores),
        "gpu_mem_cap": mem_cap, "gpu_compute_cap": compute_cap,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit recommendation as JSON only")
    ap.add_argument("--concurrent", type=int, default=None,
                    help="also report n_envs per run if you plan to launch this many jobs at once")
    args = ap.parse_args()

    gpus, cpu = detect_gpus(), detect_cpu()
    rec = recommend(gpus, cpu, args.concurrent)

    if args.json:
        print(json.dumps(rec, indent=2)); return

    print("=== Hardware ===")
    if gpus:
        for i, g in enumerate(gpus):
            print(f"  GPU {i}: {g['name']}  {g['mem_total_gb']:.0f} GB "
                  f"({g['mem_free_gb']:.0f} free)  util {g['util_pct']:.0f}%")
    else:
        print("  GPU: none detected (CPU-only)")
    print(f"  CPU: {cpu['cores']} cores  load {cpu['load1']:.1f}")
    print("\n=== Workload profile ===")
    print(f"  CPU/render-bound: ~{PER_RUN_GPU_GB:.0f} GB & ~{SINGLE_RUN_GPU_UTIL*100:.0f}% GPU per run; "
          f"~{CORES_PER_ENV:.1f} cores per env.")
    print("\n=== Recommendation ===")
    print("  NOTE: more --n-envs = more data/sec & CPU use, NOT a faster single run "
          "(per-step slows). To fill the GPU, run jobs concurrently.")
    print(f"  Single run     : --n-envs {rec['recommended_n_envs']}   "
          f"(uses ~{rec['free_cores']} free cores; for data diversity, not speed)")
    print(f"  Sweep / fill GPU: up to {rec['max_concurrent_runs']} runs at once "
          f"(GPU mem fits {rec['gpu_mem_cap']}, compute fits ~{rec['gpu_compute_cap']}), "
          f"each at --n-envs {rec['n_envs_per_run_if_full']}")
    if rec["n_envs_if_concurrent"] is not None:
        print(f"  For {args.concurrent} concurrent : --n-envs {rec['n_envs_if_concurrent']} each")
    print(f"\n  e.g.  python src/train.py --name <kw> --n-envs {rec['recommended_n_envs']}")


if __name__ == "__main__":
    main()
