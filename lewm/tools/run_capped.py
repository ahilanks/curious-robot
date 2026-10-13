"""Run a script (e.g. the authors' eval.py) UNMODIFIED with capped CPU thread pools.

On this 128-core box an uncapped eval.py spawned ~385 threads (torch / OpenMP / OpenCV pools sized to every core),
spin-waited on them and starved the concurrent training run (4.9 -> 0.6 it/s). This launcher caps the pools, then
executes the script as __main__ with the remaining argv (Hydra overrides pass through untouched).

    THREADS=4 taskset -c 96-111 .venv/bin/python tools/run_capped.py eval.py --config-name=pusht.yaml policy=...
"""
import os
import runpy
import sys

n = os.environ.get("THREADS", "4")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = n

import cv2  # noqa: E402
import torch  # noqa: E402

cv2.setNumThreads(int(n))
torch.set_num_threads(int(n))
torch.set_num_interop_threads(int(n))

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
