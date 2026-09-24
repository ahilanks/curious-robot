"""Run the method x seed grid of toy_signals.py in parallel processes, then plot.

    python run_all.py                       # 5 methods x 5 seeds, defaults, plots into runs/
    python run_all.py --seeds 3 --steps 8000 --lp-alpha 0.1   # unknown flags go to toy_signals.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from toy_signals import METHODS, build_parser, run

HERE = os.path.dirname(os.path.abspath(__file__))


def _one(argv):
    args = build_parser().parse_args(argv)
    res = run(args)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f"{args.method}_s{args.seed}.npz"), **res)
    return (f"{args.method:8s} s{args.seed}  final {res['eval_mse'][-1]:.3f}  min {res['eval_mse'].min():.3f}  "
            f"oracle {res['oracle']:.3f}  ({res['wall']:.0f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    a, extra = p.parse_known_args()
    jobs = [["--method", m, "--seed", str(s), "--out", a.out, *extra] for m in a.methods for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        for line in ex.map(_one, jobs):
            print(line, flush=True)
    if a.plot:
        subprocess.run([sys.executable, os.path.join(HERE, "plot.py"), "--runs", a.out], check=False)


if __name__ == "__main__":
    main()
