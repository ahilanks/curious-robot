"""Build the canonical diverse probe set (uniform arm poses + randomized objects) and
upload it to the HF Hub so every run computes encoder/eff_rank_probe on identical inputs.

One-time:  python src/make_probe.py [--n 256] [--probe-id probe_v1] [--no-hf]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.probe import build_probe_set, save_probe_local, upload_probe_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--wrist-resolution", type=int, default=224)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--probe-id", default="probe_v1")
    ap.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--out", default="runs/probe")
    a = ap.parse_args()

    px, prop, report = build_probe_set(a.n, a.wrist_resolution, a.seed)
    print(f"[probe] built pixels={px.shape} proprio={prop.shape}", flush=True)
    print("[probe] diversity report:\n" + json.dumps(report, indent=2), flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.probe_id}.npz"
    save_probe_local(path, px, prop, report)
    print(f"[probe] saved local -> {path}", flush=True)

    if not a.no_hf:
        repo = a.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
        if repo and os.environ.get("HF_TOKEN"):
            upload_probe_hf(path, repo, a.probe_id)
            print(f"[probe] uploaded -> {repo}/probe/{a.probe_id}.npz", flush=True)
        else:
            print("[probe] HF not configured (need HF_UPLOAD_REPO_ID + HF_TOKEN); skipped upload", flush=True)


if __name__ == "__main__":
    main()
