"""Encode every frame of a dataset with the frozen LeWM encoder -> latent cache."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rp1.wm import load_lewm
from rp1.cache import build_latent_cache

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--h5', required=True)
p.add_argument('--out', required=True)
p.add_argument('--extra', nargs='*', default=[])
p.add_argument('--workers', type=int, default=24)
a = p.parse_args()
model = load_lewm(a.ckpt)
build_latent_cache(model, a.h5, a.out, extra_cols=a.extra, workers=a.workers)
