"""Build a latent cache: encode every frame of an HDF5 dataset with the frozen
LeWM encoder (+projector). Output .npz holds
    z        (T_total, D) float32   latents in dataset row order
    ep_idx   (T_total,)   int32
    step_idx (T_total,)   int32
    ep_len   (E,)         int32
    ep_off   (E,)         int64
plus optional extra state columns for analysis (e.g. pos_agent).
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the compression filters)
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .wm import encode_pixels, normalize_uint8_batch


class _ChunkReader(Dataset):
    """Yields contiguous row-chunks of the pixels dataset (workers decompress)."""

    def __init__(self, h5_path, chunk=500):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, 'r') as f:
            self.n = f['pixels'].shape[0]
        self.chunk = chunk
        self.starts = list(range(0, self.n, chunk))
        self._f = None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        if self._f is None:
            self._f = h5py.File(self.h5_path, 'r', swmr=True, rdcc_nbytes=64 * 1024 * 1024)
        s = self.starts[i]
        e = min(s + self.chunk, self.n)
        px = self._f['pixels'][s:e]  # (n,H,W,3) uint8
        return s, torch.from_numpy(np.ascontiguousarray(px))


@torch.no_grad()
def build_latent_cache(model, h5_path, out_path, extra_cols=(), chunk=500, workers=24,
                       device='cuda', batch=500):
    h5_path, out_path = Path(h5_path), Path(out_path)
    with h5py.File(h5_path, 'r') as f:
        ep_len = f['ep_len'][:].astype(np.int32)
        ep_off = f['ep_offset'][:].astype(np.int64)
        ep_idx = f['ep_idx'][:].astype(np.int32)
        step_idx = f['step_idx'][:].astype(np.int32)
        extras = {c: f[c][:] for c in extra_cols}
        n = f['pixels'].shape[0]
    reader = _ChunkReader(h5_path, chunk)
    dl = DataLoader(reader, batch_size=None, shuffle=False, num_workers=workers,
                    prefetch_factor=4, persistent_workers=False)
    D = model.projector.net[-1].out_features if hasattr(model.projector, 'net') else 192
    z = np.zeros((n, D), dtype=np.float32)
    t0 = time.time()
    done = 0
    for s, px in dl:
        px = px.to(device, non_blocking=True)
        outs = []
        for j in range(0, px.shape[0], batch):
            x = normalize_uint8_batch(px[j:j + batch])
            outs.append(encode_pixels(model, x).float().cpu())
        zz = torch.cat(outs).numpy()
        z[s:s + zz.shape[0]] = zz
        done += zz.shape[0]
        if (done // chunk) % 200 == 0:
            el = time.time() - t0
            print(f'  encoded {done}/{n}  {done / el:.0f} img/s  eta {(n - done) / (done / el):.0f}s', flush=True)
    print(f'cache built: {n} latents in {time.time() - t0:.0f}s')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, z=z, ep_idx=ep_idx, step_idx=step_idx, ep_len=ep_len, ep_off=ep_off, **extras)
    return out_path


class LatentCache:
    """In-memory view of a cache file with episode indexing helpers.
    Time unit for the critic / actor is the *block* (5 primitive steps)."""

    def __init__(self, path, block: int = 5, device='cuda'):
        d = np.load(path)
        self.z = torch.from_numpy(d['z']).to(device)  # raw latents (the WM's space)
        self.ep_len = d['ep_len'].astype(np.int64)
        self.ep_off = d['ep_off'].astype(np.int64)
        self.ep_idx = d['ep_idx']
        self.step_idx = d['step_idx']
        self.extras = {k: d[k] for k in d.files if k not in ('z', 'ep_len', 'ep_off', 'ep_idx', 'step_idx')}
        self.block = block
        self.device = device
        self.n_ep = len(self.ep_len)
        self.mean = self.z.mean(0, keepdim=True)
        self.std = self.z.std(0, keepdim=True) + 1e-6

    def episodes(self, lo, hi):
        return np.arange(lo, min(hi, self.n_ep))

    def row(self, ep, t):
        """global row index of (episode, primitive step)."""
        return self.ep_off[ep] + t
