"""Disk-quota path for Reacher / Cube: stream the HDF5 straight out of the
.tar.zst (no extraction), encode every frame with LeWM into a latent cache,
and write a small *eval subset* HDF5 holding only the held-out episodes that the
paper protocol samples (start/goal states + goal images), plus a tasks.json.

One merged forward pass over the file reads the small columns (raw chunks) and
the pixel chunks in ascending byte order.
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py, hdf5plugin  # noqa
from rp1.zsth5 import open_h5_in_tar_zst
from rp1.wm import load_lewm, encode_pixels, normalize_uint8_batch
from stable_worldmodel.data.formats.hdf5 import HDF5Writer

p = argparse.ArgumentParser()
p.add_argument('--tar', required=True)
p.add_argument('--ckpt', required=True)
p.add_argument('--out_cache', required=True)
p.add_argument('--out_subset', required=True)
p.add_argument('--out_tasks', required=True)
p.add_argument('--horizons', type=int, nargs='+', default=[25, 100])
p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
p.add_argument('--n', type=int, default=50)
p.add_argument('--first_heldout_ep', type=int, default=8000)
p.add_argument('--skip_cols', nargs='*', default=['render_time', 'reward', 'distance_to_target', 'id'])
a = p.parse_args()

t0 = time.time()
f, fobj = open_h5_in_tar_zst(a.tar)
keys = list(f.keys())
print('member', fobj.member_name, f'{fobj.size/1e9:.1f} GB  keys {keys}', flush=True)
small_cols = [k for k in keys if k != 'pixels' and k not in a.skip_cols and f[k].dtype.kind not in 'OSU']
dpx = f['pixels']
print('pixels', dpx.shape, dpx.dtype, dpx.chunks, ' small cols', small_cols, flush=True)

# ---- chunk index of every dataset (metadata only) --------------------------------
items = []  # (byte_offset, col, chunk_row0)
for c in small_cols + ['pixels']:
    d = f[c]
    n = d.id.get_num_chunks()
    t1 = time.time()
    for i in range(n):
        ci = d.id.get_chunk_info(i)
        items.append((ci.byte_offset, c, ci.chunk_offset[0]))
    print(f'  index {c}: {n} chunks in {time.time()-t1:.1f}s restarts {fobj.restarts}', flush=True)
items.sort()
print(f'index built: {len(items)} chunks, {time.time()-t0:.0f}s, restarts {fobj.restarts}', flush=True)

model = load_lewm(a.ckpt)
N = dpx.shape[0]
D = 192
z = np.zeros((N, D), dtype=np.float32)
small = {c: np.zeros(f[c].shape, dtype=f[c].dtype) for c in small_cols}
ep_len = f['ep_len'][:]; ep_off = f['ep_offset'][:]          # tiny, already cached
n_ep = len(ep_len)
# ---- which episodes does the eval protocol need?  (needs ep_idx/step_idx == derivable from ep_len/ep_off)
ep_idx_full = np.repeat(np.arange(n_ep), ep_len)
step_idx_full = np.concatenate([np.arange(l) for l in ep_len])
tasks, eval_eps = {}, set()
for h in a.horizons:
    for seed in a.seeds:
        valid = np.nonzero((ep_idx_full >= a.first_heldout_ep) & (step_idx_full <= ep_len[ep_idx_full] - h - 1))[0]
        rng = np.random.default_rng(seed)
        rows = np.sort(valid[rng.choice(len(valid), a.n, replace=False)])
        tasks[f'{h}_{seed}'] = [(int(ep_idx_full[r]), int(step_idx_full[r])) for r in rows]
        eval_eps.update(int(e) for e in ep_idx_full[rows])
eval_eps = sorted(eval_eps)
eval_rows = np.zeros(N, dtype=bool)
for e in eval_eps:
    eval_rows[ep_off[e]:ep_off[e] + ep_len[e]] = True
print(f'{len(eval_eps)} eval episodes ({eval_rows.sum()} frames) for {len(tasks)} task sets', flush=True)
px_keep = {}

# ---- single merged forward pass -----------------------------------------------------
t1 = time.time(); done_px = 0
for k, (off, c, r0) in enumerate(items):
    d = f[c]
    rows = d.chunks[0]
    r1 = min(r0 + rows, d.shape[0])
    if c == 'pixels':
        px = torch.from_numpy(np.ascontiguousarray(d[r0:r1])).cuda(non_blocking=True)
        with torch.no_grad():
            zz = encode_pixels(model, normalize_uint8_batch(px)).float().cpu().numpy()
        z[r0:r1] = zz
        if eval_rows[r0:r1].any():
            px_keep[r0] = px.cpu().numpy()
        done_px += r1 - r0
        if (k % 2000) == 0:
            el = time.time() - t1
            print(f'  [{k}/{len(items)}] frames {done_px}/{N}  {done_px/el:.0f} fps  streamed {fobj.bytes_streamed/1e9:.0f}GB restarts {fobj.restarts}  {el:.0f}s', flush=True)
    else:
        _, buf = d.id.read_direct_chunk((r0,) + (0,) * (d.ndim - 1))
        arr = np.frombuffer(buf, dtype=d.dtype).reshape((rows,) + d.shape[1:])[: r1 - r0]
        small[c][r0:r1] = arr
print(f'pass done in {time.time()-t1:.0f}s; restarts {fobj.restarts}; streamed {fobj.bytes_streamed/1e9:.0f}GB', flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out_cache)), exist_ok=True)
np.savez(a.out_cache, z=z, ep_idx=ep_idx_full.astype(np.int32), step_idx=step_idx_full.astype(np.int32),
         ep_len=ep_len.astype(np.int32), ep_off=ep_off.astype(np.int64))
print('saved cache', a.out_cache, flush=True)

# ---- eval subset h5 (episodes renumbered 0..len(eval_eps)-1) --------------------------
def frames_of(e):
    s, l = ep_off[e], ep_len[e]
    out = []
    for r in range(s, s + l):
        c0 = (r // dpx.chunks[0]) * dpx.chunks[0]
        out.append(px_keep[c0][r - c0])
    return out

new_idx = {e: i for i, e in enumerate(eval_eps)}
with HDF5Writer(a.out_subset, mode='overwrite') as w:
    for e in eval_eps:
        s, l = ep_off[e], ep_len[e]
        ep = {c: list(small[c][s:s + l]) for c in small_cols if c not in ('ep_idx',)}
        ep['ep_idx'] = [np.int32(new_idx[e])] * l
        ep['pixels'] = frames_of(e)
        w.write_episode(ep)
json.dump({k: [[new_idx[e], s] for e, s in v] for k, v in tasks.items()}, open(a.out_tasks, 'w'))
print('saved subset', a.out_subset, 'tasks', a.out_tasks, f'total {time.time()-t0:.0f}s', flush=True)
