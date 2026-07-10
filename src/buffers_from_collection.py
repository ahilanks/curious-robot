"""Convert probe collections / ring-format state snapshots into the FLAT
save_buffer npz format that offline_train.load_buffer expects
(keys: pixels, proprio, action, r, d, is_start + env_lengths per stream).

Two modes:
  --collection runs/probe_contact_scale/collection.npz   (probe_contact_scale output)
      Each episode becomes its OWN stream (no cross-episode windows). The final
      frame of each episode (px[60]) is dropped -- rows are (obs_i, action_i) and
      the last obs has no action, same information loss as an online reset.
      proprio is zeros: this campaign trains with no_proprio=True, and the stage-1/2
      probes validated the zero-proprio encode path against the training band.
      r is zeros: safety reward was not recorded; offline SAC updates on these rows
      are junk but the consolidation target is the WM/predictor, which never sees r.
  --state runs/hf/pc_treat/state_latest.npz               (train.py ring snapshot)
      Unrolls each ring oldest-to-newest into a stream (anti-forgetting mix data).
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--collection", default=None)
ap.add_argument("--state", default=None)
ap.add_argument("--max-rings", type=int, default=None,
                help="state mode: only unroll the first N rings (mix-ratio / RAM control; "
                     "offline_train pads every stream to the LONGEST one, so a few 3750-row "
                     "rings next to sixty-row episode streams multiplies the allocation)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert (a.collection is None) != (a.state is None), "exactly one of --collection/--state"

if a.collection:
    z = np.load(a.collection)
    px, act = z["px"], z["act"]                     # (E, D+1, H, W, 3), (E, D, a_dim)
    E, D = act.shape[0], act.shape[1]
    pixels = px[:, :D].reshape(E * D, *px.shape[2:])
    action = act.reshape(E * D, -1).astype(np.float32)
    n_dof = action.shape[-1] // 5
    proprio = np.zeros((E * D, 3 * n_dof), np.float32)
    r = np.zeros(E * D, np.float32)
    d = np.zeros(E * D, np.float32)
    is_start = np.zeros(E * D, bool)
    is_start[::D] = True
    env_lengths = np.full(E, D, np.int64)
else:
    z = np.load(a.state)
    C = z["pixels"].shape[1]
    streams = {k: [] for k in ("pixels", "proprio", "action", "r", "d", "is_start")}
    env_lengths = []
    n_rings = z["count"].shape[0] if a.max_rings is None else min(a.max_rings, z["count"].shape[0])
    for e in range(n_rings):
        n, head = int(z["count"][e]), int(z["head"][e])
        idx = (head - n + np.arange(n)) % C
        streams["pixels"].append(z["pixels"][e, idx])
        streams["proprio"].append(z["proprio"][e, idx])
        streams["action"].append(z["action"][e, idx])
        streams["r"].append(z["r"][e, idx])
        streams["d"].append(z["done"][e, idx])
        streams["is_start"].append(z["is_start"][e, idx])
        env_lengths.append(n)
    pixels = np.concatenate(streams["pixels"])
    proprio = np.concatenate(streams["proprio"])
    action = np.concatenate(streams["action"])
    r = np.concatenate(streams["r"])
    d = np.concatenate(streams["d"])
    is_start = np.concatenate(streams["is_start"])
    env_lengths = np.array(env_lengths, np.int64)

np.savez_compressed(a.out, pixels=pixels, proprio=proprio, action=action,
                    r=r, d=d, is_start=is_start, env_lengths=env_lengths)
print(f"[out] {a.out}: {len(env_lengths)} stream(s), {int(env_lengths.sum())} transitions", flush=True)
