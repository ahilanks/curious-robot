"""A Push-T eval set in the expert file's format, from random-action episodes - the from-scratch arms' own regime.

200 episodes x 100 steps from stock resets (random agent / T pose), actions N(0, 0.6^2) clipped (the arms' warm-up
and CEM-initial distribution) with the collection clip (agent target inside the arena), columns pixels / action /
state / proprio / episode_idx / step_idx + ep_len / ep_offset like pusht_expert_train.h5. Used with
`eval.dataset_name=pusht_random_eval`: start = a random step, goal = the state 25 steps later on that trajectory.

    .venv/bin/python gen_random_eval_set.py   ->  $STABLEWM_HOME/datasets/pusht_random_eval.h5
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_sim as ps  # noqa: E402


def main(n_eps=200, T=100, seed=12345):
    ps.single_thread()
    env = ps.make_env()
    rng = np.random.default_rng(seed)
    home = Path(os.environ.get("STABLEWM_HOME", Path(__file__).resolve().parent / "swm_home"))
    out = home / "datasets" / "pusht_random_eval.h5"
    rows = n_eps * (T + 1)
    with h5py.File(out, "w") as f:
        px = f.create_dataset("pixels", (rows, 224, 224, 3), np.uint8, chunks=(16, 224, 224, 3))
        act = f.create_dataset("action", (rows, 2), np.float32)
        st = f.create_dataset("state", (rows, 7), np.float32)
        pr = f.create_dataset("proprio", (rows, 4), np.float32)
        epi = f.create_dataset("episode_idx", (rows,), np.int64)
        stp = f.create_dataset("step_idx", (rows,), np.int64)
        f.create_dataset("ep_len", data=np.full(n_eps, T + 1, np.int32))
        f.create_dataset("ep_offset", data=(np.arange(n_eps) * (T + 1)).astype(np.int64))
        for e in range(n_eps):
            s0, f0 = ps.reset_state(env, 900_000 + e)
            acts = np.clip(rng.normal(0.0, 0.6, (T, 2)), -1, 1)
            frames = np.zeros((T,) + ps.FRAME, np.uint8)
            exe, sts, _ = ps.rollout(env, s0, acts, 1, frames)
            S = np.concatenate([s0[None], sts])                                  # (T+1, 10)
            st7 = np.concatenate([S[:, 0:2], S[:, 4:6], np.mod(S[:, 6:7], 2 * np.pi), S[:, 2:4]], 1)
            o = e * (T + 1)
            px[o] = f0
            px[o + 1:o + T + 1] = frames
            act[o:o + T] = exe
            act[o + T] = 0.0                                                     # terminal row (the expert file has no NaNs)
            st[o:o + T + 1] = st7
            pr[o:o + T + 1] = np.concatenate([st7[:, 0:2], st7[:, 5:7]], 1)
            epi[o:o + T + 1] = e
            stp[o:o + T + 1] = np.arange(T + 1)
    print("wrote", out, rows, "rows")


if __name__ == "__main__":
    main()
