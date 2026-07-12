"""Convert recorded guardian demos (record_guardian_demos.py npz) into a
LeRobotDataset for SmolVLA finetuning.

Feature names match lerobot/smolvla_base's config exactly:
  observation.state (6), observation.images.camera1 (parent_view),
  camera2 (overhead), camera3 (zeros -- the policy expects three slots),
  action (6, absolute joint targets rad).
"""
import argparse, glob, os, shutil, sys
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--demos", default="runs/guardian_demos")
ap.add_argument("--root", default="runs/lerobot_guardian")
ap.add_argument("--repo-id", default="local/guardian_demos")
ap.add_argument("--limit", type=int, default=0, help="0 = all episodes")
ap.add_argument("--fresh", action="store_true")
a = ap.parse_args()

if a.fresh and os.path.exists(a.root):
    shutil.rmtree(a.root)
resume_from = 0

H = W = 256
features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "observation.images.camera1": {"dtype": "video", "shape": (H, W, 3),
                                   "names": ["height", "width", "channels"]},
    "observation.images.camera2": {"dtype": "video", "shape": (H, W, 3),
                                   "names": ["height", "width", "channels"]},
    "observation.images.camera3": {"dtype": "video", "shape": (H, W, 3),
                                   "names": ["height", "width", "channels"]},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}
if os.path.exists(os.path.join(a.root, "meta", "info.json")) and not a.fresh:
    ds = LeRobotDataset(a.repo_id, root=a.root)
    resume_from = ds.num_episodes
    print(f"[resume] dataset has {resume_from} episodes; continuing", flush=True)
else:
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=30, features=features, root=a.root,
                           robot_type="so101", use_videos=True,
                           image_writer_processes=4, image_writer_threads=4,
                           metadata_buffer_size=1,   # flush per episode: the default (10) defers
                                                     # to __del__, which runs after pyarrow teardown
                           vcodec="h264",            # default libsvtav1 encodes ~90s/episode (3h
                                                     # for 150 eps); h264/x264 is 10-50x faster
                           )
eps = sorted(glob.glob(os.path.join(a.demos, "ep_*.npz")))
if a.limit:
    eps = eps[: a.limit]
eps = eps[resume_from:]
zeros = np.zeros((H, W, 3), np.uint8)
for i, p in enumerate(eps):
    z = np.load(p)
    task = str(z["instruction"]) or "move a cube on the table"
    T = z["action"].shape[0]
    for t in range(T):
        ds.add_frame({
            "observation.state": z["qpos"][t],
            "observation.images.camera1": z["parent_view"][t],
            "observation.images.camera2": z["overhead"][t],
            "observation.images.camera3": zeros,
            "action": z["action"][t],
            "task": task,
        })
    ds.save_episode()
    if (i + 1) % 10 == 0:
        print(f"[convert] {i+1}/{len(eps)} episodes", flush=True)
print(f"[done] {len(eps)} episodes -> {a.root}", flush=True)
