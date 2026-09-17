"""Does the learned critic capture temporal reachability across the wall better than latent L2?
Held-out episodes; pairs (t1<t2) within an episode; ground truth = door-routed path length."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py, hdf5plugin  # noqa
from scipy.stats import spearmanr
from rp1.cache import LatentCache
from rp1.critic import make_critic

p = argparse.ArgumentParser()
p.add_argument('--cache', default='runs/cache/tworoom.npz')
p.add_argument('--h5', default='swm_home/datasets/tworoom/tworoom.h5')
p.add_argument('--critic', default='runs/tworoom/critic_offline.pt')
p.add_argument('--n', type=int, default=30000)
a = p.parse_args()
from stable_worldmodel.envs.two_room.env import TwoRoomEnv
WALL = float(TwoRoomEnv.WALL_CENTER)
cache = LatentCache(a.cache)
ck = torch.load(a.critic, map_location='cuda')
critic = make_critic(ck['in_dim'], 256, 128, 2).cuda(); critic.load_state_dict(ck['target']); critic.eval()
with h5py.File(a.h5, 'r') as f:
    obs = f['observation'][:]          # agent(2) target(2) door_centers(3x2)
pos = cache.extras['pos_agent']
door = obs[:, 4:6]
rng = np.random.default_rng(0)
eps = rng.integers(8000, cache.n_ep, a.n)
L = cache.ep_len[eps]
t1 = (rng.random(a.n) * (L - 6)).astype(int)
t2 = t1 + 1 + (rng.random(a.n) * (L - 1 - t1 - 1)).astype(int)
r1, r2 = cache.ep_off[eps] + t1, cache.ep_off[eps] + t2
p1, p2, d = pos[r1], pos[r2], door[r1]
euclid = np.linalg.norm(p1 - p2, axis=1)
cross = (p1[:, 0] - WALL) * (p2[:, 0] - WALL) < 0
path = np.where(cross, np.linalg.norm(p1 - d, axis=1) + np.linalg.norm(d - p2, axis=1), euclid)
with torch.no_grad():
    z1, z2 = cache.z[torch.as_tensor(r1)], cache.z[torch.as_tensor(r2)]
    V = critic(z1, z2).cpu().numpy()
    L2 = (z1 - z2).pow(2).sum(-1).cpu().numpy()
dt = (t2 - t1) / cache.block
print(f'{a.n} held-out pairs, {cross.mean()*100:.1f}% cross-wall; wall x={WALL}')
for name, s in [('V (critic)', V), ('latent L2', L2), ('euclid px', euclid)]:
    print(f'  spearman[{name:11s}] vs door-path {spearmanr(s, path)[0]:.3f} | vs euclid {spearmanr(s, euclid)[0]:.3f} | vs elapsed blocks {spearmanr(s, dt)[0]:.3f}')
print('matched-euclid bins: same-room vs cross-wall (mean V | mean latent L2 | mean path px)')
for lo, hi in [(20, 40), (40, 60), (60, 80), (80, 100), (100, 130)]:
    m = (euclid >= lo) & (euclid < hi)
    for lab, sel in [('same', m & ~cross), ('cross', m & cross)]:
        if sel.sum() > 20:
            print(f'  euclid [{lo:3d},{hi:3d}) {lab:5s} n={sel.sum():5d}  V {V[sel].mean():5.2f}  L2 {L2[sel].mean():6.2f}  path {path[sel].mean():6.1f}')
