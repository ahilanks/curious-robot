"""World-model-only diagnostic for a trained RP1 checkpoint: on held-out (z_t, z_{t+h}) pairs,
compare the critic value / latent distance of (a) the dataset's own action blocks, (b) the RP1 plan,
(c) a zero plan. Large gaps between v(RP1) and v(dataset) with RP1 << dataset suggest WM exploitation."""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py, hdf5plugin  # noqa
from rp1.wm import load_lewm, rollout_latent, fit_action_scaler
from rp1.cache import LatentCache
from rp1.critic import make_critic
from rp1.planner import Refiner, RP1Planner
from rp1.configs import DOMAINS, COMMON

p = argparse.ArgumentParser()
p.add_argument('--domain', required=True); p.add_argument('--h', type=int, default=25)
p.add_argument('--cache', required=True); p.add_argument('--h5', required=True); p.add_argument('--ckpt', required=True)
p.add_argument('--rp1', required=True); p.add_argument('--n', type=int, default=256)
a = p.parse_args()
cfg = DOMAINS[a.domain]
model = load_lewm(a.ckpt)
cache = LatentCache(a.cache)
ck = torch.load(a.rp1, map_location='cuda')
crit = make_critic(ck['in_dim'], 256, 128, 2).cuda(); crit.load_state_dict(ck['critic_teacher']); crit.eval().requires_grad_(False)
ref = Refiner(ck['N'] * ck['A'], COMMON['hidden']).cuda(); ref.load_state_dict(ck['refiner']); ref.eval()
planner = RP1Planner(model, crit, ref, N=ck['N'], A=ck['A'], K=ck['K'], amax=ck['actor_cfg']['amax'])
with h5py.File(a.h5, 'r') as f:
    act = f['action'][:]
scaler = fit_action_scaler(act)
act_z = scaler.transform(np.nan_to_num(act)).astype(np.float32)
rng = np.random.default_rng(0)
eps = rng.integers(8000, cache.n_ep, a.n); L = cache.ep_len[eps]
t = (rng.random(a.n) * (L - 1 - a.h)).astype(int)
r0 = cache.ep_off[eps] + t; rg = r0 + a.h
z0, zg = cache.z[torch.as_tensor(r0)], cache.z[torch.as_tensor(rg)]
N, A = ck['N'], ck['A']; blk = A // cfg['action_dim']
plan_ds = torch.as_tensor(np.stack([act_z[r:r + N * blk].reshape(N, A) for r in r0])).cuda()
with torch.no_grad():
    out = planner.plan(z0, zg, train=False)
    plan_rp1 = out['plans'][-1]
    rows = []
    for name, plan in [('dataset actions', plan_ds), ('RP1 plan', plan_rp1), ('zero plan', torch.zeros_like(plan_ds))]:
        zN = rollout_latent(model, z0, plan)
        v = crit(zN, zg); l2 = (zN - zg).pow(2).sum(-1); l2_true = (zN - cache.z[torch.as_tensor(rg)]).pow(2).sum(-1)
        rows.append((name, v.mean().item(), v.median().item(), l2.mean().item(), plan.abs().mean().item(), (plan.abs() > ck['actor_cfg']['amax'] - 1e-3).float().mean().item()))
    v_start = crit(z0, zg); v_goal = crit(zg, zg)
print(f'{a.n} held-out pairs, h={a.h}: V(z0,zg) mean {v_start.mean():.2f}   V(zg,zg) mean {v_goal.mean():.2f}   RP1 v_k: {[round(v.mean().item(), 2) for v in out["values"]]}')
print(f"{'plan':16s} {'V(zN,zg)':>9s} {'median':>7s} {'||zN-zg||^2':>12s} {'mean|a|':>8s} {'sat':>5s}")
for r in rows: print(f'{r[0]:16s} {r[1]:9.2f} {r[2]:7.2f} {r[3]:12.1f} {r[4]:8.2f} {r[5]:5.2f}')
# per-block action magnitude of RP1 vs dataset
print('per-block mean|a|: dataset', [round(x, 2) for x in plan_ds.abs().mean((0, 2)).tolist()], ' RP1', [round(x, 2) for x in plan_rp1.abs().mean((0, 2)).tolist()])
print('corr(RP1 plan, dataset plan) per block:', [round(torch.corrcoef(torch.stack([plan_rp1[:, i].flatten(), plan_ds[:, i].flatten()]))[0, 1].item(), 2) for i in range(N)])
