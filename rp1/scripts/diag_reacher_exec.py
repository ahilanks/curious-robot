"""Execute plans open-loop in the real Reacher env from dataset start states and measure the
final per-joint error to the goal qpos (h=25): RP1 plan vs dataset actions vs zero actions.
Tells whether RP1 failures are near-misses (precision) or gross misses (WM exploitation)."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np, torch, h5py, hdf5plugin  # noqa
import stable_worldmodel as swm
from stable_worldmodel.world.world import _extract_init_goal, _apply_callables
from rp1.configs import DOMAINS, COMMON
from rp1.evalproto import make_world, sample_eval_tasks
from rp1.wm import load_lewm, fit_action_scaler, encode_pixels, normalize_uint8_batch
from rp1.critic import make_critic
from rp1.planner import Refiner, RP1Planner

p = argparse.ArgumentParser()
p.add_argument('--rp1', required=True); p.add_argument('--n', type=int, default=50); p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
cfg = DOMAINS['reacher']; h = 25
ds = swm.data.load_dataset('reacher', cache_dir=os.environ.get('STABLEWM_HOME'), keys_to_cache=['action', 'ep_idx', 'step_idx'])
acts = ds.get_col_data('action'); scaler = fit_action_scaler(acts)
eps, starts, rows = sample_eval_tasks(ds, h, a.seed, a.n)
model = load_lewm('downloads/ckpt/lewm-reacher')
ck = torch.load(a.rp1, map_location='cuda')
crit = make_critic(ck['in_dim'], 256, 128, 2).cuda(); crit.load_state_dict(ck['critic_teacher']); crit.eval().requires_grad_(False)
ref = Refiner(ck['N'] * ck['A'], COMMON['hidden']).cuda(); ref.load_state_dict(ck['refiner']); ref.eval()
planner = RP1Planner(model, crit, ref, N=ck['N'], A=ck['A'], K=ck['K'], amax=ck['actor_cfg']['amax'])
w = make_world(cfg, a.n, h)
init_state, goal_state, _ = _extract_init_goal(ds, eps, starts, h)

def setup():
    w.reset(seed=init_state.get('seed'))
    merged = {**init_state, **goal_state}
    for i in range(a.n):
        _apply_callables(w.envs.envs[i].unwrapped, cfg['callables'], {k: v[i] for k, v in merged.items()})
    return np.stack([w.envs.envs[i].unwrapped.env.physics.render(224, 224, camera_id=0) for i in range(a.n)])

def run_open_loop(raw_actions):  # (n, T, 2) raw actions -> final per-joint abs error (n,2), first-hit within T at tau 0.1/0.05
    setup()
    goal_q = goal_state['goal_qpos'][:, :2]
    hit01 = np.zeros(a.n, bool); hit005 = np.zeros(a.n, bool)
    for t in range(raw_actions.shape[1]):
        for i in range(a.n):
            w.envs.envs[i].step(raw_actions[i, t].astype(np.float32))
            q = w.envs.envs[i].unwrapped.env.physics.data.qpos[:2].copy()
            err = np.abs(q - goal_q[i])
            hit01[i] |= np.all(err < 0.1); hit005[i] |= np.all(err < 0.05)
    errs = np.stack([np.abs(w.envs.envs[i].unwrapped.env.physics.data.qpos[:2] - goal_q[i]) for i in range(a.n)])
    return errs, hit01, hit005

goal_q = goal_state['goal_qpos'][:, :2]
frames = setup()
goal_frames = np.stack([ds.get_col_data('pixels')[r + h] if False else h5py.File('swm_home/datasets/reacher/reacher.h5', 'r')['pixels'][r + h] for r in rows])
with torch.no_grad():
    z0 = encode_pixels(model, normalize_uint8_batch(torch.from_numpy(frames).cuda()))
    zg = encode_pixels(model, normalize_uint8_batch(torch.from_numpy(goal_frames).cuda()))
    out = planner.plan(z0, zg, train=False)
plan = out['plans'][-1].detach().cpu().numpy().reshape(a.n, h, 2)  # (n, 25, 2) z-scored
rp1_raw = scaler.inverse_transform(plan.reshape(-1, 2)).reshape(a.n, h, 2)
ds_raw = np.stack([np.nan_to_num(acts[r:r + h]) for r in rows]).astype(np.float64)
zero_raw = np.zeros_like(ds_raw)
print(f'n={a.n} tasks seed {a.seed}; RP1 v_k (WM): {[round(v.mean().item(), 2) for v in out["values"]]}')
print(f"{'plan':18s} {'final |err| j1':>14s} {'j2':>7s} {'median max':>11s} {'hit@0.1':>8s} {'hit@0.05':>9s} {'mean|raw a|':>12s}")
for name, ra in [('dataset actions', ds_raw), ('RP1 plan', rp1_raw), ('zero actions', zero_raw)]:
    errs, h1, h05 = run_open_loop(ra)
    print(f'{name:18s} {errs[:, 0].mean():14.3f} {errs[:, 1].mean():7.3f} {np.median(errs.max(1)):11.3f} {h1.mean()*100:8.1f} {h05.mean()*100:9.1f} {np.abs(ra).mean():12.3f}')
    if name == 'RP1 plan':
        mx = errs.max(1); print('   RP1 final max-joint error distribution:', np.round(np.percentile(mx, [10, 25, 50, 75, 90]), 3), ' frac>0.3rad:', (mx > 0.3).mean().round(2))
w.close()

# ---- WM error under each plan: encode the REAL final frame and compare with the WM's predicted terminal latent
from rp1.wm import rollout_latent
w = make_world(cfg, a.n, h)
init_state, goal_state, _ = _extract_init_goal(ds, eps, starts, h)
def run_and_encode(raw_actions):
    setup()
    for t in range(raw_actions.shape[1]):
        for i in range(a.n):
            w.envs.envs[i].step(raw_actions[i, t].astype(np.float32))
    fr = np.stack([w.envs.envs[i].unwrapped.env.physics.render(224, 224, camera_id=0) for i in range(a.n)])
    with torch.no_grad():
        return encode_pixels(model, normalize_uint8_batch(torch.from_numpy(fr).cuda()))
print(f"\n{'plan':18s} {'||zN_pred-zg||^2':>16s} {'||zN_pred-z_real||^2':>20s} {'||z_real-zg||^2':>16s} {'V(zN_pred,zg)':>13s} {'V(z_real,zg)':>12s}")
for name, ra, zplan in [('dataset actions', ds_raw, torch.as_tensor(scaler.transform(ds_raw.reshape(-1, 2)).reshape(a.n, h // 5, 10), dtype=torch.float32).cuda()),
                        ('RP1 plan', rp1_raw, out['plans'][-1].detach())]:
    z_real = run_and_encode(ra)
    with torch.no_grad():
        zN = rollout_latent(model, z0, zplan)
        print(f'{name:18s} {(zN - zg).pow(2).sum(-1).mean():16.1f} {(zN - z_real).pow(2).sum(-1).mean():20.1f} {(z_real - zg).pow(2).sum(-1).mean():16.1f} {crit(zN, zg).mean():13.2f} {crit(z_real, zg).mean():12.2f}')
w.close()
