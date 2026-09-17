"""Evaluate planners under the paper protocol. Appends one JSON line per (planner, eval seed) to --out."""
import argparse, os, sys, time, json
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, hdf5plugin  # noqa
import stable_worldmodel as swm
from rp1.baselines import CEMSolver, MPPISolver, GradientSolverFixed as GradientSolver
from rp1.wm import load_lewm, fit_action_scaler, LatentCostModel
from rp1.critic import make_critic
from rp1.planner import Refiner, RP1Planner, RP1Solver
from rp1.configs import DOMAINS, COMMON
from rp1.evalproto import sample_eval_tasks, NoopPolicy, ReplayPolicy, make_world, make_wm_policy, run_eval

p = argparse.ArgumentParser()
p.add_argument('--domain', required=True)
p.add_argument('--h', type=int, required=True)
p.add_argument('--planner', required=True, choices=['rp1', 'cem', 'mppi', 'adam', 'noop', 'replay', 'random'])
p.add_argument('--objective', default='latent', choices=['latent', 'value'])
p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
p.add_argument('--n', type=int, default=50)
p.add_argument('--ckpt', required=True, help='LeWM checkpoint folder')
p.add_argument('--dataset', default=None, help='override dataset name/path (e.g. an eval subset h5)')
p.add_argument('--critic', default=None, help='offline critic ckpt (value objective)')
p.add_argument('--rp1', nargs='*', default=[], help='RP1 checkpoints (one per planner seed)')
p.add_argument('--solver_batch', type=int, default=10)
p.add_argument('--out', required=True)
p.add_argument('--tag', default='')
p.add_argument('--reacher_tau', type=float, default=None)
p.add_argument('--video', default=None)
p.add_argument('--first_heldout_ep', type=int, default=8000)
p.add_argument('--tasks', default=None, help='tasks.json from cache_from_tar.py (subset dataset mode)')
a = p.parse_args()
TASKS = json.load(open(a.tasks)) if a.tasks else None

cfg = DOMAINS[a.domain]
ds = swm.data.load_dataset(a.dataset or cfg['dataset'], cache_dir=os.environ.get('STABLEWM_HOME'), keys_to_cache=['action', 'ep_idx', 'step_idx'])
scaler = fit_action_scaler(ds.get_col_data('action'))
A = cfg['action_dim']
budget = 2 * a.h
world = make_world(cfg, a.n, budget)
if a.reacher_tau is not None:
    from rp1.evalproto import set_reacher_tau, reacher_taus
    set_reacher_tau(world, a.reacher_tau)

model = load_lewm(a.ckpt) if a.planner in ('rp1', 'cem', 'mppi', 'adam') else None
critic = None
if a.planner in ('cem', 'mppi', 'adam') and a.objective == 'value':
    ck = torch.load(a.critic, map_location='cuda')
    critic = make_critic(ck['in_dim'], 256, 128, 2).cuda()
    critic.load_state_dict(ck['target'])
    critic.eval().requires_grad_(False)


def build_policy(seed, rows, rp1_ckpt=None):
    if a.planner == 'noop':
        return NoopPolicy(A)
    if a.planner == 'random':
        return swm.policy.RandomPolicy(seed=seed)
    if a.planner == 'replay':
        return ReplayPolicy(ds, rows, A)
    if a.planner == 'rp1':
        ck = torch.load(rp1_ckpt, map_location='cuda')
        crit = make_critic(ck['in_dim'], 256, 128, 2).cuda(); crit.load_state_dict(ck['critic_teacher']); crit.eval().requires_grad_(False)
        ref = Refiner(ck['N'] * ck['A'], COMMON['hidden']).cuda(); ref.load_state_dict(ck['refiner']); ref.eval()
        planner = RP1Planner(model, crit, ref, N=ck['N'], A=ck['A'], K=ck['K'], amax=ck['actor_cfg']['amax'])
        return make_wm_policy(RP1Solver(planner), scaler)
    cost = LatentCostModel(model, a.objective, critic)
    if a.planner == 'cem':
        solver = CEMSolver(cost, batch_size=a.solver_batch, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device='cuda', seed=seed)
    elif a.planner == 'mppi':
        solver = MPPISolver(cost, batch_size=a.solver_batch, num_samples=300, var_scale=1.0, n_steps=30, topk=30, temperature=0.5, device='cuda', seed=seed)
    elif a.planner == 'adam':
        ns, nsteps = (100, 30) if a.domain == 'tworoom' else (300, 10)
        solver = GradientSolver(cost, n_steps=nsteps, batch_size=a.solver_batch, var_scale=1.0, num_samples=ns, device='cuda', seed=seed,
                                optimizer_cls=torch.optim.AdamW, optimizer_kwargs={'lr': 0.1})
    return make_wm_policy(solver, scaler)


rp1_list = a.rp1 if a.planner == 'rp1' else [None]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
for rp1_ckpt in rp1_list:
    for seed in a.seeds:
        if TASKS is not None:
            pairs = TASKS[f'{a.h}_{seed}']
            eps = [e for e, _ in pairs]; starts = [s for _, s in pairs]
            rows = np.array([int(ds.offsets[e]) + s for e, s in pairs])
        else:
            eps, starts, rows = sample_eval_tasks(ds, a.h, seed, a.n, a.first_heldout_ep)
        policy = build_policy(seed, rows, rp1_ckpt)
        t0 = time.time()
        res = run_eval(world, policy, ds, eps, starts, a.h, cfg['callables'], video=a.video)
        dt = time.time() - t0
        if a.reacher_tau is not None:
            taus = set(reacher_taus(world))
            assert taus == {a.reacher_tau}, f'qpos_threshold changed during evaluate(): {taus}'
        rec = dict(domain=a.domain, h=a.h, planner=a.planner, objective=a.objective if a.planner in ('cem', 'mppi', 'adam') else None,
                   eval_seed=seed, rp1_ckpt=rp1_ckpt, tag=a.tag, n=a.n, success_rate=res['success_rate'],
                   successes=[bool(x) for x in res['episode_successes']], time_s=dt, reacher_tau=a.reacher_tau)
        print(json.dumps({k: v for k, v in rec.items() if k != 'successes'}), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
world.close()
