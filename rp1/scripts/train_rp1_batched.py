"""Train all RP1 planner runs of a domain at once (rp1.batched): every (goal horizon h, seed)
pair is one run; runs sharing h share an actor config (a RunGroup). Writes one checkpoint per
run, {out_prefix}_h{h}_s{seed}.pt, in the format scripts/train_rp1.py writes (read by eval.py)."""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from rp1.wm import load_lewm
from rp1.cache import LatentCache
from rp1.critic import TDSampler
from rp1.planner import ActorSampler
from rp1.batched import BatchedRefiner, BatchedMRN, BatchedRP1Trainer, RunGroup, use_math_sdpa
from rp1.configs import DOMAINS, COMMON

p = argparse.ArgumentParser()
p.add_argument('--domain', required=True)
p.add_argument('--h', type=int, nargs='+', required=True, help='goal horizons (one run group each)')
p.add_argument('--cache', required=True)
p.add_argument('--critic', required=True)
p.add_argument('--ckpt', required=True)
p.add_argument('--out_prefix', required=True)
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--steps', type=int, default=None)
p.add_argument('--train_eps', type=int, nargs=2, default=[0, 8000])
p.add_argument('--override', type=str, default='{}', help='json dict overriding every actor cfg')
p.add_argument('--no_graph', action='store_true')
p.add_argument('--no_compile', action='store_true')
p.add_argument('--tf32', action='store_true', help='TF32 matmuls (~2x faster, ~1e-2 rel. gradient error); default exact fp32')
p.add_argument('--log_every', type=int, default=100)
a = p.parse_args()
torch.manual_seed(a.seeds[0]); np.random.seed(a.seeds[0])
if a.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.set_float32_matmul_precision('high')
use_math_sdpa()
cfg = DOMAINS[a.domain]; ccfg = dict(cfg['critic'])
ccfg.update(td_batch=COMMON['td_batch'], critic_ratio=COMMON['critic_ratio'], ema=COMMON['ema'])
acfgs = {}
for h in a.h:
    ac = dict(cfg['actor'][h]); ac.update(json.loads(a.override))
    if a.steps: ac['steps'] = a.steps
    acfgs[h] = ac
R = len(a.h) * len(a.seeds)
print('critic cfg', ccfg, '\nactor cfgs', acfgs, '\nseeds', a.seeds, 'runs', R, flush=True)

model = load_lewm(a.ckpt)
cache = LatentCache(a.cache, block=cfg.get('block', 5))
std = cfg['standardize_latents']
D = cache.z.shape[1]
ck = torch.load(a.critic, map_location='cuda')
critic = BatchedMRN(R, D, 256, 128, 2, cache.mean if std else None, cache.std if std else None).cuda()
teacher = BatchedMRN(R, D, 256, 128, 2, cache.mean if std else None, cache.std if std else None).cuda()
critic.load_single(ck['critic']); teacher.load_single(ck['target'])
N = COMMON['horizon']; A = cfg['action_dim'] * COMMON['action_block']
groups, td_s = [], []
for h in a.h:
    ac = acfgs[h]
    ref = BatchedRefiner(len(a.seeds), N * A, COMMON['hidden'], seeds=a.seeds).cuda()
    act_s = [ActorSampler(cache, range(*a.train_eps), max_delta=ac['max_delta'], cross_prob=COMMON['cross_prob'], seed=s) for s in a.seeds]
    groups.append(RunGroup(f'h{h}', ref, ac, act_s, a.seeds))
    td_s += [TDSampler(cache, range(*a.train_eps), n_step=COMMON['n_step'], gamma=ccfg['gamma'], cross_prob=COMMON['cross_prob'], seed=s + 100) for s in a.seeds]
tr = BatchedRP1Trainer(model, groups, critic, teacher, td_s, ccfg, N, A, COMMON['K'], use_graph=not a.no_graph, use_compile=not a.no_compile)
t0 = time.time(); hist = []
for i in range(tr.steps):
    info = tr.step()
    hist.append(info)
    if i % a.log_every == 0 or i == tr.steps - 1:
        print(f'[{i:5d}] J {info["loss"]:.3f} v0 {info["v0"]:.2f} -> vK {info["vK"]:.2f} per-run {[round(x, 2) for x in info["vK_per_run"]]}'
              f'  lr {["%.1e" % x for x in info["lr"]]} gn {info["gnorm"]:.2f} sat {info["amax_frac"]:.2f}'
              + (f' critic {info["critic_loss"]:.3f}' if 'critic_loss' in info else '') + f'  {time.time() - t0:.0f}s', flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
r = 0
for gi, (h, grp) in enumerate(zip(a.h, groups)):
    for si, s in enumerate(a.seeds):
        out = f'{a.out_prefix}_h{h}_s{s}.pt'
        torch.save({'refiner': grp.ref.export_single(si), 'critic_teacher': teacher.export_single(r), 'critic': critic.export_single(r),
                    'actor_cfg': acfgs[h], 'domain': a.domain, 'h': h, 'N': N, 'A': A, 'K': COMMON['K'], 'standardize': std,
                    'in_dim': D, 'seed': s, 'tf32': a.tf32,
                    'hist': [{k: v for k, v in hh.items() if k not in ('vK_per_run', 'lr')} | {'vK_run': hh['vK_per_run'][r], 'lr': hh['lr'][gi]} for hh in hist[::10]]}, out)
        print('saved', out, flush=True)
        r += 1
print(f'done {tr.steps} steps x {R} runs in {time.time() - t0:.0f}s')
