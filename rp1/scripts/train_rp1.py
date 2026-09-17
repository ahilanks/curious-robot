"""Train the RP1 refiner (App. B.2) with a co-trained critic initialised from the offline value."""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from rp1.wm import load_lewm
from rp1.cache import LatentCache
from rp1.critic import make_critic, TDSampler, CriticTrainer
from rp1.planner import Refiner, RP1Planner, ActorSampler, RP1Trainer
from rp1.configs import DOMAINS, COMMON

p = argparse.ArgumentParser()
p.add_argument('--domain', required=True)
p.add_argument('--h', type=int, required=True, help='goal horizon (primitive steps) selecting the actor config')
p.add_argument('--cache', required=True)
p.add_argument('--critic', required=True, help='offline value checkpoint')
p.add_argument('--ckpt', required=True, help='LeWM checkpoint folder')
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=None)
p.add_argument('--train_eps', type=int, nargs=2, default=[0, 8000])
p.add_argument('--override', type=str, default='{}', help='json dict overriding actor cfg')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
cfg = DOMAINS[a.domain]; acfg = dict(cfg['actor'][a.h]); acfg.update(json.loads(a.override)); ccfg = cfg['critic']
if a.steps: acfg['steps'] = a.steps
acfg.update(td_batch=COMMON['td_batch'], critic_ratio=COMMON['critic_ratio'], critic_live_steps=ccfg['live_steps'],
            value_expansion=ccfg['value_expansion'])
print('actor cfg', acfg)

model = load_lewm(a.ckpt)
cache = LatentCache(a.cache, block=cfg.get('block', 5))
std = cfg['standardize_latents']
critic = make_critic(cache.z.shape[1], 256, 128, 2, cache.mean if std else None, cache.std if std else None).cuda()
ck = torch.load(a.critic, map_location='cuda')
critic.load_state_dict(ck['critic'])
td_sampler = TDSampler(cache, range(*a.train_eps), n_step=COMMON['n_step'], gamma=ccfg['gamma'], cross_prob=COMMON['cross_prob'], seed=a.seed + 100)
ct = CriticTrainer(critic, td_sampler, gamma=ccfg['gamma'], tau=ccfg['expectile'], tau_final=ccfg['expectile_final'],
                   lr=ccfg['lr'], lr_final=ccfg['lr_final'], polyak=COMMON['ema'], total_steps=ccfg['live_steps'])
ct.target.load_state_dict(ck['target'])  # EMA teacher starts from the offline target net

N = COMMON['horizon']; A = cfg['action_dim'] * COMMON['action_block']
refiner = Refiner(N * A, COMMON['hidden']).cuda()
planner = RP1Planner(model, ct.target, refiner, N=N, A=A, K=COMMON['K'], amax=acfg['amax'])
sampler = ActorSampler(cache, range(*a.train_eps), max_delta=acfg['max_delta'], cross_prob=COMMON['cross_prob'], seed=a.seed)
tr = RP1Trainer(planner, sampler, ct, acfg)
t0 = time.time(); hist = []
for i in range(acfg['steps']):
    info = tr.step()
    hist.append(info)
    if i % 100 == 0 or i == acfg['steps'] - 1:
        print(f'[{i:5d}] loss {info["loss"]:.3f} v0 {info["v0"]:.2f} -> vK {info["vK"]:.2f}  lr {info["lr"]:.1e} gn {info["gnorm"]:.2f} sat {info["amax_frac"]:.2f}'
              + (f' critic {info["critic_loss"]:.3f}' if 'critic_loss' in info else '') + f'  {time.time()-t0:.0f}s', flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({'refiner': refiner.state_dict(), 'critic_teacher': ct.target.state_dict(), 'critic': ct.critic.state_dict(),
            'actor_cfg': acfg, 'domain': a.domain, 'h': a.h, 'N': N, 'A': A, 'K': COMMON['K'], 'standardize': std,
            'in_dim': cache.z.shape[1], 'seed': a.seed, 'hist': hist[::10]}, a.out)
print('saved', a.out)
