"""Offline value initialisation (App. B.1): n-step TD + HER + expectile-Huber on cached latents."""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from rp1.cache import LatentCache
from rp1.critic import make_critic, TDSampler, CriticTrainer
from rp1.configs import DOMAINS

p = argparse.ArgumentParser()
p.add_argument('--domain', required=True)
p.add_argument('--cache', required=True)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=None)
p.add_argument('--train_eps', type=int, nargs=2, default=[0, 8000])
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
cfg = DOMAINS[a.domain]; ov = cfg['offline_value']
cache = LatentCache(a.cache, block=cfg.get('block', 5))
std = cfg['standardize_latents']
critic = make_critic(cache.z.shape[1], 256, 128, 2, cache.mean if std else None, cache.std if std else None).cuda()
sampler = TDSampler(cache, range(*a.train_eps), n_step=ov['n_step'], gamma=ov['gamma'], cross_prob=0.3, seed=a.seed)
steps = a.steps or ov['steps']
tr = CriticTrainer(critic, sampler, gamma=ov['gamma'], tau=ov['expectile'], lr=ov['lr'], polyak=0.005, total_steps=steps)
t0 = time.time()
for i in range(steps):
    info = tr.step(ov['batch'])
    if i % 500 == 0 or i == steps - 1:
        print(f'[{i:5d}] loss {info["loss"]:.4f} v {info["v_mean"]:.2f} y {info["y_mean"]:.2f} exact {info["exact_frac"]:.2f}  {time.time()-t0:.0f}s', flush=True)

# --- diagnostics on held-out episodes: predicted V vs true in-episode block offset
with torch.no_grad():
    eps = np.arange(8000, cache.n_ep)
    rng = np.random.default_rng(0)
    rows = []
    for d in range(1, 21):
        e = rng.choice(eps, 2000); L = cache.ep_len[e]
        ok = (L - 1) >= d * cache.block
        e = e[ok]; L = L[ok]
        t = (rng.random(len(e)) * (L - 1 - d * cache.block)).astype(np.int64)
        r0 = cache.ep_off[e] + t; r1 = r0 + d * cache.block
        v = tr.target(cache.z[torch.as_tensor(r0)], cache.z[torch.as_tensor(r1)])
        rows.append((d, v.mean().item(), v.std().item()))
    print('held-out V(z_t, z_{t+d blocks}) by d:', ' '.join(f'{d}:{m:.1f}±{s:.1f}' for d, m, s in rows))
    # cross-episode (unrelated) pairs
    e1 = rng.choice(eps, 2000); e2 = rng.choice(eps, 2000)
    r0 = cache.ep_off[e1] + (rng.random(2000) * cache.ep_len[e1]).astype(np.int64)
    r1 = cache.ep_off[e2] + (rng.random(2000) * cache.ep_len[e2]).astype(np.int64)
    v = tr.target(cache.z[torch.as_tensor(r0)], cache.z[torch.as_tensor(r1)])
    print(f'cross-episode V mean {v.mean().item():.1f} ± {v.std().item():.1f}')
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({'critic': tr.critic.state_dict(), 'target': tr.target.state_dict(), 'cfg': ov, 'domain': a.domain,
            'standardize': std, 'in_dim': cache.z.shape[1]}, a.out)
print('saved', a.out)
