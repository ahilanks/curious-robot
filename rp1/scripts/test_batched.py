"""Equivalence test: batched (surrogate + CUDA graph) refiner gradient == single-run (double backward) gradient."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rp1.wm import load_lewm
from rp1.cache import LatentCache
from rp1.critic import make_critic
from rp1.planner import Refiner, RP1Planner
from rp1.batched import BatchedRefiner, BatchedMRN, ValueGrad
torch.manual_seed(0)
model = load_lewm('downloads/ckpt/lewm-tworooms')
cache = LatentCache('runs/cache/tworoom.npz')
ck = torch.load('runs/tworoom/critic_offline.pt', map_location='cuda')
N, A, K, B, lam, amax = 5, 10, 8, 32, 0.3, 1.8
# --- single-run reference
crit1 = make_critic(192, 256, 128, 2).cuda(); crit1.load_state_dict(ck['target']); crit1.requires_grad_(False)
ref1 = Refiner(N * A, 512).cuda()
planner = RP1Planner(model, crit1, ref1, N=N, A=A, K=K, amax=amax)
rng = np.random.default_rng(0)
r0 = torch.as_tensor(rng.integers(0, len(cache.z), B)); r1 = torch.as_tensor(rng.integers(0, len(cache.z), B))
z0, zg = cache.z[r0], cache.z[r1]
out = planner.plan(z0, zg, train=True)
vals = torch.stack(out['values'][1:], 0)
loss = out['values'][-1].mean() + lam * vals.mean()
loss.backward()
g_ref = {k: p.grad.clone() for k, p in ref1.named_parameters()}
# --- batched (R=2: run 0 = same weights, run 1 = different weights) with graph
R = 2
refb = BatchedRefiner(R, N * A, 512, seeds=[0, 1]).cuda()
sd = ref1.state_dict()
for j, l in enumerate((refb.l1, refb.l2, refb.l3)):
    l.load_single(sd[f'net.{2*j}.weight'], sd[f'net.{2*j}.bias'], r=0)
critb = BatchedMRN(R, 192, 256, 128, 2).cuda(); critb.load_single(ck['target']); critb.requires_grad_(False)
from rp1.batched import use_math_sdpa; use_math_sdpa()
vg = ValueGrad(model, critb, R, B, N, A, 192, use_graph=True, use_compile=True)
z0b = torch.stack([z0, z0.flip(0)]); zgb = torch.stack([zg, zg.flip(0)])
a = torch.zeros(R, B, N, A, device='cuda'); surrogate = 0.0; values = []
for k in range(K + 1):
    v, g = vg(z0b, zgb, a); values.append(v)
    if k >= 1:
        surrogate = surrogate + (lam / K + (1.0 if k == K else 0.0)) * (g * a).sum((2, 3)).mean(1).sum()
    if k == K: break
    delta = refb(a.flatten(2), v, g.flatten(2)).view(R, B, N, A)
    a = (a + delta).clamp(-amax, amax)
surrogate.backward()
vals_b = torch.stack(values, 0)
loss_b = vals_b[-1, 0].mean() + lam * vals_b[1:, 0].mean()
print(f'loss single {loss.item():.6f}  batched(run0) {loss_b.item():.6f}')
for j, (name, l) in enumerate(zip(('net.0', 'net.2', 'net.4'), (refb.l1, refb.l2, refb.l3))):
    gw = l.weight.grad[0].t(); gb = l.bias.grad[0, 0]
    for nm, gb_, gr in ((name + '.weight', gw, g_ref[name + '.weight']), (name + '.bias', gb, g_ref[name + '.bias'])):
        rel = (gb_ - gr).norm() / (gr.norm() + 1e-12)
        print(f'  {nm:14s} |g_ref| {gr.norm():.4e}  rel diff {rel.item():.2e}')
# graph replay determinism / independence of run 1 from run 0
v1, g1 = vg(z0b, zgb, torch.zeros_like(a)); v2, g2 = vg(z0b, zgb, torch.zeros_like(a))
print('replay deterministic:', torch.allclose(v1, v2), torch.allclose(g1, g2), ' v(run1)==v(run0 flipped):', torch.allclose(v1[1], v1[0].flip(0), atol=1e-4))
