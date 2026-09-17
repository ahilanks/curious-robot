"""Micro-benchmark of the (rollout -> critic -> d/da) call that dominates RP1 training."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rp1.wm import load_lewm, rollout_latent
from rp1.cache import LatentCache
from rp1.critic import make_critic
from rp1.batched import BatchedMRN, ValueGrad
torch.backends.cuda.matmul.allow_tf32 = True; torch.set_float32_matmul_precision('high')
R, B, N, A, D = int(sys.argv[1]) if len(sys.argv) > 1 else 6, 128, 5, 10, 192
model = load_lewm('downloads/ckpt/lewm-tworooms')
ck = torch.load('runs/tworoom/critic_offline.pt', map_location='cuda')
crit = BatchedMRN(R, D, 256, 128, 2).cuda(); crit.load_single(ck['target']); crit.requires_grad_(False)
z0 = torch.randn(R, B, D, device='cuda'); zg = torch.randn(R, B, D, device='cuda'); a = torch.randn(R, B, N, A, device='cuda') * 0.5

def bench(label, fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); dt = (time.time() - t) / n * 1000
    print(f'{label:34s} {dt:7.1f} ms/call  -> {dt * 9 / 1000:5.2f} s/step (9 calls)', flush=True)
    return dt

vg = ValueGrad(model, crit, R, B, N, A, D, use_graph=False)
bench('eager', lambda: vg(z0, zg, a))
vg_g = ValueGrad(model, crit, R, B, N, A, D, use_graph=True)
bench('cuda graph', lambda: vg_g(z0, zg, a))
torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_flash_sdp(False)
vg_m = ValueGrad(model, crit, R, B, N, A, D, use_graph=True)
bench('cuda graph + math sdpa', lambda: vg_m(z0, zg, a))
# compiled forward (AOTAutograd backward), autograd.grad outside
sa = a.reshape(R * B, N, A).clone().requires_grad_(True)
def fwd(z0f, zgf, af):
    zN = rollout_latent(model, z0f, af)
    return crit(zN.view(R, B, -1), zgf.view(R, B, -1))
cf = torch.compile(fwd, mode='default', dynamic=False)
z0f, zgf = z0.reshape(R * B, D), zg.reshape(R * B, D)
def compiled_call():
    v = cf(z0f, zgf, sa); (g,) = torch.autograd.grad(v.sum(), sa); return v, g
t = time.time(); compiled_call(); torch.cuda.synchronize(); print(f'compile warmup {time.time()-t:.0f}s')
bench('compiled fwd+bwd (math sdpa)', compiled_call)
# compiled + manual cuda graph
try:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): compiled_call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ov, og = compiled_call()
    def replay():
        g.replay(); return ov.clone(), og.clone()
    bench('compiled + cuda graph', replay)
    v1, g1 = replay(); v2, g2 = vg(z0, zg, a)
    print('graph/compiled vs eager max abs diff: v', (v1.view(R, B) - v2).abs().max().item(), 'g', (g1.view(R, B, N, A) - g2).abs().max().item())
except Exception as e:
    print('compiled+graph FAILED:', type(e).__name__, str(e)[:300])
