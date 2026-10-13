"""From-scratch LeWM on Push-T: data collected by CEM that tries candidate action sequences IN THE REAL SIMULATOR and
scores each real outcome with a learning signal. No policy, no learned proxy for the signal.

Per round, every one of --n-envs envs plans one 25-step segment (5 action blocks x frameskip 5 = LeWM's planning
horizon, executed open-loop as in its Push-T eval). CEM (--cem-samples x --cem-iters, top --cem-elites refit) samples
commanded actions, runs each candidate from the env's exact current state in a cloned simulator (pusht_sim: state
restore verified bit-exact incl. frames; agent clipped to the arena during collection), renders the 5 frames at the
frameskip points and scores them with the arm's signal under the CURRENT world model. The best-scoring candidate is
executed (re-simulated from the same state, so the executed trajectory is the one that was scored).

Signals, on a candidate x = (real history frames <= 3 at stride 5, its 5 outcome frames, executed action blocks);
L(x; theta) = mean over the 5 new transitions and 192 latent dims of the one-step latent error, each predicted from
<= 3 context frames (eval mode: running BatchNorm statistics, no dropout):
  pred    L(x; theta_now)                                                        naive prediction error
  lp0     |L(x; theta_past) - L(x; theta_now)|                                   zeroth order: realised loss change
  lp1     |<P * grad L(x; theta_now), theta_past - theta_now>|                   first order (arXiv 2609.30063 Eq. 2)
          P = lr / (sqrt(v_hat) + eps) from the world model's AdamW state; one forward-mode JVP scores all candidates
  rnd     pixel RND novelty (Burda et al. 2018 nets, 84x84 grey, frozen warm-up obs normalisation), mean of 5 frames
  random  no search: the first CEM sample (N(0, cem_std^2) clipped) is executed
theta_past = the snapshot at floor(e / 2) of the e world-model updates so far (ring of snapshots every --snap-every).

World model: LeWM exactly as lewm/train.py builds and trains it (config/train lewm.yaml + model/lewm.yaml, the
lejepa_forward loss with SIGReg weight 0.09, AdamW lr 5e-5 wd 1e-3 on all parameters, grad-norm clip 1, bf16 autocast,
batch 128 of 4-frame windows at stride 5 with 4 action blocks of 5, the per-step linear-warmup + cosine schedule of a
--sched-total-steps run truncated at --grad-steps). Actions are z-scored with the fixed statistics of U[-1, 1]
(mean 0, std 1/sqrt 3) - no expert statistics. --grad-steps 13933 = one epoch of the offline 10-epoch run, so its
epoch-1 checkpoint is the matched comparison.

    taskset -c 0-17 .venv/bin/python pusht_explore.py --signal pred --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

LEWM = Path(__file__).resolve().parent
sys.path.insert(0, str(LEWM))
import pusht_sim as ps  # noqa: E402  (no torch import: the worker pool must fork before CUDA exists)

FS = 5                 # frameskip / action block length (config/train/data/pusht.yaml)
HIST = 3               # history_size
NUM_STEPS = 4          # num_preds + history_size frames per training window
SPAN = NUM_STEPS * FS  # 20 rows per training window
A_STD = 1.0 / math.sqrt(3.0)
SIGNALS = ("pred", "lp0", "lp1", "rnd", "random")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signal", required=True, choices=SIGNALS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--ep-plans", type=int, default=8, help="plans per episode (episode = ep_plans x 25 steps)")
    p.add_argument("--plan-blocks", type=int, default=5)
    p.add_argument("--env-steps", type=int, default=200_000)
    p.add_argument("--warmup-rounds", type=int, default=10, help="random-action rounds before training/scoring")
    p.add_argument("--grad-steps", type=int, default=13_933)
    p.add_argument("--sched-total-steps", type=int, default=139_330, help="10 epochs x 13,933 (offline run)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--cem-samples", type=int, default=32)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--cem-elites", type=int, default=4)
    p.add_argument("--cem-std", type=float, default=0.6)
    p.add_argument("--cem-min-std", type=float, default=0.05)
    p.add_argument("--snap-every", type=int, default=100)
    p.add_argument("--score-chunk", type=int, default=256, help="candidates per scoring forward pass")
    p.add_argument("--rnd-lr", type=float, default=1e-4)
    p.add_argument("--bn-batches", type=int, default=8, help="BatchNorm recalibration batches before each scoring")
    p.add_argument("--bn-batch", type=int, default=64)
    p.add_argument("--wandb-id", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--ckpt-at", default="0.25,0.5,1.0", help="fractions of --grad-steps to checkpoint at")
    p.add_argument("--name", default=None)
    p.add_argument("--max-rounds", type=int, default=0, help="debug: stop after this many rounds")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--check-lp1", action="store_true", help="debug: compare the JVP with a finite difference")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing in the ViT encoder: same maths (activations recomputed in the backward "
                        "pass), several-fold less GPU memory, ~30%% more encoder compute")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- replay buffer
class Replay:
    """Episodes stored contiguously (fixed length rows_per_ep = steps + 1): row t = (frame o_t, action a_t)."""

    def __init__(self, n_eps_max: int, rows_per_ep: int):
        cap = n_eps_max * rows_per_ep
        self.rows_per_ep = rows_per_ep
        self.frames = np.empty((cap,) + ps.FRAME, np.uint8)
        self.actions = np.zeros((cap, 2), np.float32)
        self.states = np.zeros((cap, ps.STATE_DIM), np.float64)
        self.contacts = np.zeros(cap, np.int32)
        self.ep_start: list[int] = []
        self.ep_rows: list[int] = []      # rows with a frame written
        self.n_eps = 0

    def new_episode(self) -> int:
        e = self.n_eps
        self.ep_start.append(e * self.rows_per_ep)
        self.ep_rows.append(0)
        self.n_eps += 1
        return e

    def write(self, e: int, t: int, frame, state, contact=0):
        r = self.ep_start[e] + t
        self.frames[r] = frame
        self.states[r] = state
        self.contacts[r] = contact
        self.ep_rows[e] = max(self.ep_rows[e], t + 1)

    def set_actions(self, e: int, t0: int, acts: np.ndarray):
        r = self.ep_start[e] + t0
        self.actions[r:r + len(acts)] = acts

    def n_windows(self) -> int:
        return int(np.maximum(np.asarray(self.ep_rows) - (SPAN - 1), 0).sum()) if self.n_eps else 0

    def sample_windows(self, rng, B: int):
        counts = np.maximum(np.asarray(self.ep_rows) - (SPAN - 1), 0)
        cum = np.cumsum(counts)
        u = rng.integers(0, cum[-1], B)
        e = np.searchsorted(cum, u, side="right")
        s = u - (cum[e] - counts[e])
        rows = np.asarray(self.ep_start)[e] + s
        px = self.frames[rows[:, None] + np.arange(0, SPAN, FS)[None]]             # (B, 4, H, W, 3)
        act = self.actions[rows[:, None] + np.arange(SPAN)[None]]                 # (B, 20, 2)
        return px, act

    def sample_frames(self, rng, B: int):
        rows = np.concatenate([np.asarray(self.ep_start)[e] + np.arange(n)
                               for e, n in enumerate(self.ep_rows) if n > 0])
        return self.frames[rows[rng.integers(0, len(rows), B)]]


def main():
    args = parse()
    name = args.name or f"pusht_fs_{args.signal}_s{args.seed}"
    T_plan = args.plan_blocks * FS
    ep_steps = args.ep_plans * T_plan
    E, S = args.n_envs, args.cem_samples
    n_rounds = args.env_steps // (E * T_plan)
    if args.max_rounds:
        n_rounds = min(n_rounds, args.max_rounds)
    n_eps_max = -(-n_rounds // args.ep_plans) * E

    # --- 1) simulator workers, forked before torch / CUDA exist in this process
    pool = ps.WorkerPool(args.workers, max_cands=E * S, n_render=args.plan_blocks, n_envs=E, T=T_plan)

    import torch
    import torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch import nn
    from torch.func import functional_call, jvp

    import stable_worldmodel as swm
    from module import SIGReg

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_f = open(out_dir / "metrics.jsonl", "a")

    # --- 2) world model, built exactly like lewm/train.py
    with initialize_config_dir(config_dir=str(LEWM / "config" / "train"), version_base=None):
        cfg = compose(config_name="lewm", overrides=["data=pusht"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.action_encoder.input_dim = FS * 2
    model = instantiate(cfg.model).to(dev)
    if args.grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    W_steps = max(1, int(0.01 * args.sched_total_steps))

    def lr_factor(step):  # spt LinearWarmupCosineAnnealingLR, stepped after every optimizer step
        if step < W_steps:
            return step / W_steps
        return 0.5 * (1 + math.cos(math.pi * (step - W_steps) / (args.sched_total_steps - W_steps)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def to_px(u8):  # (..., H, W, 3) uint8 -> (..., 3, H, W) ImageNet-normalised float (spt ToImage)
        x = torch.as_tensor(u8, device=dev)
        lead = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:]).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean) / std
        return x.reshape(*lead, *x.shape[1:])

    def to_act(a):  # (..., 5k, 2) raw executed actions -> (..., k, 10) normalised blocks
        x = torch.as_tensor(np.asarray(a, np.float32), device=dev) / A_STD
        return x.reshape(*x.shape[:-2], x.shape[-2] // FS, FS * 2)

    def train_step(px_u8, act_raw):  # lewm/train.py lejepa_forward, verbatim maths
        model.train()
        batch = {"pixels": to_px(px_u8), "action": to_act(act_raw)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encode(batch)
            emb, act_emb = out["emb"], out["act_emb"]
            pred = model.predict(emb[:, :HIST], act_emb[:, :HIST])
            pred_loss = (pred - emb[:, 1:]).pow(2).mean()
            sig = sigreg(emb.transpose(0, 1))
            loss = pred_loss + args.sigreg_weight * sig
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        return float(pred_loss), float(sig), float(gn)

    # --- 3) the candidate loss L(x; theta) as a module, so functional_call / jvp can swap parameters
    class CandLoss(nn.Module):
        def __init__(self, wm):
            super().__init__()
            self.wm = wm

        def enc(self, x):  # JEPA.encode: CLS of the ViT -> projector
            h = self.wm.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            return self.wm.projector(h)

        def forward(self, hist_px, hist_act, cand_px, cand_act, env_of):
            Eh, m = hist_px.shape[:2]
            n = cand_px.shape[0]
            zh = self.enc(hist_px.flatten(0, 1)).view(Eh, m, -1)
            zc = self.enc(cand_px.flatten(0, 1)).view(n, cand_px.shape[1], -1)
            Z = torch.cat([zh[env_of], zc], 1)                                   # (n, m + 5, D)
            A = torch.cat([hist_act[env_of], cand_act], 1) if m > 1 else cand_act  # (n, m + 4, 10)
            ae = self.wm.action_encoder(A)
            L = 0.0
            for t in range(m, m + cand_px.shape[1]):
                lo = max(0, t - HIST)
                pred = self.wm.predict(Z[:, lo:t], ae[:, lo:t])[:, -1]
                L = L + (pred - Z[:, t]).pow(2).mean(-1)
            return L / cand_px.shape[1]

    cand_loss = CandLoss(model)
    past_model, past_loss, past_step = None, None, -1
    snaps: dict[int, dict] = {}
    if args.signal in ("lp0", "lp1"):
        snaps[0] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if args.signal == "lp0":
        past_model = instantiate(cfg.model).to(dev).eval()
        past_loss = CandLoss(past_model)

    def theta_past_step(e):
        return (e // 2 // args.snap_every) * args.snap_every

    # --- 4) RND (pixel novelty)
    if args.signal == "rnd":
        def convs():
            return nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.LeakyReLU(), nn.Conv2d(32, 64, 4, 2), nn.LeakyReLU(),
                                 nn.Conv2d(64, 64, 3, 1), nn.LeakyReLU(), nn.Flatten())
        rnd_t = nn.Sequential(convs(), nn.Linear(3136, 512)).to(dev).eval()
        rnd_p = nn.Sequential(convs(), nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512)).to(dev)
        for p_ in rnd_t.parameters():
            p_.requires_grad_(False)
        rnd_opt = torch.optim.Adam(rnd_p.parameters(), lr=args.rnd_lr)
        rnd_stats = {}

        def rnd_in(u8):  # (N, H, W, 3) uint8 -> (N, 1, 84, 84) normalised grey
            x = torch.as_tensor(u8, device=dev).float()
            g = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114).unsqueeze(1) / 255.0
            g = F.interpolate(g, size=(84, 84), mode="area")
            if "mu" in rnd_stats:
                g = ((g - rnd_stats["mu"]) / rnd_stats["sd"]).clamp_(-5, 5)
            return g

        def rnd_novelty(u8):
            with torch.no_grad():
                x = rnd_in(u8)
                return (rnd_p(x) - rnd_t(x)).pow(2).mean(-1)

    @torch.no_grad()
    def precond_tangent():
        """tangent = P * (theta_past - theta_now), P = lr / (sqrt(v_hat) + eps) from AdamW."""
        sd_past = snaps[theta_past_step(n_updates)]
        lr_now = opt.param_groups[0]["lr"]
        b1, b2 = opt.param_groups[0]["betas"]
        eps = opt.param_groups[0]["eps"]
        tang = {}
        for k, p_ in cand_loss.named_parameters():
            st = opt.state.get(p_, {})
            if "exp_avg_sq" not in st:
                tang[k] = torch.zeros_like(p_)
                continue
            t = float(st["step"])
            vhat = st["exp_avg_sq"] / (1 - b2 ** t)
            P = lr_now / (vhat.sqrt() + eps)
            tang[k] = P * (sd_past[k[len("wm."):]].to(dev) - p_.detach())
        return tang

    lp1_checked: list = []
    rng_bn = np.random.default_rng(args.seed + 7_777)

    def bn_modules(m):
        return [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]

    @torch.no_grad()
    def recalibrate_bn(m):
        """Running BatchNorm statistics := cumulative average of batch statistics over --bn-batches x --bn-batch
        windows of this arm's replay (train-mode statistics, no dropout). Early in training the EMA running
        statistics lag far behind the model (09-17 ledger: eval-mode latent mean off by ~1 per dim); eval-mode
        candidate losses came out ~50x the training loss (13.4 vs 0.23) before this. Returns the saved EMA
        statistics so they can be restored after scoring (checkpoints keep the trainer's own running stats)."""
        bns = bn_modules(m)
        saved = [(b.running_mean.clone(), b.running_var.clone(), b.num_batches_tracked.clone(), b.momentum) for b in bns]
        m.eval()
        for b in bns:
            b.reset_running_stats()
            b.momentum = None
            b.train()
        for _ in range(args.bn_batches):
            px, ac = rb.sample_windows(rng_bn, args.bn_batch)
            out = m.encode({"pixels": to_px(px), "action": to_act(ac)})
            m.predict(out["emb"][:, :HIST], out["act_emb"][:, :HIST])
        for b in bns:
            b.eval()
        return saved

    @torch.no_grad()
    def restore_bn(m, saved):
        for b, (rm, rv, nb, mom) in zip(bn_modules(m), saved):
            b.running_mean.copy_(rm)
            b.running_var.copy_(rv)
            b.num_batches_tracked.copy_(nb)
            b.momentum = mom

    def check_jvp(hp, ha, cp, ca, eo, tang, d_tf32):
        """float64 copy of the model: JVP vs central differences, (a) along a random tangent of measurable size,
        (b) along the LP tangent (scaled so the step is ~1e-4 of the weights), (c) the TF32 JVP actually used."""
        import copy
        import types
        m64 = copy.deepcopy(cand_loss).double()

        def embedder_fwd64(self, x):  # module.Embedder.forward minus its x.float() cast (float64 copy only)
            return self.embed(self.patch_embed(x.permute(0, 2, 1)).permute(0, 2, 1))

        m64.wm.action_encoder.forward = types.MethodType(embedder_fwd64, m64.wm.action_encoder)
        p64 = {k: v.detach() for k, v in m64.named_parameters()}
        b64 = dict(m64.named_buffers())
        args64 = (hp.double(), ha.double() if ha is not None else None, cp.double(), ca.double(), eo)

        def f64(p_):
            return functional_call(m64, (p_, b64), args64)

        def compare(tag, v):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                _, dj = jvp(f64, (p64,), (v,))
            with torch.no_grad():
                for h in (1e-1, 1e-2):
                    fd = (f64({k: p64[k] + h * v[k] for k in p64}) - f64({k: p64[k] - h * v[k] for k in p64})) / (2 * h)
                    rel = float((dj - fd).abs().max() / fd.abs().max().clamp_min(1e-300))
                    print(f"[check-lp1] {tag}: h={h:g} jvp {dj.abs().mean().item():.4e} fd {fd.abs().mean().item():.4e} "
                          f"max rel err {rel:.2e}", flush=True)
            return dj

        g = torch.Generator(device=dev).manual_seed(0)
        vr = {k: 1e-3 * v.abs().mean().clamp_min(1e-3) * torch.randn(v.shape, generator=g, device=dev, dtype=v.dtype)
              for k, v in p64.items()}
        compare("random tangent", vr)
        tnorm = math.sqrt(sum(float(t.double().pow(2).sum()) for t in tang.values()))
        wnorm = math.sqrt(sum(float(v.pow(2).sum()) for v in p64.values()))
        c = 1e-4 * wnorm / max(tnorm, 1e-30)
        dl = compare(f"LP tangent x{c:.3g}", {k: c * tang[k].double() for k in p64}) / c
        rel = float((d_tf32.double() - dl).abs().max() / dl.abs().max().clamp_min(1e-300))
        print(f"[check-lp1] TF32 jvp used for scoring vs float64 jvp: max rel err {rel:.2e}; |tangent| / |theta| "
              f"= {tnorm / wnorm:.3e}", flush=True)

    def score(hist_u8, hist_act_raw, cand_u8, cand_act_raw):
        """-> (E * S,) scores. hist_u8 (E, m, H, W, 3); hist_act_raw (E, (m-1)*5, 2); cand (E*S, 5, ...), (E*S, 25, 2)."""
        n = cand_u8.shape[0]
        out = torch.empty(n, device=dev)
        if args.signal == "rnd":
            for i in range(0, n, args.score_chunk):
                sl = slice(i, min(n, i + args.score_chunk))
                c = cand_u8[sl]
                out[sl] = rnd_novelty(c.reshape(-1, *ps.FRAME)).view(c.shape[0], -1).mean(1)
            return out.cpu().numpy()
        model.eval()
        hp = to_px(hist_u8)
        ha = to_act(hist_act_raw) if hist_u8.shape[1] > 1 else None
        env_all = torch.arange(n, device=dev) // S
        tang = precond_tangent() if args.signal == "lp1" else None
        params = {k: v.detach() for k, v in cand_loss.named_parameters()}
        buffers = dict(cand_loss.named_buffers())
        for i in range(0, n, args.score_chunk):
            sl = slice(i, min(n, i + args.score_chunk))
            cp = to_px(cand_u8[sl])
            ca = to_act(cand_act_raw[sl])
            eo = env_all[sl]
            if args.signal == "pred":
                with torch.no_grad():
                    out[sl] = cand_loss(hp, ha, cp, ca, eo)
            elif args.signal == "lp0":
                with torch.no_grad():
                    out[sl] = (past_loss(hp, ha, cp, ca, eo) - cand_loss(hp, ha, cp, ca, eo)).abs()
            elif args.signal == "lp1":
                def f(p_):
                    return functional_call(cand_loss, (p_, buffers), (hp, ha, cp, ca, eo))
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    _, d = jvp(f, (params,), (tang,))
                if args.check_lp1 and i == 0 and not lp1_checked and n_updates >= 2 * args.snap_every:
                    lp1_checked.append(True)
                    check_jvp(hp, ha, cp[:4], ca[:4], eo[:4], tang, d[:4])
                out[sl] = d.detach().abs()
        return out.cpu().numpy()

    # --- 5) buffers and env state
    rb = Replay(n_eps_max + E, ep_steps + 1)
    states = np.zeros((E, ps.STATE_DIM))
    ep_of = np.zeros(E, np.int64)
    t_ep = 0
    n_updates = 0
    ckpt_steps = sorted({int(round(float(f) * args.grad_steps)) for f in args.ckpt_at.split(",")})
    train_rounds = max(1, n_rounds - args.warmup_rounds)
    seed_ctr = args.seed * 1_000_000

    def start_episodes():
        nonlocal seed_ctr, t_ep
        st, fr = pool.reset(list(range(seed_ctr, seed_ctr + E)))
        seed_ctr += E
        for i in range(E):
            ep_of[i] = rb.new_episode()
            rb.write(ep_of[i], 0, fr[i], st[i])
        states[:] = st
        t_ep = 0

    if args.wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT"), entity=os.environ.get("WANDB_ENTITY"), name=name,
                   id=args.wandb_id or name, resume="allow", group="pusht_fs", config=vars(args))

    def save_ckpt(tag):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(model, run_name=name, config=cfg.model, filename=f"weights_{tag}.pt")

    start_episodes()
    t0_all = time.time()
    exec_mismatch = 0
    for rnd_i in range(n_rounds):
        t_round = time.time()
        warm = rnd_i < args.warmup_rounds
        # history at stride 5 ending at the current frame (<= 3 frames)
        hist_t = [t for t in (t_ep - 2 * FS, t_ep - FS, t_ep) if t >= 0]
        m = len(hist_t)
        hrows = np.array([[rb.ep_start[ep_of[i]] + t for t in hist_t] for i in range(E)])
        hist_u8 = rb.frames[hrows]                                                     # (E, m, H, W, 3)
        hist_act = (rb.actions[hrows[:, :1] + np.arange((m - 1) * FS)[None]] if m > 1
                    else np.zeros((E, 0, 2), np.float32))
        # ---- plan
        t_sim = t_score = 0.0
        stats = {}
        if args.signal == "random" or warm:
            best_act = np.clip(rng.normal(0.0, args.cem_std, (E, T_plan, 2)), -1, 1)
            best_final = None
        else:
            mu = np.zeros((E, T_plan, 2))
            sig = np.full((E, T_plan, 2), args.cem_std)
            best_score = np.full(E, -np.inf)
            best_act = np.zeros((E, T_plan, 2))
            best_final = np.zeros((E, ps.STATE_DIM))
            all_scores = []
            saved_bn = None
            if args.signal in ("pred", "lp0", "lp1"):
                ts = time.time()
                saved_bn = recalibrate_bn(model)
                if args.signal == "lp0":
                    recalibrate_bn(past_model)
                t_score += time.time() - ts
            for it in range(args.cem_iters):
                cand = np.clip(mu[:, None] + sig[:, None] * rng.standard_normal((E, S, T_plan, 2)), -1, 1)
                ts = time.time()
                exe, st, _ = pool.run("cand", np.repeat(states, S, 0), cand.reshape(E * S, T_plan, 2), FS)
                t_sim += time.time() - ts
                ts = time.time()
                sc = score(hist_u8, hist_act, pool.cand[:E * S], exe).reshape(E, S)
                t_score += time.time() - ts
                all_scores.append(sc)
                top = np.argsort(-sc, axis=1)[:, :args.cem_elites]
                el = np.take_along_axis(cand, top[:, :, None, None], axis=1)
                mu, sig = el.mean(1), np.maximum(el.std(1), args.cem_min_std)
                j = sc.argmax(1)
                better = sc[np.arange(E), j] > best_score
                best_score[better] = sc[np.arange(E), j][better]
                best_act[better] = cand[np.arange(E), j][better]
                best_final[better] = st.reshape(E, S, T_plan, -1)[np.arange(E), j, -1][better]
            if saved_bn is not None:
                restore_bn(model, saved_bn)
            a_sc = np.stack(all_scores)
            stats = {"sig/chosen": float(best_score.mean()), "sig/mean": float(a_sc.mean()),
                     "sig/it0_mean": float(a_sc[0].mean()), "sig/it0_std": float(a_sc[0].std(1).mean()),
                     "sig/gain_vs_it0": float(best_score.mean() / max(1e-12, a_sc[0].mean()))}
        # ---- execute (re-simulated from the same state: identical to the scored candidate)
        ts = time.time()
        exe, st, ct = pool.run("exec", states, best_act, 1)
        t_sim += time.time() - ts
        if best_final is not None:
            exec_mismatch += int((np.abs(st[:, -1] - best_final) > 0).any(1).sum())
        for i in range(E):
            rb.set_actions(ep_of[i], t_ep, exe[i])
            for k in range(T_plan):
                rb.write(ep_of[i], t_ep + 1 + k, pool.exec[i, k], st[i, k], ct[i, k])
        blk_disp = np.linalg.norm(st[:, -1, 4:6] - states[:, 4:6], axis=1)
        blk_rot = np.abs(st[:, -1, 6] - states[:, 6])
        states[:] = st[:, -1]
        t_ep += T_plan
        if t_ep >= ep_steps:
            start_episodes()
        # ---- RND observation normalisation (frozen after warm-up) + world-model / RND updates
        if args.signal == "rnd" and rnd_i == args.warmup_rounds - 1:
            fr = torch.as_tensor(rb.sample_frames(rng, 4096), device=dev).float()
            g = (fr[..., 0] * 0.299 + fr[..., 1] * 0.587 + fr[..., 2] * 0.114).unsqueeze(1) / 255.0
            g = F.interpolate(g, size=(84, 84), mode="area")
            rnd_stats["mu"], rnd_stats["sd"] = g.mean(0, keepdim=True), g.std(0, keepdim=True) + 1e-3
        ts = time.time()
        losses = []
        if not warm:
            k = rnd_i - args.warmup_rounds
            target = round((k + 1) * args.grad_steps / train_rounds)
            while n_updates < min(target, args.grad_steps) and rb.n_windows() > 0:
                px, ac = rb.sample_windows(rng, args.batch)
                losses.append(train_step(px, ac))
                n_updates += 1
                if args.signal == "rnd":
                    x = rnd_in(rb.sample_frames(rng, args.batch))
                    rl = (rnd_p(x) - rnd_t(x)).pow(2).mean()
                    rnd_opt.zero_grad(set_to_none=True)
                    rl.backward()
                    rnd_opt.step()
                    stats["rnd/pred_loss"] = float(rl)
                if args.signal in ("lp0", "lp1") and n_updates % args.snap_every == 0:
                    snaps[n_updates] = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
                    keep = theta_past_step(n_updates)
                    for k_ in [k_ for k_ in snaps if k_ < keep]:
                        del snaps[k_]
                if n_updates in ckpt_steps:
                    save_ckpt(f"step_{n_updates}")
        if args.signal == "lp0" and theta_past_step(n_updates) != past_step:
            past_step = theta_past_step(n_updates)
            past_model.load_state_dict({k_: v.to(dev) for k_, v in snaps[past_step].items()})
            past_model.eval()
        t_train = time.time() - ts
        # ---- log
        rec = {"round": rnd_i, "env_steps": (rnd_i + 1) * E * T_plan, "updates": n_updates, "warm": warm,
               "m_hist": m, "t_round": time.time() - t_round, "t_sim": t_sim, "t_score": t_score, "t_train": t_train,
               "lr": opt.param_groups[0]["lr"], "exec_mismatch": exec_mismatch,
               "beh/blk_disp_px": float(blk_disp.mean()), "beh/blk_rot_rad": float(blk_rot.mean()),
               "beh/contact_frac": float((ct > 0).mean()), "beh/blk_moved_frac": float((blk_disp > 1.0).mean()),
               **stats}
        if losses:
            lo = np.array(losses)
            rec.update({"wm/pred_loss": float(lo[:, 0].mean()), "wm/sigreg": float(lo[:, 1].mean()),
                        "wm/grad_norm": float(lo[:, 2].mean())})
        if args.signal in ("lp0", "lp1"):
            rec["lp/past_step"] = theta_past_step(n_updates)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if args.wandb:
            wandb.log(rec, step=rnd_i)
        if rnd_i % 10 == 0 or rnd_i == n_rounds - 1:
            el = time.time() - t0_all
            print(f"[{name}] round {rnd_i + 1}/{n_rounds} env_steps {rec['env_steps']} updates {n_updates} "
                  f"pred {rec.get('wm/pred_loss', float('nan')):.4f} sig {rec.get('sig/chosen', float('nan')):.4g} "
                  f"blk {rec['beh/blk_disp_px']:.1f}px contact {rec['beh/contact_frac']:.3f} | round {rec['t_round']:.1f}s "
                  f"(sim {t_sim:.1f} score {t_score:.1f} train {t_train:.1f}) | {el / 60:.1f} min | "
                  f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if n_updates not in ckpt_steps:
        save_ckpt(f"step_{n_updates}")
    np.savez_compressed(out_dir / "replay_index.npz", states=rb.states[: rb.n_eps * rb.rows_per_ep],
                        actions=rb.actions[: rb.n_eps * rb.rows_per_ep], contacts=rb.contacts[: rb.n_eps * rb.rows_per_ep],
                        ep_rows=np.asarray(rb.ep_rows), rows_per_ep=rb.rows_per_ep)
    print(f"[{name}] done: {n_updates} updates, {rb.n_eps} episodes, exec mismatches {exec_mismatch}", flush=True)
    pool.close()
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
