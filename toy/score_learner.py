"""E2 for run_selfplay.py: how well does the agent's own world model (the latent learner) predict the human demos --
above all, the T's movement?

The learner predicts embeddings, not states, so it is scored through a read-out: an MLP z -> 8-D observation fitted
with the encoder frozen (diagnostic only, like model/decoder.py: never in a gradient path of the agent) on one fixed
dataset shared by every run -- pusht_eval's scripted pusher (free space, pushes and wall contact), 480k transitions,
seed 777 (the test-set normaliser uses seed 12345). The predictor then rolls out open loop in latent space from
enc(s_0) on each window's actions, every step is read out, and pusht_eval.evaluate scores it on the frozen test sets,
on E1's scale:
  DEMO skill  mean_h log(MSE "T static" / MSE model) on the T's corners (the agent is not scored)
  PUSH        the same on windows where the agent pushes the T in open space
  hit         PUSH windows whose T truly moves > 5 px by h = 10 where the prediction also moves it > 5 px
  invented    FREE windows (nothing touches the T) where the prediction moves it > 5 px by h = 10
Also reported: the read-out's own error at h = 0 on DEMO (decode(enc(s)) vs s), i.e. what the latent keeps at all.

"latent" -- the same question in the learner's own latent space, no read-out: the model rolls out open loop from
enc(s_0) on the window's actions and is compared to enc(s_h). Score = fraction of the change missed,
    missed = E |z_hat_h - z_h|^2 / E |z_h - z_0|^2        (0 = perfect, 1 = as bad as predicting no change),
a ratio within one latent, so comparable across runs. Split into windows whose T truly moves > 5 px by h ("T moves":
does it model the block's movement?) and < 1 px ("T still"); "shuffled" = the same with another window's actions (a
model that uses its actions misses more). On DEMO (unseen) and one step ahead on the run's own replay (its training
distribution), which separates generalisation from fit.
    python toy/score_learner.py toy/runs/pusht/ptf_spep_s0 [...]      # -> <run>/eval_learner.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from push_t import PushT
from pusht_eval import (DEV, MOVE_TOL, TESTS, _kp_err, _sha1, _strip, evaluate, headline, load_replay, load_tests,
                        scripted_data)
from run_selfplay import LatentWM, mlp

PROBE_DATA = Path(__file__).resolve().parent / "data" / "probe_scripted.pt"


def probe_states(n=480_000):
    if not PROBE_DATA.exists():
        s = scripted_data("fixed", n=n, seed=777)[0]
        torch.save(s, PROBE_DATA)
    return torch.load(PROBE_DATA)


def unit_angle(o):
    return torch.cat([o[..., :6], o[..., 6:8] / o[..., 6:8].norm(dim=-1, keepdim=True).clamp_min(1e-6)], -1)


class LearnerWM:
    """The run's frozen learner + a read-out fitted on the shared probe data; called like a pusht_eval predictor."""

    def __init__(self, fin, steps=10_000, batch=4096, lr=1e-3, seed=0):
        a = fin["args"]
        self.m = LatentWM(PushT.obs_dim, PushT.act_dim, a["z_dim"], a["wm_hidden"]).to(DEV)
        self.m.load_state_dict(fin["learner"])
        self.m.eval().requires_grad_(False)
        torch.manual_seed(seed)
        o = PushT.obs_from_state(probe_states().to(DEV))
        with torch.no_grad():
            z = self.m.enc(o)
        self.read = mlp(a["z_dim"], PushT.obs_dim, 256).to(DEV)
        opt = torch.optim.Adam(self.read.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
        for _ in range(steps):
            idx = torch.randint(0, len(z), (batch,), device=DEV)
            loss = (self.read(z[idx]) - o[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        self.fit_loss = loss.item()
        self.read.eval().requires_grad_(False)

    @torch.no_grad()
    def decode(self, z):
        return PushT.state_from_obs(unit_angle(self.read(z)))

    @torch.no_grad()
    def __call__(self, s0, act):
        z = self.m.enc(PushT.obs_from_state(s0))
        out = [s0]
        for t in range(act.shape[1]):
            z = z + self.m.pred(torch.cat([z, act[:, t]], -1))
            out.append(self.decode(z))
        return torch.stack(out, 1)


def missed(zh, zt, z0, moved):
    """Fraction of the change missed on T-moving (> MOVE_TOL px) and T-still (< 1 px) windows."""
    e, d = (zh - zt).pow(2).sum(-1), (zt - z0).pow(2).sum(-1)
    out = {}
    for name, mask in (("T moves", moved > MOVE_TOL), ("T still", moved < 1.0)):
        out[name] = {"n": int(mask.sum()), "missed": (e[mask].mean() / d[mask].mean()).item()}
    return out


@torch.no_grad()
def latent_scores(m, tests, run=None, hs=(1, 5, 10, 25)):
    enc = lambda st: m.enc(PushT.obs_from_state(st.to(DEV)))
    out = {"DEMO": {}}
    for h in hs:
        S, act = tests["DEMO"][h]["S"].to(DEV), tests["DEMO"][h]["act"].to(DEV)
        perm = torch.randperm(len(act), generator=torch.Generator().manual_seed(h)).to(DEV)
        z0, zt, moved = enc(S[:, 0]), enc(S[:, h]), _kp_err(S[:, 0, 4:7], S[:, h, 4:7])
        q = {}
        for key, a in (("model", act), ("shuffled", act[perm])):
            z = z0
            for t in range(h):
                z = z + m.pred(torch.cat([z, a[:, t]], -1))
            q[key] = missed(z, zt, z0, moved)
        out["DEMO"][h] = q
    if run is not None and (Path(run) / "replay.pt").exists():
        s, a, s2, _ = load_replay(run)
        i = torch.randperm(len(s), generator=torch.Generator().manual_seed(0))[:400_000]
        s, a, s2 = s[i].to(DEV), a[i].to(DEV), s2[i].to(DEV)
        z0 = enc(s)
        out["own"] = {1: {"model": missed(z0 + m.pred(torch.cat([z0, a], -1)), enc(s2), z0,
                                          _kp_err(s[:, 4:7].double(), s2[:, 4:7].double()))}}
    return out


def table(runs):
    """Per-arm mean (sd) of the fraction of change missed, from <run>/eval_learner.json."""
    import numpy as np
    arms = {}
    for r in map(Path, runs):
        f = r / "eval_learner.json"
        if f.exists():
            arms.setdefault(r.name.rsplit("_s", 1)[0], []).append(json.load(open(f))["latent"])
    cols = [("DEMO", "1", "model", "T moves"), ("DEMO", "5", "model", "T moves"), ("DEMO", "10", "model", "T moves"),
            ("DEMO", "1", "model", "T still"), ("DEMO", "10", "model", "T still"), ("DEMO", "1", "shuffled", "T moves"),
            ("own", "1", "model", "T moves"), ("own", "1", "model", "T still")]
    print("fraction of the latent change missed (0 = perfect, 1 = no better than 'nothing changes'); mean (sd) over seeds")
    print(f"{'arm':18s} {'n':>2s} " + " ".join(f"{f'{d} h{h} {w}' + (' shuf' if k == 'shuffled' else ''):>17s}"
                                          for d, h, k, w in cols))
    for arm, L in arms.items():
        cells = []
        for d, h, k, w in cols:
            v = np.array([x[d][h][k][w]["missed"] for x in L if d in x])
            cells.append(f"{v.mean():.3f} ({v.std(ddof=1) if len(v) > 1 else 0:.3f})" if len(v) else "-")
        print(f"{arm:18s} {len(L):2d} " + " ".join(f"{c:>17s}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--table", action="store_true", help="only print the per-arm table from existing eval_learner.json")
    p.add_argument("--steps", type=int, default=10_000, help="read-out fit steps")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.table:
        return table(args.runs)
    tests = load_tests()
    s_demo = tests["DEMO"][1]["S"][:, 0].to(DEV)                  # every DEMO window start, for the h = 0 read-out
    for run in map(Path, args.runs):
        out = run / "eval_learner.json"
        if out.exists() and not args.force:
            print(f"[score_learner] {run.name}: exists, skipped")
            continue
        t0 = time.time()
        wm = LearnerWM(torch.load(run / "final.pt", weights_only=False), steps=args.steps)
        with torch.no_grad():
            r0 = wm.decode(wm.m.enc(PushT.obs_from_state(s_demo)))
        probe = {"fit_loss": wm.fit_loss,
                 "read0_T_px": _kp_err(r0[:, 4:7], s_demo[:, 4:7]).pow(2).mean().sqrt().item(),
                 "read0_agent_px": (r0[:, :2] - s_demo[:, :2]).norm(dim=-1).pow(2).mean().sqrt().item()}
        res = {"tests_sha1": _sha1(TESTS), "probe": probe, "latent": latent_scores(wm.m, tests, run), "E2": evaluate(wm, tests)}
        json.dump(_strip(res), open(out, "w"), indent=1)
        lat = "  ".join(f"h{h} {q['model']['T moves']['missed']:.3f}/{q['model']['T still']['missed']:.3f}"
                        for h, q in res["latent"]["DEMO"].items())
        print(f"[score_learner] {run.name} ({time.time() - t0:.0f}s): DEMO change missed, T moves/T still: {lat}\n"
              f"  read-out at h=0 T {probe['read0_T_px']:.1f} px, agent {probe['read0_agent_px']:.1f} px "
              f"(fit {probe['fit_loss']:.2e}); decoded E2: {headline(res['E2'])}", flush=True)


if __name__ == "__main__":
    main()
