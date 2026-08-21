"""Live goal dashboard for --goal-explore runs (--live-view PORT).

A tiny threaded HTTP server on localhost that mirrors what the goal-explore loop is
doing RIGHT NOW: the wrist frame the encoder sees, the goal photo z* is re-encoded
from, ||z - z*|| against the reach eps, and a running strip of every goal the run
has set (with the retention switch reason — seed / arrive / stall / age / refresh).

Design constraints:
  * The trainer thread only ever calls `update()` (cheap: two small uint8 copies +
    a dict); ALL encoding/serving happens on server threads. An exception anywhere
    in the viewer must never kill the run — every entry point is try/except'd.
  * No new dependencies: stdlib http.server + cv2 (already required by UsbCamera)
    for JPEG encoding. Single self-contained HTML page, no external assets.
  * Goal switches are detected HERE (byte-compare against the last pushed goal),
    so train.py's loop needs no bookkeeping beyond passing its existing state.

Open http://localhost:PORT while the run is live.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_JPEG_QUALITY = 87


def _jpeg(img_rgb: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


class LiveViewer:
    """Shared state between the trainer thread (update/set_phase) and HTTP threads."""

    def __init__(self, port: int, run_name: str = "", max_goals: int = 48):
        self.port = int(port)
        self.run_name = run_name
        self._lock = threading.Lock()
        self._wrist: np.ndarray | None = None
        self._wrist_id = 0
        self._goal_jpg: dict[int, bytes] = {}          # goal_id -> encoded photo (history strip)
        self._goals: deque[dict] = deque(maxlen=max_goals)
        self._goal_id = 0
        self._last_goal_px: np.ndarray | None = None   # byte-compare target for switch detection
        self._last_retain = {"arrive": 0, "stall": 0, "age": 0}
        self._cur_min = float("inf")                   # min dist seen under the CURRENT goal
        self._dist_hist: deque[tuple[int, float]] = deque(maxlen=900)
        self._info: dict = {}
        self._decoded: dict[str, np.ndarray] = {}      # decoder's-eye frames by name
        self._phase = "wake"
        self._t_update = 0.0
        self._server: ThreadingHTTPServer | None = None

    # ---- trainer-side API (must stay cheap + unkillable) ---------------------------------
    def update(self, step: int, wrist_px: np.ndarray, goal_px: np.ndarray | None,
               dist: float | None, eps: float, info: dict | None = None,
               switched: bool = False, decoded: dict | None = None) -> None:
        """Once per decision step. goal_px=None -> no goal yet (self-goal / warming buffer).
        `switched=True` -> the trainer KNOWS a new pursuit window opened this step (retention
        reset) — trust it even when the re-picked goal photo is byte-identical to the last.
        Otherwise a byte-compare catches cadence re-rolls. Switch reason comes from whichever
        --goal-retain counter moved, else 'refresh' (re-roll) or 'seed' (first goal)."""
        try:
            info = dict(info or {})
            with self._lock:
                self._wrist = np.ascontiguousarray(wrist_px)
                self._wrist_id += 1
                if decoded is not None:               # decoder's-eye frames (now / plan / goal)
                    self._decoded = {k: np.ascontiguousarray(v) for k, v in decoded.items()}
                if dist is not None and np.isfinite(dist):
                    self._dist_hist.append((int(step), round(float(dist), 4)))
                    self._cur_min = min(self._cur_min, float(dist))
                if goal_px is not None:
                    changed = (switched or self._last_goal_px is None
                               or not np.array_equal(goal_px, self._last_goal_px))
                    if changed:
                        reason = "seed" if self._last_goal_px is None else "refresh"
                        retain = {k: int(info.get(f"retain_{k}", 0)) for k in ("arrive", "stall", "age")}
                        for k in ("arrive", "stall", "age"):
                            if retain[k] > self._last_retain[k]:
                                reason = k                     # the counter that moved names the switch
                        self._last_retain = retain
                        if self._goals:                        # stamp the outgoing goal's outcome
                            prev = self._goals[-1]
                            prev["closed_step"] = int(step)
                            prev["outcome"] = reason if reason in ("arrive", "stall", "age") else "replaced"
                            if np.isfinite(self._cur_min):     # closest this goal ever got (viewer-tracked)
                                prev["min_dist"] = round(self._cur_min, 3)
                        self._cur_min = float("inf")
                        self._goal_id += 1
                        self._goal_jpg[self._goal_id] = _jpeg(goal_px)
                        self._goals.append({
                            "id": self._goal_id, "step": int(step), "reason": reason,
                            "mse": info.get("goal_mse"), "outcome": None,
                        })
                        for gid in list(self._goal_jpg):       # drop photos that fell off the strip
                            if gid < self._goals[0]["id"]:
                                del self._goal_jpg[gid]
                        self._last_goal_px = goal_px.copy()
                else:
                    self._last_goal_px = None
                self._info = {"step": int(step), "eps": round(float(eps), 3),
                              "dist": (round(float(dist), 3) if dist is not None and np.isfinite(dist) else None),
                              "has_goal": goal_px is not None, **info}
                self._t_update = time.time()
        except Exception as e:                                  # never propagate into the control loop
            print(f"[live-view] update failed (non-fatal): {e}", flush=True)

    def set_phase(self, phase: str) -> None:
        """'wake' | 'sleep' (consolidate/cotrain burst — the arm holds still)."""
        try:
            with self._lock:
                self._phase = phase
                self._t_update = time.time()
        except Exception:
            pass

    # ---- server ---------------------------------------------------------------------------
    def start(self) -> None:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):                          # keep the train log clean
                pass

            def _send(self, code: int, ctype: str, body: bytes, cache: bool = False):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control",
                                 "max-age=3600" if cache else "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    path = self.path.split("?")[0]
                    if path == "/":
                        self._send(200, "text/html; charset=utf-8", _PAGE.encode())
                    elif path == "/state.json":
                        with viewer._lock:
                            state = {
                                "run": viewer.run_name, "phase": viewer._phase,
                                "age_s": round(time.time() - viewer._t_update, 1) if viewer._t_update else None,
                                "wrist_id": viewer._wrist_id,
                                "dist_hist": list(viewer._dist_hist),
                                "goals": list(viewer._goals),
                                "decoded": list(viewer._decoded.keys()),
                                "info": viewer._info,
                            }
                        self._send(200, "application/json", json.dumps(state).encode())
                    elif path == "/img/wrist.jpg":
                        with viewer._lock:
                            frame = None if viewer._wrist is None else viewer._wrist.copy()
                        if frame is None:
                            self._send(404, "text/plain", b"no frame yet")
                        else:
                            self._send(200, "image/jpeg", _jpeg(frame))
                    elif path.startswith("/img/dec/"):
                        name = path.rsplit("/", 1)[1].removesuffix(".jpg")
                        with viewer._lock:
                            frame = viewer._decoded.get(name)
                            frame = None if frame is None else frame.copy()
                        if frame is None:
                            self._send(404, "text/plain", b"no decoded frame")
                        else:
                            self._send(200, "image/jpeg", _jpeg(frame))
                    elif path.startswith("/img/goal/"):
                        gid = int(path.rsplit("/", 1)[1].removesuffix(".jpg"))
                        with viewer._lock:
                            jpg = viewer._goal_jpg.get(gid)
                        if jpg is None:
                            self._send(404, "text/plain", b"gone")
                        else:
                            self._send(200, "image/jpeg", jpg, cache=True)   # goal photos are immutable
                    else:
                        self._send(404, "text/plain", b"not found")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    try:
                        self._send(500, "text/plain", str(e).encode())
                    except Exception:
                        pass

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="live-view",
                         daemon=True).start()
        print(f"[live-view] serving http://localhost:{self.port}", flush=True)


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>curious robot — live goals</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --tx:#e6edf3; --dim:#8b949e;
          --acc:#58a6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--tx); font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; padding:18px; }
  h1 { font-size:16px; font-weight:600; margin-bottom:2px; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:14px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
  .chip { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:4px 10px; font-size:12px; }
  .chip b { color:var(--acc); font-weight:600; }
  .chip.sleep { border-color:var(--warn); color:var(--warn); }
  .row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
  .card h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--dim); margin-bottom:8px; }
  .imgbox img { width:300px; height:300px; image-rendering:pixelated; border-radius:4px; background:#000; display:block; }
  .gauge { min-width:210px; display:flex; flex-direction:column; justify-content:center; }
  .dist { font-size:34px; font-weight:700; }
  .dist small { font-size:13px; color:var(--dim); font-weight:400; }
  .bar { height:10px; background:#21262d; border-radius:5px; margin:10px 0; overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--acc); transition:width .3s; }
  canvas { background:#0a0d12; border-radius:4px; }
  .strip { display:flex; gap:10px; overflow-x:auto; padding-bottom:6px; }
  .g { flex:0 0 auto; width:120px; }
  .g img { width:120px; height:120px; image-rendering:pixelated; border-radius:4px; display:block; }
  .g .meta { font-size:11px; color:var(--dim); margin-top:3px; }
  .tag { display:inline-block; padding:0 5px; border-radius:4px; font-size:10px; font-weight:600; }
  .tag.arrive { background:#12261e; color:var(--ok); }  .tag.stall { background:#2b2111; color:var(--warn); }
  .tag.age { background:#21262d; color:var(--dim); }    .tag.seed,.tag.refresh { background:#0f2036; color:var(--acc); }
  .tag.live { background:#2ea04326; color:var(--ok); }
  .stale { color:var(--bad); }
</style></head><body>
<h1>curious robot — live goals <span id=run class=sub></span></h1>
<div class=sub id=meta>connecting…</div>
<div class=chips id=chips></div>
<div class=row>
  <div class=card><h2>goal photo (z*)</h2><div class=imgbox><img id=goal alt="no goal yet"></div>
    <div class=meta id=goalmeta style="font-size:11px;color:var(--dim);margin-top:5px"></div></div>
  <div class="card gauge"><h2>latent distance ‖z − z*‖</h2>
    <div class=dist><span id=dist>–</span> <small>eps <span id=eps>–</span></small></div>
    <div class=bar><i id=bari></i></div>
    <div style="font-size:11px;color:var(--dim)" id=gaugemeta></div></div>
  <div class=card><h2>wrist camera (live)</h2><div class=imgbox><img id=wrist></div></div>
</div>
<div class=row id=decrow style="display:none">
  <div class=card><h2>decode(z now)</h2><div class=imgbox><img id=dec_now class=decimg></div>
    <div style="font-size:11px;color:var(--dim);margin-top:5px">vs wrist above = what the latent preserves<span id=reconl1></span></div></div>
  <div class=card><h2>decode(plan &rarr; next z)</h2><div class=imgbox><img id=dec_plan class=decimg></div>
    <div style="font-size:11px;color:var(--dim);margin-top:5px">what the CEM plan imagines its next action does</div></div>
  <div class=card><h2>decode(z*)</h2><div class=imgbox><img id=dec_goal class=decimg></div>
    <div style="font-size:11px;color:var(--dim);margin-top:5px">vs goal photo above = what the planner can "see" of the goal</div></div>
</div>
<div class=card style="margin-bottom:14px"><h2>distance history</h2><canvas id=chart width=980 height=140></canvas></div>
<div class=card><h2>goals set (newest first)</h2><div class=strip id=strip></div></div>
<script>
let lastWrist = -1, knownGoals = new Set();
async function tick() {
  let s;
  try { s = await (await fetch('/state.json')).json(); }
  catch (e) { document.getElementById('meta').textContent = 'disconnected — is the run alive?'; return; }
  const i = s.info || {};
  document.getElementById('run').textContent = s.run || '';
  const stale = s.age_s != null && s.age_s > 8 && s.phase !== 'sleep';
  document.getElementById('meta').innerHTML =
    `step <b>${i.step ?? '–'}</b> · phase <b>${s.phase}</b> · updated ${s.age_s ?? '?'}s ago` +
    (stale ? ' <span class=stale>(stale — loop paused?)</span>' : '');
  const chips = [];
  const chip = (k, v) => chips.push(`<span class=chip>${k} <b>${v}</b></span>`);
  if (s.phase === 'sleep') chips.push('<span class="chip sleep">SLEEP — consolidating, arm holding</span>');
  if (i.curric_d != null) chip('d budget', i.curric_d);
  if (i.curric_pctl != null) chip('mse pctl', i.curric_pctl);
  if (i.eff_amax != null) chip('eff amax', i.eff_amax);
  if (i.buffer != null) chip('buffer', i.buffer);
  if (i.archive != null) chip('archive', i.archive);
  if (i.arrival != null) chip('arrival', i.arrival);
  if (i.goal_age != null) chip('goal age', i.goal_age);
  if (i.since_best != null) chip('since best', i.since_best);
  if (i.retain_arrive != null) chip('arrive/stall/age', `${i.retain_arrive}/${i.retain_stall}/${i.retain_age}`);
  if (i.guard) chip('torque guard', i.guard);
  document.getElementById('chips').innerHTML = chips.join('');
  if (s.wrist_id !== lastWrist) {
    document.getElementById('wrist').src = '/img/wrist.jpg?t=' + s.wrist_id;
    const dn = s.decoded || [];
    if (dn.length) {
      document.getElementById('decrow').style.display = 'flex';
      for (const n of ['now', 'plan', 'goal']) if (dn.includes(n))
        document.getElementById('dec_' + n).src = '/img/dec/' + n + '.jpg?t=' + s.wrist_id;
    }
    lastWrist = s.wrist_id;
  }
  document.getElementById('reconl1').textContent =
    i.recon_l1 != null ? ` · recon L1 ${i.recon_l1}/255` : '';
  const dist = i.dist, eps = i.eps;
  document.getElementById('dist').textContent = dist == null ? '–' : dist.toFixed(2);
  document.getElementById('eps').textContent = eps == null ? '–' : eps;
  if (dist != null && eps != null) {
    const span = Math.max(eps * 4, dist, 1e-6);
    document.getElementById('bari').style.width = Math.min(100, 100 * dist / span) + '%';
    document.getElementById('bari').style.background = dist < eps ? 'var(--ok)' : 'var(--acc)';
  }
  document.getElementById('gaugemeta').textContent =
    (dist != null && eps != null && dist < eps ? 'INSIDE reach eps' : '') ;
  const goals = (s.goals || []).slice().reverse();
  if (goals.length) {
    const cur = goals[0];
    document.getElementById('goal').src = '/img/goal/' + cur.id + '.jpg';
    document.getElementById('goalmeta').innerHTML =
      `#${cur.id} set @ step ${cur.step} <span class="tag ${cur.reason}">${cur.reason}</span>` +
      (cur.mse != null ? ` · wm-mse ${(+cur.mse).toFixed(3)}` : '');
  }
  const strip = document.getElementById('strip');
  const want = goals.map(g => g.id).join();
  if (strip.dataset.ids !== want || true) {   // outcomes mutate in place — rebuild text, reuse imgs
    strip.dataset.ids = want;
    strip.innerHTML = goals.map(g => {
      const out = g.outcome ? `<span class="tag ${g.outcome === 'replaced' ? 'refresh' : g.outcome}">${g.outcome}</span>` +
                              (g.min_dist != null ? ` min ${(+g.min_dist).toFixed(2)}` : '')
                            : '<span class="tag live">pursuing</span>';
      return `<div class=g><img loading=lazy src="/img/goal/${g.id}.jpg">` +
             `<div class=meta>#${g.id} @${g.step} <span class="tag ${g.reason}">${g.reason}</span><br>${out}</div></div>`;
    }).join('');
  }
  const c = document.getElementById('chart'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const h = s.dist_hist || [];
  if (h.length > 1) {
    const xs = h.map(p => p[0]), ys = h.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), ymax = Math.max(...ys, eps || 0) * 1.08;
    const X = x => 4 + (c.width - 8) * (x - x0) / Math.max(x1 - x0, 1);
    const Y = y => c.height - 4 - (c.height - 8) * y / Math.max(ymax, 1e-6);
    if (eps != null) { ctx.strokeStyle = '#3fb95066'; ctx.setLineDash([4,3]); ctx.beginPath();
      ctx.moveTo(0, Y(eps)); ctx.lineTo(c.width, Y(eps)); ctx.stroke(); ctx.setLineDash([]); }
    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.4; ctx.beginPath();
    h.forEach((p, k) => k ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])));
    ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
    ctx.fillText(ymax.toFixed(1), 6, 12); ctx.fillText(String(x1), c.width - 46, c.height - 8);
  }
}
tick(); setInterval(tick, 300);
</script></body></html>
"""
