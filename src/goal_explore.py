"""Goal archive for the --goal-explore (goal-conditioned Go-Explore) training mode.

A Go-Explore-style archive of the highest one-step-WM-MSE states the agent has seen.
Design (see train.py --goal-explore):
  * Holds RAW observations o* (uint8 pixels + float32 proprio), NEVER latents: z* is
    re-encoded from o* at sample time, so the archive is immune to encoder drift, and a
    ring index into the ReplayBuffer would dangle as the ring overwrites.
  * The GOAL obs is the surprising OUTCOME o_{t+1} (where curiosity novelty lives); the
    SOURCE (o_t, action) is kept alongside so a stored goal can be FAITHFULLY re-measured
    with the current WM (predict o_{t+1} from (o_t, a) and compare) -- the real one-step
    MSE, not a stationary proxy.
  * GLOBAL pool shared across the parallel (homogeneous) envs; each env samples its own
    goal. Sampling is softmax(score / temp) over the top-K -- a distribution over high-
    surprise goals, NOT argmax.
  * `score` is the TRUE r_cur (per-dim-mean one-step MSE) measured at capture, used for
    RANKING and P(k) so it matches "highest-MSE states seen" exactly.
  * `ref` is the SAME-CONSTRUCTION re-measure (score_obs_mse with the WM-at-capture) stored
    so eviction compares like-with-like: rescore_and_evict re-runs score_obs_mse with the
    CURRENT WM and evicts a goal once its MSE has fallen below drop_frac * ref. (Comparing a
    fabricated-context re-measure against the real-history capture r_cur would be apples-to-
    oranges, so `ref` -- not `score` -- is the eviction baseline.)

This module is intentionally torch-free (pure numpy): the WM re-measure is injected as a
`scorer` callable from train.py (which owns the WM/encoder), keeping the archive unit-
testable in isolation.
"""

import numpy as np


class GoalArchive:
    def __init__(self, K, img_hw, prop_dim, a_dim):
        self.K = int(K)
        s = (self.K,)
        self.gpx = np.zeros((*s, img_hw, img_hw, 3), np.uint8)   # goal obs o_{t+1}: pixels
        self.gprop = np.zeros((*s, prop_dim), np.float32)        # goal obs o_{t+1}: proprio
        self.spx = np.zeros((*s, img_hw, img_hw, 3), np.uint8)   # source obs o_t: pixels (for re-measure)
        self.sprop = np.zeros((*s, prop_dim), np.float32)        # source obs o_t: proprio
        self.sact = np.zeros((*s, a_dim), np.float32)            # action a_t that produced the MSE
        self.score = np.zeros(s, np.float64)                     # true r_cur at capture (ranking / P(k))
        self.ref = np.zeros(s, np.float64)                       # score_obs_mse at capture (eviction baseline)
        self.n = 0

    # ---- insertion (keep the top-K by score) ----
    def _insert_one(self, gpx, gprop, spx, sprop, sact, score, ref):
        if self.n < self.K:
            i = self.n
            self.n += 1
        else:
            i = int(np.argmin(self.score))
            if score <= self.score[i]:          # archive full and candidate is not better -> drop
                return
        self.gpx[i] = gpx
        self.gprop[i] = gprop
        self.spx[i] = spx
        self.sprop[i] = sprop
        self.sact[i] = sact
        self.score[i] = score
        self.ref[i] = ref

    def insert_batch(self, gpx, gprop, spx, sprop, sact, score, ref):
        for j in range(len(score)):
            self._insert_one(gpx[j], gprop[j], spx[j], sprop[j], sact[j],
                             float(score[j]), float(ref[j]))

    # ---- sampling (softmax over scores; a distribution, not argmax) ----
    def _probs(self, temp):
        s = self.score[:self.n]
        logits = (s - s.max()) / max(temp, 1e-6)
        p = np.exp(logits)
        return p / p.sum()

    def sample(self, temp):
        """Return one entry index drawn with P(k) = softmax(score/temp), or None if empty."""
        if self.n == 0:
            return None
        return int(np.random.choice(self.n, p=self._probs(temp)))

    def sample_indices(self, m, temp):
        if self.n == 0:
            return None
        return np.random.choice(self.n, size=m, p=self._probs(temp))

    # ---- re-measure + evict mastered goals ----
    def rescore_and_evict(self, scorer, drop_frac):
        """scorer(gpx, gprop, spx, sprop, sact) -> (n,) freshly-measured one-step MSE under
        the CURRENT WM (same construction as the stored `ref`). Evict any entry whose fresh
        MSE has fallen below drop_frac * ref (the WM has since learned that region). Returns
        #evicted."""
        if self.n == 0:
            return 0
        now = np.asarray(scorer(self.gpx[:self.n], self.gprop[:self.n],
                                self.spx[:self.n], self.sprop[:self.n],
                                self.sact[:self.n]), np.float64)
        keep = now >= drop_frac * self.ref[:self.n]
        idx = np.where(keep)[0]
        m = len(idx)
        if m != self.n:
            self.gpx[:m] = self.gpx[idx]
            self.gprop[:m] = self.gprop[idx]
            self.spx[:m] = self.spx[idx]
            self.sprop[:m] = self.sprop[idx]
            self.sact[:m] = self.sact[idx]
            self.score[:m] = self.score[idx]
            self.ref[:m] = self.ref[idx]
        evicted = self.n - m
        self.n = m
        return int(evicted)

    def mean_score(self):
        return float(self.score[:self.n].mean()) if self.n else 0.0

    # ---- checkpoint (de)serialization ----
    def state_dict(self):
        n = self.n
        return {"K": self.K, "gpx": self.gpx[:n].copy(), "gprop": self.gprop[:n].copy(),
                "spx": self.spx[:n].copy(), "sprop": self.sprop[:n].copy(),
                "sact": self.sact[:n].copy(), "score": self.score[:n].copy(),
                "ref": self.ref[:n].copy()}

    def load_state_dict(self, d):
        n = min(int(len(d["score"])), self.K)
        self.gpx[:n] = d["gpx"][:n]
        self.gprop[:n] = d["gprop"][:n]
        self.spx[:n] = d["spx"][:n]
        self.sprop[:n] = d["sprop"][:n]
        self.sact[:n] = d["sact"][:n]
        self.score[:n] = d["score"][:n]
        if "ref" in d:                          # back-compat: pre-ref checkpoints fall back to score
            self.ref[:n] = d["ref"][:n]
        else:
            self.ref[:n] = d["score"][:n]
        self.n = n
