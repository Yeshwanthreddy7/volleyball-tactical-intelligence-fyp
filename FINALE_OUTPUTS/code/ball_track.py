"""Offline ball tracking from per-frame detector candidates.

Why offline: the fine-tuned ball detector fires every frame, but most of its
candidates are false - fixed spots (net-banner logo, crowd, scoreboard) that
fire in thousands of frames. A greedy online tracker locks onto them and the
ball marker "sticks". With the whole segment available we can instead

  1. drop fixed-position false positives (global + local occupancy),
  2. link the remaining candidates into short motion-consistent tracklets
     (constant-velocity gating, Hungarian assignment per frame),
  3. pick the single best chain of tracklets over time with dynamic
     programming - consecutive tracklets must be reachable at ball speed,
  4. fill short gaps between chained tracklets with a cubic Hermite curve
     that matches the end velocities (smooth, no teleport, no freeze),
  5. light Savitzky-Golay smoothing.

Frames where the ball truly cannot be seen for > MAX_FILL frames stay empty,
so the marker disappears instead of freezing.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

CELL = 8
MAX_FILL = 36          # frames (1.2 s) a gap may be bridged
VMAX = 48.0            # px / frame, fastest spike seen in 1280x720
MIN_LEN = 4


def static_mask(frames, n):
    """Per-candidate flag: True when the spot is a fixed false positive."""
    occ = defaultdict(lambda: np.zeros(n, np.uint8))
    for fi, fr in enumerate(frames):
        for (x, y, w, c) in fr["balls"]:
            occ[(int(x // CELL), int(y // CELL))][fi] = 1
    tot = {k: a.sum() for k, a in occ.items()}
    kern = np.ones(91, np.float32)
    local = {k: np.convolve(a.astype(np.float32), kern, "same") / 91.0
             for k, a in occ.items() if tot[k] >= 25}

    flags = []
    for fi, fr in enumerate(frames):
        f = []
        for (x, y, w, c) in fr["balls"]:
            cx, cy = int(x // CELL), int(y // CELL)
            g = sum(tot.get((cx + dx, cy + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            lo = max((local[(cx + dx, cy + dy)][fi] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      if (cx + dx, cy + dy) in local), default=0.0)
            scoreboard = y < 48 and (430 < x < 850 or x < 90 or x > 1190)
            f.append(g > 0.04 * n or lo > 0.35 or scoreboard)
        flags.append(f)
    return flags


def build_tracklets(frames, flags):
    n = len(frames)
    active, done = [], []
    for fi in range(n):
        cands = [(x, y, w, c) for (x, y, w, c), s in zip(frames[fi]["balls"], flags[fi]) if not s]
        # retire tracklets not seen for > 3 frames
        still = []
        for t in active:
            (done if fi - t["f"][-1] > 3 else still).append(t)
        active = still
        used = set()
        if active and cands:
            cost = np.full((len(active), len(cands)), 1e6)
            for i, t in enumerate(active):
                dt = fi - t["f"][-1]
                pred = t["p"][-1] + t["v"] * dt
                gate = 30 + 0.6 * np.linalg.norm(t["v"]) * dt + 12 * dt
                for j, (x, y, w, c) in enumerate(cands):
                    d = math.hypot(x - pred[0], y - pred[1])
                    if d < gate:
                        cost[i, j] = d - 20 * c
            r, cidx = linear_sum_assignment(cost)
            for i, j in zip(r, cidx):
                if cost[i, j] >= 1e5:
                    continue
                t = active[i]
                x, y, w, c = cands[j]
                p = np.array([x, y], np.float32)
                dt = fi - t["f"][-1]
                nv = (p - t["p"][-1]) / dt
                t["v"] = nv if len(t["f"]) == 1 else 0.6 * nv + 0.4 * t["v"]
                t["f"].append(fi); t["p"].append(p); t["w"].append(w); t["c"].append(c)
                used.add(j)
        for j, (x, y, w, c) in enumerate(cands):
            if j not in used:
                active.append(dict(f=[fi], p=[np.array([x, y], np.float32)], w=[w], c=[c],
                                   v=np.zeros(2, np.float32)))
    done.extend(active)

    out = []
    for t in done:
        if len(t["f"]) < MIN_LEN:
            continue
        P = np.array(t["p"])
        span = np.ptp(P, axis=0).max()
        if span < 12 and len(t["f"]) < 60:
            continue                                  # a stationary blob, not a ball in play
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(t["f"]), 1)
        if np.median(steps) > VMAX:
            continue
        # ball flight is smooth (gravity only); detector noise jitters frame to frame
        f = np.array(t["f"])
        dd = np.diff(P, axis=0)
        consec = (np.diff(f)[:-1] == 1) & (np.diff(f)[1:] == 1)
        acc = np.linalg.norm(np.diff(dd, axis=0), axis=1)[consec]
        jitter = float(np.median(acc)) if len(acc) else 4.0
        cmean = float(np.mean(t["c"]))
        if jitter > 8 and cmean < 0.3:
            continue
        smooth = 1.0 if jitter < 5 else (0.5 if jitter < 8 else 0.2)
        out.append(dict(f=f, p=P, w=np.array(t["w"]), c=np.array(t["c"]), jitter=jitter,
                        score=float(smooth * (np.sum(t["c"]) + 0.05 * len(f) + min(span, 200) / 100.0))))
    out.sort(key=lambda t: t["f"][0])
    return out


def _end_vel(t, at_end):
    f, p = t["f"], t["p"]
    k = min(4, len(f) - 1)
    if at_end:
        return (p[-1] - p[-1 - k]) / max(f[-1] - f[-1 - k], 1)
    return (p[k] - p[0]) / max(f[k] - f[0], 1)


def choose_chain(tracks):
    """Weighted chain selection: maximise total score of non-overlapping,
    physically reachable tracklets."""
    m = len(tracks)
    best = np.zeros(m)
    prev = -np.ones(m, int)
    ends = np.array([t["f"][-1] for t in tracks])
    for i in range(m):
        t = tracks[i]
        s0 = t["f"][0]
        b, pb = t["score"], -1
        # predecessors that ended shortly before; unreachable ones cost a restart
        cand = np.where((ends < s0) & (ends >= s0 - 400))[0]
        for j in cand:
            gap = s0 - ends[j]
            d = np.linalg.norm(t["p"][0] - tracks[j]["p"][-1])
            linked = d <= VMAX * gap + 40
            if not linked and gap < 15:
                continue            # two balls this close in time: one of them is false
            val = best[j] + t["score"] - (0.0 if linked else 1.5)
            if val > b:
                b, pb = val, j
        # a chain that ended long before
        older = np.where(ends < s0 - 15)[0]
        if len(older):
            j = older[np.argmax(best[older])]
            if best[j] + t["score"] - 1.5 > b:
                b, pb = best[j] + t["score"] - 1.5, j
        best[i], prev[i] = b, pb
    i = int(np.argmax(best))
    chain = []
    while i >= 0:
        chain.append(i)
        i = prev[i]
    return [tracks[i] for i in reversed(chain)]


def hermite(p0, v0, p1, v1, steps):
    ts = np.linspace(0, 1, steps + 2)[1:-1]
    T = steps + 1
    h00 = 2 * ts ** 3 - 3 * ts ** 2 + 1
    h10 = ts ** 3 - 2 * ts ** 2 + ts
    h01 = -2 * ts ** 3 + 3 * ts ** 2
    h11 = ts ** 3 - ts ** 2
    return (h00[:, None] * p0 + h10[:, None] * v0 * T + h01[:, None] * p1 + h11[:, None] * v1 * T)


def track_ball(frames):
    """Returns (pos (n,2) float with NaN where unknown, size (n,), source (n,) 0=none 1=det 2=filled)."""
    n = len(frames)
    flags = static_mask(frames, n)
    tracks = build_tracklets(frames, flags)
    chain = choose_chain(tracks)

    pos = np.full((n, 2), np.nan, np.float32)
    size = np.full(n, np.nan, np.float32)
    src = np.zeros(n, np.uint8)
    for t in chain:
        for f, p, w in zip(t["f"], t["p"], t["w"]):
            pos[f], size[f], src[f] = p, w, 1
    for a, b in zip(chain, chain[1:]):
        f0, f1 = a["f"][-1], b["f"][0]
        gap = f1 - f0 - 1
        if gap <= 0 or gap > MAX_FILL:
            continue
        p0, p1 = a["p"][-1], b["p"][0]
        if np.linalg.norm(p1 - p0) > VMAX * (gap + 1) + 40:
            continue
        v0 = np.clip(_end_vel(a, True), -VMAX, VMAX)
        v1 = np.clip(_end_vel(b, False), -VMAX, VMAX)
        # tangents no longer than the gap itself: a bridge can bend, never loop
        chord = np.linalg.norm(p1 - p0)
        for v in (v0, v1):
            L = np.linalg.norm(v) * (gap + 1)
            if L > chord and L > 0:
                v *= chord / L
        pos[f0 + 1:f1] = hermite(p0, v0, p1, v1, gap)
        size[f0 + 1:f1] = np.linspace(a["w"][-1], b["w"][0], gap + 2)[1:-1]
        src[f0 + 1:f1] = 2
    # fill tiny holes inside a tracklet (detector blinked for 1-3 frames)
    known = ~np.isnan(pos[:, 0])
    idx = np.where(known)[0]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= 4:
            for k in (0, 1):
                pos[a + 1:b, k] = np.interp(np.arange(a + 1, b), [a, b], [pos[a, k], pos[b, k]])
            size[a + 1:b] = size[a]
            src[a + 1:b] = 2
    # smooth each continuous run
    known = ~np.isnan(pos[:, 0])
    runs, s = [], None
    for i in range(n + 1):
        k = i < n and known[i]
        if k and s is None:
            s = i
        elif not k and s is not None:
            runs.append((s, i)); s = None
    for a, b in runs:
        if b - a >= 7:
            pos[a:b] = savgol_filter(pos[a:b], 7, 2, axis=0)
    return pos, size, src, dict(n_tracklets=len(tracks), n_chain=len(chain))
