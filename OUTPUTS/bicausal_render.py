"""Pass 2 of the Bi-Causal style annotation: build the video from the cache.

Draws, in the style of Fig. 1(b) of Zhang et al., "Bi-Causal: Group Activity
Recognition via Bidirectional Causality" (CVPR 2024):

  * Human Relations (R): per-team relation graph between the players
    (minimum spanning tree in court metres, plus faint all-pairs edges),
    team colour (far = red, near = blue) and the team formation label,
    e.g. "2-Front 4-Back".
  * Human-Object Interaction (I): the ball, its trail, the link between the
    ball and the player touching it, a star on that key actor and the action
    name (SERVE / RECEPTION / SET / SPIKE / BLOCK / DIG / PASS / FREE BALL).
  * A side panel that reproduces the paper's t1-t2-t3 layout: for the last
    three touches the relation snapshot (R) next to the interaction crop (I),
    with red R->I and blue I->R causal arrows between consecutive steps.

Actions come from ball-touch events (sharp changes in the ball trajectory
next to a player) and the volleyball touch order, refined with pose (contact
above the head = overhand). They are rule-based, not a trained classifier.

Usage:
  python bicausal_render.py cache_vp1.pkl --out "videoplayback(1)_bicausal_annotated.mp4"
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- court model
# Image points (1280x720, fixed end-court camera) measured on each video's
# empty-court base frame (BASEFRAMES/base_frame) and their court coordinates in
# metres. X across the court (0..9), Y from the far end line (0) to the near
# end line (18). Every set was cross-checked with the perspective law: court
# width in pixels is linear in image y, and the lines it predicts for the
# unused court marks (attack lines / near end line) land on the painted ones.
COURTS = {
    "videoplayback (1).mp4": (
        [[385, 275], [895, 275], [268, 392], [1012, 392], [200, 460], [1080, 460]],
        [[0, 0], [9, 0], [0, 9], [9, 9], [0, 12], [9, 12]]),
    "videoplayback (7).mp4": (
        [[290, 410], [1015, 410], [234, 480], [1070, 480], [200, 522], [1103, 522],
         [160, 572], [1142, 572], [55, 703], [1245, 703]],
        [[0, 0], [9, 0], [0, 6], [9, 6], [0, 9], [9, 9], [0, 12], [9, 12], [0, 18], [9, 18]]),
    # From BASEFRAMES/corner_court_detection/labels/videoplayback (2).json (fit_rmse_px=0.465)
    "videoplayback (2).mp4": (
        [[279.8, 385.9], [1006.7, 386.0], [224.0, 457.0], [1062.6, 457.0],
         [187.9, 503.0], [1098.7, 502.8], [146.1, 556.2], [1140.6, 555.9],
         [32.9, 700.3], [1254.0, 699.9]],
        [[0, 0], [9, 0], [0, 6], [9, 6], [0, 9], [9, 9], [0, 12], [9, 12], [0, 18], [9, 18]]),
}
H = H_INV = None


def set_court(video_name):
    global H, H_INV
    img, crt = COURTS[video_name]
    H, _ = cv2.findHomography(np.float32(img), np.float32(crt))
    H_INV = np.linalg.inv(H)


set_court("videoplayback (1).mp4")

FAR, NEAR = 0, 1
TEAM_NAME = {FAR: "Far team", NEAR: "Near team"}
TEAM_BGR = {FAR: (60, 60, 235), NEAR: (235, 120, 30)}        # red / blue as in the paper
TEAM_RGB = {k: (v[2], v[1], v[0]) for k, v in TEAM_BGR.items()}
BALL_BGR = (0, 225, 255)
R2I_RGB = (235, 60, 60)
I2R_RGB = (40, 120, 235)

MAIN_W, MAIN_H, PANEL_W = 1280, 720, 480

ACTION_RGB = {
    "SERVE": (250, 170, 30), "RECEPTION": (60, 180, 90), "PASS": (60, 180, 90),
    "DIG": (40, 160, 160), "SET": (220, 50, 50), "SPIKE": (200, 30, 120),
    "BLOCK": (120, 70, 200), "FREE BALL": (110, 110, 110),
}


def to_court(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_img(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_INV).reshape(-1, 2)


# ---------------------------------------------------------------- players
def kp_point(kp, idx, thr=0.4):
    if kp is None:
        return None
    pts = [kp[i, :2] for i in idx if kp[i, 2] >= thr]
    return np.mean(pts, axis=0) if pts else None


def person_geometry(p):
    x1, y1, x2, y2 = p["box"]
    kp = p["kp"]
    feet = kp_point(kp, (15, 16))
    if feet is None or y2 >= MAIN_H - 4:
        feet = np.array([(x1 + x2) / 2, y2])
    node = kp_point(kp, (11, 12))
    if node is None:
        node = np.array([(x1 + x2) / 2, y1 + 0.55 * (y2 - y1)])
    head = kp_point(kp, (0, 1, 2), 0.3)
    if head is None:
        head = np.array([(x1 + x2) / 2, y1 + 0.1 * (y2 - y1)])
    sh = kp_point(kp, (5, 6), 0.3)
    if sh is None:
        sh = np.array([(x1 + x2) / 2, y1 + 0.22 * (y2 - y1)])
    wrists = [kp[i, :2] for i in (9, 10) if kp is not None and kp[i, 2] >= 0.3]
    return feet, node, head, sh, wrists


def build_players(frames):
    """Per frame list of players: dict(id, team, node, feet, court, head, sh, wrists, box)."""
    per_track = defaultdict(list)
    geo = []
    for fi, fr in enumerate(frames):
        g = []
        for p in fr["people"]:
            feet, node, head, sh, wrists = person_geometry(p)
            c = to_court(feet)[0]
            g.append(dict(id=p["id"], conf=p["conf"], box=p["box"], feet=feet, node=node,
                          head=head, sh=sh, wrists=wrists, court=c))
            if p["id"] >= 0:
                per_track[p["id"]].append(c)
        geo.append(g)

    track_team, track_ok = {}, {}
    for tid, cs in per_track.items():
        cs = np.array(cs)
        mx, my = np.median(cs[:, 0]), np.median(cs[:, 1])
        # a jump lifts the feet, which reads as "further from camera"; the
        # 75th percentile of depth ignores those airborne frames
        track_team[tid] = FAR if np.percentile(cs[:, 1], 75) < 9.0 else NEAR
        inside = np.mean((cs[:, 0] >= 0) & (cs[:, 0] <= 9) & (cs[:, 1] >= 0) & (cs[:, 1] <= 18))
        # players chase the ball outside the lines; coach / referees never enter the court
        track_ok[tid] = len(cs) >= 8 and ((-0.5 <= mx <= 9.5 and -10.0 <= my <= 20.0) or inside >= 0.3)

    out = []
    for g in geo:
        keep = []
        for q in g:
            X, Y = q["court"]
            if not (-1.5 <= X <= 10.5 and -10.5 <= Y <= 21.5):
                continue
            if q["id"] >= 0:
                if not track_ok.get(q["id"], False):
                    continue
                q["team"] = track_team[q["id"]]
            else:
                q["team"] = FAR if Y < 9 else NEAR
            keep.append(q)
        res = []
        for team in (FAR, NEAR):
            tq = [q for q in keep if q["team"] == team]
            # at most six per side: prefer players inside the court, then confidence
            tq.sort(key=lambda q: (_outside(q["court"]), -q["conf"]))
            res.extend(tq[:6])
        out.append(res)
    return out


def _outside(c):
    X, Y = c
    return max(0.0, -X, X - 9) + max(0.0, -Y, Y - 18)


# ---------------------------------------------------------------- ball
def build_ball(frames, players):
    n = len(frames)
    # 1) static-ball suppression: spare balls lying on the floor
    cell = 10
    occ = defaultdict(lambda: np.zeros(n, np.uint8))
    for fi, fr in enumerate(frames):
        for (x, y, w, c) in fr["balls"]:
            occ[(int(x // cell), int(y // cell))][fi] = 1
    win = 121
    kern = np.ones(win, np.float32)
    dens = {}
    for k, a in occ.items():
        if a.sum() >= 40:
            dens[k] = np.convolve(a.astype(np.float32), kern, "same") / win

    def static(fi, x, y):
        cx, cy = int(x // cell), int(y // cell)
        best = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                d = dens.get((cx + dx, cy + dy))
                if d is not None:
                    best = max(best, d[fi])
        return best > 0.45

    cands = []
    for fi, fr in enumerate(frames):
        cands.append([(x, y, w, c) for (x, y, w, c) in fr["balls"] if not static(fi, x, y)])

    # 2) greedy gated association into tracklets
    raw = np.full((n, 2), np.nan, np.float32)
    tid = np.full(n, -1, int)
    cur, pos, vel, last, gap = -1, None, np.zeros(2), -1, 0
    for fi in range(n):
        cs = cands[fi]
        chosen = None
        if pos is not None:
            dt = fi - last
            pred = pos + vel * dt
            gate = 45 + 22 * dt
            best = None
            for (x, y, w, c) in cs:
                d = math.hypot(x - pred[0], y - pred[1])
                if d < gate:
                    s = d - 25 * c
                    if best is None or s < best[0]:
                        best = (s, x, y)
            if best is not None:
                chosen = np.array(best[1:], np.float32)
                nv = (chosen - pos) / dt
                vel = 0.5 * vel + 0.5 * nv
            elif dt > 12:
                pos = None
        if pos is None and chosen is None:
            strong = [(c, x, y) for (x, y, w, c) in cs if c >= 0.35]
            if strong:
                c, x, y = max(strong)
                chosen = np.array([x, y], np.float32)
                cur += 1
                vel = np.zeros(2)
        if chosen is not None:
            raw[fi] = chosen
            tid[fi] = cur
            pos, last = chosen, fi

    # 3) drop short tracklets, interpolate gaps, smooth
    ball = np.full((n, 2), np.nan, np.float32)
    seg_of = np.full(n, -1, int)
    segs = []
    for t in range(cur + 1):
        idx = np.where(tid == t)[0]
        if len(idx) < 8:
            continue
        a, b = idx[0], idx[-1]
        xs = np.interp(np.arange(a, b + 1), idx, raw[idx, 0])
        ys = np.interp(np.arange(a, b + 1), idx, raw[idx, 1])
        if b - a >= 4:
            k = np.ones(3) / 3
            xs[1:-1] = np.convolve(xs, k, "valid")
            ys[1:-1] = np.convolve(ys, k, "valid")
        ball[a:b + 1, 0], ball[a:b + 1, 1] = xs, ys
        seg_of[a:b + 1] = len(segs)
        segs.append((a, b))
    detected = ~np.isnan(raw[:, 0]) & (seg_of >= 0)
    return ball, segs, detected


# ---------------------------------------------------------------- touches
def reach(q, bx, by):
    x1, y1, x2, y2 = q["box"]
    w, h = x2 - x1, y2 - y1
    dx = max(0.0, x1 - bx, bx - x2)
    if dx > 0.45 * w + 12:
        return None
    if not (y1 - 0.75 * h <= by <= y1 + 0.8 * h):
        return None
    pts = list(q["wrists"]) + [q["head"], q["sh"]]
    return min(math.hypot(bx - p[0], by - p[1]) for p in pts)


def nearest_player(players_f, bx, by):
    best = None
    for q in players_f:
        d = reach(q, bx, by)
        if d is not None and (best is None or d < best[0]):
            best = (d, q)
    return best


def find_touches(ball, segs, players):
    k = 4
    cand = []
    for (a, b) in segs:
        for t in range(a + k, b - k + 1):
            vin = ball[t] - ball[t - k]
            vout = ball[t + k] - ball[t]
            si, so = np.linalg.norm(vin), np.linalg.norm(vout)
            score = 0.0
            if si > 5 and so > 5:
                cosang = float(np.dot(vin, vout) / (si * so))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                if ang > 38:
                    score = ang / 38.0
            if so > 22 and so > 2.2 * max(si, 3):
                score = max(score, so / max(si, 3) / 2.2)
            # moving down then up in the image needs a contact (gravity cannot)
            if vin[1] > 4 and vout[1] < -6:
                score = max(score, 1.2)
            if score <= 0:
                continue
            np_ = nearest_player(players[t], *ball[t])
            if np_ is None:
                continue
            cand.append(dict(f=t, score=score, dist=np_[0], actor=np_[1], vout=vout / k, vin=vin / k))
    # gaps between tracklets that start again beside a player (contact hidden by the body)
    for (a0, b0), (a1, b1) in zip(segs, segs[1:]):
        if 0 < a1 - b0 <= 18 and b0 - a0 >= 4 and b1 - a1 >= 4:
            vin = ball[b0] - ball[b0 - 4]
            vout = ball[a1 + 4] - ball[a1]
            si, so = np.linalg.norm(vin), np.linalg.norm(vout)
            if si > 1 and so > 1 and np.dot(vin, vout) / (si * so) < 0.6:
                t = (b0 + a1) // 2
                mid = (ball[b0] + ball[a1]) / 2
                np_ = nearest_player(players[t], *mid)
                if np_ is not None:
                    cand.append(dict(f=t, score=1.0, dist=np_[0], actor=np_[1], vout=vout / 4, vin=vin / 4, ballpt=mid))
    # non-maximum suppression over 10 frames, stronger + closer wins
    cand.sort(key=lambda c: -(c["score"] - c["dist"] / 150.0))
    kept = []
    for c in cand:
        if all(abs(c["f"] - o["f"]) > 10 for o in kept):
            kept.append(c)
    kept.sort(key=lambda c: c["f"])
    for c in kept:
        c.setdefault("ballpt", ball[c["f"]].copy())
    return kept


def label_touches(touches, fps):
    """Volleyball touch-order rules -> action names. Returns list of event dicts."""
    events = []
    rally = []

    def flush():
        if rally:
            events.extend(_label_rally(rally, fps))

    for t in touches:
        if rally and t["f"] - rally[-1]["f"] > 4.0 * fps:
            flush()
            rally = []
        rally.append(t)
    flush()
    return events


def _label_rally(rally, fps):
    # a serve toss and the serve hit are both touches by the same player: keep the hit
    if len(rally) >= 2 and rally[0]["actor"]["id"] == rally[1]["actor"]["id"] and \
            rally[1]["f"] - rally[0]["f"] < 2.0 * fps and rally[0]["actor"]["id"] >= 0:
        rally = rally[1:]
    out = []
    prev_team, count, prev_label, prev_f = None, 0, None, -10 ** 9
    for i, t in enumerate(rally):
        q = t["actor"]
        team = q["team"]
        X, Y = q["court"]
        bx, by = t["ballpt"]
        x1, y1, x2, y2 = q["box"]
        overhead = by < q["head"][1] + 0.04 * (y2 - y1)
        front = abs(Y - 9.0) <= 3.6
        nxt = rally[i + 1] if i + 1 < len(rally) else None
        next_same = nxt is not None and nxt["actor"]["team"] == team
        speed = float(np.linalg.norm(t["vout"]))
        if team != prev_team:
            count = 1
        else:
            count += 1

        at_base = (team == FAR and Y < 1.2) or (team == NEAR and Y > 16.0)
        if i == 0 and at_base:
            label = "SERVE"
        elif prev_label == "SPIKE" and team != prev_team and abs(Y - 9.0) <= 1.8 and overhead and t["f"] - prev_f < 0.6 * fps:
            label = "BLOCK"
        elif prev_label == "SERVE" and team != prev_team:
            label = "RECEPTION"
        elif prev_label in ("SPIKE", "BLOCK") and count == 1 and not overhead:
            label = "DIG"
        elif next_same and overhead and count <= 2:
            label = "SET"
        elif not next_same and count >= 2:
            label = "SPIKE" if (overhead and speed > 7) else "FREE BALL"
        elif not next_same and nxt is not None and overhead and speed > 9 and front:
            label = "SPIKE"
        elif count == 1:
            label = "DIG" if prev_label in ("SPIKE", "FREE BALL") else "PASS"
        elif overhead:
            label = "SET"
        else:
            label = "PASS"
        if label == "BLOCK":
            count = 0
        out.append(dict(f=t["f"], label=label, team=team, actor_id=q["id"], actor=q,
                        ball=np.array(t["ballpt"]), count=count))
        prev_team, prev_label, prev_f = team, label, t["f"]
    return out


# ---------------------------------------------------------------- formations
def formations(players, n, win=31):
    raw = {FAR: [], NEAR: []}
    for fi in range(n):
        for team in (FAR, NEAR):
            front = back = 0
            for q in players[fi]:
                if q["team"] != team:
                    continue
                Y = q["court"][1]
                if abs(Y - 9.0) <= 3.5:
                    front += 1
                else:
                    back += 1
            raw[team].append((front, back))
    sm = {FAR: [], NEAR: []}
    h = win // 2
    for team in (FAR, NEAR):
        r = raw[team]
        for fi in range(n):
            c = Counter(r[max(0, fi - h):fi + h + 1])
            sm[team].append(c.most_common(1)[0][0])
    return sm


def fmt_formation(fb):
    f, b = fb
    return f"{f}-Front {b}-Back"


# ---------------------------------------------------------------- drawing helpers
def mst_edges(pts):
    m = len(pts)
    if m < 2:
        return []
    inside = {0}
    edges = []
    while len(inside) < m:
        best = None
        for i in inside:
            for j in range(m):
                if j in inside:
                    continue
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        edges.append((best[1], best[2]))
        inside.add(best[2])
    return edges


def orient_tree(edges, root):
    """Edges of an undirected tree, directed away from `root` (BFS order)."""
    adj = defaultdict(list)
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    out, seen, queue = [], {root}, [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                out.append((u, v))
                queue.append(v)
    return out


def draw_arrow(img, p, q, color, thick=3, shrink=0):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = np.linalg.norm(q - p)
    if d < 2 * shrink + 5:
        return
    u = (q - p) / d
    a, b = p + u * shrink, q - u * shrink
    tip = min(0.35, 22.0 / max(d, 1))
    cv2.arrowedLine(img, tuple(a.astype(int)), tuple(b.astype(int)), (30, 30, 30), thick + 3,
                    cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick,
                    cv2.LINE_AA, tipLength=tip)


def draw_curved_arrow(img, p, q, color):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = np.linalg.norm(q - p)
    if d < 30:
        return
    mid = (p + q) / 2 + np.array([0.0, -0.25 * d])          # arc upwards like a ball
    ts = np.linspace(0, 1, 24)[:, None]
    pts = (1 - ts) ** 2 * p + 2 * (1 - ts) * ts * mid + ts ** 2 * q
    pts = pts[1:-2]
    cv2.polylines(img, [pts.astype(np.int32)], False, (30, 30, 30), 6, cv2.LINE_AA)
    cv2.polylines(img, [pts.astype(np.int32)], False, color, 3, cv2.LINE_AA)
    draw_arrow(img, pts[-3], pts[-1], color, 3)


def star_pts(cx, cy, r_out, r_in, n=5):
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = -math.pi / 2 + k * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.int32)


def dashed_line(img, p, q, color, thick=2, dash=10):
    p, q = np.array(p, float), np.array(q, float)
    d = np.linalg.norm(q - p)
    if d < 1:
        return
    u = (q - p) / d
    s = 0.0
    while s < d:
        a = p + u * s
        b = p + u * min(s + dash, d)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)
        s += 2 * dash


def arrow(draw, p, q, rgb, width=3):
    draw.line([p, q], fill=rgb, width=width)
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    L = 12
    a1 = (q[0] - L * math.cos(ang - 0.45), q[1] - L * math.sin(ang - 0.45))
    a2 = (q[0] - L * math.cos(ang + 0.45), q[1] - L * math.sin(ang + 0.45))
    draw.polygon([q, a1, a2], fill=rgb)


def load_font(size, bold=False):
    names = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for nm in names:
        try:
            return ImageFont.truetype(str(Path("C:/Windows/Fonts") / nm), size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_box(draw, xy, text, font, fg, bg, outline=None, pad=6, anchor="la"):
    l, t, r, b = draw.textbbox(xy, text, font=font, anchor=anchor)
    draw.rounded_rectangle([l - pad, t - pad, r + pad, b + pad], radius=6, fill=bg,
                           outline=outline, width=3 if outline else 0)
    draw.text(xy, text, font=font, fill=fg, anchor=anchor)
    return (l - pad, t - pad, r + pad, b + pad)


def _court_sig(img):
    g = cv2.resize(cv2.cvtColor(img[300:], cv2.COLOR_BGR2GRAY), (96, 42),
                   interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def load_gt(labels_csv, video_name, start_s, end_s):
    gt = []
    if not Path(labels_csv).exists():
        return gt
    with open(labels_csv, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3 or row[0].strip() != video_name:
                continue
            try:
                m, s = row[1].split(":")
                ts = int(m) * 60 + float(s)
            except ValueError:
                continue
            if start_s - 2 <= ts <= end_s + 2:
                gt.append((ts, row[2].replace("_", " ")))
    return gt


# ---------------------------------------------------------------- main render
def render_frames(cache, out_stem, labels, max_frames=0, info=None):
    """Yield (fi, main_rgb, panel_rgb) for every frame of the cached segment.

    `info` (dict, optional) is filled with fps / start / n / events so callers
    can add their own overlays (e.g. the per-model tactical strip)."""
    with open(cache, "rb") as f:
        data = pickle.load(f)
    frames, fps, start = data["frames"], data["fps"], data["start"]
    if max_frames:
        frames = frames[:max_frames]
    n = len(frames)
    video = data["video"]
    video_name = Path(video).name
    set_court(video_name)

    print("players ...", flush=True)
    players = build_players(frames)
    print("ball ...", flush=True)
    ball, segs, detected = build_ball(frames, players)
    print(f"  ball visible in {np.mean(~np.isnan(ball[:, 0])) * 100:.0f}% of frames, {len(segs)} tracklets")
    touches = find_touches(ball, segs, players)
    events = label_touches(touches, fps)
    print(f"  {len(touches)} touches -> {len(events)} labelled events: "
          f"{Counter(e['label'] for e in events).most_common()}")
    forms = formations(players, n)
    gt = load_gt(labels, video_name, start / fps, (start + n) / fps)

    ev_csv = Path(str(out_stem) + ".events.csv")
    with open(ev_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_time", "frame", "action", "team", "actor_track_id",
                    "far_formation", "near_formation"])
        for e in events:
            ts = (start + e["f"]) / fps
            w.writerow([f"{int(ts // 60):02d}:{ts % 60:05.2f}", start + e["f"], e["label"],
                        TEAM_NAME[e["team"]], e["actor_id"],
                        fmt_formation(forms[FAR][e["f"]]), fmt_formation(forms[NEAR][e["f"]])])
    print("events ->", ev_csv)

    ev_at = {e["f"]: e for e in events}
    ev_index = {e["f"]: k for k, e in enumerate(events)}
    ev_frames = np.array([e["f"] for e in events]) if events else np.array([], int)

    f_big = load_font(34, True)
    f_mid = load_font(22, True)
    f_sm = load_font(17, True)
    f_xs = load_font(15)
    f_tiny = load_font(13, True)

    court_far = to_img(np.float32([[0, 0], [9, 0], [9, 9], [0, 9]])).astype(np.int32)
    court_near = to_img(np.float32([[0, 9], [9, 9], [9, 18], [0, 18]])).astype(np.int32)

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if info is not None:
        info.update(fps=fps, start=start, n=n, events=events, video=video, players=players)

    base_png = HERE.parent / "BASEFRAMES" / "base_frame" / (Path(video_name).stem + ".png")
    base_sig = _court_sig(cv2.imread(str(base_png))) if base_png.exists() else None

    history = []          # rendered snapshots of past events for the side panel
    trail = []
    for fi in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        raw_frame = frame.copy()
        img = frame
        if base_sig is not None and float((_court_sig(frame) * base_sig).mean()) < 0.45:
            # replay / camera change: the court model does not apply, draw nothing
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text_box(ImageDraw.Draw(pil), (MAIN_W // 2, MAIN_H - 22), "camera change - overlays paused",
                     f_mid, (255, 255, 255), (60, 60, 60), anchor="md")
            yield fi, np.array(pil), np.full((MAIN_H, PANEL_W, 3), 18, np.uint8)
            continue

        # -- team half shading (the paper tints the relation region)
        shade = img.copy()
        cv2.fillPoly(shade, [court_far], TEAM_BGR[FAR])
        cv2.fillPoly(shade, [court_near], TEAM_BGR[NEAR])
        cv2.addWeighted(shade, 0.13, img, 0.87, 0, img)

        # current / last event
        past = ev_frames[ev_frames <= fi + 6]
        cur_ev = ev_at[int(past[-1])] if len(past) and fi - past[-1] <= int(1.6 * fps) else None
        key_q = None
        if cur_ev is not None:
            cands = [q for q in players[fi] if q["team"] == cur_ev["team"]]
            same = [q for q in cands if q["id"] == cur_ev["actor_id"] and q["id"] >= 0]
            if same:
                key_q = same[0]
            elif cands:
                # tracker id changed: take the player closest to the event actor
                near = min(cands, key=lambda q: np.linalg.norm(q["node"] - cur_ev["actor"]["node"]))
                if np.linalg.norm(near["node"] - cur_ev["actor"]["node"]) <= 120:
                    key_q = near

        # -- relation graphs (R)
        overlay = img.copy()
        for team in (FAR, NEAR):
            tq = [q for q in players[fi] if q["team"] == team]
            nodes = np.array([q["node"] for q in tq], np.float32)
            for i in range(len(tq)):
                for j in range(i + 1, len(tq)):
                    cv2.line(overlay, tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int)),
                             TEAM_BGR[team], 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        for team in (FAR, NEAR):
            tq = [q for q in players[fi] if q["team"] == team]
            if len(tq) >= 2:
                nodes = np.array([q["node"] for q in tq], np.float32)
                courts = np.array([q["court"] for q in tq], np.float32)
                edges = mst_edges(courts)
                root = next((i for i, q in enumerate(tq) if q is key_q), None)
                if root is None:
                    for i, j in edges:
                        p, q = tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int))
                        cv2.line(img, p, q, (30, 30, 30), 6, cv2.LINE_AA)
                        cv2.line(img, p, q, (255, 255, 255), 3, cv2.LINE_AA)
                else:
                    # acting team: the tree is directed away from the player on
                    # the ball, i.e. the interaction (I) driving the relations (R)
                    for i, j in orient_tree(edges, root):
                        draw_arrow(img, nodes[i], nodes[j], (255, 255, 255), 3, shrink=12)
            for q in tq:
                if q is key_q:
                    continue
                c = tuple(q["node"].astype(int))
                cv2.circle(img, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, 7, TEAM_BGR[team], -1, cv2.LINE_AA)
                fx, fy = q["feet"].astype(int)
                cv2.ellipse(img, (fx, fy), (22, 7), 0, 0, 360, TEAM_BGR[team], 2, cv2.LINE_AA)

        # -- ball flow: previous actor -> current actor (interaction chain)
        if cur_ev is not None and key_q is not None:
            k_ev = ev_index[cur_ev["f"]]
            if k_ev > 0 and cur_ev["f"] - events[k_ev - 1]["f"] <= 3 * fps:
                pa = events[k_ev - 1]["actor"]["node"]
                draw_curved_arrow(img, pa, key_q["node"], BALL_BGR)

        # -- ball + interaction (I)
        b = ball[fi]
        has_ball = not np.isnan(b[0])
        if has_ball:
            trail.append((fi, b.copy()))
        trail = [(f, p) for f, p in trail if fi - f <= 14]
        for k in range(1, len(trail)):
            if trail[k][0] - trail[k - 1][0] > 3:
                continue
            a = k / len(trail)
            cv2.line(img, tuple(trail[k - 1][1].astype(int)), tuple(trail[k][1].astype(int)),
                     BALL_BGR, max(1, int(4 * a)), cv2.LINE_AA)
        if key_q is not None:
            fx, fy = key_q["feet"].astype(int)
            cv2.ellipse(img, (fx, fy), (30, 10), 0, 0, 360, BALL_BGR, 3, cv2.LINE_AA)
            if has_ball:
                tgt = min(key_q["wrists"] + [key_q["head"]], key=lambda p: np.linalg.norm(p - b))
                dashed_line(img, b, tgt, BALL_BGR, 2, 8)
            cx, cy = key_q["node"].astype(int)
            cv2.fillPoly(img, [star_pts(cx, cy, 20, 8)], (255, 255, 255), cv2.LINE_AA)
            cv2.fillPoly(img, [star_pts(cx, cy, 16, 6)], TEAM_BGR[key_q["team"]], cv2.LINE_AA)
        if has_ball:
            bp = tuple(b.astype(int))
            cv2.circle(img, bp, 13, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(img, bp, 11, BALL_BGR, 3, cv2.LINE_AA)

        # -- snapshot for the side panel at the event frame
        if fi in ev_at:
            e = ev_at[fi]
            thumb = cv2.resize(img, (224, 126), interpolation=cv2.INTER_AREA)
            ax, ay = e["actor"]["node"]
            x1, y1, x2, y2 = e["actor"]["box"]
            side = int(max(y2 - y1, x2 - x1) * 1.5) + 40
            cx0 = int(np.clip((x1 + x2) / 2 - side / 2, 0, MAIN_W - side))
            cy0 = int(np.clip(min(y1, e["ball"][1]) - 0.25 * side, 0, MAIN_H - side))
            crop = raw_frame[cy0:cy0 + side, cx0:cx0 + side].copy()
            bx, by = e["ball"] - np.array([cx0, cy0])
            cv2.circle(crop, (int(bx), int(by)), max(6, side // 18), BALL_BGR, 3, cv2.LINE_AA)
            crop = cv2.resize(crop, (126, 126), interpolation=cv2.INTER_AREA)
            history.append(dict(e=e, thumb=thumb, crop=crop,
                                form=fmt_formation(forms[e["team"]][fi])))
            history = history[-3:]

        # -- compose: main frame + separate side panel
        panel = np.full((MAIN_H, PANEL_W, 3), 18, np.uint8)
        row_y = [70, 250, 430]
        for k, h in enumerate(history):
            y = row_y[k]
            panel[y:y + 126, 16:16 + 224] = h["thumb"]
            panel[y:y + 126, 330:330 + 126] = h["crop"]
            cv2.rectangle(panel, (15, y - 1), (241, y + 127), TEAM_BGR[h["e"]["team"]], 2)
            cv2.rectangle(panel, (329, y - 1), (457, y + 127), (230, 230, 230), 2)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        ppil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        dp = ImageDraw.Draw(ppil)

        # main-frame text: formations (R) per team
        text_box(d, (14, 50), f"Far team:  {fmt_formation(forms[FAR][fi])}", f_mid,
                 (20, 20, 20), (255, 255, 255), TEAM_RGB[FAR])
        text_box(d, (14, MAIN_H - 22), f"Near team:  {fmt_formation(forms[NEAR][fi])}", f_mid,
                 (20, 20, 20), (255, 255, 255), TEAM_RGB[NEAR], anchor="ld")
        ts = (start + fi) / fps
        text_box(d, (MAIN_W - 14, 50), f"{video_name}   {int(ts // 60):02d}:{ts % 60:04.1f}", f_sm,
                 (255, 255, 255), (20, 20, 20), anchor="ra")
        # action (I)
        if cur_ev is not None:
            lab = cur_ev["label"]
            text_box(d, (MAIN_W // 2, MAIN_H - 22), f"{lab}  -  {TEAM_NAME[cur_ev['team']]}", f_big,
                     (255, 255, 255), ACTION_RGB[lab], (255, 255, 255), pad=10, anchor="md")
            if key_q is not None:
                x1, y1, x2, y2 = key_q["box"]
                text_box(d, ((x1 + x2) / 2, y1 - 30), lab.lower().replace(" ", "_"), f_sm,
                         (255, 255, 255), ACTION_RGB[lab], anchor="md", pad=4)

        # panel text
        dp.text((16, 12), "Bi-Causal view", font=f_mid, fill=(255, 255, 255))
        dp.text((16, 40), "Human relation (R)", font=f_xs, fill=(200, 200, 200))
        dp.text((330, 40), "Human-object\ninteraction (I)", font=f_tiny, fill=(200, 200, 200))
        for k, h in enumerate(history):
            y = row_y[k]
            e = h["e"]
            dp.text((6, y + 132), f"t{k + 1}", font=f_sm, fill=(255, 255, 255))
            ets = (start + e["f"]) / fps
            dp.text((40, y + 134), f"{int(ets // 60):02d}:{ets % 60:04.1f}  {TEAM_NAME[e['team']]}",
                   font=f_xs, fill=(190, 190, 190))
            text_box(dp, (20, y + 120), h["form"], f_tiny, (20, 20, 20), (255, 255, 255), anchor="ld", pad=3)
            text_box(dp, (393, y + 120), e["label"].lower().replace(" ", " "), f_tiny, (255, 255, 255),
                     ACTION_RGB[e["label"]], anchor="md", pad=3)
            if k + 1 < len(history):
                yn = row_y[k + 1]
                # R(t_k) -> I(t_k+1) red, I(t_k) -> R(t_k+1) blue, as in Fig. 1(b)
                arrow(dp, (244, y + 70), (326, yn + 50), R2I_RGB)
                arrow(dp, (326, y + 90), (244, yn + 40), I2R_RGB)
        if not history:
            dp.text((16, 120), "waiting for the first ball touch ...", font=f_xs, fill=(160, 160, 160))

        # legend
        ly = 612
        dp.line([(16, ly + 8), (40, ly + 8)], fill=R2I_RGB, width=3)
        dp.text((46, ly), "R->I causal effect", font=f_xs, fill=(230, 230, 230))
        dp.line([(200, ly + 8), (224, ly + 8)], fill=I2R_RGB, width=3)
        dp.text((230, ly), "I->R causal effect", font=f_xs, fill=(230, 230, 230))
        ly += 26
        dp.ellipse([16, ly + 2, 30, ly + 16], fill=TEAM_RGB[FAR], outline=(255, 255, 255), width=2)
        dp.text((36, ly), "far team", font=f_xs, fill=(230, 230, 230))
        dp.ellipse([116, ly + 2, 130, ly + 16], fill=TEAM_RGB[NEAR], outline=(255, 255, 255), width=2)
        dp.text((136, ly), "near team", font=f_xs, fill=(230, 230, 230))
        dp.polygon([tuple(p) for p in star_pts(232, ly + 9, 9, 4)], fill=(255, 220, 0))
        dp.text((246, ly), "key actor", font=f_xs, fill=(230, 230, 230))
        dp.ellipse([330, ly + 2, 344, ly + 16], outline=(255, 225, 0), width=3)
        dp.text((350, ly), "ball", font=f_xs, fill=(230, 230, 230))
        ly += 26
        dp.text((16, ly), "white arrows = relation graph (MST) from the player on the ball", font=f_xs,
               fill=(170, 170, 170))
        # ground-truth tactical label from labels.csv
        g = [lab for (t, lab) in gt if abs(t - ts) <= 1.0]
        ly += 24
        dp.text((16, ly), "Annotated tactic: " + (g[0] if g else "-"), font=f_sm,
               fill=(255, 210, 90) if g else (120, 120, 120))

        yield fi, np.array(pil), np.array(ppil)
        if fi % 600 == 0:
            print(f"render {fi}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--out", default=str(HERE / "videoplayback(1)_bicausal_annotated.mp4"))
    ap.add_argument("--labels", default=str(HERE.parent / "fyp" / "labels.csv"))
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    info = {}
    writer = None
    for fi, main_rgb, panel_rgb in render_frames(args.cache, Path(args.out).with_suffix(""),
                                                 args.labels, args.max_frames, info):
        frame_rgb = np.hstack([main_rgb, panel_rgb])
        if writer is None:
            writer = imageio_ffmpeg.write_frames(
                args.out, (frame_rgb.shape[1], frame_rgb.shape[0]), fps=info["fps"], codec="libx264",
                quality=None, pix_fmt_out="yuv420p",
                output_params=["-crf", "16", "-preset", "medium", "-movflags", "+faststart"],
                macro_block_size=16)
            writer.send(None)
        writer.send(np.ascontiguousarray(frame_rgb))
    writer.close()
    print("video ->", args.out)


if __name__ == "__main__":
    main()
