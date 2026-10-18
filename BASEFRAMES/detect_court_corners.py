"""
Detect the four corners of the full volleyball court (18 m x 9 m) in every
base frame, and save annotated images and label files.

Input : BASEFRAMES/base_frame/<video>.png
Output: BASEFRAMES/corner_court_detection/
          annotated/<video>.png   corners + fitted court model drawn on the frame
          labels/<video>.json     corners, homography, lines used, fit error
          yolo_pose/<video>.txt   YOLO keypoint label (court box + 4 corners)
          corners.csv             one row per video
          _contact_sheet.png      all annotated frames on one page

Method
------
1. White-line mask: top-hat on grey (thin bright structures) AND low saturation
   AND high brightness. Court lines are white on every court in the dataset.
2. Hough segments -> the two sidelines (steep, one per side, largest support),
   refined by a robust fit to the mask pixels along each line.
3. Cross lines (near-horizontal) between the sidelines: clustered and refined
   the same way. Candidates include the end lines, attack lines, centre line,
   and distractors (net-banner edge, floor logos).
4. Label the cross lines. Along the court they sit at Y = 0, 6, 9, 12, 18 m
   (far end, far attack, centre, near attack, near end). Any 3 parallel lines
   fit any 3 positions, so the fit alone proves nothing. Instead every triple
   of candidate lines x every labelling gives a court model, and each model is
   VERIFIED against the picture:
     - paint must lie along all five predicted court lines (where in frame),
     - lines that run on solidly past the sidelines (net tape, ad boards) do
       not count as paint,
     - the sidelines must be painted from Y=0 to Y=18 and NOT beyond,
     - camera behind the near end: court widens and line spacing grows
       towards the camera.
   The best-supported model is then re-fitted on every line it explains.
5. Corners = homography image of (0,0) (9,0) (9,18) (0,18). A corner outside
   the picture (camera cut-off) is still computed and flagged visible=false.

Run:
    python BASEFRAMES/detect_court_corners.py                          # every base frame
    python BASEFRAMES/detect_court_corners.py --image path/to/frame.png
    python BASEFRAMES/detect_court_corners.py --video path/to/new.mp4  # base frame + corners
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WORK_W, WORK_H = 1280, 720
COURT_Y = [0.0, 6.0, 9.0, 12.0, 18.0]
COURT_Y_NAME = {0.0: "far end line", 6.0: "far attack line", 9.0: "centre line",
                12.0: "near attack line", 18.0: "near end line"}
CORNER_NAMES = ["far_left", "far_right", "near_right", "near_left"]
CORNER_COURT = np.float32([[0, 0], [9, 0], [9, 18], [0, 18]])


# --------------------------------------------------------------------------- lines
def line_mask(img, ridge_thr=14, sat_max=75, val_min=140):
    """Thin white court lines: a ridge (brighter than the pixels 3 px away on
    both sides, horizontally or vertically) that is also unsaturated and bright.
    Works on light wooden floors, where a top-hat filter misses 1-2 px lines."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # whiteness = min(B, G, R): high only where all three channels are high, so
    # a white line stands out against blue free zones AND beige / orange floors
    g = cv2.GaussianBlur(img.min(axis=2), (3, 3), 0).astype(np.int16)
    d = 3
    up = np.roll(g, d, 0); dn = np.roll(g, -d, 0)
    lf = np.roll(g, d, 1); rt = np.roll(g, -d, 1)
    ridge = np.maximum(np.minimum(g - up, g - dn), np.minimum(g - lf, g - rt))
    th = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    white = (hsv[..., 1] < sat_max) & (hsv[..., 2] > val_min)
    m = white & ((ridge > ridge_thr) | (th > 30))
    m[:d] = m[-d:] = False
    m[:, :d] = m[:, -d:] = False
    return m.astype(np.uint8) * 255


def to_abc(x1, y1, x2, y2):
    """Line a*x + b*y + c = 0 with (a, b) unit length."""
    a, b = y2 - y1, x1 - x2
    n = math.hypot(a, b)
    a, b = a / n, b / n
    return np.array([a, b, -(a * x1 + b * y1)])


def intersect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def x_at(line, y):
    a, b, c = line
    return -(b * y + c) / a


def y_at(line, x):
    a, b, c = line
    return -(a * x + c) / b


def refine(mask, line, band=4.0, xlim=None, iters=3):
    """Robust re-fit of a line to the mask pixels within `band` px of it."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if xlim is not None:
        lo, hi = xlim
        keep = (pts[:, 0] > lo(pts[:, 1])) & (pts[:, 0] < hi(pts[:, 1]))
        pts = pts[keep]
    n_used = 0
    for _ in range(iters):
        d = np.abs(pts @ line[:2] + line[2])
        sel = pts[d < band]
        if len(sel) < 20:
            break
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        line = to_abc(x0, y0, x0 + vx, y0 + vy)
        n_used = len(sel)
        band = max(2.5, band * 0.7)
    return line, n_used


def find_sidelines(mask, segs):
    cands = {"left": [], "right": []}
    for x1, y1, x2, y2 in segs:
        ang = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        if not 20 <= ang <= 85:
            continue
        slope = (y2 - y1) / (x2 - x1 + 1e-9)
        side = "left" if slope < 0 else "right"
        cands[side].append((to_abc(x1, y1, x2, y2), math.hypot(x2 - x1, y2 - y1)))
    out = {}
    for side, cs in cands.items():
        # cluster by x at two reference rows
        clusters = []
        for l, L in cs:
            key = np.array([x_at(l, 400), x_at(l, 650)])
            for cl in clusters:
                if np.all(np.abs(cl["key"] - key) < 12):
                    cl["len"] += L
                    cl["lines"].append((l, L))
                    break
            else:
                clusters.append(dict(key=key, len=L, lines=[(l, L)]))
        if not clusters:
            raise RuntimeError(f"no {side} sideline found")
        best = max(clusters, key=lambda c: c["len"])
        l0 = max(best["lines"], key=lambda t: t[1])[0]
        out[side], _ = refine(mask, l0, band=6)
    return out["left"], out["right"]


def sideline_extent(mask, line, gap=25):
    """Painted extent of a sideline: (top_pt, bottom_pt, top_cut, bottom_cut).

    A sideline is painted only between the two end lines, so where the run of
    line pixels starts and stops IS the corner (unless the picture cuts it)."""
    h, w = mask.shape
    hits = []
    for y in range(h):
        x = x_at(line, y)
        xi = int(round(x))
        if xi < 2 or xi >= w - 2:
            hits.append(False)
            continue
        hits.append(bool(mask[y, xi - 2:xi + 3].any()))
    hits = np.array(hits)
    ys = np.nonzero(hits)[0]
    if len(ys) == 0:
        return None
    # runs allowing gaps (net, logos), keep the run with most pixels
    runs, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y - prev > gap:
            runs.append((start, prev))
            start = y
        prev = y
    runs.append((start, prev))
    y0, y1 = max(runs, key=lambda r: hits[r[0]:r[1] + 1].sum())
    top = np.array([x_at(line, y0), float(y0)])
    bot = np.array([x_at(line, y1), float(y1)])
    top_cut = y0 <= 3 or not (3 < top[0] < w - 4)
    bot_cut = y1 >= h - 4 or not (3 < bot[0] < w - 4)
    return top, bot, top_cut, bot_cut


def find_cross_lines(mask, segs, left, right, min_cov=0.30):
    """Near-horizontal lines between the sidelines, found by a direct scan:
    for each tilt angle (-5..5 deg) and each row, the share of the court width
    covered by line pixels. Peaks above `min_cov` are court-line candidates."""
    h, w = mask.shape
    xc = w / 2
    ys, xs = np.nonzero(mask)
    xl, xr = x_at(left, ys), x_at(right, ys)
    inside = (xs > xl + 4) & (xs < xr - 4)
    xs, ys = xs[inside].astype(float), ys[inside].astype(float)
    rows = np.arange(h)
    width = np.clip(x_at(right, rows) - x_at(left, rows) - 8, 1, None)
    best_cov = np.zeros(h)
    best_ang = np.zeros(h)
    for ang in np.arange(-5.0, 5.01, 0.2):
        t = math.tan(math.radians(ang))
        yint = np.round(ys - t * (xs - xc)).astype(int)
        ok = (yint >= 0) & (yint < h)
        # one vote per column per row: dedupe (row, x) pairs
        key = np.unique(yint[ok] * 10000 + xs[ok].astype(int))
        cnt = np.bincount(key // 10000, minlength=h)[:h]
        cov = cnt / width
        better = cov > best_cov
        best_cov[better] = cov[better]
        best_ang[better] = ang
    # 3-row smoothing of a 1-2 px line, then peaks with 6 px suppression
    cov_s = np.maximum.reduce([best_cov, np.roll(best_cov, 1), np.roll(best_cov, -1)])
    order = np.argsort(-best_cov)
    taken = np.zeros(h, bool)
    lines = []
    for y in order:
        if best_cov[y] < min_cov:
            break
        if taken[max(0, y - 6):y + 7].any():
            continue
        taken[y] = True
        t = math.tan(math.radians(best_ang[y]))
        l0 = to_abc(xc - 100, y - 100 * t, xc + 100, y + 100 * t)
        inside_lo = lambda yy: x_at(left, yy) + 6
        inside_hi = lambda yy: x_at(right, yy) - 6
        l, _ = refine(mask, l0, band=3, xlim=(inside_lo, inside_hi))
        pl, pr = intersect(l, left), intersect(l, right)
        if pl is None or pr is None or pr[0] <= pl[0]:
            continue
        lines.append(dict(line=l, pl=pl, pr=pr, coverage=float(cov_s[y]), y_mid=float(y_at(l, xc))))
    lines.sort(key=lambda d: d["y_mid"])
    merged = []
    for d in lines:
        if merged and abs(merged[-1]["y_mid"] - d["y_mid"]) < 4:
            if d["coverage"] > merged[-1]["coverage"]:
                merged[-1] = d
        else:
            merged.append(d)
    return merged


# --------------------------------------------------------------------------- labelling
def _coverage(soft, H, X, Y, tol):
    h, w = soft.shape
    pts = cv2.perspectiveTransform(np.float32([np.stack([X, np.full_like(X, Y)], 1)]), H)[0]
    xi, yi = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= tol) & (yi < h - tol)
    if inside.mean() < 0.5:
        return None
    xi, yi = xi[inside], yi[inside]
    hit = np.zeros(len(xi), bool)
    for dy in range(-tol, tol + 1):
        hit |= soft[yi + dy, xi] > 0
    return float(hit.mean())


def line_evidence(soft, H, Y, tol=1):
    """Share of the predicted court line Y (between the sidelines) that lies on
    painted-line pixels. None when the line is outside the picture.

    A court line stops at the sidelines (attack lines continue only as short
    dashes). A line that runs on solidly past both sidelines is the net tape
    or an advertising board, not a court line, and scores 0."""
    inner = _coverage(soft, H, np.linspace(0.25, 8.75, 240), Y, tol)
    if inner is None:
        return None
    outer = _coverage(soft, H, np.concatenate([np.linspace(-2.5, -0.6, 40), np.linspace(9.6, 11.5, 40)]), Y, tol)
    if outer is not None and outer > 0.7:
        return 0.0
    return inner


def sideline_score(mask, H, tol=2):
    """Sidelines are painted exactly from Y=0 to Y=18. Reward paint inside that
    span, penalise paint just beyond either end line (the end line is then in
    the wrong place) - only where the picture shows it."""
    def cov(Ys):
        vals = []
        for X in (0.0, 9.0):
            pts = cv2.perspectiveTransform(np.float32([np.stack([np.full_like(Ys, X), Ys], 1)]), H)[0]
            xi, yi = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
            ok = (xi >= tol) & (xi < mask.shape[1] - tol) & (yi >= 0) & (yi < mask.shape[0])
            if ok.sum() < 5:
                continue
            hit = np.zeros(ok.sum(), bool)
            for dx in range(-tol, tol + 1):
                hit |= mask[yi[ok], xi[ok] + dx] > 0
            vals.append(hit.mean())
        return float(np.mean(vals)) if vals else None
    inside = cov(np.linspace(0.3, 17.7, 120).astype(np.float32))
    beyond_far = cov(np.linspace(-2.0, -0.4, 20).astype(np.float32))
    beyond_near = cov(np.linspace(18.4, 20.0, 20).astype(np.float32))
    sc = 0.0
    if inside is not None:
        sc += 2.0 * (inside - 0.6)
    for b in (beyond_far, beyond_near):
        if b is not None:
            sc -= 3.0 * max(0.0, b - 0.4)
    return sc


def plausible(H):
    """Camera behind the near end line: the court gets wider and the line
    spacing per metre grows from far (Y=0) to near (Y=18)."""
    Ys = np.float32(COURT_Y)
    L = cv2.perspectiveTransform(np.float32([np.stack([np.zeros(5), Ys], 1)]), H)[0]
    R = cv2.perspectiveTransform(np.float32([np.stack([np.full(5, 9.0), Ys], 1)]), H)[0]
    width = R[:, 0] - L[:, 0]
    ymid = (L[:, 1] + R[:, 1]) / 2
    if not (np.all(np.diff(ymid) > 0) and np.all(np.diff(width) > 0) and width[0] > 20):
        return False
    per_m = np.diff(ymid) / np.diff(Ys)
    return bool(np.all(np.diff(per_m) > -0.05 * per_m[:-1]))


def fit_assignment(lines, left, right, soft, mask, min_evidence=0.35):
    """Pick the court model the picture supports best.

    Every triple of candidate lines, labelled with every order-preserving
    triple of court positions (0, 6, 9, 12, 18 m), gives a homography. Three
    parallel lines always fit, so the fit itself proves nothing; instead each
    model predicts all five court lines and is scored by how much painted line
    lies along them (verification). Then the winner is re-fitted on every
    line it explains."""
    n = len(lines)
    best = None
    for tri in itertools.combinations(range(n), 3):
        for ys in itertools.combinations(COURT_Y, 3):
            src, dst = [], []
            for i, Y in zip(tri, ys):
                src += [[0, Y], [9, Y]]
                dst += [lines[i]["pl"], lines[i]["pr"]]
            H, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
            if H is None or not plausible(H):
                continue
            ev = {Y: line_evidence(soft, H, Y) for Y in COURT_Y}
            score = sum(e - min_evidence for e in ev.values() if e is not None)
            score += sideline_score(mask, H)
            if best is None or score > best["score"]:
                best = dict(score=score, H=H, ev=ev)
    if best is None:
        return None
    # re-fit: for every court line the model explains, take the detected line
    # closest to its prediction (or refine the mask around the prediction)
    H = best["H"]
    src, dst, used = [], [], []
    for Y, e in best["ev"].items():
        if e is None or e < min_evidence:
            continue
        pred = cv2.perspectiveTransform(np.float32([[[0, Y], [9, Y]]]), H)[0]
        y_pred = float(pred[:, 1].mean())
        near = [l for l in lines if abs((l["pl"][1] + l["pr"][1]) / 2 - y_pred) < 5]
        if near:
            l = min(near, key=lambda l: abs((l["pl"][1] + l["pr"][1]) / 2 - y_pred))
            pl, pr, cov = l["pl"], l["pr"], l["coverage"]
        else:
            l0 = to_abc(pred[0, 0], pred[0, 1], pred[1, 0], pred[1, 1])
            lo = lambda yy: x_at(left, yy) + 6
            hi = lambda yy: x_at(right, yy) - 6
            lr, _ = refine(soft, l0, band=3, xlim=(lo, hi))
            pl, pr, cov = intersect(lr, left), intersect(lr, right), e
        src += [[0, Y], [9, Y]]
        dst += [pl, pr]
        used.append(dict(Y=Y, pl=pl, pr=pr, coverage=float(cov), evidence=float(e)))
    src_a, dst_a = np.float32(src), np.float32(dst)
    H2, _ = cv2.findHomography(src_a, dst_a, 0)
    proj = cv2.perspectiveTransform(src_a[None], H2)[0]
    err = float(np.sqrt(np.mean(np.sum((proj - dst_a) ** 2, 1))))
    return dict(H=H2, err=err, used=used, evidence=best["ev"], score=best["score"])


def model_check(H, left, right):
    """Sideline residual: the model's X=0 / X=9 lines vs the detected sidelines."""
    res = []
    for X, side in ((0, left), (9, right)):
        pts = cv2.perspectiveTransform(np.float32([[[X, Y] for Y in np.linspace(0, 18, 19)]]), H)[0]
        res += list(np.abs(pts @ side[:2] + side[2]))
    return float(np.mean(res)), float(np.max(res))


# --------------------------------------------------------------------------- per image
def detect(img_full):
    Hf, Wf = img_full.shape[:2]
    s = Wf / WORK_W
    img = cv2.resize(img_full, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    mask = line_mask(img)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=50, minLineLength=50, maxLineGap=12)
    segs = [] if segs is None else segs.reshape(-1, 4).astype(float)
    left, right = find_sidelines(mask, segs)
    soft = line_mask(img, ridge_thr=6, sat_max=90, val_min=140)
    lines = find_cross_lines(soft, segs, left, right, min_cov=0.4)
    best = fit_assignment(lines, left, right, soft, mask)
    if best is None:
        raise RuntimeError("could not label the cross lines")
    H = best["H"]
    side_mean, side_max = model_check(H, left, right)
    S = np.diag([s, s, 1.0])
    H_full = S @ H
    H_full /= H_full[2, 2]
    corners = cv2.perspectiveTransform(CORNER_COURT[None], H_full)[0]
    used = [dict(court_y_m=float(u["Y"]), name=COURT_Y_NAME[u["Y"]],
                 left_px=(u["pl"] * s).round(1).tolist(), right_px=(u["pr"] * s).round(1).tolist(),
                 paint_evidence=round(u["evidence"], 3))
            for u in best["used"]]
    not_seen = [dict(court_y_m=float(Y), name=COURT_Y_NAME[Y],
                     reason="outside picture" if e is None else f"paint evidence {e:.2f} (hidden / faint)")
                for Y, e in best["evidence"].items() if e is None or e < 0.35]
    return dict(
        H=H_full, corners=corners, mask=mask, lines_used=used, lines_not_seen=not_seen,
        fit_rmse_px=best["err"] * s, sideline_mean_px=side_mean * s, sideline_max_px=side_max * s,
        left=left, right=right, scale=s,
    )


def draw(img, res):
    out = img.copy()
    H = res["H"]
    sc = img.shape[1] / WORK_W
    th = max(2, int(round(2 * sc)))
    # full court model: boundary, centre line, attack lines
    def poly(pts, col, t):
        p = cv2.perspectiveTransform(np.float32([pts]), H)[0]
        cv2.polylines(out, [p.round().astype(np.int32)], False, col, t, cv2.LINE_AA)
    over = out.copy()
    quad = cv2.perspectiveTransform(CORNER_COURT[None], H)[0].round().astype(np.int32)
    cv2.fillPoly(over, [quad], (0, 200, 255))
    cv2.addWeighted(over, 0.18, out, 0.82, 0, out)
    for Y in (6, 9, 12):
        poly([[0, Y], [9, Y]], (255, 220, 0), th)
    poly([[0, 0], [9, 0], [9, 18], [0, 18], [0, 0]], (0, 255, 0), th + 1)
    h_img, w_img = img.shape[:2]
    fs = 0.75 * sc
    for name, (x, y) in zip(CORNER_NAMES, res["corners"]):
        vis = 0 <= x < w_img and 0 <= y < h_img
        col = (0, 0, 255) if vis else (0, 140, 255)
        if vis:
            cv2.circle(out, (int(round(x)), int(round(y))), int(10 * sc), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (int(round(x)), int(round(y))), int(7 * sc), col, -1, cv2.LINE_AA)
            tx, ty = int(np.clip(x + 12 * sc, 5, w_img - 330 * sc)), int(np.clip(y - 12 * sc, 25 * sc, h_img - 10))
            label = f"{name} ({x:.0f},{y:.0f})"
        else:
            tx = int(np.clip(x, 5, w_img - 360 * sc))
            ty = int(np.clip(y, 25 * sc, h_img - 12 * sc))
            label = f"{name} ({x:.0f},{y:.0f}) off-frame"
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), int(5 * sc), cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, col, int(2 * sc), cv2.LINE_AA)
    txt = (f"court fit: {len(res['lines_used'])} court lines, rmse {res['fit_rmse_px']:.2f}px, "
           f"sideline dev {res['sideline_mean_px']:.2f}px")
    cv2.putText(out, txt, (int(10 * sc), h_img - int(14 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc,
                (0, 0, 0), int(5 * sc), cv2.LINE_AA)
    cv2.putText(out, txt, (int(10 * sc), h_img - int(14 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc,
                (255, 255, 255), int(2 * sc), cv2.LINE_AA)
    return out


def yolo_pose_line(corners, w, h):
    vis = [(0 <= x < w and 0 <= y < h) for x, y in corners]
    xs, ys = np.clip(corners[:, 0], 0, w - 1), np.clip(corners[:, 1], 0, h - 1)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    parts = [0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    for (x, y), v in zip(corners, vis):
        parts += [x / w, y / h, 2 if v else 0] if v else [0.0, 0.0, 0]
    return " ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts)


def process(path: Path, out_dir: Path):
    img = cv2.imread(str(path))
    res = detect(img)
    h, w = img.shape[:2]
    stem = path.stem
    for sub in ("annotated", "labels", "yolo_pose"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "annotated" / f"{stem}.png"), draw(img, res))
    corners = res["corners"]
    label = {
        "image": str(path.name), "width": w, "height": h,
        "court_size_m": [9, 18],
        "corners_px": {n: [round(float(x), 2), round(float(y), 2)] for n, (x, y) in zip(CORNER_NAMES, corners)},
        "corners_visible": {n: bool(0 <= x < w and 0 <= y < h) for n, (x, y) in zip(CORNER_NAMES, corners)},
        "corner_court_m": {n: c.tolist() for n, c in zip(CORNER_NAMES, CORNER_COURT)},
        "homography_court_m_to_px": res["H"].round(8).tolist(),
        "lines_used": res["lines_used"], "lines_not_seen": res["lines_not_seen"],
        "fit_rmse_px": round(res["fit_rmse_px"], 3),
        "sideline_mean_dev_px": round(res["sideline_mean_px"], 3),
        "sideline_max_dev_px": round(res["sideline_max_px"], 3),
    }
    (out_dir / "labels" / f"{stem}.json").write_text(json.dumps(label, indent=2))
    (out_dir / "yolo_pose" / f"{stem}.txt").write_text(yolo_pose_line(corners, w, h) + "\n")
    return label


def contact_sheet(out_dir: Path):
    ims = sorted((out_dir / "annotated").glob("*.png"))
    tiles = []
    for p in ims:
        t = cv2.resize(cv2.imread(str(p)), (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(t, p.stem, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(t, p.stem, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out_dir / "_contact_sheet.png"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=str(HERE / "base_frame"))
    ap.add_argument("--image", default=None, help="process one image only")
    ap.add_argument("--video", default=None,
                    help="new video: extract its base frame into --frames-dir first, then detect its corners")
    ap.add_argument("--out-dir", default=str(HERE / "corner_court_detection"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.video:
        from extract_base_frames import process_video
        vid = Path(args.video)
        process_video(vid, Path(args.frames_dir), n_sample=400, n_clusters=8, n_keep=120, overwrite=True)
        args.image = str(Path(args.frames_dir) / (vid.stem + ".png"))
    paths = [Path(args.image)] if args.image else sorted(Path(args.frames_dir).glob("*.png"))
    rows = []
    for p in paths:
        try:
            lab = process(p, out_dir)
        except Exception as exc:                      # noqa: BLE001
            print(f"FAIL {p.name}: {exc}")
            continue
        c = lab["corners_px"]
        used = ", ".join(f"{u['court_y_m']:g}" for u in lab["lines_used"])
        print(f"{p.name:24s} lines[{used}]  rmse {lab['fit_rmse_px']:.2f}px  "
              f"side {lab['sideline_mean_dev_px']:.2f}px  " +
              "  ".join(f"{n}=({v[0]:.0f},{v[1]:.0f}){'' if lab['corners_visible'][n] else '*'}"
                        for n, v in c.items()))
        rows.append([p.name, lab["width"], lab["height"]] +
                    [v for n in CORNER_NAMES for v in c[n]] +
                    [lab["corners_visible"][n] for n in CORNER_NAMES] +
                    [lab["fit_rmse_px"], len(lab["lines_used"])])
    header = (["image", "width", "height"] + [f"{n}_{a}" for n in CORNER_NAMES for a in ("x", "y")] +
              [f"{n}_visible" for n in CORNER_NAMES] + ["fit_rmse_px", "n_court_lines"])
    csv_path = out_dir / "corners.csv"
    if args.image and csv_path.exists():
        with open(csv_path, newline="") as f:
            old = [r for r in csv.reader(f)][1:]
        names = {r[0] for r in rows}
        rows = sorted([r for r in old if r[0] not in names] + rows, key=lambda r: r[0])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    contact_sheet(out_dir)
    print("->", out_dir)


if __name__ == "__main__":
    main()
