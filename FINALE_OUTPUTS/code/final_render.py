"""FINALE_OUTPUTS renderer: one annotated video per tactical model (9 ML + 4 DL)
for videoplayback (7), 03:00-08:00.

Per frame (shared by all models):
  * a bounding box on every player: far team (Japan) red, near team
    (Argentina) blue,
  * a graph per team in its own colour: nodes on the players, edges =
    minimum spanning tree + 2 nearest neighbours, measured in court metres,
  * the ball: bounding box + trail from the offline tracker (ball_track.py);
    hidden when the ball truly is not visible, never frozen.

Per model:
  * its call for the near team's current 1-second window - Coordinated
    Attack / Coordinated Defense / Delayed Support / Spacing Breakdown - as a
    banner and a tinted outline around the near team,
  * a side panel with the model's held-out accuracy, confidence, 13-model
    majority and the class definitions,
  * a 5-minute timeline of its calls with a playhead.

Usage:
  python final_render.py cache_vp7.pkl --out-dir ../FINALE_OUTPUTS/videos
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import bicausal_render as br
from ball_track import track_ball
from render_13_models import MODELS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MAIN_W, MAIN_H, PANEL_W, STRIP_H = 1280, 720, 400, 64
FAR, NEAR = br.FAR, br.NEAR
TEAM_BGR = {FAR: (50, 50, 230), NEAR: (230, 120, 20)}          # red / blue
TEAM_RGB = {k: (v[2], v[1], v[0]) for k, v in TEAM_BGR.items()}
TEAM_LABEL = {FAR: "Japan (far side)", NEAR: "Argentina (near side)"}
BALL_BGR = (0, 230, 255)

CLASSES = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
# kept clear of the two team colours and the yellow ball
CLASS_RGB = {
    "Coordinated Attack": (255, 130, 0),
    "Coordinated Defense": (20, 175, 120),
    "Delayed Support": (225, 80, 160),
    "Spacing Breakdown": (140, 90, 220),
    None: (110, 110, 110),
}
CLASS_DEF = {
    "Coordinated Attack": "fast front-row approach, base held at the net",
    "Coordinated Defense": "the team shifts together as one rigid unit",
    "Delayed Support": "nearest player reacts late after ball contact",
    "Spacing Breakdown": "formation too spread out or too bunched up",
}


def _iou_max(boxes):
    m = len(boxes)
    out = np.zeros(m)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            a, b = boxes[i], boxes[j]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            if inter > 0:
                out[i] = max(out[i], inter / ((a[2] - a[0]) * (a[3] - a[1])))
    return out


class TeamAssigner:
    """Team per player for videoplayback (7): Japan wears red, Argentina black.

    Each unoccluded view of a tracked player votes on the torso colour (red ->
    Japan, dark -> Argentina); votes decay slowly per tracker id, so a moment
    of overlap at the net cannot flip a player. The white / light-blue liberos
    never vote and fall back to their side of the net."""

    def __init__(self, decay=0.97):
        self.votes = {}
        self.crew = {}
        self.decay = decay

    def __call__(self, img, players_f):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cover = _iou_max([q["box"] for q in players_f])
        out = []
        for q, cov in zip(players_f, cover):
            x1, y1, x2, y2 = q["box"].astype(int)
            h, w = y2 - y1, x2 - x1
            t = hsv[max(0, y1 + int(0.2 * h)):max(1, y1 + int(0.45 * h)),
                    max(0, x1 + int(0.3 * w)):max(1, x2 - int(0.3 * w))]
            v = 0
            if t.size:
                H, S, V = t[..., 0], t[..., 1], t[..., 2]
                red = float((((H < 10) | (H > 165)) & (S > 110) & (V > 70)).mean())
                dark = float((V < 70).mean())
                teal = float(((H > 70) & (H < 110) & (S > 120) & (V > 60)).mean())
                tid0 = q["id"]
                crew = self.crew.get(tid0, 0.0)
                crew = crew + 1.0 if teal > 0.3 else max(0.0, crew - 0.1)
                if tid0 >= 0:
                    self.crew[tid0] = crew
                if teal > 0.3 or crew >= 2.0:
                    continue                # floor moppers / ball crew (teal shirts), not players
                if red > 0.25:
                    v = 1
                elif dark > 0.2 and red < 0.05:
                    v = -1
            tid = q["id"]
            score = self.votes.get(tid, 0.0) * self.decay if tid >= 0 else 0.0
            if cov < 0.25:
                score += v
            if tid >= 0:
                self.votes[tid] = score
            if score >= 1.5:
                team = FAR
            elif score <= -1.5:
                team = NEAR
            elif v and cov < 0.25:
                team = FAR if v > 0 else NEAR
            else:
                team = FAR if q["court"][1] < 9.0 else NEAR
            out.append(dict(q, team=team))
        res = []
        for team in (FAR, NEAR):
            tq = [q for q in out if q["team"] == team]
            tq.sort(key=lambda q: (br._outside(q["court"]), -q["conf"]))
            res.extend(tq[:6])
        return res


def graph_edges(court_pts):
    m = len(court_pts)
    edges = set(tuple(sorted(e)) for e in br.mst_edges(court_pts))
    for i in range(m):
        d = np.linalg.norm(court_pts - court_pts[i], axis=1)
        for j in np.argsort(d)[1:3]:
            if d[j] < 6.5:
                edges.add(tuple(sorted((i, int(j)))))
    return sorted(edges)


def draw_base(img, players_f, ball_p, ball_w, trail):
    """Boxes, team graphs and ball - identical for every model."""
    for team in (FAR, NEAR):
        tq = [q for q in players_f if q["team"] == team]
        if len(tq) >= 2:
            nodes = np.array([q["node"] for q in tq], np.float32)
            courts = np.array([q["court"] for q in tq], np.float32)
            for i, j in graph_edges(courts):
                p, q = tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int))
                cv2.line(img, p, q, (25, 25, 25), 7, cv2.LINE_AA)
                cv2.line(img, p, q, TEAM_BGR[team], 4, cv2.LINE_AA)
    for q in players_f:
        x1, y1, x2, y2 = q["box"].astype(int)
        col = TEAM_BGR[q["team"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 25), 4, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        c = tuple(q["node"].astype(int))
        cv2.circle(img, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, 7, col, -1, cv2.LINE_AA)
    # ball trail + box
    for k in range(1, len(trail)):
        if trail[k][0] - trail[k - 1][0] > 2:
            continue
        a = k / len(trail)
        cv2.line(img, tuple(trail[k - 1][1].astype(int)), tuple(trail[k][1].astype(int)),
                 BALL_BGR, max(1, int(5 * a)), cv2.LINE_AA)
    if ball_p is not None:
        r = int(max(10, (ball_w if ball_w == ball_w else 20) * 0.65))
        x, y = ball_p.astype(int)
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), (25, 25, 25), 5, cv2.LINE_AA)
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), BALL_BGR, 2, cv2.LINE_AA)
        cv2.putText(img, "ball", (x - r, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 4, cv2.LINE_AA)
        cv2.putText(img, "ball", (x - r, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BALL_BGR, 2, cv2.LINE_AA)


def static_panel(name, fam, rank, m, fonts):
    """Parts of the side panel that never change for a model."""
    f_big, f_mid, f_sm, f_xs = fonts
    im = Image.new("RGB", (PANEL_W, MAIN_H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((18, 16), "MODEL", font=f_xs, fill=(150, 150, 150))
    d.text((18, 34), name, font=f_big, fill=(255, 255, 255))
    d.text((18, 70), f"{'Machine learning' if fam == 'ML' else 'Deep learning'}  |  rank {rank} of 13",
           font=f_sm, fill=(190, 190, 190))
    d.text((18, 104), "Accuracy on unseen matches (LOVO)", font=f_xs, fill=(150, 150, 150))
    d.text((18, 124), f"{m.test_accuracy_LOVO * 100:.1f}%", font=f_big, fill=(255, 215, 90))
    d.text((140, 130), f"macro-F1 {m.test_macro_f1_LOVO:.3f}", font=f_sm, fill=(230, 230, 230))
    d.text((18, 160), f"train accuracy {m.train_accuracy * 100:.1f}%", font=f_xs, fill=(150, 150, 150))
    d.line([(18, 190), (PANEL_W - 18, 190)], fill=(60, 60, 60), width=1)
    d.text((18, 200), "NEAR TEAM TACTIC (this second)", font=f_xs, fill=(150, 150, 150))
    # legend
    y = 420
    d.line([(18, y - 10), (PANEL_W - 18, y - 10)], fill=(60, 60, 60), width=1)
    d.text((18, y), "Tactic classes", font=f_sm, fill=(230, 230, 230))
    y += 26
    for c in CLASSES:
        d.rectangle([18, y + 3, 32, y + 17], fill=CLASS_RGB[c])
        d.text((40, y), c, font=f_sm, fill=(235, 235, 235))
        d.text((40, y + 20), CLASS_DEF[c], font=f_xs, fill=(150, 150, 150))
        y += 46
    y += 4
    d.line([(18, y), (PANEL_W - 18, y)], fill=(60, 60, 60), width=1)
    y += 10
    for team in (FAR, NEAR):
        d.rectangle([18, y + 3, 32, y + 17], outline=TEAM_RGB[team], width=3)
        d.line([(40, y + 10), (64, y + 10)], fill=TEAM_RGB[team], width=4)
        d.text((72, y), TEAM_LABEL[team] + " - box + graph", font=f_xs, fill=(220, 220, 220))
        y += 24
    d.rectangle([18, y + 3, 32, y + 17], outline=(255, 230, 0), width=3)
    d.text((40, y), "ball - box + trail", font=f_xs, fill=(220, 220, 220))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--windows", default=str(ROOT / "vp7_3to8_results" / "vp7_3to8_all_models.csv"))
    ap.add_argument("--metrics", default=str(ROOT / "model_comparison" / "trainvstest_metrics.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "FINALE_OUTPUTS" / "videos"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        data = pickle.load(f)
    frames, fps, start, video = data["frames"], data["fps"], data["start"], data["video"]
    if args.max_frames:
        frames = frames[:args.max_frames]
    n = len(frames)
    br.set_court(Path(video).name)
    print("players ...", flush=True)
    players = br.build_players(frames)
    print("ball ...", flush=True)
    ball, bsize, bsrc, st = track_ball(frames)
    print(f"  ball shown in {np.mean(bsrc > 0) * 100:.0f}% of frames "
          f"(detected {np.mean(bsrc == 1) * 100:.0f}%, bridged {np.mean(bsrc == 2) * 100:.0f}%)", flush=True)

    win = pd.read_csv(args.windows)
    met = pd.read_csv(args.metrics).set_index("model")
    order = met.sort_values(["test_accuracy_LOVO", "test_macro_f1_LOVO"], ascending=False).index
    rank = pd.Series(range(1, len(order) + 1), index=order)
    models = [m for m in MODELS if not args.only or m[0] in args.only]
    fs, fe = win["frame_start"].to_numpy(), win["frame_end"].to_numpy()

    fonts = (br.load_font(28, True), br.load_font(22, True), br.load_font(17, True), br.load_font(14))
    f_big, f_mid, f_sm, f_xs = fonts
    f_banner = br.load_font(30, True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, Hh = MAIN_W + PANEL_W, MAIN_H + STRIP_H
    writers, panels, tls = {}, {}, {}
    tl_x0, tl_x1 = 16, W - 16
    for col, name, fam, mname in models:
        r = int(rank[mname])
        fn = f"{r:02d}_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_{fam}.mp4"
        w = imageio_ffmpeg.write_frames(
            str(out_dir / fn), (W, Hh), fps=fps, codec="libx264", quality=None, pix_fmt_out="yuv420p",
            output_params=["-crf", "18", "-preset", "veryfast", "-threads", "2", "-movflags", "+faststart"],
            macro_block_size=16)
        w.send(None)
        writers[col] = (w, out_dir / fn)
        panels[col] = static_panel(name, fam, r, met.loc[mname], fonts)
        bar = np.full((16, tl_x1 - tl_x0, 3), 50, np.uint8)
        for lab, a, b in zip(win[col], fs, fe):
            x0 = max(0, int((a - start) / n * bar.shape[1]))
            x1 = min(bar.shape[1], int((b + 1 - start) / n * bar.shape[1]))
            if x1 > x0:
                bar[:, x0:x1] = CLASS_RGB.get(lab, CLASS_RGB[None])
        tls[col] = bar

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    trail = []
    assign_teams = TeamAssigner()
    for fi in range(n):
        ok, img = cap.read()
        if not ok:
            break
        bp = None if np.isnan(ball[fi, 0]) else ball[fi]
        if bp is not None:
            trail.append((fi, bp.copy()))
        trail = [(f, p) for f, p in trail if fi - f <= 12]
        pl = assign_teams(img, players[fi])
        draw_base(img, pl, bp, bsize[fi], trail)
        base_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        absf = start + fi
        k = int(np.searchsorted(fs, absf, side="right")) - 1
        in_win = 0 <= k < len(fs) and absf <= fe[k] + 1
        near = [q for q in pl if q["team"] == NEAR]
        hull = (cv2.convexHull(np.array([q["feet"] for q in near], np.float32)).astype(np.int32)
                if len(near) >= 3 else None)
        ts = absf / fps
        tstr = f"{int(ts // 60):02d}:{ts % 60:04.1f}"

        for col, name, fam, mname in models:
            lab = win[col].iloc[k] if in_win else None
            conf = float(win[col + "__conf"].iloc[k]) if in_win else None
            rgb = CLASS_RGB.get(lab, CLASS_RGB[None])
            main = base_rgb.copy()
            if hull is not None and lab is not None:
                over = main.copy()
                cv2.fillPoly(over, [hull], rgb)
                cv2.addWeighted(over, 0.16, main, 0.84, 0, main)
                cv2.polylines(main, [hull], True, rgb, 3, cv2.LINE_AA)
            pm = Image.fromarray(main)
            d = ImageDraw.Draw(pm)
            banner = (lab.upper() if lab else "NO PREDICTION FOR THIS SECOND")
            br.text_box(d, (MAIN_W // 2, MAIN_H - 20), banner, f_banner, (255, 255, 255), rgb,
                        (255, 255, 255), pad=10, anchor="md")
            br.text_box(d, (MAIN_W - 12, 46), f"videoplayback (7)   {tstr}", f_sm,
                        (255, 255, 255), (20, 20, 20), anchor="ra", pad=5)

            panel = panels[col].copy()
            dp = ImageDraw.Draw(panel)
            if lab:
                br.text_box(dp, (18, 228), lab, f_mid, (255, 255, 255), rgb, pad=8)
                dp.text((18, 276), f"confidence {conf:.2f}", font=f_sm, fill=(230, 230, 230))
                dp.rectangle([18, 302, PANEL_W - 18, 316], outline=(90, 90, 90))
                dp.rectangle([19, 303, 19 + int((PANEL_W - 38) * conf), 315], fill=rgb)
                dp.text((18, 332), "13-model majority for this second:", font=f_xs, fill=(150, 150, 150))
                dp.text((18, 352), f"{win['majority_label'].iloc[k]}  ({win['n_agree_majority'].iloc[k]}/13 agree)",
                        font=f_sm, fill=(230, 230, 230))
            else:
                dp.text((18, 232), "no 1-second window here", font=f_sm, fill=(150, 150, 150))

            strip = np.full((STRIP_H, W, 3), 14, np.uint8)
            strip[30:46, tl_x0:tl_x1] = tls[col]
            px = tl_x0 + int((tl_x1 - tl_x0) * fi / max(n - 1, 1))
            cv2.line(strip, (px, 24), (px, 52), (255, 255, 255), 3)
            ps = Image.fromarray(strip)
            ds = ImageDraw.Draw(ps)
            ds.text((16, 6), f"{name}: tactic timeline 03:00 - 08:00", font=f_xs, fill=(200, 200, 200))
            for mm in range(4, 8):
                x = tl_x0 + int((tl_x1 - tl_x0) * ((mm * 60 * fps - start) / n))
                ds.text((x, 48), f"{mm:02d}:00", font=f_xs, fill=(150, 150, 150), anchor="ma")

            frame = np.vstack([np.hstack([np.array(pm), np.array(panel)]), np.array(ps)])
            writers[col][0].send(np.ascontiguousarray(frame))
        if fi % 600 == 0:
            print(f"render {fi}/{n}", flush=True)

    for col, (w, path) in writers.items():
        w.close()
        print("video ->", path)
    np.savez(out_dir.parent / "ball_track_vp7.npz", pos=ball, size=bsize, src=bsrc, start=start)


if __name__ == "__main__":
    main()
