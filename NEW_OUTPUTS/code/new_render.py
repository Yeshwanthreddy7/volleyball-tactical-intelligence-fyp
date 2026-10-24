"""NEW_OUTPUTS renderer: one-team relation graph + team tactics + opponent actions.

Changes from PAPER_OUTPUT (paper_output_render.py / bicausal_render.py):

  * Focus team = near (bottom, blue) team. Only this team gets the relation
    graph (all-pair links + MST edges in court metres) and a hull.
  * The focus team is labelled with the four tactical classes only:
    Coordinated Attack / Coordinated Defense / Delayed Support /
    Spacing Breakdown (team_tactics.py, deployed HGB model). Its individual
    touch names (SET, DIG, PASS ...) are not shown.
  * Opponent = far (top, red) team. Its players are drawn as nodes only, with
    no connections. Only three opponent actions are shown: SERVE, SPIKE and
    BLOCK. All other opponent touches are hidden.
  * The random-init paper-feature HUD is removed (its edge heatmap covered
    both teams).

The detection cache, court model, ball tracker, touch detector and action
rules are the ones from bicausal_render.py, unchanged.

Usage:
  python new_render.py --cache ../../PAPER_OUTPUT/cache/cache_vp1.pkl --out ../videos/vp1_team_tactics.mp4
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                              # finalyearproject/
sys.path.insert(0, str(ROOT / "OUTPUTS"))
sys.path.insert(0, str(HERE))
import bicausal_render as bc                        # noqa: E402
import team_tactics as tt                           # noqa: E402

FOCUS, OPP = bc.NEAR, bc.FAR
OPP_ACTIONS = ("SERVE", "SPIKE", "BLOCK")
PANEL_W = 480

TEAM_BGR = bc.TEAM_BGR
TEAM_RGB = bc.TEAM_RGB
BALL_BGR = bc.BALL_BGR
# kept clear of the two team colours (red / blue) and the yellow ball
TAC_RGB = {
    "Coordinated Attack": (255, 130, 0),
    "Coordinated Defense": (20, 175, 120),
    "Delayed Support": (225, 80, 160),
    "Spacing Breakdown": (140, 90, 220),
}
TAC_BGR = {k: (v[2], v[1], v[0]) for k, v in TAC_RGB.items()}
TAC_DEF = {
    "Coordinated Attack": "front row approaches together, base held",
    "Coordinated Defense": "team shifts as one unit to cover",
    "Delayed Support": "teammates late to cover the ball player",
    "Spacing Breakdown": "gaps / overlaps in court coverage",
}
OPP_TAG_RGB = (190, 40, 40)
NO_CALL_RGB = (90, 90, 90)


# ---------------------------------------------------------------- live-play mask
def camera_ok_mask(video, start, n):
    """True where the frame matches the empty-court base frame (no replay / cut)."""
    base_png = ROOT / "BASEFRAMES" / "base_frame" / (Path(video).stem + ".png")
    if not base_png.exists():
        return np.ones(n, bool)
    base_sig = bc._court_sig(cv2.imread(str(base_png)))
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok = np.zeros(n, bool)
    for fi in range(n):
        r, frame = cap.read()
        if not r:
            raise RuntimeError(f"camera mask: could not read frame {start + fi} of {video}")
        ok[fi] = float((bc._court_sig(frame) * base_sig).mean()) >= 0.45
    cap.release()
    return ok


def live_mask(n, fps, touches, ball, players, cam_ok):
    """Rally time: around ball touches (both teams) or where the ball is tracked."""
    rally = np.zeros(n, bool)
    fs = [t["f"] for t in touches]
    group = []
    for f in fs + [None]:
        if f is not None and (not group or f - group[-1] <= 4.0 * fps):
            group.append(f)
            continue
        if group:
            a = max(0, int(group[0] - 1.0 * fps))
            b = min(n - 1, int(group[-1] + 1.5 * fps))
            rally[a:b + 1] = True
        group = [f] if f is not None else []
    vis = (~np.isnan(ball[:, 0])).astype(np.float32)
    w = int(2 * fps) + 1
    dens = np.convolve(vis, np.ones(w) / w, "same")
    rally |= dens >= 0.4
    n_focus = np.array([sum(q["team"] == FOCUS for q in p) for p in players])
    live = rally & cam_ok & (n_focus >= 3)
    # close gaps < 1 s inside live play, then drop live islands < 1 s
    k = int(fps)
    live = _fill_runs(live, False, k)
    live = _fill_runs(live, True, k) & cam_ok
    return live


def _fill_runs(mask, value, max_len):
    """Flip runs of `value` shorter than max_len that have the other value on both sides."""
    m = mask.copy()
    i, n = 0, len(m)
    while i < n:
        j = i
        while j < n and m[j] == m[i]:
            j += 1
        if m[i] == value and i > 0 and j < n and j - i < max_len:
            m[i:j] = not value
        i = j
    return m


# ---------------------------------------------------------------- drawing
def hull_outline(img, pts, bgr):
    if len(pts) < 3:
        return
    hull = cv2.convexHull(np.asarray(pts, np.float32)).astype(np.int32)
    over = img.copy()
    cv2.fillPoly(over, [hull], bgr)
    cv2.addWeighted(over, 0.16, img, 0.84, 0, img)
    cv2.polylines(img, [hull], True, (30, 30, 30), 6, cv2.LINE_AA)
    cv2.polylines(img, [hull], True, bgr, 3, cv2.LINE_AA)


def draw_node(img, q, team):
    c = tuple(q["node"].astype(int))
    cv2.circle(img, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, c, 7, TEAM_BGR[team], -1, cv2.LINE_AA)
    fx, fy = q["feet"].astype(int)
    cv2.ellipse(img, (fx, fy), (22, 7), 0, 0, 360, TEAM_BGR[team], 2, cv2.LINE_AA)


def fmt_t(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def render(cache, out_path, max_frames=0, segment_frames=4000):
    with open(cache, "rb") as f:
        data = pickle.load(f)
    frames, fps, start = data["frames"], data["fps"], data["start"]
    if max_frames:
        frames = frames[:max_frames]
    n = len(frames)
    video = data["video"]
    video_name = Path(video).name
    bc.set_court(video_name)
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    bc.MAIN_W, bc.MAIN_H = W, Hh

    print(f"[{video_name}] {n} frames @ {fps:.2f} fps", flush=True)
    players = bc.build_players(frames)
    ball, segs, _ = bc.build_ball(frames, players)
    touches = bc.find_touches(ball, segs, players)
    events = bc.label_touches(touches, fps)
    opp_events = [e for e in events if e["team"] == OPP and e["label"] in OPP_ACTIONS]
    print(f"  all events {Counter(e['label'] for e in events).most_common()}", flush=True)
    print(f"  opponent events kept {Counter(e['label'] for e in opp_events).most_common()}", flush=True)
    forms = bc.formations(players, n)
    del data, frames                                 # raw detections no longer needed (RAM)

    print("  camera / replay mask ...", flush=True)
    cam_ok = camera_ok_mask(video, start, n)
    # the court-similarity score hovers near its threshold on some frames:
    # remove sub-0.5 s flicker in both directions
    cam_ok = _fill_runs(_fill_runs(cam_ok, False, int(0.5 * fps)), True, int(0.5 * fps))
    live = live_mask(n, fps, touches, ball, players, cam_ok)
    print("  team tactics (HGB) ...", flush=True)
    rows = tt.raw_rows(players, ball, bc.to_court, FOCUS)
    ws, we, P = tt.window_probs(rows, fps)
    lab, prob = tt.frame_calls(n, fps, ws, we, P, live)
    dist = Counter(tt.CLASSES[v] if v >= 0 else "no call" for v in lab)
    print(f"  per-frame tactic share { {k: f'{v / n:.1%}' for k, v in dist.most_common()} }", flush=True)

    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    # CSVs: opponent actions + focus-team tactic segments
    with open(str(stem) + "_opponent_actions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_time", "frame", "action", "team", "actor_track_id"])
        for e in opp_events:
            ts = (start + e["f"]) / fps
            w.writerow([fmt_t(ts), start + e["f"], e["label"], "Far team (opponent)", e["actor_id"]])
    segs_out = tt.segments(lab, prob, fps, start)
    with open(str(stem) + "_team_tactics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_frame", "end_frame", "start_s", "end_s",
                                          "tactic", "mean_confidence"])
        w.writeheader()
        w.writerows(segs_out)

    f_big = bc.load_font(30, True)
    f_mid = bc.load_font(22, True)
    f_sm = bc.load_font(17, True)
    sc = Hh / 720.0
    fm_big = bc.load_font(int(30 * sc), True)
    fm_mid = bc.load_font(int(22 * sc), True)
    fm_sm = bc.load_font(int(17 * sc), True)
    f_xs = bc.load_font(15)
    f_tiny = bc.load_font(13, True)

    court_near = bc.to_img(np.float32([[0, 9], [9, 9], [9, 18], [0, 18]])).astype(np.int32)

    # timeline strip of the tactic calls (panel width, drawn once)
    TL_X0, TL_W, TL_H = 16, PANEL_W - 32, 22
    cols = np.full((TL_H, TL_W, 3), 45, np.uint8)
    for x in range(TL_W):
        a = int(x * n / TL_W)
        b = max(a + 1, int((x + 1) * n / TL_W))
        c = Counter(lab[a:b].tolist()).most_common(1)[0][0]
        if c >= 0:
            cols[:, x] = TAC_BGR[tt.CLASSES[c]]
    timeline = cols

    ev_frames = np.array([e["f"] for e in opp_events]) if opp_events else np.array([], int)
    ev_at = {e["f"]: e for e in opp_events}
    history = []
    trail = []

    seg_dir = out_path.parent / (stem.name + "_segments")
    seg_dir.mkdir(parents=True, exist_ok=True)
    writer, seg_paths, seg_idx, in_seg = None, [], 0, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fi in range(n):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"render: could not read frame {start + fi} of {video}")
        raw = frame.copy()
        img = frame
        ts = (start + fi) / fps

        if not cam_ok[fi]:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            bc.text_box(ImageDraw.Draw(pil), (W // 2, Hh - 22), "camera change - overlays paused",
                        fm_mid, (255, 255, 255), (60, 60, 60), anchor="md")
            main_rgb = np.array(pil)
            panel = np.full((Hh, PANEL_W, 3), 18, np.uint8)
        else:
            tac = tt.CLASSES[lab[fi]] if lab[fi] >= 0 else None
            tq = [q for q in players[fi] if q["team"] == FOCUS]
            oq = [q for q in players[fi] if q["team"] == OPP]

            # focus half tint
            shade = img.copy()
            cv2.fillPoly(shade, [court_near], TEAM_BGR[FOCUS])
            cv2.addWeighted(shade, 0.12, img, 0.88, 0, img)

            # focus team: hull in tactic colour + relation graph
            nodes = np.array([q["node"] for q in tq], np.float32)
            if tac is not None and len(tq) >= 3:
                hull_outline(img, [q["feet"] for q in tq], TAC_BGR[tac])
            overlay = img.copy()
            for i in range(len(tq)):
                for j in range(i + 1, len(tq)):
                    cv2.line(overlay, tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int)),
                             TEAM_BGR[FOCUS], 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            if len(tq) >= 2:
                edge_bgr = TAC_BGR[tac] if tac else (255, 255, 255)
                for i, j in bc.mst_edges(np.array([q["court"] for q in tq], np.float32)):
                    p, q = tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int))
                    cv2.line(img, p, q, (30, 30, 30), 7, cv2.LINE_AA)
                    cv2.line(img, p, q, edge_bgr, 4, cv2.LINE_AA)
            for q in tq:
                draw_node(img, q, FOCUS)

            # opponent: current SERVE / SPIKE / BLOCK
            past = ev_frames[ev_frames <= fi + 6]
            cur = ev_at[int(past[-1])] if len(past) and fi - past[-1] <= int(1.6 * fps) else None
            key_q = None
            if cur is not None:
                same = [q for q in oq if q["id"] == cur["actor_id"] and q["id"] >= 0]
                if same:
                    key_q = same[0]
                elif oq:
                    near = min(oq, key=lambda q: np.linalg.norm(q["node"] - cur["actor"]["node"]))
                    if np.linalg.norm(near["node"] - cur["actor"]["node"]) <= 120:
                        key_q = near
            # opponent: nodes only, no connections
            for q in oq:
                if q is not key_q:
                    draw_node(img, q, OPP)

            # ball
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
                    bc.dashed_line(img, b, tgt, BALL_BGR, 2, 8)
                cx, cy = key_q["node"].astype(int)
                cv2.fillPoly(img, [bc.star_pts(cx, cy, 20, 8)], (255, 255, 255), cv2.LINE_AA)
                cv2.fillPoly(img, [bc.star_pts(cx, cy, 16, 6)], TEAM_BGR[OPP], cv2.LINE_AA)
            if has_ball:
                bp = tuple(b.astype(int))
                cv2.circle(img, bp, 13, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, bp, 11, BALL_BGR, 3, cv2.LINE_AA)

            # snapshot of each opponent action for the side panel
            if fi in ev_at:
                e = ev_at[fi]
                x1, y1, x2, y2 = e["actor"]["box"]
                side = int(max(y2 - y1, x2 - x1) * 1.5) + 40
                side = min(side, W, Hh)
                cx0 = int(np.clip((x1 + x2) / 2 - side / 2, 0, W - side))
                cy0 = int(np.clip(min(y1, e["ball"][1]) - 0.25 * side, 0, Hh - side))
                crop = raw[cy0:cy0 + side, cx0:cx0 + side].copy()
                bx, by = e["ball"] - np.array([cx0, cy0])
                cv2.circle(crop, (int(bx), int(by)), max(6, side // 18), BALL_BGR, 3, cv2.LINE_AA)
                crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
                history.append(dict(e=e, crop=crop))
                history = history[-3:]

            # ---- main-frame text
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            bc.text_box(d, (14, int(50 * sc)), "Far team (opponent): SERVE / SPIKE / BLOCK", fm_sm,
                        (20, 20, 20), (255, 255, 255), TEAM_RGB[OPP])
            bc.text_box(d, (W - 14, int(50 * sc)), f"{video_name}   {fmt_t(ts)}", fm_sm,
                        (255, 255, 255), (20, 20, 20), anchor="ra")
            bc.text_box(d, (14, Hh - 22), f"Near team:  {bc.fmt_formation(forms[FOCUS][fi])}", fm_sm,
                        (20, 20, 20), (255, 255, 255), TEAM_RGB[FOCUS], anchor="ld")
            if tac is not None:
                bc.text_box(d, (W // 2, Hh - 22), f"{tac.upper()}  -  Near team", fm_big,
                            (255, 255, 255), TAC_RGB[tac], (255, 255, 255), pad=10, anchor="md")
            if cur is not None:
                bc.text_box(d, (W // 2, int(100 * sc)), f"{cur['label']}  -  Far team (opponent)", fm_big,
                            (255, 255, 255), OPP_TAG_RGB, (255, 255, 255), pad=10, anchor="ma")
                if key_q is not None:
                    x1, y1, x2, y2 = key_q["box"]
                    bc.text_box(d, ((x1 + x2) / 2, y1 - 30), cur["label"].lower(), fm_sm,
                                (255, 255, 255), OPP_TAG_RGB, anchor="md", pad=4)
            main_rgb = np.array(pil)

            # ---- side panel
            panel = np.full((Hh, PANEL_W, 3), 18, np.uint8)
            tl_y = 228
            panel[tl_y:tl_y + TL_H, TL_X0:TL_X0 + TL_W] = timeline
            px = TL_X0 + int(fi / max(n - 1, 1) * (TL_W - 1))
            cv2.line(panel, (px, tl_y - 4), (px, tl_y + TL_H + 3), (255, 255, 255), 2)
            row_y = [318, 446, 574]
            for k in range(len(history)):
                y = row_y[k]
                cv2.rectangle(panel, (15, y - 1), (129, y + 113), TEAM_BGR[OPP], 2)
            ppil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            dp = ImageDraw.Draw(ppil)
            dp.text((16, 10), "Team tactics view", font=f_mid, fill=(255, 255, 255))
            dp.text((16, 40), "Near team (focus) - 4 tactical classes", font=f_xs, fill=(200, 200, 200))
            if tac is not None:
                bc.text_box(dp, (16, 70), tac, f_sm, (255, 255, 255), TAC_RGB[tac], pad=5)
            else:
                bc.text_box(dp, (16, 70), "no call (between rallies)", f_sm, (220, 220, 220),
                            NO_CALL_RGB, pad=5)
            by_ = 104
            for c, name in enumerate(tt.CLASSES):
                p = float(prob[fi, c]) if lab[fi] >= 0 else 0.0
                dp.text((16, by_), name, font=f_xs, fill=(225, 225, 225))
                dp.rectangle([200, by_ + 3, 200 + 200, by_ + 15], fill=(50, 50, 50))
                dp.rectangle([200, by_ + 3, 200 + int(200 * p), by_ + 15], fill=TAC_RGB[name])
                dp.text((408, by_), f"{p:.2f}", font=f_xs, fill=(225, 225, 225))
                by_ += 22
            dp.text((16, 200), "Tactic timeline (this segment)", font=f_xs, fill=(200, 200, 200))
            dp.text((16, 260), "Opponent actions (far team): SERVE / SPIKE / BLOCK", font=f_xs,
                    fill=(200, 200, 200))
            dp.line([(16, 284), (PANEL_W - 16, 284)], fill=(70, 70, 70), width=1)
            if not history:
                dp.text((16, 300), "no opponent serve / spike / block yet", font=f_xs, fill=(150, 150, 150))
            for k, h in enumerate(reversed(history)):      # newest at the top
                y = row_y[k]
                e = h["e"]
                ppil.paste(Image.fromarray(cv2.cvtColor(h["crop"], cv2.COLOR_BGR2RGB)), (16, y))
                bc.text_box(dp, (146, y + 8), e["label"], f_mid, (255, 255, 255), OPP_TAG_RGB, pad=5)
                dp.text((146, y + 50), f"at {fmt_t((start + e['f']) / fps)}", font=f_xs, fill=(210, 210, 210))
                dp.text((146, y + 72), f"player #{e['actor_id']}  (far team)", font=f_xs, fill=(170, 170, 170))
            main_rgb_panel = np.array(ppil)
            panel = main_rgb_panel
            # legend
            lpil = Image.fromarray(panel)
            dl = ImageDraw.Draw(lpil)
            ly = Hh - 26
            dl.ellipse([16, ly + 2, 30, ly + 16], fill=TEAM_RGB[FOCUS], outline=(255, 255, 255), width=2)
            dl.text((36, ly), "near (graph)", font=f_tiny, fill=(230, 230, 230))
            dl.ellipse([136, ly + 2, 150, ly + 16], fill=TEAM_RGB[OPP], outline=(255, 255, 255), width=2)
            dl.text((156, ly), "far (nodes only)", font=f_tiny, fill=(230, 230, 230))
            dl.polygon([tuple(p) for p in bc.star_pts(292, ly + 9, 9, 4)], fill=(255, 220, 0))
            dl.text((305, ly), "actor", font=f_tiny, fill=(230, 230, 230))
            dl.ellipse([360, ly + 2, 374, ly + 16], outline=(255, 225, 0), width=3)
            dl.text((380, ly), "ball", font=f_tiny, fill=(230, 230, 230))
            panel = np.array(lpil)

        frame_rgb = np.ascontiguousarray(np.hstack([main_rgb, panel]))
        if writer is None:
            seg_path = seg_dir / f"seg{seg_idx:04d}.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(seg_path), (frame_rgb.shape[1], frame_rgb.shape[0]), fps=fps, codec="libx264",
                quality=None, pix_fmt_out="yuv420p", output_params=["-crf", "18", "-preset", "medium"],
                macro_block_size=8)
            writer.send(None)
            seg_paths.append(seg_path)
        writer.send(frame_rgb)
        in_seg += 1
        if in_seg >= segment_frames:
            writer.close()
            writer, in_seg, seg_idx = None, 0, seg_idx + 1
        if fi % 1000 == 0:
            print(f"  render {fi}/{n}", flush=True)
    if writer is not None:
        writer.close()
    cap.release()
    written = fi + 1
    if written != n:
        raise RuntimeError(f"wrote {written} of {n} frames")

    list_file = seg_dir / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(list_file), "-c", "copy", "-movflags", "+faststart",
                    str(out_path)], check=True)
    for p in seg_paths:
        p.unlink()
    list_file.unlink()
    seg_dir.rmdir()
    chk = cv2.VideoCapture(str(out_path))
    got = int(chk.get(cv2.CAP_PROP_FRAME_COUNT))
    chk.release()
    if abs(got - n) > 2:
        out_path.unlink()
        raise RuntimeError(f"output has {got} frames, expected {n}")
    print(f"video -> {out_path}  ({got} frames = {got / fps:.1f} s)", flush=True)
    return dict(video=video_name, frames=n, fps=fps,
                opp=Counter(e["label"] for e in opp_events),
                tactics={k: round(v / n, 4) for k, v in dist.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(render(args.cache, args.out, args.max_frames))


if __name__ == "__main__":
    main()
