"""NEW_OUTPUTS renderer: clean one-team relation graph + continuous ball.

The annotated video shows only:
  * your team (near side): at most six players standing inside the near half
    of the court, each linked to its two neighbours around the team (a closed
    ring, so six players give exactly six links);
  * the ball, with gaps of up to 2 s in live play bridged along a flight arc,
    so it is tracked continuously.

  * a caption naming your team's tactic (SHOWN_TACTICS: Coordinated Attack,
    Coordinated Defense, Delayed Support, Spacing Breakdown) while the model
    is at least 50% sure.

Nothing is drawn for the opponent or for anyone outside the court, and there
is no side panel. The tactic CSV (*_team_tactics.csv) holds exactly the
captioned segments, so the website's report lists the same moments the video
shows.
The opponent CSV is still written for research use; the website does not show it.

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                              # finalyearproject/
sys.path.insert(0, str(ROOT / "OUTPUTS"))
sys.path.insert(0, str(HERE))
import bicausal_render as bc                        # noqa: E402
import team_tactics as tt                           # noqa: E402

FOCUS, OPP = bc.NEAR, bc.FAR
OPP_ACTIONS = ("SERVE", "SPIKE", "BLOCK")
COURT_MARGIN_M = 0.5        # tolerance for homography error at the lines
BALL_MAX_GAP_S = 2.0
# the only actions shown: caption in the video + rows in the tactic CSV / report
SHOWN_TACTICS = ("Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown")
CAPTION_BGR = {"Coordinated Attack": (40, 110, 235), "Coordinated Defense": (150, 120, 20),
               "Delayed Support": (150, 60, 140), "Spacing Breakdown": (60, 60, 215)}

TEAM_BGR = bc.TEAM_BGR
BALL_BGR = bc.BALL_BGR


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


def in_near_half(q, m=COURT_MARGIN_M):
    X, Y = q["court"]
    return -m <= X <= 9 + m and 9 - m <= Y <= 18 + m


def team_ring(court_pts):
    """Index pairs linking each player to its neighbours around the team centroid."""
    k = len(court_pts)
    if k < 2:
        return []
    if k == 2:
        return [(0, 1)]
    c = court_pts - court_pts.mean(axis=0)
    order = np.argsort(np.arctan2(c[:, 1], c[:, 0]))
    return [(int(order[i]), int(order[(i + 1) % k])) for i in range(k)]


def ball_over_court(ball, m=1.5):
    """Drop ball points that project beside the court (spare balls at the benches)."""
    out = ball.copy()
    have = ~np.isnan(ball[:, 0])
    if have.any():
        c = bc.to_court(ball[have])
        # points near the horizon project to huge depths; only judge sane projections
        sane = (c[:, 1] > -15) & (c[:, 1] < 35)
        beside = sane & ((c[:, 0] < -m) | (c[:, 0] > 9 + m))
        idx = np.where(have)[0][beside]
        out[idx] = np.nan
    return out


def continuous_ball(ball, ok, max_gap, ctx=6):
    """Bridge detection gaps of up to max_gap frames (both ends tracked, all frames ok).

    The gap follows a flight arc: x linear and y quadratic in time, fitted to
    the tracked points on both sides; a straight line when too few points.
    """
    out = ball.copy()
    have = ~np.isnan(ball[:, 0])
    idx = np.where(have)[0]
    for a, c in zip(idx[:-1], idx[1:]):
        if not (1 < c - a <= max_gap and ok[a:c + 1].all()):
            continue
        gap = np.arange(a + 1, c)
        before = idx[(idx <= a) & (idx > a - 2 * ctx)][-ctx:]
        after = idx[(idx >= c) & (idx < c + 2 * ctx)][:ctx]
        known = np.concatenate([before, after])
        if len(before) >= 2 and len(after) >= 2:
            fx = np.polyval(np.polyfit(known, ball[known, 0], 1), gap)
            fy = np.polyval(np.polyfit(known, ball[known, 1], 2), gap)
            # pin the arc to both ends so it joins the track without a jump
            t = (gap - a) / (c - a)
            ex = ball[a] + t[:, None] * (ball[c] - ball[a])
            fa = np.array([np.polyval(np.polyfit(known, ball[known, 0], 1), [a, c]),
                           np.polyval(np.polyfit(known, ball[known, 1], 2), [a, c])])
            corr = (fa[:, 0] + t[:, None] * (fa[:, 1] - fa[:, 0]))
            out[gap, 0] = fx - corr[:, 0] + ex[:, 0]
            out[gap, 1] = fy - corr[:, 1] + ex[:, 1]
        else:
            t = ((gap - a) / (c - a))[:, None]
            out[gap] = ball[a] + t * (ball[c] - ball[a])
    return out


# ---------------------------------------------------------------- drawing
def draw_node(img, q, team):
    c = tuple(q["node"].astype(int))
    cv2.circle(img, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, c, 7, TEAM_BGR[team], -1, cv2.LINE_AA)
    fx, fy = q["feet"].astype(int)
    cv2.ellipse(img, (fx, fy), (22, 7), 0, 0, 360, TEAM_BGR[team], 2, cv2.LINE_AA)


def shown_only(lab):
    """Per-frame labels with every class outside SHOWN_TACTICS set to -1 (no call)."""
    keep = [tt.CLASSES.index(t) for t in SHOWN_TACTICS]
    return np.where(np.isin(lab, keep), lab, -1)


def draw_caption(img, tactic):
    """Tactic name in a box at the top left, under where broadcast score bugs sit."""
    h = img.shape[0]
    scale = max(0.6, h / 720 * 0.9)
    thick = max(1, int(round(scale * 2)))
    text = tactic.upper()
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x, y = int(0.02 * img.shape[1]), int(0.11 * h)
    pad = int(10 * scale)
    box = img[y - th - pad:y + base + pad, x - pad:x + tw + pad]
    box[:] = (box * 0.35 + np.array(CAPTION_BGR[tactic]) * 0.65).astype(np.uint8)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


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
    del data, frames                                 # raw detections no longer needed (RAM)

    print("  camera / replay mask ...", flush=True)
    cam_ok = camera_ok_mask(video, start, n)
    # the court-similarity score hovers near its threshold on some frames:
    # remove sub-0.5 s flicker in both directions
    cam_ok = _fill_runs(_fill_runs(cam_ok, False, int(0.5 * fps)), True, int(0.5 * fps))
    live = live_mask(n, fps, touches, ball, players, cam_ok)
    print("  team tactics (HGB) ...", flush=True)
    rows = tt.raw_rows(players, ball, (bc.MAIN_W, bc.MAIN_H), FOCUS)
    ws, we, P = tt.window_probs(rows, fps)
    lab, prob = tt.frame_calls(n, fps, ws, we, P, live)
    lab = shown_only(lab)
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

    ball_draw = continuous_ball(ball_over_court(ball), cam_ok, int(BALL_MAX_GAP_S * fps))
    trail, held, hold = [], {}, int(0.5 * fps)

    seg_dir = out_path.parent / (stem.name + "_segments")
    seg_dir.mkdir(parents=True, exist_ok=True)
    writer, seg_paths, seg_idx, in_seg = None, [], 0, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fi in range(n):
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"render: could not read frame {start + fi} of {video}")

        if not cam_ok[fi]:
            trail, held = [], {}            # replay / camera cut: plain frame
        else:
            tq = [q for q in players[fi] if q["team"] == FOCUS and in_near_half(q)][:6]
            for q in tq:
                if q["id"] >= 0:
                    held[q["id"]] = (fi, q)
            # a player missed for a few frames keeps the last position (no flicker)
            for tid, (f0, q0) in sorted(held.items(), key=lambda kv: -kv[1][0]):
                if len(tq) >= 6:
                    break
                if 0 < fi - f0 <= hold and all(q["id"] != tid for q in tq) and all(
                        np.linalg.norm(q["court"] - q0["court"]) > 1.0 for q in tq):
                    tq.append(q0)
            held = {k: v for k, v in held.items() if fi - v[0] <= hold}
            if tq:
                nodes = np.array([q["node"] for q in tq], np.float32)
                for i, j in team_ring(np.array([q["court"] for q in tq], np.float32)):
                    p, r = tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int))
                    cv2.line(img, p, r, (30, 30, 30), 6, cv2.LINE_AA)
                    cv2.line(img, p, r, TEAM_BGR[FOCUS], 3, cv2.LINE_AA)
                for q in tq:
                    draw_node(img, q, FOCUS)

            b = ball_draw[fi]
            if not np.isnan(b[0]):
                trail.append((fi, b.copy()))
            trail = [(f, p) for f, p in trail if fi - f <= 10]
            for k in range(1, len(trail)):
                if trail[k][0] - trail[k - 1][0] > 1:
                    continue
                a = k / len(trail)
                cv2.line(img, tuple(trail[k - 1][1].astype(int)), tuple(trail[k][1].astype(int)),
                         BALL_BGR, max(1, int(4 * a)), cv2.LINE_AA)
            if not np.isnan(b[0]):
                bp = tuple(b.astype(int))
                cv2.circle(img, bp, 13, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, bp, 11, BALL_BGR, 3, cv2.LINE_AA)
            if lab[fi] >= 0:
                draw_caption(img, tt.CLASSES[lab[fi]])

        frame_rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
