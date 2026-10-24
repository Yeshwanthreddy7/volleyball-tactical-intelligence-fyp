"""NEW_OUTPUTS renderer, revision 2 (first used for videoplayback (2).mp4).

Our team = near (bottom) side. Opponent = far (top) side.

Main frame
  * Our team only: a node on each PLAYER plus the relation graph (MST in
    court metres + faint all-pair links), tinted with the current tactic.
    Referees, line judges, coaches and anyone else on the floor get no node
    (filter_our_players: position + jersey colour + track quality).
  * Opponent: no nodes and no links. Only its SERVE / SPIKE / BLOCK are shown,
    as a tag above the player, a ring at the feet and the ball link.
  * Banners: our tactic at the bottom, the opponent action at the top.

Side panel, split in two halves
  * left  - our team: current tactic, held-out accuracy of each of the four
            tactics (lovo_per_class_accuracy.json), agreement with the hand
            labels of this video, and the timings of every tactic segment.
  * right - opponent: current action and the timings of every SERVE / SPIKE /
            BLOCK. These come from the rule-based touch detector; there are no
            hand labels for them, so no accuracy is shown.

Usage:
  python render_v2.py --cache ../../PAPER_OUTPUT/cache/cache_vp2.pkl --out ../vp2_final/vp2_team_tactics.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
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
from new_render import _fill_runs, fmt_t           # noqa: E402

OUR, OPP = bc.NEAR, bc.FAR
OPP_ACTIONS = ("SERVE", "SPIKE", "BLOCK")
PANEL_W = 600
HALF = PANEL_W // 2

TEAM_BGR = bc.TEAM_BGR
BALL_BGR = bc.BALL_BGR
TAC_RGB = {
    "Coordinated Attack": (255, 130, 0),
    "Coordinated Defense": (20, 175, 120),
    "Delayed Support": (225, 80, 160),
    "Spacing Breakdown": (140, 90, 220),
}
TAC_BGR = {k: (v[2], v[1], v[0]) for k, v in TAC_RGB.items()}
OPP_RGB = (190, 40, 40)
OPP_ACT_RGB = {"SERVE": (215, 120, 20), "SPIKE": (190, 40, 40), "BLOCK": (120, 60, 170)}
GREY = (90, 90, 90)


# ---------------------------------------------------------------- one video pass
def scan_video(video, start, n, players, sample_every=3):
    """Camera-ok mask (court matches the empty-court base frame) and the mean
    torso colour (CIE-Lab) of every tracked person, in one sequential read."""
    base_png = ROOT / "BASEFRAMES" / "base_frame" / (Path(video).stem + ".png")
    base_sig = bc._court_sig(cv2.imread(str(base_png))) if base_png.exists() else None
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cam_ok = np.ones(n, bool)
    colours = defaultdict(list)
    for fi in range(n):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"scan: could not read frame {start + fi} of {video}")
        if base_sig is not None:
            cam_ok[fi] = float((bc._court_sig(frame) * base_sig).mean()) >= 0.45
        if fi % sample_every or not cam_ok[fi]:
            continue
        for q in players[fi]:
            if q["id"] < 0:
                continue
            x1, y1, x2, y2 = q["box"].astype(int)
            w, h = x2 - x1, y2 - y1
            crop = frame[max(0, y1 + int(0.2 * h)):y1 + int(0.5 * h),
                         max(0, x1 + int(0.25 * w)):x2 - int(0.25 * w)]
            if crop.size >= 30:
                colours[q["id"]].append((fi, cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)))
    cap.release()
    return cam_ok, {k: (np.array([f for f, _ in v]), np.array([c for _, c in v]))
                    for k, v in colours.items() if len(v) >= 3}


def all_people(frames, W):
    """Every tracked person of every frame with court position, before any team
    split or six-per-side cap (bicausal_render.build_players assigns teams by
    depth, which sends our blockers at the net to the far side)."""
    out = []
    for fr in frames:
        g = []
        for p in fr["people"]:
            if p["id"] < 0:
                continue
            feet, node, head, sh, wrists = bc.person_geometry(p)
            x1, _, x2, _ = p["box"]
            g.append(dict(id=p["id"], conf=p["conf"], box=p["box"], feet=feet, node=node, head=head,
                          sh=sh, wrists=wrists, court=bc.to_court(feet)[0], team=OUR,
                          edge=bool(x1 <= 3 or x2 >= W - 3)))
        out.append(g)
    return out


def filter_our_players(people, colours, fps):
    """Keep only our team's real players, decided per detection.

    Kit colours: the opponent's kit is learned from long tracks deep in the far
    half, ours from long tracks deep in the near half (CIE-Lab, hue/chroma only,
    so a dark libero shirt is not confused with a dark opponent kit).

    Whole tracks rejected
      * shorter than 20 frames,
      * median position outside the sidelines (referees, coaches, bench, staff),
      * standing still at a corner behind the end line (line judges).
    Each remaining detection is ours when
      * its LOCAL kit colour (track samples within +-0.5 s; tracker ids can swap
        between teams at the net) is closer to our kit than the opponent's,
      * it is on our side: between the sidelines (+ margin) and not deep in the
        far half (a jumping hitter's / blocker's feet project past the net),
      * it is not cut off at the image edge.
    At most six per frame.
    """
    per = defaultdict(list)
    for ps in people:
        for q in ps:
            per[q["id"]].append(q["court"])
    per = {k: np.array(v) for k, v in per.items()}
    med = {k: np.median(colours[k][1], 0) for k in colours}

    def ref(box):
        acc = []
        for tid, cs in per.items():
            mx, my = np.median(cs, 0)
            if len(cs) >= 150 and tid in med and box[0] <= mx <= box[1] and box[2] <= my <= box[3]:
                acc += [med[tid]] * len(cs)
        return np.median(acc, 0)

    our_ab = ref((0.5, 8.5, 11.0, 17.5))[1:]
    opp_ab = ref((0.5, 8.5, 0.5, 7.0))[1:]
    print(f"  kit hue (Lab a,b): ours {np.round(our_ab)}  opponent {np.round(opp_ab)}", flush=True)

    bad, why = set(), Counter()
    for tid, cs in per.items():
        mx, my = np.median(cs, 0)
        if len(cs) < 20:
            bad.add(tid); why["short track"] += 1
        elif tid not in colours:
            bad.add(tid); why["no colour sample"] += 1
        elif not -0.6 <= mx <= 9.6:
            bad.add(tid); why["outside the sidelines (official / bench / staff)"] += 1
        elif my > 17.5 and (mx < 0.6 or mx > 8.4) and cs.std(0).max() < 0.6:
            bad.add(tid); why["line judge (static at corner)"] += 1
    print(f"  rejected tracks {dict(why)}", flush=True)

    win = int(0.5 * fps)
    out, n_opp = [], 0
    for fi, ps in enumerate(people):
        tq = []
        for q in ps:
            if q["id"] in bad or q["edge"]:
                continue
            X, Y = q["court"]
            if not (-1.0 <= X <= 10.0 and 2.5 <= Y <= 21.5):
                continue
            fs, cs = colours[q["id"]]
            near = np.abs(fs - fi) <= win
            ab = np.median(cs[near] if near.any() else cs, 0)[1:]
            d_our, d_opp = np.linalg.norm(ab - our_ab), np.linalg.norm(ab - opp_ab)
            # past the net (Y < 9) the kit must be clearly ours
            if d_opp < d_our or (Y < 9.0 and d_our > 0.6 * d_opp):
                n_opp += 1
                continue
            tq.append(q)
        tq.sort(key=lambda q: (bc._outside(q["court"]), -q["conf"]))
        out.append(tq[:6])
    print(f"  {n_opp} detections on our side rejected as opponent kit; "
          f"mean {np.mean([len(t) for t in out]):.2f} of our players per frame", flush=True)
    return out


def rally_mask(n, fps, touches, ours, cam_ok):
    """Live play = from 1 s before the first to 1.5 s after the last ball touch
    of each rally (>= 2 touches, <= 4 s apart). Only real touches count: the ball
    tracker sometimes follows a ball in the crowd during celebrations."""
    live = np.zeros(n, bool)
    group = []
    for f in [t["f"] for t in touches] + [None]:
        if f is not None and (not group or f - group[-1] <= 4.0 * fps):
            group.append(f)
            continue
        if len(group) >= 2:          # one lone touch = ball thrown back between points
            live[max(0, int(group[0] - 1.0 * fps)):min(n, int(group[-1] + 1.5 * fps) + 1)] = True
        group = [f] if f is not None else []
    live &= cam_ok & (np.array([len(t) for t in ours]) >= 3)
    live = _fill_runs(live, False, int(fps))
    return _fill_runs(live, True, int(fps)) & cam_ok


# ---------------------------------------------------------------- accuracy
def load_lovo_accuracy():
    p = HERE.parent / "lovo_per_class_accuracy.json"
    d = json.loads(p.read_text())
    return {k: v["accuracy"] for k, v in d["per_class"].items()}, d["overall_accuracy"]


def video_label_agreement(labels_csv, video_name, start, n, fps, lab):
    """Compare the per-frame call with the hand labels (labels.csv) of this segment.
    A label counts as matched when the call at its anchor time (+-0.5 s majority)
    is the labelled class."""
    rows = []
    norm = {"Coordinated_Defence": "Coordinated Defense", "Coordinated_Attack": "Coordinated Attack",
            "Delayed_Support": "Delayed Support", "Spacing_Breakdown": "Spacing Breakdown"}
    with open(labels_csv, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 3 or r[0].strip() != video_name:
                continue
            try:
                m, s = r[1].split(":")
                ts = int(m) * 60 + float(s)
            except ValueError:
                continue
            fi = int(round(ts * fps)) - start
            if 0 <= fi < n and r[2].strip() in norm:
                a, b = max(0, fi - int(0.5 * fps)), min(n, fi + int(0.5 * fps) + 1)
                calls = [v for v in lab[a:b] if v >= 0]
                pred = tt.CLASSES[Counter(calls).most_common(1)[0][0]] if calls else None
                rows.append(dict(time=fmt_t(ts), label=norm[r[2].strip()], predicted=pred or "no call",
                                 match=pred == norm[r[2].strip()]))
    return rows


# ---------------------------------------------------------------- drawing
def tag(draw, xy, text, font, bg, anchor="la", pad=5, fg=(255, 255, 255)):
    return bc.text_box(draw, xy, text, font, fg, bg, pad=pad, anchor=anchor)


def render(cache, out_path, labels_csv, max_frames=0, segment_frames=4000):
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
    cap.release()
    bc.MAIN_W, bc.MAIN_H = W, Hh

    print(f"[{video_name}] {n} frames @ {fps:.2f} fps", flush=True)
    players = bc.build_players(frames)
    ball, segs, _ = bc.build_ball(frames, players)
    touches = bc.find_touches(ball, segs, players)
    events = bc.label_touches(touches, fps)
    opp_events = [e for e in events if e["team"] == OPP and e["label"] in OPP_ACTIONS]
    print(f"  opponent actions {Counter(e['label'] for e in opp_events).most_common()}", flush=True)
    people = all_people(frames, W)
    del data, frames

    print("  video scan (camera mask + jersey colours) ...", flush=True)
    cam_ok, colours = scan_video(video, start, n, people)
    cam_ok = _fill_runs(_fill_runs(cam_ok, False, int(0.5 * fps)), True, int(0.5 * fps))
    ours = filter_our_players(people, colours, fps)

    # tactics from our filtered players only
    our_view = ours
    live = rally_mask(n, fps, touches, our_view, cam_ok)
    rows = tt.raw_rows(our_view, ball, bc.to_court, OUR)
    ws, we, P = tt.window_probs(rows, fps)
    lab, prob = tt.frame_calls(n, fps, ws, we, P, live)
    dist = Counter(tt.CLASSES[v] if v >= 0 else "no call" for v in lab)
    print(f"  tactic share { {k: f'{v / n:.1%}' for k, v in dist.most_common()} }", flush=True)

    acc_cls, acc_all = load_lovo_accuracy()
    agree = video_label_agreement(labels_csv, video_name, start, n, fps, lab)
    n_match = sum(r["match"] for r in agree)
    print(f"  hand labels in segment: {len(agree)}, matched {n_match}", flush=True)

    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    tac_segs = tt.segments(lab, prob, fps, start)
    with open(str(stem) + "_our_team_tactics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_frame", "end_frame", "start_s", "end_s", "tactic",
                                          "mean_confidence"])
        w.writeheader()
        w.writerows(tac_segs)
    with open(str(stem) + "_opponent_actions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_time", "frame", "action", "actor_track_id"])
        for e in opp_events:
            w.writerow([fmt_t((start + e["f"]) / fps), start + e["f"], e["label"], e["actor_id"]])
    with open(str(stem) + "_hand_label_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "label", "predicted", "match"])
        w.writeheader()
        w.writerows(agree)

    sc = Hh / 720.0
    fm_big = bc.load_font(int(30 * sc), True)
    fm_sm = bc.load_font(int(17 * sc), True)
    fm_mid = bc.load_font(int(22 * sc), True)
    f_title = bc.load_font(21, True)
    f_head = bc.load_font(16, True)
    f_txt = bc.load_font(14)
    f_txtb = bc.load_font(14, True)
    f_small = bc.load_font(12)

    # per-frame index of the tactic segment, for the timing lists
    seg_start_f = [s["start_frame"] - start for s in tac_segs]
    ev_frames = np.array([e["f"] for e in opp_events]) if opp_events else np.array([], int)
    ev_at = {e["f"]: e for e in opp_events}

    seg_dir = out_path.parent / (stem.name + "_segments")
    seg_dir.mkdir(parents=True, exist_ok=True)
    writer, seg_paths, seg_idx, in_seg = None, [], 0, 0
    trail = []

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fi in range(n):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"render: could not read frame {start + fi} of {video}")
        img = frame
        ts = (start + fi) / fps
        tac = tt.CLASSES[lab[fi]] if lab[fi] >= 0 else None
        cur = None
        if cam_ok[fi]:
            tq = ours[fi]
            nodes = np.array([q["node"] for q in tq], np.float32)
            overlay = img.copy()
            for i in range(len(tq)):
                for j in range(i + 1, len(tq)):
                    cv2.line(overlay, tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int)),
                             TEAM_BGR[OUR], 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            if len(tq) >= 2:
                edge = TAC_BGR[tac] if tac else (255, 255, 255)
                for i, j in bc.mst_edges(np.array([q["court"] for q in tq], np.float32)):
                    p, q = tuple(nodes[i].astype(int)), tuple(nodes[j].astype(int))
                    cv2.line(img, p, q, (30, 30, 30), 7, cv2.LINE_AA)
                    cv2.line(img, p, q, edge, 4, cv2.LINE_AA)
            for q in tq:
                c = tuple(q["node"].astype(int))
                cv2.circle(img, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, 7, TEAM_BGR[OUR], -1, cv2.LINE_AA)

            # opponent action (no nodes for the opponent)
            past = ev_frames[ev_frames <= fi + 6]
            cur = ev_at[int(past[-1])] if len(past) and fi - past[-1] <= int(1.6 * fps) else None
            key = None
            if cur is not None:
                opp_now = [q for q in players[fi] if q["team"] == OPP]
                same = [q for q in opp_now if q["id"] == cur["actor_id"] and q["id"] >= 0]
                if same:
                    key = same[0]
                elif opp_now:
                    nq = min(opp_now, key=lambda q: np.linalg.norm(q["node"] - cur["actor"]["node"]))
                    if np.linalg.norm(nq["node"] - cur["actor"]["node"]) <= 120:
                        key = nq

            b = ball[fi]
            has_ball = not np.isnan(b[0])
            if has_ball:
                trail.append((fi, b.copy()))
            trail = [(f, p) for f, p in trail if fi - f <= 14]
            for k in range(1, len(trail)):
                if trail[k][0] - trail[k - 1][0] > 3:
                    continue
                cv2.line(img, tuple(trail[k - 1][1].astype(int)), tuple(trail[k][1].astype(int)),
                         BALL_BGR, max(1, int(4 * k / len(trail))), cv2.LINE_AA)
            if key is not None:
                fx, fy = key["feet"].astype(int)
                cv2.ellipse(img, (fx, fy), (30, 10), 0, 0, 360, BALL_BGR, 3, cv2.LINE_AA)
                if has_ball:
                    tgt = min(key["wrists"] + [key["head"]], key=lambda p: np.linalg.norm(p - b))
                    bc.dashed_line(img, b, tgt, BALL_BGR, 2, 8)
            if has_ball:
                bp = tuple(b.astype(int))
                cv2.circle(img, bp, 13, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, bp, 11, BALL_BGR, 3, cv2.LINE_AA)

            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            bc.text_box(d, (W - 14, int(50 * sc)), fmt_t(ts), fm_sm, (255, 255, 255), (20, 20, 20), anchor="ra")
            if tac is not None:
                bc.text_box(d, (W // 2, Hh - 22), f"{tac.upper()}  -  Our team", fm_big,
                            (255, 255, 255), TAC_RGB[tac], (255, 255, 255), pad=10, anchor="md")
            if cur is not None:
                bc.text_box(d, (W // 2, int(60 * sc)), f"{cur['label']}  -  Opponent", fm_big,
                            (255, 255, 255), OPP_ACT_RGB[cur["label"]], (255, 255, 255), pad=10, anchor="ma")
                if key is not None:
                    x1, y1, x2, y2 = key["box"]
                    bc.text_box(d, ((x1 + x2) / 2, y1 - 12), cur["label"].lower(), fm_sm,
                                (255, 255, 255), OPP_ACT_RGB[cur["label"]], anchor="md", pad=4)
        else:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            bc.text_box(d, (W // 2, Hh - 22), "replay / camera change - overlays paused",
                        fm_mid, (255, 255, 255), (60, 60, 60), anchor="md")
        main_rgb = np.array(pil)

        # ------------------------------------------------ side panel (two halves)
        ppil = Image.new("RGB", (PANEL_W, Hh), (18, 18, 22))
        dp = ImageDraw.Draw(ppil)
        dp.text((14, 10), "Team tactics view", font=f_title, fill=(255, 255, 255))
        dp.line([(HALF, 44), (HALF, Hh - 10)], fill=(80, 80, 90), width=2)
        dp.line([(10, 42), (PANEL_W - 10, 42)], fill=(80, 80, 90), width=1)

        # left: our team
        x0 = 12
        dp.text((x0, 52), "OUR TEAM (near side)", font=f_head, fill=(120, 170, 255))
        if tac is not None:
            tag(dp, (x0, 80), tac, f_txtb, TAC_RGB[tac])
        else:
            tag(dp, (x0, 80), "no call (ball dead)", f_txtb, GREY)
        dp.text((x0, 110), "Model accuracy per tactic", font=f_txtb, fill=(230, 230, 230))
        dp.text((x0, 127), "(held-out videos, 948 clips)", font=f_small, fill=(160, 160, 160))
        y = 146
        for name in tt.CLASSES:
            hl = name == tac
            dp.rectangle([x0, y + 3, x0 + 10, y + 13], fill=TAC_RGB[name])
            dp.text((x0 + 16, y), name, font=f_txtb if hl else f_txt,
                    fill=(255, 255, 255) if hl else (200, 200, 200))
            dp.text((HALF - 12, y), f"{acc_cls[name] * 100:.0f}%", font=f_txtb if hl else f_txt,
                    fill=(255, 255, 255) if hl else (200, 200, 200), anchor="ra")
            y += 19
        dp.text((x0, y + 2), f"overall {acc_all * 100:.0f}%", font=f_small, fill=(170, 170, 170))
        if agree:
            dp.text((x0, y + 18), f"this video: {n_match} of {len(agree)} hand labels", font=f_small,
                    fill=(170, 170, 170))
        y += 42
        dp.line([(x0, y), (HALF - 10, y)], fill=(70, 70, 80), width=1)
        dp.text((x0, y + 6), "Timings (our team)", font=f_txtb, fill=(230, 230, 230))
        y += 28
        k = int(np.searchsorted(seg_start_f, fi, side="right"))   # segments started so far
        room = int((Hh - 12 - y) // 34)
        for s in tac_segs[max(0, k - room):k][::-1]:
            active = s["start_frame"] - start <= fi <= s["end_frame"] - start
            dp.rectangle([x0, y + 3, x0 + 10, y + 27], fill=TAC_RGB[s["tactic"]])
            dp.text((x0 + 16, y), f"{fmt_t(s['start_s'])} - {fmt_t(s['end_s'])}", font=f_txtb if active else f_txt,
                    fill=(255, 255, 255) if active else (190, 190, 190))
            dp.text((x0 + 16, y + 15), s["tactic"], font=f_small,
                    fill=TAC_RGB[s["tactic"]] if active else (150, 150, 150))
            y += 34

        # right: opponent
        x1 = HALF + 12
        dp.text((x1, 52), "OPPONENT (far side)", font=f_head, fill=(255, 120, 120))
        if cur is not None:
            tag(dp, (x1, 80), cur["label"], f_txtb, OPP_ACT_RGB[cur["label"]])
        else:
            tag(dp, (x1, 80), "no action now", f_txtb, GREY)
        dp.text((x1, 110), "Actions shown: SERVE, SPIKE, BLOCK", font=f_small, fill=(170, 170, 170))
        cnt = Counter(e["label"] for e in opp_events if e["f"] <= fi)
        yy = 130
        for a in OPP_ACTIONS:
            dp.rectangle([x1, yy + 3, x1 + 10, yy + 13], fill=OPP_ACT_RGB[a])
            dp.text((x1 + 16, yy), a, font=f_txt, fill=(200, 200, 200))
            dp.text((PANEL_W - 12, yy), f"{cnt[a]} so far", font=f_txt, fill=(200, 200, 200), anchor="ra")
            yy += 19
        yy = 146 + 4 * 19 + 42                     # aligned with the left list
        dp.line([(x1, yy), (PANEL_W - 10, yy)], fill=(70, 70, 80), width=1)
        dp.text((x1, yy + 6), "Timings (opponent)", font=f_txtb, fill=(230, 230, 230))
        yy += 28
        done = [e for e in opp_events if e["f"] <= fi + 6]
        room = int((Hh - 12 - yy) // 22)
        for e in done[-room:][::-1]:
            active = cur is not None and e is cur
            dp.rectangle([x1, yy + 3, x1 + 10, yy + 13], fill=OPP_ACT_RGB[e["label"]])
            dp.text((x1 + 16, yy), fmt_t((start + e["f"]) / fps), font=f_txtb if active else f_txt,
                    fill=(255, 255, 255) if active else (190, 190, 190))
            dp.text((x1 + 90, yy), e["label"], font=f_txtb if active else f_txt,
                    fill=OPP_ACT_RGB[e["label"]] if active else (170, 170, 170))
            yy += 22
        panel = np.array(ppil)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=str(ROOT / "fyp" / "labels.csv"))
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render(args.cache, args.out, args.labels, args.max_frames)


if __name__ == "__main__":
    main()
