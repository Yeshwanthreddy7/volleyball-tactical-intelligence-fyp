"""
render_team_relation.py - cheap second pass: replay a pipeline.py
`--dump-tracks` render-state (boxes + ball + court, already filtered to one
team by `--team-side`) and add a Bi-Causal-paper-style overlay on top:

  * Human Relation (R): minimum-spanning-tree edges between the analysed
    team's player boxes each frame (reuses `mst_edges`/`orient_tree` from
    bicausal_render.py), oriented away from whichever player is nearest the
    ball that frame.
  * Human-Object Interaction (I): an arrow from the ball to the nearest
    player when within `--interact-thresh-px` pixels.

The base layer (boxes, jersey numbers, ball marker, tactic banner, court
outline) is `pipeline._annotate_frame`, identical to fyp/rebanner.py, so
this script never re-runs detection - one `pipeline.py --dump-tracks` pass
feeds both the plain re-banner and this overlay.

Usage:
  python OUTPUTS/render_team_relation.py VIDEO.mp4 tracks.npz \
      --windows-csv preds.csv --output out.mp4 \
      --start-frame 0 --max-frames 27000
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fyp"))
sys.path.insert(0, HERE)

from pipeline import _annotate_frame, BALL_TRAIL_LEN, FOURCC  # noqa: E402
from bicausal_render import mst_edges, orient_tree, draw_arrow  # noqa: E402

R_COLOR = (255, 210, 60)    # cyan-ish blue, "Human Relation" edges
I_COLOR = (60, 60, 255)     # red, "Human-Object Interaction" arrow


class _QuadShim:
    def __init__(self) -> None:
        self.corners = None

    def draw(self, frame: np.ndarray) -> None:
        if self.corners is not None:
            cv2.polylines(frame, [self.corners.astype(np.int32)], isClosed=True,
                          color=(0, 220, 220), thickness=2, lineType=cv2.LINE_AA)


_CANON = {
    "coordinated attack": "Coordinated Attack",
    "coordinated defense": "Coordinated Defense",
    "coordinated defence": "Coordinated Defense",
    "delayed support": "Delayed Support",
    "spacing breakdown": "Spacing Breakdown",
    "unclassified": "Unclassified",
    "transition / dead ball": "Unclassified",
}


def _label_lookup(windows_csv: str):
    df = pd.read_csv(windows_csv)
    spans = []
    for _, r in df.iterrows():
        lab = _CANON.get(str(r["label"]).strip().lower(), str(r["label"]))
        conf = float(r["confidence"]) if "confidence" in df.columns else float("nan")
        spans.append((int(r["frame_start"]), int(r["frame_end"]), lab, conf))
    spans.sort()

    def at(fi: int):
        hit = ("Unclassified", float("nan"))
        for a, b, lab, conf in spans:
            if a <= fi <= b:
                hit = (lab, conf)
        return hit
    return at


def _draw_legend(img: np.ndarray) -> None:
    x0, y0 = 16, img.shape[0] - 70
    cv2.rectangle(img, (x0 - 10, y0 - 28), (x0 + 330, y0 + 36), (20, 20, 20), -1)
    cv2.rectangle(img, (x0 - 10, y0 - 28), (x0 + 330, y0 + 36), (90, 90, 90), 1)
    cv2.line(img, (x0, y0), (x0 + 30, y0), R_COLOR, 3, cv2.LINE_AA)
    cv2.putText(img, "R: Human Relation (MST)", (x0 + 40, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + 28), (x0 + 30, y0 + 28), I_COLOR, 3, cv2.LINE_AA)
    cv2.putText(img, "I: Human-Object Interaction", (x0 + 40, y0 + 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("dump_npz", help="render-state .npz from pipeline --dump-tracks")
    ap.add_argument("--windows-csv", required=True, help="--output-csv from pipeline.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--interact-thresh-px", type=int, default=180,
                     help="draw the I arrow only when the ball is within this "
                          "many pixels of the nearest player box centre.")
    args = ap.parse_args()

    d = np.load(args.dump_npz)
    box, ball, court = d["box"], d["ball"], d["court"]
    fps = float(d["fps"][0])
    box_by_f: dict[int, list] = {}
    for row in box:
        box_by_f.setdefault(int(row[0]), []).append(row)
    ball_by_f = {int(r[0]): r for r in ball}
    court_by_f = {int(r[0]): r[1:].reshape(4, 2) for r in court}
    dump_frames = sorted(box_by_f) or sorted(ball_by_f)
    f0 = args.start_frame if args.start_frame is not None else (dump_frames[0] if dump_frames else 0)
    f_last = dump_frames[-1] if dump_frames else 0
    if args.max_frames:
        f_last = min(f_last, f0 + args.max_frames - 1)

    label_at = _label_lookup(args.windows_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, FOURCC, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    cal = _QuadShim()
    trail: deque = deque(maxlen=BALL_TRAIL_LEN)
    last_court = None
    fi = f0
    n = 0
    while fi <= f_last:
        ok, frame = cap.read()
        if not ok:
            break
        rows = box_by_f.get(fi, [])
        xyxy = np.asarray([r[1:5] for r in rows], dtype=float).reshape(-1, 4)
        ids = np.asarray([int(r[5]) for r in rows], dtype=int)
        b = ball_by_f.get(fi)
        ball_c = np.asarray(b[1:3], dtype=float) if b is not None else None
        ball_pred = bool(b[3]) if b is not None else False
        if ball_c is not None and not (
            np.isfinite(ball_c).all()
            and -200 <= ball_c[0] <= w + 200
            and -200 <= ball_c[1] <= h + 200
        ):
            # pipeline.py's BallTracker coast-extrapolation occasionally
            # diverges to huge/NaN coordinates after a long undetected run;
            # treat those frames as "no ball" rather than feeding garbage
            # into cv2.circle (int overflow -> OpenCV argument error) or the
            # nearest-player HOI distance calc.
            ball_c = None
            ball_pred = False
        if fi in court_by_f:
            last_court = court_by_f[fi]
        cal.corners = last_court
        if ball_c is not None:
            trail.append((int(ball_c[0]), int(ball_c[1]), ball_pred))
        lab, conf = label_at(fi)
        conf = 1.0 if conf != conf else conf

        out = _annotate_frame(
            frame, xyxy, ids, ball_c, ball_pred, trail, lab, conf, fi, fps,
            calibrator=cal, hud_metrics=None, current_entropy=0.0,
            is_anomaly=(lab != "Unclassified" and conf < 0.5),
            pose_summary=None,
        )

        # ---- Bi-Causal-paper-style overlay: R (relation MST) + I (HOI arrow)
        centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0,
                             (xyxy[:, 1] + xyxy[:, 3]) / 2.0], axis=1) if len(xyxy) else np.zeros((0, 2))
        nearest_idx = None
        if len(centers) and ball_c is not None:
            d2 = np.linalg.norm(centers - ball_c[None, :], axis=1)
            nearest_idx = int(np.argmin(d2))
            if d2[nearest_idx] <= args.interact_thresh_px:
                draw_arrow(out, ball_c, centers[nearest_idx], I_COLOR, thick=3, shrink=6)
            else:
                nearest_idx = None

        if len(centers) >= 2:
            root = nearest_idx if nearest_idx is not None else 0
            edges = orient_tree(mst_edges(centers), root)
            for i, j in edges:
                draw_arrow(out, centers[i], centers[j], R_COLOR, thick=2, shrink=14)

        _draw_legend(out)

        writer.write(out)
        fi += 1
        n += 1
        if n % 500 == 0:
            print(f"  {n} frames…")
    cap.release()
    writer.release()
    print(f"wrote {args.output}  ({n} frames)")


if __name__ == "__main__":
    main()
