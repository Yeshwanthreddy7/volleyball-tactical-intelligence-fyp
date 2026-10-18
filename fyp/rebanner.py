"""
rebanner.py - re-render an annotated video with a DIFFERENT model's tactical
banner, reusing a render-state dump from `pipeline.py --dump-tracks`.

Detection + tracking + strict-render are the expensive parts and are identical
for every classifier, so comparing N models on one video needs ONE
`pipeline.py` pass (with --dump-tracks) and then N cheap rebanner passes
(no YOLO, ~1 min each).

Usage
-----
  # one detection pass
  python fyp/pipeline.py VIDEO.mp4 tactical_ml_v4.joblib --strict-render \
      --auto-court <serve flags> --output-video hgb.mp4 \
      --output-csv hgb_windows.csv --frame-csv frames.csv \
      --dump-tracks tracks.npz --start-frame S --max-frames N

  # per-model predictions for every classifier (9 ML + 4 DL)
  python classify_all_models.py frames.csv all_models.csv

  # a banner-only video per model
  python fyp/rebanner.py VIDEO.mp4 tracks.npz --all-models-csv all_models.csv \
      --model inceptiontime --start-frame S --output annot_inceptiontime.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import _annotate_frame, BALL_TRAIL_LEN, FOURCC  # noqa: E402


class _QuadShim:
    """Minimal stand-in for CourtCalibrator: _annotate_frame only touches
    `.corners` and `.draw()`."""

    def __init__(self) -> None:
        self.corners: "np.ndarray | None" = None

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
    "no manual label": "No manual label",
}


def _label_lookup(args) -> "callable":
    """Return f(frame_idx) -> (canonical label, confidence)."""
    if args.all_models_csv:
        df = pd.read_csv(args.all_models_csv)
        if args.model not in df.columns:
            sys.exit(f"--model {args.model!r} not a column in {args.all_models_csv} "
                     f"(have: {[c for c in df.columns if not c.endswith('__conf')]})")
        col = args.model
        conf_col = f"{args.model}__conf"
    else:
        df = pd.read_csv(args.windows_csv)
        col = "label"
        conf_col = "confidence"
    have_conf = conf_col in df.columns
    spans = []
    for _, r in df.iterrows():
        lab = _CANON.get(str(r[col]).strip().lower(), str(r[col]))
        conf = float(r[conf_col]) if have_conf else float("nan")
        spans.append((int(r["frame_start"]), int(r["frame_end"]), lab, conf))
    spans.sort()

    def at(fi: int) -> "tuple[str, float]":
        hit = ("Unclassified", float("nan"))
        for a, b, lab, conf in spans:
            if a <= fi <= b:
                hit = (lab, conf)
        return hit
    return at, (spans[0][0] if spans else 0), (spans[-1][1] if spans else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("dump_npz", help="render-state .npz from pipeline --dump-tracks")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--windows-csv", help="single-model --output-csv from pipeline.py")
    g.add_argument("--all-models-csv", help="wide CSV from classify_all_models.py")
    ap.add_argument("--model", default="", help="column name when --all-models-csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-frame", type=int, default=None,
                    help="defaults to the first frame present in the dump")
    ap.add_argument("--max-frames", type=int, default=0)
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

    label_at, _, _ = _label_lookup(args)

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
        if fi in court_by_f:
            last_court = court_by_f[fi]
        cal.corners = last_court
        if ball_c is not None:
            trail.append((int(ball_c[0]), int(ball_c[1]), ball_pred))
        lab, conf = label_at(fi)
        conf = 1.0 if conf != conf else conf  # NaN -> 1.0 (unknown / gated)
        out = _annotate_frame(
            frame, xyxy, ids, ball_c, ball_pred, trail, lab, conf, fi, fps,
            calibrator=cal, hud_metrics=None, current_entropy=0.0,
            is_anomaly=(lab not in ("Unclassified", "No manual label") and conf < 0.5),
            pose_summary=None,
        )
        writer.write(out)
        fi += 1
        n += 1
        if n % 500 == 0:
            print(f"  {n} frames…")
    cap.release()
    writer.release()
    tag = args.model or "windows-csv"
    print(f"wrote {args.output}  ({n} frames, banner = {tag})")


if __name__ == "__main__":
    main()
