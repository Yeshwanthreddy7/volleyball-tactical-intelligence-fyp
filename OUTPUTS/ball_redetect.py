"""Re-run only the ball detector on a cached segment, at a low confidence.

Fast, motion-blurred balls score 0.05-0.10 with fyp/volleyball_best.pt, so the
0.10 cut in bicausal_extract.py dropped them. The offline tracker
(ball_track.py) keeps only motion-consistent chains, so a low cut is safe.

Usage:  python ball_redetect.py cache_vp7.pkl --conf 0.02
(rewrites the cache in place, keeping the player detections)
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--ball-model", default=str(HERE.parent / "fyp" / "volleyball_best.pt"))
    args = ap.parse_args()
    LOGGER.setLevel(logging.ERROR)

    with open(args.cache, "rb") as f:
        data = pickle.load(f)
    ball = YOLO(args.ball_model)
    bid = next(i for i, n in ball.names.items() if "ball" in n.lower())
    cap = cv2.VideoCapture(data["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, data["start"])
    t0 = time.time()
    for i, fr in enumerate(data["frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        r = ball.predict(frame, imgsz=1280, conf=args.conf, classes=[bid], verbose=False)[0]
        fr["balls"] = [(float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(b[2] - b[0]), float(c))
                       for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        if i % 1000 == 0:
            print(f"{i}/{len(data['frames'])}  {time.time() - t0:.0f}s", flush=True)
    data["ball_conf"] = args.conf
    with open(args.cache, "wb") as f:
        pickle.dump(data, f)
    print("done", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
