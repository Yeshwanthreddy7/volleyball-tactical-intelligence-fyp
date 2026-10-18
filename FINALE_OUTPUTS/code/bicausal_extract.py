"""Pass 1 of the Bi-Causal style annotation: run the detectors once and cache.

For every frame of the requested segment this stores
  * tracked people from YOLO11m-pose + BoT-SORT (id, box, 17 keypoints),
  * every ball candidate from the project's volleyball fine-tune.

Pass 2 (bicausal_render.py) works only from this cache, so the rendering
rules can be tuned without re-running the GPU.

Usage:
  python bicausal_extract.py "../fyp/Datasets/videoplayback (1).mp4" --start 60 --seconds 300
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=60.0, help="start time in seconds")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--pose-model", default=str(HERE / "yolo11m-pose.pt"))
    ap.add_argument("--ball-model", default=str(HERE.parent / "fyp" / "volleyball_best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default=str(HERE / "cache_vp1.pkl"))
    args = ap.parse_args()

    LOGGER.setLevel(logging.ERROR)
    pose = YOLO(args.pose_model)
    ball = YOLO(args.ball_model)
    ball_id = next(i for i, n in ball.names.items() if "ball" in n.lower())

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(round(args.start * fps))
    n = int(round(args.seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    t0 = time.time()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        r = pose.track(frame, persist=True, tracker="botsort.yaml", imgsz=args.imgsz,
                       conf=0.25, classes=[0], verbose=False)[0]
        people = []
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else -np.ones(len(r.boxes), int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for j in range(len(xyxy)):
                people.append({
                    "id": int(ids[j]),
                    "box": xyxy[j].astype(np.float32),
                    "conf": float(conf[j]),
                    "kp": kp[j].astype(np.float32) if kp is not None else None,
                })
        b = ball.predict(frame, imgsz=args.imgsz, conf=0.10, classes=[ball_id], verbose=False)[0]
        balls = []
        if len(b.boxes):
            for bx, c in zip(b.boxes.xyxy.cpu().numpy(), b.boxes.conf.cpu().numpy()):
                balls.append((float((bx[0] + bx[2]) / 2), float((bx[1] + bx[3]) / 2),
                              float(bx[2] - bx[0]), float(c)))
        frames.append({"people": people, "balls": balls})
        if i % 300 == 0:
            el = time.time() - t0
            print(f"frame {i}/{n}  {el:.0f}s elapsed  ~{el / max(i, 1) * (n - i) / 60:.1f} min left", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump({"video": args.video, "fps": fps, "start": start, "frames": frames}, f)
    print(f"saved {len(frames)} frames -> {args.out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
