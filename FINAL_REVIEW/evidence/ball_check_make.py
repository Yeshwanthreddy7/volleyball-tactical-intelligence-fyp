"""Ball-detector check, step 1 (added 2026-11-15).

Takes 96 moments of live play (12 random labelled moments from each of the 8
labelled videos, 0.5 s after the labelled anchor time, so the rally is on),
runs the project's ball detector (fyp/volleyball_best.pt, imgsz 1280, conf 0.25)
and saves:
  * ball_check/sheet_*.jpg - the best detection of each frame as a zoomed crop
    (number in the corner) so a person can mark each one "ball" / "not ball"
  * ball_check/detections.json - frame, box, confidence per sample
Step 2 (ball_check_score.py) turns the human marks into precision / detection rate.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject")
OUT = Path(__file__).resolve().parent / "ball_check"
OUT.mkdir(exist_ok=True)
random.seed(7)

lab = pd.read_csv(ROOT / "fyp" / "labels.csv")


def secs(t):
    p = [float(x) for x in str(t).split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


model = YOLO(str(ROOT / "fyp" / "volleyball_best.pt"))
samples = []
for vf, g in lab.groupby("video_file"):
    path = ROOT / "fyp" / "Datasets" / vf
    if not path.exists() or "(2)" in vf:          # (2) is partly corrupt
        continue
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    for t in random.sample(list(g["anchor_time"]), 12):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((secs(t) + 0.5) * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        r = model.predict(fr, imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]   # 0 = 'ball'
        s = {"video": vf, "time": str(t), "n_det": int(len(r.boxes))}
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            b = r.boxes.xyxy[k].cpu().numpy()
            s.update(box=[float(x) for x in b], conf=float(r.boxes.conf[k]))
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            h, w = fr.shape[:2]
            x0, y0 = max(cx - 60, 0), max(cy - 60, 0)
            crop = fr[y0:min(cy + 60, h), x0:min(cx + 60, w)]
            crop = cv2.resize(cv2.copyMakeBorder(crop, 0, 120 - crop.shape[0], 0, 120 - crop.shape[1],
                                                 cv2.BORDER_CONSTANT), (240, 240), interpolation=cv2.INTER_CUBIC)
            s["crop"] = crop
        samples.append(s)
    cap.release()

det = [s for s in samples if "crop" in s]
for i, s in enumerate(det):
    s["id"] = i
for sh in range(0, len(det), 30):
    tiles = []
    for s in det[sh:sh + 30]:
        c = s["crop"].copy()
        cv2.rectangle(c, (0, 0), (52, 30), (0, 0, 0), -1)
        cv2.putText(c, str(s["id"]), (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(c)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(OUT / f"sheet_{sh // 30}.jpg"), np.vstack(rows))
for s in samples:
    s.pop("crop", None)
(OUT / "detections.json").write_text(json.dumps(samples, indent=1))
print(len(samples), "frames;", len(det), "with a detection")
