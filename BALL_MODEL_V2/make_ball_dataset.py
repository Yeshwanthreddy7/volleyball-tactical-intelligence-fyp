"""Build a ball-detection training set from every video in fyp/Datasets by
pseudo-labelling (no hand labels exist for these videos).

Two teachers label each sampled frame:
  A  fyp/volleyball_best.pt   (Roboflow fine-tune, class 'ball')
  B  yolo11n.pt               (stock COCO, class 32 'sports ball')
A candidate ball is accepted when it is confident enough AND temporally
consistent: a ball detection must also exist within MAX_JUMP of it in the
frame STEP frames before or after (a real ball moves continuously). A box that
sits on exactly the same pixels in all three frames is rejected as a static
false positive (logo, light, net antenna). One ball per frame at most.

Frames whose teachers see nothing at all are kept as negatives (capped);
frames with only weak, unconfirmed candidates are dropped, because labelling
them "no ball" could teach the model to miss real balls.

Split: per video, the last 15 % of sampled frames (by time) go to validation,
with a gap, so near-identical neighbouring frames never straddle the split.

Out:  %USERPROFILE%/vta_build/ball_v2/data/{images,labels}/{train,val}/ + data.yaml + stats.json
Run:  python BALL_MODEL_V2/make_ball_dataset.py [--per-video 450]
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "fyp" / "Datasets"
OUT = Path(os.environ["USERPROFILE"]) / "vta_build" / "ball_v2" / "data"
STEP = 3                 # neighbour frames used for the continuity check
CONF_STRONG = 0.45       # one teacher alone
CONF_AGREE = 0.20        # both teachers agree (each at least this)
CONF_ANY = 0.10          # minimum for a neighbour-frame confirmation
NEG_MAX_CONF = 0.15      # a negative frame: no candidate above this anywhere
NEG_SHARE = 0.25         # negatives at most this share of the kept frames
VAL_SHARE = 0.15


def dets(model, frames, cls, imgsz):
    out = []
    for r in model.predict(frames, imgsz=imgsz, conf=CONF_ANY, classes=[cls], verbose=False):
        b = r.boxes
        out.append([(*map(float, xyxy), float(c)) for xyxy, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())])
    return out


def centre(d):
    return np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])


def pick(a_mid, b_mid, a_nb, b_nb, max_jump):
    """Choose the accepted ball in the middle frame, or None."""
    cands = []
    for d in a_mid:
        agree = max((e[4] for e in b_mid if np.linalg.norm(centre(d) - centre(e)) < 20), default=0)
        strong = d[4] >= CONF_STRONG or (d[4] >= CONF_AGREE and agree >= CONF_AGREE)
        cands.append((d, strong, d[4] + agree))
    for e in b_mid:                       # stock-only candidates (COCO sports ball)
        if not any(np.linalg.norm(centre(e) - centre(d)) < 20 for d in a_mid) and e[4] >= CONF_STRONG:
            cands.append((e, True, e[4]))
    best = None
    for d, strong, score in cands:
        if not strong:
            continue
        c = centre(d)
        nb = [centre(x) for x in a_nb + b_nb]
        moved = [np.linalg.norm(c - n) for n in nb]
        if not any(m <= max_jump for m in moved):
            continue                      # no continuity: likely a one-frame false positive
        if len(moved) >= 2 and all(m < 2 for m in moved):
            continue                      # pixel-static in every frame: logo / light, not a ball
        if best is None or score > best[1]:
            best = (d, score)
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=450)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    teacher_a = YOLO(str(ROOT / "fyp" / "volleyball_best.pt"))
    a_cls = next(i for i, n in teacher_a.names.items() if "ball" in n.lower())
    teacher_b = YOLO(str(ROOT / "yolo11n.pt"))
    b_cls = 32
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {}
    t0 = time.time()
    for vid in sorted(VIDEOS.glob("*.mp4")):
        cap = cv2.VideoCapture(str(vid))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        max_jump = 0.06 * w
        targets = np.linspace(n * 0.02, n * 0.98, args.per_video).astype(int)
        n_val_from = int(len(targets) * (1 - VAL_SHARE))
        pos, neg, dropped, unread = [], [], 0, 0
        for k, t in enumerate(targets):
            if k in (n_val_from - 1, n_val_from):      # gap between train and val
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, t - STEP))
            frames = []
            for _ in range(2 * STEP + 1):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            if len(frames) < 2 * STEP + 1:
                unread += 1
                continue
            trio = [frames[0], frames[STEP], frames[2 * STEP]]
            da, db = dets(teacher_a, trio, a_cls, args.imgsz), dets(teacher_b, trio, b_cls, args.imgsz)
            ball = pick(da[1], db[1], da[0] + da[2], db[0] + db[2], max_jump)
            split = "val" if k >= n_val_from else "train"
            if ball is not None:
                pos.append((t, split, trio[1], ball))
            elif max((d[4] for d in da[1] + db[1]), default=0) < NEG_MAX_CONF:
                neg.append((t, split, trio[1], None))
            else:
                dropped += 1
        cap.release()
        rng = np.random.default_rng(0)
        keep_neg = int(NEG_SHARE / (1 - NEG_SHARE) * len(pos))
        if len(neg) > keep_neg:
            neg = [neg[i] for i in sorted(rng.choice(len(neg), keep_neg, replace=False))]
        for t, split, img, ball in pos + neg:
            stem = f"{vid.stem.replace(' ', '_').replace('(', '').replace(')', '')}_{t:07d}"
            h, wi = img.shape[:2]
            cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(OUT / "labels" / split / f"{stem}.txt", "w") as f:
                if ball is not None:
                    x1, y1, x2, y2 = ball[:4]
                    f.write(f"0 {(x1 + x2) / 2 / wi:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / wi:.6f} {(y2 - y1) / h:.6f}\n")
        stats[vid.name] = {"sampled": len(targets), "ball_frames": len(pos), "negative_frames": len(neg),
                           "dropped_uncertain": dropped, "unreadable": unread}
        print(vid.name, stats[vid.name], f"{time.time() - t0:.0f}s", flush=True)

    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print("done", OUT)


if __name__ == "__main__":
    main()
