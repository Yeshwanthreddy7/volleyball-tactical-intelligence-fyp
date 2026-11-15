"""Train the v2 ball detector on the pseudo-labelled set made by make_ball_dataset.py,
then compare it with the old fine-tune (fyp/volleyball_best.pt) on the validation frames.

Base: YOLO11s (COCO). One class: ball. imgsz 1280 because the ball is only
15-20 px wide on a 720p frame.

Out: fyp/volleyball_ball_v2.pt  and  BALL_MODEL_V2/results.json
Run: python BALL_MODEL_V2/train_ball_v2.py [--epochs 40]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ["USERPROFILE"]) / "vta_build" / "ball_v2"
DATA = WORK / "data" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue from runs/ball_v2/weights/last.pt")
    ap.add_argument("--eval-only", action="store_true", help="skip training; copy best.pt and compare")
    args = ap.parse_args()

    last = WORK / "runs" / "ball_v2" / "weights" / "last.pt"
    if args.eval_only:
        pass  # use the best.pt already saved by an earlier (stopped) run
    elif args.resume and last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(str(WORK / "yolo11s.pt"))
        model.train(data=str(DATA), imgsz=1280, epochs=args.epochs, batch=args.batch, patience=10,
                    workers=args.workers, cache=False, device=0, project=str(WORK / "runs"), name="ball_v2",
                    exist_ok=True, close_mosaic=5, degrees=0, mosaic=1.0, fliplr=0.5, hsv_h=0.02, plots=True,
                    verbose=False)
    best = WORK / "runs" / "ball_v2" / "weights" / "best.pt"
    dst = ROOT / "fyp" / "volleyball_ball_v2.pt"
    shutil.copy2(best, dst)

    # same validation frames, same pseudo-labels, both models
    res = {}
    for name, w in [("old_volleyball_best", ROOT / "fyp" / "volleyball_best.pt"), ("new_ball_v2", dst)]:
        m = YOLO(str(w))
        ball = next(i for i, n in m.names.items() if "ball" in n.lower())
        r = m.val(data=str(DATA), imgsz=1280, batch=4, classes=[ball], device=0, plots=False, verbose=False,
                  project=str(WORK / "runs"), name=f"val_{name}", exist_ok=True)
        # the old model's 'ball' id may differ from the label id 0: map through classes filter
        res[name] = {"precision": float(r.box.mp), "recall": float(r.box.mr),
                     "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map)}
        print(name, res[name])
    res["note"] = ("Validation labels are pseudo-labels from the old model + stock COCO with a temporal "
                   "check, not hand labels. Treat these as agreement scores, not true precision/recall.")
    (Path(__file__).parent / "results.json").write_text(json.dumps(res, indent=1))
    print("saved", dst)


if __name__ == "__main__":
    main()
