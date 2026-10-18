"""
Detect point-scored events on vp(7) by watching the on-screen score digits.

The broadcast score box reads e.g. "Z5 (1) 06-11 (0) Z3". When the "06-11"
digits change, a rally just ended -> a real point boundary. We don't OCR;
we detect the pixel change in the digit ROI (robust to font/anti-aliasing).

Out: results/vp7_score_events.json  - list of frame indices where the score
     changed (debounced), plus the ROI used.
"""
import argparse
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(os.path.dirname(HERE), "fyp", "Datasets", "videoplayback (7).mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", nargs=4, type=int, default=[585, 5, 720, 48],
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--thresh", type=float, default=0.055,
                    help="fraction of ROI pixels that must change to call an event")
    ap.add_argument("--debounce", type=int, default=90,
                    help="min frames between two counted events")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vp7_score_events.json"))
    a = ap.parse_args()
    x0, y0, x1, y1 = a.roi

    cap = cv2.VideoCapture(VIDEO)
    N = int(cap.get(7))
    prev = None
    prev_f = -10 ** 9
    events = []
    diffs = []
    f = -1
    while True:
        ok, im = cap.read()          # sequential read is ~10x faster than seek
        if not ok:
            break
        f += 1
        if f >= N:                   # cv2 5.0 keeps returning frames past EOF
            break
        if f % a.step:
            continue
        if f % 6000 == 0:
            print(f"  {f}/{N}", flush=True)
        g = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if prev is not None:
            d = np.mean(np.abs(b.astype(int) - prev.astype(int)) > 0)
            diffs.append((f, round(float(d), 4)))
            if d > a.thresh and (f - prev_f) > a.debounce:
                events.append(int(f))
                prev_f = f
        prev = b
        f += a.step
    cap.release()

    json.dump({"roi": a.roi, "step": a.step, "thresh": a.thresh,
               "n_events": len(events), "events": events},
              open(a.out, "w"), indent=1)
    # also dump the diff trace for threshold tuning
    np.save(os.path.join(HERE, "cache", "vp7_score_diffs.npy"), np.array(diffs))
    print(f"{len(events)} score-change events over {N} frames "
          f"({N/30/60:.0f} min) -> {a.out}")
    if events:
        gaps = np.diff(events) / 30
        print(f"  inter-event gap: median {np.median(gaps):.0f}s  "
              f"p10 {np.percentile(gaps,10):.0f}s  p90 {np.percentile(gaps,90):.0f}s")
        print(f"  first few (s): {[round(e/30) for e in events[:12]]}")


if __name__ == "__main__":
    main()
