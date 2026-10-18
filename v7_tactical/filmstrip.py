"""Fast visual check - overlay the windows.csv label straight onto raw video
frames, no rebanner. For rapid smoothing-parameter iteration."""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(os.path.dirname(HERE), "fyp", "Datasets", "videoplayback (7).mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=os.path.join(HERE, "results", "vp7_windows.csv"))
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--span", type=int, default=4500)
    ap.add_argument("--every", type=int, default=90)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "strip.jpg"))
    a = ap.parse_args()

    w = pd.read_csv(a.windows)
    fs = w["frame_start"].values
    fe = w["frame_end"].values
    lab = w["label"].values
    praw = w["p_attack_raw"].values

    def at(f):
        m = np.where((fs <= f) & (fe >= f))[0]
        if len(m) == 0:
            return "?", 0.0
        i = m[-1]
        return lab[i], praw[i]

    cap = cv2.VideoCapture(VIDEO)
    N = int(cap.get(7))
    tiles = []
    f = a.start
    while f < min(a.start + a.span, N):
        cap.set(1, f)
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (420, 236))
        L, p = at(f)
        short = {"Coordinated Attack": "ATTACK", "Coordinated Defense": "DEFENSE",
                 "Unclassified": "break"}.get(L, L)
        col = (0, 165, 255) if short == "ATTACK" else (
            (255, 130, 0) if short == "DEFENSE" else (150, 150, 150))
        cv2.rectangle(im, (0, 214), (420, 236), (0, 0, 0), -1)
        cv2.putText(im, f"{f/30:6.1f}s {short} {p:.2f}", (4, 231),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        tiles.append(im)
        f += a.every
    cap.release()

    rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = np.zeros((rows * 236, a.cols * 420, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, a.cols)
        sheet[r * 236:(r + 1) * 236, c * 420:(c + 1) * 420] = t
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 78])
    print(f"wrote {a.out}  {len(tiles)} frames  {a.start/30:.0f}s..{f/30:.0f}s  every {a.every/30:.1f}s")


if __name__ == "__main__":
    main()
