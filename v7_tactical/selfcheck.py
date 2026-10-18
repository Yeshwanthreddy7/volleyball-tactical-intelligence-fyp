"""
Stage 6 - self-check the vp(7) Attack/Defense annotation without ground truth.

(a) SANITY STATS vs volleyball priors:
      - live time split Attack/Defense should be roughly balanced (35-65%)
      - a rally phase lasts ~1.5-6 s -> median run length in that band
      - ~30-55% of a broadcast is dead time (BREAK)
      - ensemble confidence on live windows should be decisive (>0.55 mean)
(b) CONTACT SHEET: pulls N frames from the annotated video, stratified across
    time and predicted label, into one JPG grid for eyeball review.

Usage:
  python v7_tactical/selfcheck.py --video results/vp7_annotated.mp4 \
      --windows results/vp7_windows.csv --n 24
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def sanity(win: pd.DataFrame, fps: float) -> dict:
    lab = win["label"].values
    live = lab != "Unclassified"
    n = len(lab)
    # run lengths (in windows) per state
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        runs.append((lab[i], j - i))
        i = j
    stride_s = (win["frame_start"].iloc[1] - win["frame_start"].iloc[0]) / fps if n > 1 else 1
    live_runs = [r * stride_s for s, r in runs if s != "Unclassified"]
    att = (lab == "Coordinated Attack").sum()
    dfn = (lab == "Coordinated Defense").sum()
    rep = {
        "windows": int(n),
        "dead_frac": round(float((~live).mean()), 3),
        "attack_frac_live": round(float(att / max(att + dfn, 1)), 3),
        "defense_frac_live": round(float(dfn / max(att + dfn, 1)), 3),
        "n_state_runs": len(runs),
        "median_live_run_s": round(float(np.median(live_runs)) if live_runs else 0, 2),
        "p90_live_run_s": round(float(np.percentile(live_runs, 90)) if live_runs else 0, 2),
        "switches_per_min": round(len([1 for a, b in zip(lab, lab[1:]) if a != b])
                                  / max(n * stride_s / 60, 1e-6), 2),
        "mean_conf_live": round(float(win.loc[live, "confidence"].mean()), 3),
        "mean_p_attack_raw": round(float(win["p_attack_raw"].mean()), 3),
    }
    flags = []
    if not 0.30 <= rep["attack_frac_live"] <= 0.70:
        flags.append(f"class imbalance in predictions: attack {rep['attack_frac_live']}")
    if rep["median_live_run_s"] < 1.0:
        flags.append(f"phases too choppy: median run {rep['median_live_run_s']}s")
    if rep["mean_conf_live"] < 0.5:
        flags.append(f"low ensemble confidence: {rep['mean_conf_live']}")
    if not 0.15 <= rep["dead_frac"] <= 0.70:
        flags.append(f"implausible dead-time fraction: {rep['dead_frac']}")
    rep["flags"] = flags or ["none"]
    return rep


def contact_sheet(video: str, win: pd.DataFrame, n: int, out: str) -> None:
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = []
    for lab in ("Coordinated Attack", "Coordinated Defense", "Unclassified"):
        sub = win[win["label"] == lab]
        if sub.empty:
            continue
        k = max(1, int(round(n * len(sub) / len(win))))
        idx = np.linspace(0, len(sub) - 1, k).astype(int)
        for _, r in sub.iloc[idx].iterrows():
            mid = int((r["frame_start"] + r["frame_end"]) // 2)
            picks.append((mid, lab, r.get("p_attack_ema", np.nan)))
    picks.sort()
    tiles = []
    for fr, lab, p in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(fr, total - 1))
        ok, im = cap.read()
        if not ok:
            continue
        im = cv2.resize(im, (480, 270))
        cv2.putText(im, f"f{fr} t{fr/30:.0f}s {lab.split()[-1]} p={p:.2f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    cap.release()
    if not tiles:
        print("no tiles")
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 270, cols * 480, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 270:(r + 1) * 270, c * 480:(c + 1) * 480] = t
    cv2.imwrite(out, sheet)
    print(f"wrote {out}  ({len(tiles)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE, "results", "vp7_annotated.mp4"))
    ap.add_argument("--windows", default=os.path.join(HERE, "results", "vp7_windows.csv"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--sheet", default=os.path.join(HERE, "results", "vp7_contact.jpg"))
    args = ap.parse_args()

    win = pd.read_csv(args.windows)
    rep = sanity(win, args.fps)
    print("=== SANITY ===")
    for k, v in rep.items():
        print(f"  {k:22s} {v}")
    if os.path.exists(args.video):
        contact_sheet(args.video, win, args.n, args.sheet)
    else:
        print(f"(no annotated video at {args.video} yet - stats only)")


if __name__ == "__main__":
    main()
