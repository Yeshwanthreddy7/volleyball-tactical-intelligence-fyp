"""Render one Bi-Causal annotated video per tactical model (9 ML + 4 DL).

The Bi-Causal overlay (relation graph with arrows, ball / interaction, action
names, t1-t3 R<->I side panel) is drawn once per frame by
bicausal_render.render_frames(); every model then gets its own copy with

  * its tactical call for the near team's current 1-second window
    (Coordinated Attack / Coordinated Defense / Delayed Support /
    Spacing Breakdown), drawn as a coloured hull around the near team and a
    banner with the model's confidence,
  * its held-out accuracy (leave-one-video-out, from
    model_comparison/trainvstest_metrics.csv),
  * a 5-minute timeline of all its calls with a playhead.

The per-window predictions are the saved outputs of the 13 trained models on
videoplayback (7) 03:00-08:00 (vp7_3to8_results/vp7_3to8_all_models.csv).

Usage:
  python render_13_models.py cache_vp7.pkl --out-dir ../FINAL_OUTPUT/videos
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import bicausal_render as br

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# csv column -> (display name, family, name in trainvstest_metrics.csv)
MODELS = [
    ("hist_gradient_boosting", "HistGradientBoosting", "ML", "HistGradientBoosting"),
    ("gradient_boosting", "GradientBoosting", "ML", "GradientBoosting"),
    ("random_forest", "RandomForest", "ML", "RandomForest"),
    ("extra_trees", "ExtraTrees", "ML", "ExtraTrees"),
    ("rbf_svm", "RBF-SVM", "ML", "RBF-SVM"),
    ("linear_svm", "Linear SVM", "ML", "Linear SVM"),
    ("logistic_regression", "LogisticRegression", "ML", "LogisticRegression"),
    ("knn", "k-NN", "ML", "k-NN"),
    ("mlp", "MLP", "ML", "MLP"),
    ("inceptiontime", "InceptionTime", "DL", "InceptionTime"),
    ("tcn", "TCN", "DL", "TCN"),
    ("bilstm", "BiLSTM+Attention", "DL", "BiLSTM+Attn"),
    ("mamba", "Mamba (SSM)", "DL", "Mamba (SSM)"),
]
CLASSES = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
CLASS_RGB = {
    "Coordinated Attack": (230, 70, 50),
    "Coordinated Defense": (40, 120, 230),
    "Delayed Support": (240, 170, 20),
    "Spacing Breakdown": (150, 70, 200),
    None: (90, 90, 90),
}
STRIP_H = 112


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--windows", default=str(ROOT / "vp7_3to8_results" / "vp7_3to8_all_models.csv"))
    ap.add_argument("--metrics", default=str(ROOT / "model_comparison" / "trainvstest_metrics.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "FINAL_OUTPUT" / "videos"))
    ap.add_argument("--labels", default=str(ROOT / "fyp" / "labels.csv"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="subset of model csv names (for tests)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    win = pd.read_csv(args.windows)
    met = pd.read_csv(args.metrics).set_index("model")
    models = [m for m in MODELS if not args.only or m[0] in args.only]
    rank = met["test_accuracy_LOVO"].rank(ascending=False, method="min").astype(int)

    fs = win["frame_start"].to_numpy()
    fe = win["frame_end"].to_numpy()
    maj = win["majority_label"].tolist()
    nmaj = win["n_agree_majority"].tolist()

    f_mid = br.load_font(22, True)
    f_big = br.load_font(26, True)
    f_sm = br.load_font(17, True)
    f_xs = br.load_font(15)

    info = {}
    gen = br.render_frames(args.cache, out_dir / "videoplayback(7)_bicausal", args.labels,
                           args.max_frames, info)
    writers, timelines = {}, {}
    tl_x0, tl_x1 = 16, 1330
    for fi, main_rgb, panel_rgb in gen:
        start, fps, n = info["start"], info["fps"], info["n"]
        absf = start + fi
        k = int(np.searchsorted(fs, absf, side="right")) - 1
        in_win = 0 <= k < len(fs) and absf <= fe[k] + 1
        if not writers:
            W = main_rgb.shape[1] + panel_rgb.shape[1]
            Hh = main_rgb.shape[0] + STRIP_H
            for col, name, fam, _ in models:
                path = out_dir / f"{col}__{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}.mp4"
                w = imageio_ffmpeg.write_frames(
                    str(path), (W, Hh), fps=fps, codec="libx264", quality=None, pix_fmt_out="yuv420p",
                    output_params=["-crf", "18", "-preset", "veryfast", "-threads", "2",
                                   "-movflags", "+faststart"],
                    macro_block_size=16)
                w.send(None)
                writers[col] = (w, path)
                timelines[col] = _timeline(win[col].tolist(), fs, fe, start, n, tl_x1 - tl_x0)

        near = [q for q in info["players"][fi] if q["team"] == br.NEAR]
        hull = None
        if len(near) >= 3:
            hull = cv2.convexHull(np.array([q["feet"] for q in near], np.float32)).astype(np.int32)

        for col, name, fam, mname in models:
            lab = win[col].iloc[k] if in_win else None
            conf = float(win[col + "__conf"].iloc[k]) if in_win else None
            rgb = CLASS_RGB.get(lab, CLASS_RGB[None])

            main = main_rgb.copy()
            if hull is not None and lab is not None:
                over = main.copy()
                cv2.fillPoly(over, [hull], rgb)
                cv2.addWeighted(over, 0.18, main, 0.82, 0, main)
                cv2.polylines(main, [hull], True, rgb, 3, cv2.LINE_AA)

            strip = Image.new("RGB", (main.shape[1] + panel_rgb.shape[1], STRIP_H), (14, 14, 18))
            d = ImageDraw.Draw(strip)
            m = met.loc[mname]
            d.text((16, 10), f"MODEL: {name}", font=f_mid, fill=(255, 255, 255))
            d.text((16 + d.textlength(f"MODEL: {name}", font=f_mid) + 12, 14),
                   f"{'Machine learning' if fam == 'ML' else 'Deep learning'}   |   rank {rank[mname]} of 13 by test accuracy",
                   font=f_xs, fill=(170, 170, 170))
            acc_txt = (f"Test accuracy {m.test_accuracy_LOVO * 100:.1f}%   macro-F1 {m.test_macro_f1_LOVO:.3f}"
                       f"   (train acc {m.train_accuracy * 100:.1f}%)")
            d.text((strip.width - 16, 10), acc_txt, font=f_sm, fill=(255, 215, 90), anchor="ra")
            d.text((16, 50), "Near team tactic:", font=f_sm, fill=(220, 220, 220))
            x = 16 + d.textlength("Near team tactic:", font=f_sm) + 14
            chip = (lab or "no window (between predictions)").upper()
            br.text_box(d, (x, 46), chip, f_big, (255, 255, 255), rgb, pad=6)
            x += d.textlength(chip, font=f_big) + 24
            if conf is not None:
                d.text((x, 50), f"confidence {conf:.2f}", font=f_sm, fill=(220, 220, 220))
            if in_win:
                d.text((strip.width - 16, 50), f"13-model majority: {maj[k]}  ({nmaj[k]}/13 agree)",
                       font=f_xs, fill=(170, 170, 170), anchor="ra")
            # timeline + playhead
            tl = timelines[col]
            strip_np = np.array(strip)
            strip_np[88:102, tl_x0:tl_x1] = tl
            px = tl_x0 + int((tl_x1 - tl_x0) * fi / max(n - 1, 1))
            cv2.line(strip_np, (px, 84), (px, 106), (255, 255, 255), 3)
            strip = Image.fromarray(strip_np)
            d = ImageDraw.Draw(strip)
            lx = tl_x1 + 14
            for c in CLASSES:
                d.rectangle([lx, 89, lx + 12, 101], fill=CLASS_RGB[c])
                short = {"Coordinated Attack": "Attack", "Coordinated Defense": "Defense",
                         "Delayed Support": "Delayed", "Spacing Breakdown": "Spacing"}[c]
                d.text((lx + 16, 87), short, font=f_xs, fill=(210, 210, 210))
                lx += 22 + d.textlength(short, font=f_xs) + 10

            frame = np.vstack([np.hstack([main, panel_rgb]), np.array(strip)])
            writers[col][0].send(np.ascontiguousarray(frame))

    for col, (w, path) in writers.items():
        w.close()
        print("video ->", path)


def _timeline(labels, fs, fe, start, n, width):
    bar = np.full((14, width, 3), 50, np.uint8)
    for lab, a, b in zip(labels, fs, fe):
        x0 = int((a - start) / n * width)
        x1 = int((b + 1 - start) / n * width)
        x0, x1 = max(0, x0), min(width, max(x1, x0 + 1))
        if x1 > x0:
            bar[:, x0:x1] = CLASS_RGB.get(lab, CLASS_RGB[None])
    return bar


if __name__ == "__main__":
    main()
