"""
Stage 1 - build the model-input cache for the binary Attack/Defense task.

Reads every Coordinated_Attack__* / Coordinated_Defense__* CSV in
training_csv_v4, runs the SAME feature pipeline the live server uses
(features.build_model_sequence -> 18 channels, 29 frames), and also stores a
left-right MIRRORED copy of every clip (court is symmetric across the sideline,
so a mirrored attack is still an attack).  Mirror is the single most valuable
augmentation and it is expensive to rebuild per epoch, so it is baked in here.

Out: v7_tactical/cache/binary_cache.npz
  seq       (N, 29, 18) float32   - canonical
  seq_mir   (N, 29, 18) float32   - horizontally mirrored
  y         (N,)        int64     - 0 = Coordinated Defense, 1 = Coordinated Attack
  groups    (N,)        <U12      - source-video key, e.g. '(3)', '(plain)', '?'
  files     (N,)        <U256
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "fyp"))

from features import build_model_sequence, COURT_W_CM          # noqa: E402
from train_utils import video_key                              # noqa: E402
RAW_COLS = ["ball_x", "ball_y"] + [f"p{i}_{a}" for i in range(1, 7) for a in "xy"]
LABELS = {"Coordinated Defense": 0, "Coordinated Attack": 1}


def _mirror_raw(raw: np.ndarray) -> np.ndarray:
    """Flip x about the court centre line; leave (0,0) sentinels untouched."""
    out = raw.copy().reshape(len(raw), 7, 2)
    valid = ~np.all(out == 0.0, axis=-1)
    out[..., 0][valid] = COURT_W_CM - out[..., 0][valid]
    return out.reshape(len(raw), 14)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=os.path.join(ROOT, "training_csv_v4"))
    ap.add_argument("--out", default=os.path.join(HERE, "cache", "binary_cache.npz"))
    a = ap.parse_args()
    CSV_DIR, OUT = a.csv_dir, a.out
    files = sorted(glob.glob(os.path.join(CSV_DIR, "Coordinated_Attack__*.csv"))
                   + glob.glob(os.path.join(CSV_DIR, "Coordinated_Defense__*.csv")))
    seq, seq_mir, y, groups, keep = [], [], [], [], []
    for f in files:
        df = pd.read_csv(f)
        lab = df["target_label"].iloc[0]
        if lab not in LABELS:
            continue
        raw = df[RAW_COLS].values.astype(np.float32)
        seq.append(build_model_sequence(raw, 29))
        seq_mir.append(build_model_sequence(_mirror_raw(raw), 29))
        y.append(LABELS[lab])
        groups.append(video_key(os.path.basename(f)))
        keep.append(os.path.basename(f))

    seq = np.asarray(seq, np.float32)
    seq_mir = np.asarray(seq_mir, np.float32)
    y = np.asarray(y, np.int64)
    groups = np.asarray(groups)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, seq=seq, seq_mir=seq_mir, y=y, groups=groups,
                        files=np.asarray(keep))
    print(f"wrote {OUT}")
    print(f"  N={len(y)}  attack={int(y.sum())}  defense={int((y == 0).sum())}")
    for g in sorted(set(groups)):
        m = groups == g
        print(f"  {g:10s} n={m.sum():4d}  attack={int(y[m].sum()):3d}  "
              f"defense={int((y[m] == 0).sum()):3d}")


if __name__ == "__main__":
    main()
