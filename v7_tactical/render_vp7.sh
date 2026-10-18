#!/bin/bash
# Fires when vp7 detection is done: classify windows with the geom ensemble,
# smooth, render the Attack/Defense banner video, self-check.
set -e
cd "$(dirname "$0")/.."
R=v7_tactical/results

echo "=== [1/3] classify + smooth windows ==="
python -u v7_tactical/classify_windows.py \
  --frames $R/vp7_frames.csv --out $R/vp7_windows.csv \
  --ckpt-glob "v7_tactical/models/binary_geom_seed*.pt" \
  --stride 10 --ema 0.30 --hi 0.58 --lo 0.42 --min-dwell-s 1.2

echo "=== [2/3] render annotated video ==="
python -u fyp/rebanner.py "fyp/Datasets/videoplayback (7).mp4" \
  $R/vp7_tracks.npz --windows-csv $R/vp7_windows.csv \
  --output $R/vp7_annotated.mp4

echo "=== [3/3] self-check ==="
python -u v7_tactical/selfcheck.py \
  --video $R/vp7_annotated.mp4 --windows $R/vp7_windows.csv --n 28 \
  --sheet $R/vp7_contact_v1.jpg

echo "=== DONE: $R/vp7_annotated.mp4 ==="
ls -la $R/vp7_annotated.mp4
