#!/bin/bash
set -e
cd "/c/Users/yashwanth/OneDrive/Desktop/GROUP_56/finalyearproject"
echo "=== Pass 1: pipeline.py detection + tracking + team filter (GPU) ==="
python fyp/pipeline.py "fyp/Datasets/videoplayback (1).mp4" tactical_ml_v4.joblib \
  --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 \
  --team-split colour --auto-court --court-coords linear --tracker botsort \
  --team-side bottom \
  --strict-render --ball-max-coast 8 \
  --start-frame 0 --max-frames 27000 \
  --dump-tracks DEMOO/render_work/vp1_15min_tracks.npz \
  --output-csv DEMOO/render_work/vp1_15min_preds.csv
echo "=== Pass 1 done ==="
echo "=== Pass 2: Bi-Causal-style R/I overlay (CPU, cheap) ==="
python OUTPUTS/render_team_relation.py "fyp/Datasets/videoplayback (1).mp4" DEMOO/render_work/vp1_15min_tracks.npz \
  --windows-csv DEMOO/render_work/vp1_15min_preds.csv \
  --output "DEMOO/videoplayback(1)_15min_bicausal_annotated.mp4" \
  --start-frame 0 --max-frames 27000
echo "=== ALL DONE ==="
