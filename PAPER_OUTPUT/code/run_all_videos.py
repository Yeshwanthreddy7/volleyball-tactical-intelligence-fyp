"""Orchestrator: runs the existing 3-stage Bi-Causal paper-output pipeline
(bicausal_extract.py -> paper_features.py -> paper_output_render.py) over a
fixed 15-minute segment of every video in fyp/Datasets, writing everything
into PAPER_OUTPUT/{cache,features,video,logs}/.

Same recipe already used for videoplayback (2) (see PAPER_OUTPUT/video/
videoplayback(2)_bicausal_paper.mp4) -- random-init paper machinery (see
paper_features.py docstring), not a trained classifier. Scaled to all 11
source videos here.

Each step is skipped if its output already exists, and bicausal_extract.py
has its own internal checkpoint/resume, so re-running this script after an
interruption (or across sessions) just continues where it left off.

Usage: python run_all_videos.py [--start 60] [--seconds 900]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent            # finalyearproject/
DATASETS = ROOT / "fyp" / "Datasets"
PAPER_OUT = HERE.parent              # PAPER_OUTPUT/
CACHE_DIR = PAPER_OUT / "cache"
FEATURES_DIR = PAPER_OUT / "features"
VIDEO_DIR = PAPER_OUT / "video"
LOG_DIR = PAPER_OUT / "logs"

for d in (CACHE_DIR, FEATURES_DIR, VIDEO_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# (source filename, slug used for every output path)
VIDEOS = [
    ("usa.mp4", "usa"),
    ("videoplayback.mp4", "plain"),
    ("videoplayback (1).mp4", "vp1"),
    ("videoplayback (2).mp4", "vp2"),
    ("videoplayback (3).mp4", "vp3"),
    ("videoplayback (4).mp4", "vp4"),
    ("videoplayback (5).mp4", "vp5"),
    ("videoplayback (6).mp4", "vp6"),
    ("videoplayback (7).mp4", "vp7"),
    ("videoplayback (8).mp4", "vp8"),
    ("videoplayback (9).mp4", "vp9"),
]


def run(cmd, log_path):
    print(f"  $ {' '.join(cmd)}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n=== {time.ctime()} ===\n$ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=60.0, help="segment start, seconds")
    ap.add_argument("--seconds", type=float, default=900.0, help="segment length, seconds (15 min)")
    ap.add_argument("--only", help="comma-separated slugs to restrict to, e.g. vp3,vp5")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    py = sys.executable

    for fname, slug in VIDEOS:
        if only and slug not in only:
            continue
        video_path = DATASETS / fname
        cache_path = CACHE_DIR / f"cache_{slug}.pkl"
        feat_dir = FEATURES_DIR / slug
        out_video = VIDEO_DIR / f"{slug}_bicausal_paper.mp4"
        log_path = LOG_DIR / f"{slug}_pipeline.log"
        print(f"\n### {slug} ({fname}) ###", flush=True)

        if not video_path.exists():
            print(f"  MISSING source video, skipping: {video_path}", flush=True)
            continue

        # Stage 1: detection cache (resumable internally)
        rc = run([py, str(HERE.parent.parent / "OUTPUTS" / "bicausal_extract.py"),
                  str(video_path), "--start", str(args.start), "--seconds", str(args.seconds),
                  "--out", str(cache_path)], log_path)
        if rc != 0:
            print(f"  extract FAILED (rc={rc}), see {log_path}", flush=True)
            continue

        # Stage 2: paper-equation feature tensors (random-init weights, see docstring)
        manifest = feat_dir / "manifest.json"
        if manifest.exists():
            print("  features already done, skipping", flush=True)
        else:
            feat_dir.mkdir(parents=True, exist_ok=True)
            rc = run([py, str(HERE / "paper_features.py"),
                      "--cache", str(cache_path), "--video-name", fname,
                      "--out", str(feat_dir)], log_path)
            if rc != 0:
                print(f"  features FAILED (rc={rc}), see {log_path}", flush=True)
                continue

        # Stage 3: annotated video with the live paper-feature HUD
        if out_video.exists():
            print("  video already done, skipping", flush=True)
            continue
        rc = run([py, str(HERE / "paper_output_render.py"),
                  "--cache", str(cache_path), "--features", str(feat_dir),
                  "--out", str(out_video)], log_path)
        if rc != 0:
            print(f"  render FAILED (rc={rc}), see {log_path}", flush=True)
            continue
        print(f"  done -> {out_video}", flush=True)

    print("\nall videos processed.", flush=True)


if __name__ == "__main__":
    main()
