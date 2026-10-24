"""Render the NEW_OUTPUTS video for all 11 source videos, a few in parallel.

Uses the same detection caches as PAPER_OUTPUT (PAPER_OUTPUT/cache/cache_<slug>.pkl,
the 5-minute segment from 00:00) so the two output sets cover the same frames.
A video is skipped when its .mp4 already exists.

Usage: python run_all.py [--jobs 4] [--only vp1,vp3]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_OUT = HERE.parent
ROOT = NEW_OUT.parent
CACHE_DIR = ROOT / "PAPER_OUTPUT" / "cache"
VIDEO_DIR = NEW_OUT / "videos"
LOG_DIR = NEW_OUT / "logs"

SLUGS = ["usa", "plain", "vp1", "vp2", "vp3", "vp4", "vp5", "vp6", "vp7", "vp8", "vp9"]


def one(slug):
    out = VIDEO_DIR / f"{slug}_team_tactics.mp4"
    if out.exists():
        return slug, "skipped (exists)"
    t = time.time()
    with open(LOG_DIR / f"{slug}.log", "w") as f:
        rc = subprocess.run([sys.executable, "-u", str(HERE / "new_render.py"),
                             "--cache", str(CACHE_DIR / f"cache_{slug}.pkl"), "--out", str(out)],
                            stdout=f, stderr=subprocess.STDOUT).returncode
    return slug, f"rc={rc} in {(time.time() - t) / 60:.1f} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only")
    args = ap.parse_args()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    slugs = args.only.split(",") if args.only else SLUGS
    # 1080p / 50 fps videos (plain, vp3, vp5, vp6) are the heavy ones: interleave
    # them with 720p ones so two heavy renders do not share the 8 GB of RAM
    heavy = [s for s in slugs if s in ("plain", "vp3", "vp5", "vp6")]
    light = [s for s in slugs if s not in heavy]
    slugs = [x for pair in zip(heavy, light) for x in pair] + light[len(heavy):] + heavy[len(light):]
    with ThreadPoolExecutor(args.jobs) as ex:
        for slug, msg in ex.map(one, slugs):
            print(slug, msg, flush=True)


if __name__ == "__main__":
    main()
