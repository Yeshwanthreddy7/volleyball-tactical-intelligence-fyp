"""Cut short, web-friendly demo clips from the project's existing annotated
videos into webapp/storage/demos/ and write demos.json for the Demos page.

Usage:  python webapp/scripts/make_demos.py [--seconds 45]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from app import config  # noqa: E402

R = config.REPO_ROOT
# (source, start seconds, output stem, title, description)
SOURCES = [
    (R / "DEMOO" / "videoplayback(1)_15min_bicausal_annotated.mp4", 95, "bicausal_vp1",
     "Bi-Causal view: relations and interactions",
     "Team relation graphs (R), the player-ball interaction (I), formations and timestamped actions, "
     "with the t1-t2-t3 causal side panel."),
    (R / "PAPER_OUTPUT" / "video" / "vp7_bicausal_paper.mp4", 40, "paper_vp7",
     "Paper feature read-out (videoplayback 7)",
     "The same view with the live Bi-Causal paper features (O, P, K, E, I, R, Z) computed per 10-frame window."),
    (R / "PAPER_OUTPUT" / "video" / "vp9_bicausal_paper.mp4", 120, "paper_vp9",
     "Another match and court (videoplayback 9)",
     "The same pipeline on a different match, court colour and camera position."),
    (R / "PAPER_OUTPUT" / "video" / "usa_bicausal_paper.mp4", 60, "paper_usa",
     "Paper feature read-out (USA match)",
     "A different broadcast, court colour and camera height, processed by the same pipeline."),
]


def run(args):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--extra", nargs=3, action="append", metavar=("MP4", "TITLE", "DESCRIPTION"),
                    help="add another finished video (e.g. a website analysis output)")
    a = ap.parse_args()
    out = config.DEMOS_DIR
    items = []
    sources = list(SOURCES)
    for i, (mp4, title, desc) in enumerate(a.extra or []):
        sources.insert(0, (Path(mp4), 0, f"extra{i}", title, desc))
    for src, start, stem, title, desc in sources:
        if not src.exists():
            print("skip (missing):", src)
            continue
        mp4, jpg = out / f"{stem}.mp4", out / f"{stem}.jpg"
        print("->", mp4.name)
        try:
            run(["-ss", str(start), "-i", str(src), "-t", str(a.seconds), "-vf", "scale=-2:720:flags=lanczos",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p",
                 "-an", "-movflags", "+faststart", str(mp4)])
            run(["-ss", "5", "-i", str(mp4), "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "4", str(jpg)])
        except subprocess.CalledProcessError:
            print("skip (unreadable / unfinished video):", src)
            continue
        items.append({"file": mp4.name, "poster": jpg.name, "title": title, "description": desc,
                      "source": str(src.relative_to(R)) if src.is_relative_to(R) else src.name})
    (out / "demos.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"{len(items)} demos ->", out)


if __name__ == "__main__":
    main()
