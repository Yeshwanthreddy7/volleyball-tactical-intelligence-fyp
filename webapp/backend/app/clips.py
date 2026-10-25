"""Short per-tactic clips cut from an analysis' annotated video.

For each of the four tactics, the best segments (longest x most confident) of
the user's own video are cut into small 480p MP4s under
<job>/output/clips/<tactic-slug>/, so the Demos page can show one folder of
examples per tactic.

Backfill existing analyses:  python -m app.clips all     (from webapp/backend)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

from . import config

TACTICS = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
MAX_PER_TACTIC = 8
MIN_SECONDS = 1.0
PAD = 0.5            # seconds of context before / after each segment
MAX_SECONDS = 12.0


def slug(tactic: str) -> str:
    return tactic.lower().replace(" ", "_")


def make_clips(job_dir: Path, result: dict) -> dict:
    out = job_dir / "output"
    src = out / "annotated_high.mp4" if (out / "annotated_high.mp4").exists() else out / "annotated_ultra.mp4"
    if not src.exists():
        return {}
    offset = result["video"]["analysed_start"]
    length = result["video"]["analysed_end"] - offset
    segs = result.get("tactics", {}).get("segments", [])
    clips: dict[str, list] = {}
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    for tactic in TACTICS:
        cand = [s for s in segs if s["tactic"] == tactic and s["end"] - s["start"] >= MIN_SECONDS]
        cand.sort(key=lambda s: -(min(s["end"] - s["start"], MAX_SECONDS) * s["confidence"]))
        chosen = sorted(cand[:MAX_PER_TACTIC], key=lambda s: s["start"])
        d = out / "clips" / slug(tactic)
        d.mkdir(parents=True, exist_ok=True)
        items = []
        for i, s in enumerate(chosen, 1):
            a = max(0.0, s["start"] - offset - PAD)
            b = min(length, min(s["end"], s["start"] + MAX_SECONDS) - offset + PAD)
            name = f"clip_{i:02d}.mp4"
            rc = subprocess.run(
                [exe, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{a:.2f}", "-i", str(src),
                 "-t", f"{b - a:.2f}", "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "26", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(d / name)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=config.NO_WINDOW).returncode
            if rc == 0 and (d / name).exists():
                items.append({"file": name, "start": s["start"], "end": s["end"],
                              "confidence": s["confidence"]})
        clips[slug(tactic)] = items
    return clips


def backfill(job_id: str) -> None:
    """Add clips to a finished analysis that was made before this feature existed."""
    from .db import store

    job_dir = config.JOBS_DIR / job_id
    rp = job_dir / "result.json"
    if not rp.exists():
        return
    result = json.loads(rp.read_text(encoding="utf-8"))
    if result.get("clips"):
        return
    result["clips"] = make_clips(job_dir, result)
    rp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    store.update(job_id, {"result": result})
    n = sum(len(v) for v in result["clips"].values())
    print(f"{job_id}: {n} clips")


if __name__ == "__main__":
    ids = sys.argv[1:] or ["all"]
    if ids == ["all"]:
        ids = [d.name for d in config.JOBS_DIR.iterdir() if d.is_dir()]
    for jid in ids:
        backfill(jid)
