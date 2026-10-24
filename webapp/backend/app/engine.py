"""Analysis engine: one uploaded video -> annotated MP4 renditions + reports.

Runs as its own process (`python -m app.engine <job_dir>`) so a crash, the GPU
memory and the module-level state of the research renderer stay isolated from
the web server, and a cancel is a plain process kill.

It drives the project's own research code, unmodified:

  1. court    BASEFRAMES/extract_base_frames.py  empty-court base frame
              BASEFRAMES/detect_court_corners.py court lines -> homography
  2. detect   OUTPUTS/bicausal_extract.py        YOLO11m-pose + BoT-SORT and the
                                                 volleyball ball model
  3. render   NEW_OUTPUTS/code/new_render.py     near (focus) team: the four
              NEW_OUTPUTS/code/team_tactics.py   tactics from tactical_ml_v6.joblib;
                                                 far (opponent) team: SERVE / SPIKE /
                                                 BLOCK only (bicausal_render rules)
  4. encode   ffmpeg: master MP4 (H.264 + original AAC audio, faststart) and
              720p / 480p / 360p renditions
  5. reports  app/reports.py: team tactical report + opponent action report

Progress goes to <job_dir>/progress.json, the final summary to result.json.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import imageio_ffmpeg
from PIL import ImageFont

from . import config, reports

for _p in (config.RENDER_DIR, config.REPO_ROOT / "OUTPUTS", config.REPO_ROOT / "fyp",
           config.REPO_ROOT / "BASEFRAMES"):
    sys.path.insert(0, str(_p))

RENDITIONS = [("high", 720, 22), ("medium", 480, 24), ("low", 360, 26)]


class EngineError(RuntimeError):
    """A failure with a message that is safe to show to the user."""


class Progress:
    def __init__(self, job_dir: Path):
        self.path = job_dir / "progress.json"
        self.state = {"stage": "queued", "progress": 0.0, "message": ""}

    def set(self, stage: str, progress: float, message: str = "", **extra) -> None:
        self.state.update(stage=stage, progress=round(float(progress), 2), message=message,
                          updated_at=time.time(), **extra)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        tmp.replace(self.path)


class _LineTap(io.TextIOBase):
    """stdout replacement: copies every line to a log and hands it to a callback."""

    def __init__(self, log, callback):
        self._log, self._cb, self._buf = log, callback, ""

    def write(self, s):
        self._log.write(s)
        self._log.flush()
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._cb(line)
        return len(s)

    def flush(self):
        self._log.flush()


# ---------------------------------------------------------------- 0. probe
def probe(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise EngineError("The file could not be opened as a video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, _ = cap.read()
    cap.release()
    if not ok or w <= 0 or h <= 0:
        raise EngineError("No readable video frames were found in the file.")
    if not (5.0 <= fps <= 240.0):
        fps = 30.0
    return {"fps": fps, "frames": total, "width": w, "height": h,
            "duration": total / fps if total > 0 else 0.0}


# ---------------------------------------------------------------- 1. court
def calibrate_court(video: Path, job_dir: Path, w: int, h: int, bc) -> dict:
    """Court homography points for this video, in the format bicausal_render.COURTS uses."""
    import detect_court_corners as dcc
    import extract_base_frames as ebf

    # new_render.camera_ok_mask reads ROOT/BASEFRAMES/base_frame/<stem>.png;
    # ROOT is pointed at the job folder before rendering
    bf_dir = job_dir / "BASEFRAMES" / "base_frame"
    bf_dir.mkdir(parents=True, exist_ok=True)
    ebf.process_video(video, bf_dir, n_sample=400, n_clusters=8, n_keep=120, overwrite=True)
    png = bf_dir / f"{video.stem}.png"
    if png.exists():
        try:
            lab = dcc.process(png, job_dir / "court")
            lines = lab["lines_used"]
            if len(lines) >= 2:
                img_pts, crt_pts = [], []
                for ln in lines:
                    img_pts.append(ln["left_px"]); crt_pts.append([0, ln["court_y_m"]])
                    img_pts.append(ln["right_px"]); crt_pts.append([9, ln["court_y_m"]])
                return {"method": "auto", "img": img_pts, "court": crt_pts,
                        "lines": [ln["name"] for ln in lines],
                        "fit_rmse_px": lab.get("fit_rmse_px"),
                        "annotated": f"court/annotated/{video.stem}.png"}
        except Exception as exc:  # noqa: BLE001 - detector failure -> generic court
            print(f"court detection failed: {exc}", flush=True)
    # generic end-court broadcast view: the hand-measured videoplayback (1)
    # model, scaled to this frame size
    img, crt = bc.COURTS["videoplayback (1).mp4"]
    sx, sy = w / 1280.0, h / 720.0
    return {"method": "fallback", "img": [[x * sx, y * sy] for x, y in img], "court": crt,
            "lines": [], "fit_rmse_px": None, "annotated": None}


# ---------------------------------------------------------------- 2. detection
def run_extract(video: Path, job_dir: Path, start: float, seconds: float, imgsz: int,
                progress: Progress, lo: float, hi: float) -> Path:
    cache = job_dir / "cache.pkl"
    cmd = [config.PYTHON_EXE, "-u", str(config.EXTRACT_SCRIPT), str(video),
           "--start", f"{start:.3f}", "--seconds", f"{seconds:.3f}", "--imgsz", str(imgsz),
           "--pose-model", str(config.POSE_MODEL), "--ball-model", str(config.BALL_MODEL),
           "--out", str(cache), "--checkpoint-every", "1500"]
    pat = re.compile(r"frame (\d+)/(\d+)")
    with open(job_dir / "extract.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=str(job_dir), encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = pat.search(line)
            if m:
                done, total = int(m.group(1)), max(1, int(m.group(2)))
                progress.set("detecting", lo + (hi - lo) * done / total,
                             f"Detecting players and ball: frame {done:,} of {total:,}")
        rc = proc.wait()
    if rc != 0 or not cache.exists():
        raise EngineError("Player / ball detection failed (see extract.log).")
    return cache


# ---------------------------------------------------------------- 3. render
def _linux_font(size, bold=False):
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
        p = Path(d) / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def render_video(cache: Path, job_dir: Path, progress: Progress, lo: float, hi: float):
    import new_render as nr

    nr.ROOT = job_dir
    out = job_dir / "render" / "annotated_raw.mp4"
    out.parent.mkdir(exist_ok=True)
    pat = re.compile(r"render (\d+)/(\d+)")
    steps = {"camera / replay mask": "Detecting replays and camera changes",
             "team tactics": "Classifying your team's tactics"}

    def on_line(line):
        m = pat.search(line)
        if m:
            done, total = int(m.group(1)), max(1, int(m.group(2)))
            progress.set("rendering", lo + 3 + (hi - lo - 3) * done / total,
                         f"Rendering annotated video: frame {done:,} of {total:,}")
            return
        for key, msg in steps.items():
            if key in line:
                progress.set("analyzing", lo + (1 if "camera" in key else 2), msg)

    progress.set("analyzing", lo, "Finding ball touches and opponent actions")
    with open(job_dir / "render.log", "a", encoding="utf-8") as log, \
            contextlib.redirect_stdout(_LineTap(log, on_line)):
        summary = nr.render(str(cache), str(out))
    stem = out.with_suffix("")
    return out, summary, Path(f"{stem}_team_tactics.csv"), Path(f"{stem}_opponent_actions.csv")


def read_outputs(tactics_csv: Path, opp_csv: Path, fps: float):
    segments = []
    with open(tactics_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segments.append({"tactic": r["tactic"], "start": float(r["start_s"]),
                             # end_frame is inclusive: the segment lasts until the next frame
                             "end": (int(r["end_frame"]) + 1) / fps,
                             "start_frame": int(r["start_frame"]), "end_frame": int(r["end_frame"]),
                             "confidence": float(r["mean_confidence"])})
    events = []
    with open(opp_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            events.append({"time": int(r["frame"]) / fps, "frame": int(r["frame"]),
                           "action": r["action"], "team": "opponent", "side": "far",
                           "player_track_id": int(r["actor_track_id"])})
    return segments, events


# ---------------------------------------------------------------- 4. encode
def ffmpeg(args, log_path: Path):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    with open(log_path, "a", encoding="utf-8") as log:
        rc = subprocess.run([exe, "-hide_banner", "-y", *args], stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise EngineError("Video encoding failed (see ffmpeg.log).")


def encode_outputs(raw: Path, video: Path, start: float, seconds: float, job_dir: Path,
                   progress: Progress, lo: float, hi: float) -> dict:
    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)
    log = job_dir / "ffmpeg.log"
    master = out_dir / "annotated_ultra.mp4"
    progress.set("encoding", lo, "Adding the original audio")
    # the original commentary / crowd sound, cut to the analysed segment
    ffmpeg(["-i", str(raw), "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(video),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart", str(master)], log)
    cap = cv2.VideoCapture(str(master))
    mw, mh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    files = {"ultra": {"file": master.name, "width": mw, "height": mh, "size": master.stat().st_size}}
    todo = [r for r in RENDITIONS if r[1] < mh]
    for i, (name, height, crf) in enumerate(todo):
        progress.set("encoding", lo + (hi - lo) * (i + 1) / (len(todo) + 1), f"Encoding the {height}p version")
        out = out_dir / f"annotated_{name}.mp4"
        ffmpeg(["-i", str(master), "-vf", f"scale=-2:{height}:flags=lanczos", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "copy",
                "-movflags", "+faststart", str(out)], log)
        width = int(round(mw * height / mh / 2) * 2)
        files[name] = {"file": out.name, "width": width, "height": height, "size": out.stat().st_size}
    ffmpeg(["-ss", f"{min(seconds / 3, 30):.2f}", "-i", str(master), "-frames:v", "1",
            "-vf", "scale=960:-2", "-q:v", "4", str(out_dir / "poster.jpg")], log)
    raw.unlink(missing_ok=True)
    return files


# ---------------------------------------------------------------- main
def run(job_dir: Path) -> dict:
    params = json.loads((job_dir / "params.json").read_text(encoding="utf-8"))
    progress = Progress(job_dir)
    video = job_dir / params["video_file"]
    imgsz = 1280 if params.get("mode", "accurate") == "accurate" else 960

    progress.set("probing", 1, "Reading video")
    meta = probe(video)
    start = max(0.0, float(params.get("start") or 0.0))
    end = float(params.get("end") or 0.0) or meta["duration"]
    end = min(end, meta["duration"]) if meta["duration"] else end
    seconds = end - start
    if seconds < 3:
        raise EngineError("The selected part of the video is shorter than 3 seconds.")
    if seconds > config.MAX_VIDEO_SECONDS:
        raise EngineError(f"Please select at most {config.MAX_VIDEO_SECONDS / 60:.0f} minutes of video.")

    import bicausal_render as bc
    if os.name != "nt":
        bc.load_font = _linux_font

    progress.set("calibrating", 2, "Finding the court lines")
    court = calibrate_court(video, job_dir, meta["width"], meta["height"], bc)
    bc.COURTS[video.name] = (court["img"], court["court"])

    progress.set("detecting", 6, "Loading detection models")
    cache = run_extract(video, job_dir, start, seconds, imgsz, progress, 6, 62)

    raw, summary, tac_csv, opp_csv = render_video(cache, job_dir, progress, 62, 94)
    fps = float(summary["fps"])
    segments, events = read_outputs(tac_csv, opp_csv, fps)
    files = encode_outputs(raw, video, start, seconds, job_dir, progress, 94, 99)

    progress.set("reporting", 99, "Writing reports")
    n_frames = int(summary["frames"])
    start_frame = int(round(start * fps))
    result = {
        "video": {**meta, "analysed_start": start_frame / fps,
                  "analysed_end": (start_frame + n_frames) / fps, "analysed_frames": n_frames},
        "team": "near",
        "mode": params.get("mode", "accurate"),
        "court": {k: court[k] for k in ("method", "lines", "fit_rmse_px", "annotated")},
        "events": events,
        "tactics": {"model": "HistGradientBoosting, tactical_ml_v6.joblib (NEW_OUTPUTS/code/team_tactics.py)",
                    "segments": segments,
                    "frame_share": {k: float(v) for k, v in summary["tactics"].items()}},
        "files": files,
    }
    result["summary"] = reports.summarise(result)
    reports.write_all(result, job_dir / "output", params.get("original_name", video.name))
    (job_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    cache.unlink(missing_ok=True)       # hundreds of MB, only needed while rendering
    progress.set("done", 100, "Analysis complete")
    return result


def main():
    job_dir = Path(sys.argv[1]).resolve()
    progress = Progress(job_dir)
    try:
        run(job_dir)
    except EngineError as exc:
        progress.set("failed", progress.state.get("progress", 0), str(exc))
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        progress.set("failed", progress.state.get("progress", 0),
                     f"Unexpected error: {exc.__class__.__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
