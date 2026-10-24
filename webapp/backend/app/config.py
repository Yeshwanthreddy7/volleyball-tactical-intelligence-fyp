"""Runtime configuration. Every value can be overridden by an environment
variable so the same code runs on a laptop, in Docker and on a cloud GPU box."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
WEBAPP_DIR = BACKEND_DIR.parent
# finalyearproject/ - the research repo whose code the website runs
REPO_ROOT = Path(os.environ.get("VTA_REPO_ROOT", WEBAPP_DIR.parent)).resolve()

STORAGE_DIR = Path(os.environ.get("VTA_STORAGE_DIR", WEBAPP_DIR / "storage")).resolve()
UPLOAD_DIR = STORAGE_DIR / "uploads"
JOBS_DIR = STORAGE_DIR / "jobs"
DEMOS_DIR = Path(os.environ.get("VTA_DEMOS_DIR", STORAGE_DIR / "demos")).resolve()
FRONTEND_DIST = Path(os.environ.get("VTA_FRONTEND_DIST", WEBAPP_DIR / "frontend" / "dist")).resolve()

for _d in (UPLOAD_DIR, JOBS_DIR, DEMOS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("VTA_MAX_UPLOAD_MB", "500")) * 1024 * 1024
# Cloudflare (free plan) rejects request bodies over 100 MB, so the browser
# sends the file in chunks no bigger than this.
CHUNK_BYTES = 32 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MAX_VIDEO_SECONDS = float(os.environ.get("VTA_MAX_VIDEO_SECONDS", str(60 * 60)))

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "volleyball_tactical_analysis")

_exe = Path(sys.executable)
# the desktop app runs under pythonw.exe; the engine needs python.exe (a hidden
# console that ffmpeg / detection child processes inherit instead of opening windows)
if _exe.name.lower() == "pythonw.exe" and (_exe.parent / "python.exe").exists():
    _exe = _exe.parent / "python.exe"
PYTHON_EXE = os.environ.get("VTA_PYTHON", str(_exe))
# Windows: start child processes without a console window
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Research-repo artefacts the analysis engine runs (all paths relative to REPO_ROOT)
EXTRACT_SCRIPT = REPO_ROOT / "OUTPUTS" / "bicausal_extract.py"
# focus-team tactics + opponent SERVE / SPIKE / BLOCK renderer (new_render.py, team_tactics.py)
RENDER_DIR = REPO_ROOT / "NEW_OUTPUTS" / "code"
POSE_MODEL = REPO_ROOT / "OUTPUTS" / "yolo11m-pose.pt"
BALL_MODEL = REPO_ROOT / "fyp" / "volleyball_best.pt"
TACTICAL_MODEL = Path(os.environ.get("VTA_TACTICAL_MODEL", REPO_ROOT / "tactical_ml_v6.joblib"))

SITE_NAME = "Volleyball Tactical Analysis"
SITE_DOMAIN = os.environ.get("VTA_DOMAIN", "volleyballtacticalanalyze.com")
GITHUB_URL = os.environ.get(
    "VTA_GITHUB_URL", "https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp")
