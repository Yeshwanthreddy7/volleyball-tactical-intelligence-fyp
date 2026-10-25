"""Stage (and optionally upload) the full website as a Hugging Face Docker Space.

The Space runs the same FastAPI + React app and analysis engine as the laptop,
on the free CPU tier, so anyone can upload a video at any time.

    python webapp/scripts/build_hf_space.py                       # stage only
    python webapp/scripts/build_hf_space.py --repo USER/volleyballanalyzer --token hf_...

The file list is the Docker whitelist in .dockerignore ("!" lines), so the
Space gets exactly what webapp/Dockerfile would.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = Path(os.environ.get("USERPROFILE", Path.home())) / "vta_build" / "hfspace"
SKIP_DIRS = {"node_modules", "dist", "__pycache__"}

README = """---
title: Volleyball Tactical Analysis
emoji: 🏐
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
short_description: Upload a volleyball match, get annotated video + reports
---

Final Year Project, Group 56. Upload a broadcast volleyball video to get an
annotated MP4, a team tactical report and an opponent serve / spike / block report.
"""


def whitelist() -> list[str]:
    lines = (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return [ln[1:].strip() for ln in lines if ln.startswith("!")]


def copy(src: Path, rel: Path) -> None:
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(REPO).parts):
                copy(p, p.relative_to(REPO))
        return
    dst = STAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for pattern in whitelist():
        matches = [REPO / pattern] if "*" not in pattern else [
            p for p in REPO.glob(pattern) if fnmatch.fnmatch(p.relative_to(REPO).as_posix(), pattern)]
        for p in matches:
            if p.exists():
                copy(p, p.relative_to(REPO))
    shutil.copy2(REPO / "webapp" / "Dockerfile.hf", STAGE / "Dockerfile")
    shutil.copy2(REPO / "webapp" / "hf_requirements.txt", STAGE / "webapp" / "hf_requirements.txt")
    # Spaces read their config from README.md front matter; keep the project
    # README (a docs-portal page) below it
    readme = STAGE / "README.md"
    readme.write_text(README + "\n" + (readme.read_text(encoding="utf-8") if readme.exists() else ""),
                      encoding="utf-8")
    (STAGE / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n"
                                          "*.joblib filter=lfs diff=lfs merge=lfs -text\n"
                                          "*.mp4 filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file()) / 1e6
    print(f"staged -> {STAGE}  ({size:.0f} MB)")


def upload(repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Volleyball Tactical Analysis", delete_patterns=["*"])
    user, name = repo_id.split("/")
    print(f"uploaded -> https://huggingface.co/spaces/{repo_id}")
    print(f"site     -> https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="USER/SPACE_NAME on Hugging Face")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or HF_TOKEN env)")
    a = ap.parse_args()
    stage()
    if a.repo:
        if not a.token:
            raise SystemExit("Need --token or HF_TOKEN to upload.")
        upload(a.repo, a.token)
