"""Stage and (optionally) upload the Hugging Face Space that runs the whole app
in the cloud, so anyone can upload a video without the project laptop.

  python webapp/hfspace/build_hf_space.py                         # stage only
  python webapp/hfspace/build_hf_space.py --user NAME --token hf_xxx   # stage + upload

Space URL: https://<user>-volleyballanalyzer.hf.space
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEBAPP = HERE.parent
REPO = WEBAPP.parent
SPACE = "volleyballanalyzer"

# research code, models and portal content the app needs (same set as the installer)
FILES = [
    "OUTPUTS/bicausal_extract.py", "OUTPUTS/bicausal_render.py", "OUTPUTS/yolo11m-pose.pt",
    "fyp/volleyball_best.pt", "tactical_ml_v6.joblib", "ml_results_v6/ml_results.json",
    "deep_results_v6/deep_results.json", "PANEL_REVIEW/architecture_diagram.svg",
    "BASEFRAMES/_contact_sheet.png", "lovo_metrics_chart.png", "train_test_split_chart.png",
    "ALL_13_MODELS/README.md", "BINARY_ATTACK_DEFENSE/README.md", "FINALE_OUTPUTS/README.md",
    "PANEL_REVIEW/README.md", "webapp/backend/requirements.txt",
]
GLOBS = [
    ("NEW_OUTPUTS/code", "*.py"), ("fyp", "*.py"), ("BASEFRAMES", "*.py"),
    ("BASEFRAMES/corner_court_detection/labels", "*.json"), (".", "*.md"), ("docs", "*.md"),
    ("figures_atlas", "**/*"), ("figures_perclass_f1", "*"), ("figures_workflow", "*"),
    ("model_comparison", "*"), ("PAPER_OUTPUT/action_examples", "*.png"),
    ("webapp/backend/app", "*.py"), ("webapp/frontend/dist", "**/*"), ("webapp/storage/demos", "*"),
]
README = """---
title: Volleyball Tactical Analysis
emoji: 🏐
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
short_description: Volleyball team tactics + opponent actions from match video
---

Upload a volleyball match clip and get an annotated video, a team tactical report
(Coordinated Attack / Coordinated Defense / Delayed Support / Spacing Breakdown)
and an opponent SERVE / SPIKE / BLOCK report.

Group 56 final year project - https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp
"""


def stage(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def put(src: Path) -> None:
        rel = src.relative_to(REPO)
        if "__pycache__" in rel.parts or src.is_dir():
            return
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for f in FILES:
        if (REPO / f).exists():
            put(REPO / f)
    for d, pat in GLOBS:
        for p in (REPO / d).glob(pat):
            put(p)
    shutil.copy2(HERE / "Dockerfile", out / "Dockerfile")
    reqs = (WEBAPP / "installer" / "requirements-desktop.txt").read_text(encoding="utf-8")
    reqs = "\n".join(l for l in reqs.splitlines() if not l.startswith("pywebview"))
    (out / "requirements-space.txt").write_text(reqs + "\n", encoding="utf-8")
    shutil.copy2(REPO / "README.md", out / "PROJECT_README.md")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n*.joblib filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.jpg filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"staged {out} ({size:.0f} MB)")


def upload(out: Path, user: str, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{user}/{SPACE}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Volleyball Tactical Analysis")
    url = f"https://{user.lower().replace('_', '-')}-{SPACE}.hf.space"
    print("Space:", f"https://huggingface.co/spaces/{repo_id}")
    print("App  :", url)
    return url


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "vta_build" / "hfspace"))
    ap.add_argument("--user")
    ap.add_argument("--token")
    a = ap.parse_args()
    out = Path(a.out)
    stage(out)
    if a.user and a.token:
        upload(out, a.user, a.token)
