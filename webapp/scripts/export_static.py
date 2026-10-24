"""Export the documentation portal as a static site for Vercel.

Vercel cannot run the analysis (no GPU, ~4.5 MB request limit, short function
timeouts), so the deployed site serves the portal from pre-exported JSON and
images, and the Analyze page points visitors to the desktop app / live server.
The /api/project/* URLs the frontend already uses are mapped to these static
files by vercel.json rewrites, so the frontend code is the same everywhere.

Usage:  python webapp/scripts/export_static.py [--out DIR] [--desktop-url URL] [--live-url URL]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from app import config, project  # noqa: E402

# demo clips small enough for a static deployment (rest stay on the desktop app)
DEMO_KEEP = 3


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def copy_image(rel: str, out: Path) -> str:
    """Copy one repo image to out/repo/<rel>; big PNG photos become JPEG. Returns the served path."""
    src = config.REPO_ROOT / rel
    dst = out / "repo" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png" and src.stat().st_size > 1_500_000:
        jpg = dst.with_suffix(".jpg")
        Image.open(src).convert("RGB").save(jpg, quality=85, optimize=True)
        return rel[: -len(src.suffix)] + ".jpg"
    shutil.copy2(src, dst)
    return rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "vta_build" / "vercel"))
    ap.add_argument("--desktop-url", default="", help="download link of VolleyballTacticalAnalysis-Setup.exe")
    ap.add_argument("--live-url", default="", help="URL of a running analysis server (e.g. a tunnel link)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(config.FRONTEND_DIST, out)
    api = out / "static-api"

    write_json(api / "health.json", {
        "status": "static", "database": "none (static site)", "gpu": {"available": False, "name": None},
        "queue": 0, "max_upload_mb": 500, "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
        "desktop_url": a.desktop_url, "live_url": a.live_url})
    write_json(api / "info.json", {"name": config.SITE_NAME, "domain": config.SITE_DOMAIN,
                                   "github": config.GITHUB_URL, "static": True,
                                   "desktop_url": a.desktop_url, "live_url": a.live_url})
    write_json(api / "metrics.json", project.metrics())
    write_json(api / "analyses.json", [])

    renamed: dict[str, str] = {}
    figs = []
    for f in project.figures():
        served = copy_image(f["path"], out)
        renamed[f["path"]] = served
        figs.append({**f, "path": served})
    write_json(api / "figures.json", figs)
    # figures the Results page asks for by fixed path
    for rel in ("model_comparison/trainvstest_macrof1.png", "model_comparison/trainvstest_accuracy.png",
                "figures_perclass_f1/perclass_f1_all4.png", "PAPER_OUTPUT/action_examples/SPIKE.png"):
        if (config.REPO_ROOT / rel).exists() and rel not in renamed:
            renamed[rel] = copy_image(rel, out)

    docs = project.docs_index()
    write_json(api / "docs.json", docs)
    img_re = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
    for d in docs:
        full = project.doc(d["slug"])
        base = full["base"] if full["base"] != "." else ""
        for href in img_re.findall(full["markdown"]):
            if re.match(r"^(https?:|data:)", href):
                continue
            rel = "/".join(p for p in (base + "/" + href).split("/") if p and p != ".")
            if project.repo_file(rel) and rel not in renamed:
                renamed[rel] = copy_image(rel, out)
        write_json(api / "docs" / f"{d['slug']}.json", full)

    demos = project.demos()[:DEMO_KEEP]
    (out / "demos").mkdir(exist_ok=True)
    for d in demos:
        for k in ("file", "poster"):
            if d.get(k):
                shutil.copy2(config.DEMOS_DIR / d[k], out / "demos" / d[k])
    write_json(api / "demos.json", demos)

    rewrites = [
        {"source": "/api/health", "destination": "/static-api/health.json"},
        {"source": "/api/analyses", "destination": "/static-api/analyses.json"},
        {"source": "/api/project/info", "destination": "/static-api/info.json"},
        {"source": "/api/project/metrics", "destination": "/static-api/metrics.json"},
        {"source": "/api/project/docs", "destination": "/static-api/docs.json"},
        {"source": "/api/project/docs/:slug", "destination": "/static-api/docs/:slug.json"},
        {"source": "/api/project/figures", "destination": "/static-api/figures.json"},
        {"source": "/api/project/demos", "destination": "/static-api/demos.json"},
        {"source": "/api/project/demos/:name", "destination": "/demos/:name"},
    ]
    # large PNGs were converted to JPEG: keep the original URLs working
    for rel, served in renamed.items():
        if rel != served:
            rewrites.append({"source": "/api/project/file", "has": [{"type": "query", "key": "path", "value": re.escape(rel)}],
                             "destination": f"/repo/{served}"})
    rewrites += [
        {"source": "/api/project/file", "has": [{"type": "query", "key": "path", "value": "(?<p>.*)"}],
         "destination": "/repo/:p"},
        {"source": "/((?!api/|assets/|static-api/|repo/|demos/).*)", "destination": "/index.html"},
    ]
    vercel = {
        "cleanUrls": False,
        "rewrites": rewrites,
        "headers": [
            {"source": "/assets/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]},
            {"source": "/sw.js", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
        ],
    }
    write_json(out / "vercel.json", vercel)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"static site -> {out}  ({size:.0f} MB, {len(docs)} docs, {len(figs)} figures, {len(demos)} demos)")


if __name__ == "__main__":
    main()
