"""Volleyball Tactical Analysis - web API + static frontend.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000   (from webapp/backend)
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, project
from .db import now_iso, store
from .worker import worker

MAX_QUEUE = 20
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("text/javascript", ".js")
OUTPUT_FILES = {
    "annotated_ultra.mp4", "annotated_high.mp4", "annotated_medium.mp4", "annotated_low.mp4",
    "poster.jpg", "opponent_report.txt", "opponent_report.csv", "opponent_report.json",
    "tactical_report.txt", "tactical_report.csv", "tactical_report.json",
}

app = FastAPI(title=config.SITE_NAME, version="1.0.0", docs_url="/api/docs-ui", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    worker.start()
    worker.recover()


def _owner(client_id: str | None) -> str:
    if not client_id or not re.fullmatch(r"[A-Za-z0-9-]{16,64}", client_id):
        raise HTTPException(400, "Missing or invalid X-Client-Id header.")
    return hashlib.sha256(client_id.encode()).hexdigest()[:32]


def _safe_name(name: str) -> str:
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9 ()_.-]+", "_", stem).strip(" ._") or "video"
    return stem[:80] + ext


def _public(job: dict) -> dict:
    job = dict(job)
    job.pop("owner", None)
    if job.get("status") == "queued":
        job["queue_position"] = worker.position(job["id"])
    return job


# ------------------------------------------------------------------ health
_GPU: dict | None = None


def _gpu() -> dict:
    global _GPU
    if _GPU is None:
        _GPU = {"available": False, "name": None}
        exe = shutil.which("nvidia-smi")
        if exe:
            try:
                out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10,
                                     creationflags=config.NO_WINDOW).stdout.strip()
                if out:
                    _GPU = {"available": True, "name": out.splitlines()[0]}
            except (OSError, subprocess.TimeoutExpired):
                pass
    return _GPU


@app.get("/api/health")
def health():
    return {"status": "ok", "database": store.kind, "gpu": _gpu(), "queue": worker.queue_length(),
            "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024), "chunk_bytes": config.CHUNK_BYTES,
            "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
            "tactical_model": config.TACTICAL_MODEL.name, "models_present": all(
                p.exists() for p in (config.POSE_MODEL, config.BALL_MODEL, config.TACTICAL_MODEL,
                                     config.RENDER_DIR / "new_render.py"))}


# ------------------------------------------------------------------ chunked upload
class UploadInit(BaseModel):
    filename: str = Field(max_length=255)
    size: int = Field(gt=0)


@app.post("/api/uploads")
def upload_init(body: UploadInit, x_client_id: str | None = Header(None)):
    owner = _owner(x_client_id)
    ext = Path(body.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Use one of: "
                                 + ", ".join(sorted(config.ALLOWED_EXTENSIONS)))
    if body.size > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_BYTES // 2**20} MB.")
    uid = uuid.uuid4().hex
    d = config.UPLOAD_DIR / uid
    d.mkdir(parents=True)
    n_chunks = -(-body.size // config.CHUNK_BYTES)
    (d / "meta.json").write_text(json.dumps({"filename": body.filename, "size": body.size, "owner": owner,
                                             "chunks": n_chunks, "received": []}))
    with open(d / "data.part", "wb") as f:
        f.truncate(body.size)
    return {"upload_id": uid, "chunk_bytes": config.CHUNK_BYTES, "chunks": n_chunks}


def _upload_meta(uid: str, owner: str) -> tuple[Path, dict]:
    if not re.fullmatch(r"[0-9a-f]{32}", uid):
        raise HTTPException(404, "Upload not found.")
    d = config.UPLOAD_DIR / uid
    if not (d / "meta.json").exists():
        raise HTTPException(404, "Upload not found or expired.")
    meta = json.loads((d / "meta.json").read_text())
    if meta["owner"] != owner:
        raise HTTPException(404, "Upload not found.")
    return d, meta


@app.put("/api/uploads/{uid}/chunks/{index}")
async def upload_chunk(uid: str, index: int, request: Request, x_client_id: str | None = Header(None)):
    d, meta = _upload_meta(uid, _owner(x_client_id))
    if not 0 <= index < meta["chunks"]:
        raise HTTPException(400, "Bad chunk index.")
    offset = index * config.CHUNK_BYTES
    expected = min(config.CHUNK_BYTES, meta["size"] - offset)
    written = 0
    with open(d / "data.part", "r+b") as f:
        f.seek(offset)
        async for part in request.stream():
            written += len(part)
            if written > expected:
                raise HTTPException(413, "Chunk larger than declared.")
            f.write(part)
    if written != expected:
        raise HTTPException(400, f"Chunk incomplete ({written} of {expected} bytes).")
    meta = json.loads((d / "meta.json").read_text())
    if index not in meta["received"]:
        meta["received"].append(index)
    (d / "meta.json").write_text(json.dumps(meta))
    return {"received": len(meta["received"]), "chunks": meta["chunks"]}


# ------------------------------------------------------------------ analyses
class AnalysisCreate(BaseModel):
    upload_id: str
    mode: str = Field("accurate", pattern="^(accurate|fast)$")
    start: float = Field(0.0, ge=0)
    end: float | None = Field(None, ge=0)


@app.post("/api/analyses", status_code=201)
def create_analysis(body: AnalysisCreate, x_client_id: str | None = Header(None)):
    owner = _owner(x_client_id)
    d, meta = _upload_meta(body.upload_id, owner)
    if len(meta["received"]) != meta["chunks"]:
        raise HTTPException(400, f"Upload incomplete: {len(meta['received'])} of {meta['chunks']} chunks.")
    if body.end is not None and body.end and body.end <= body.start:
        raise HTTPException(400, "End time must be after start time.")
    if worker.queue_length() >= MAX_QUEUE:
        raise HTTPException(429, "The analysis queue is full. Please try again later.")
    job_id = uuid.uuid4().hex
    video_file = _safe_name(meta["filename"])
    doc = {"id": job_id, "owner": owner, "original_name": meta["filename"], "size": meta["size"],
           "team": "near", "mode": body.mode, "start": body.start, "end": body.end,
           "status": "queued", "stage": "queued", "progress": 0, "message": "Waiting in queue",
           "created_at": now_iso()}
    store.insert(doc)
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    shutil.move(str(d / "data.part"), str(job_dir / video_file))
    shutil.rmtree(d, ignore_errors=True)
    (job_dir / "params.json").write_text(json.dumps({
        "video_file": video_file, "original_name": meta["filename"],
        "mode": body.mode, "start": body.start, "end": body.end}))
    worker.submit(job_id)
    return _public(store.get(job_id))


@app.get("/api/analyses")
def list_analyses(x_client_id: str | None = Header(None)):
    owner = _owner(x_client_id)
    return [_public(j) for j in store.list(200) if j.get("owner") == owner][:50]


def _job(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "Analysis not found.")
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Analysis not found.")
    return job


@app.get("/api/analyses/{job_id}")
def get_analysis(job_id: str):
    return _public(_job(job_id))


@app.post("/api/analyses/{job_id}/cancel")
def cancel_analysis(job_id: str, x_client_id: str | None = Header(None)):
    job = _job(job_id)
    if job.get("owner") != _owner(x_client_id):
        raise HTTPException(403, "Only the uploader can cancel this analysis.")
    if job["status"] not in ("queued", "processing"):
        raise HTTPException(409, f"Analysis is already {job['status']}.")
    worker.cancel(job_id)
    return _public(store.get(job_id))


@app.delete("/api/analyses/{job_id}", status_code=204)
def delete_analysis(job_id: str, x_client_id: str | None = Header(None)):
    job = _job(job_id)
    if job.get("owner") != _owner(x_client_id):
        raise HTTPException(403, "Only the uploader can delete this analysis.")
    if job["status"] in ("queued", "processing"):
        worker.cancel(job_id)
    shutil.rmtree(config.JOBS_DIR / job_id, ignore_errors=True)
    store.delete(job_id)


@app.get("/api/analyses/{job_id}/files/{name}")
def analysis_file(job_id: str, name: str, download: bool = False):
    job = _job(job_id)
    if name not in OUTPUT_FILES:
        raise HTTPException(404, "File not found.")
    p = config.JOBS_DIR / job_id / "output" / name
    if not p.exists():
        raise HTTPException(404, "File not available for this analysis.")
    media = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".txt": "text/plain; charset=utf-8",
             ".csv": "text/csv; charset=utf-8", ".json": "application/json"}[p.suffix]
    headers = {"Cache-Control": "private, max-age=3600"}
    if download:
        stem = Path(job.get("original_name", "video")).stem
        fname = re.sub(r'[^A-Za-z0-9 ()_.-]+', "_", f"{stem} - {name}")
        return FileResponse(p, media_type=media, filename=fname, headers=headers)
    return FileResponse(p, media_type=media, headers=headers)


@app.get("/api/analyses/{job_id}/court.png")
def analysis_court(job_id: str):
    job = _job(job_id)
    rel = ((job.get("result") or {}).get("court") or {}).get("annotated")
    p = config.JOBS_DIR / job_id / rel if rel else None
    if not p or not p.exists():
        raise HTTPException(404, "No court image.")
    return FileResponse(p, media_type="image/png")


# ------------------------------------------------------------------ project portal
@app.get("/api/project/metrics")
def project_metrics():
    return project.metrics()


@app.get("/api/project/docs")
def project_docs():
    return project.docs_index()


@app.get("/api/project/docs/{slug}")
def project_doc(slug: str):
    d = project.doc(slug)
    if not d:
        raise HTTPException(404, "Document not found.")
    return d


@app.get("/api/project/figures")
def project_figures():
    return project.figures()


@app.get("/api/project/file")
def project_file(path: str):
    p = project.repo_file(path)
    if not p:
        raise HTTPException(404, "File not found.")
    return FileResponse(p, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/project/demos")
def project_demos():
    return project.demos()


@app.get("/api/project/demos/{name}")
def project_demo_file(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(mp4|jpg|txt)", name):
        raise HTTPException(404, "Not found.")
    p = config.DEMOS_DIR / name
    if not p.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(p, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/project/info")
def project_info():
    return {"name": config.SITE_NAME, "domain": config.SITE_DOMAIN, "github": config.GITHUB_URL}


# ------------------------------------------------------------------ frontend (SPA)
@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "Not found."}, status_code=404)
    dist = config.FRONTEND_DIST
    if full_path:
        p = (dist / full_path).resolve()
        try:
            p.relative_to(dist)
            if p.is_file():
                cache = "public, max-age=31536000, immutable" if "/assets/" in p.as_posix() else "no-cache"
                return FileResponse(p, headers={"Cache-Control": cache})
        except ValueError:
            pass
    index = dist / "index.html"
    if not index.exists():
        return JSONResponse({"detail": "Frontend not built. Run `npm run build` in webapp/frontend."},
                            status_code=503)
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
