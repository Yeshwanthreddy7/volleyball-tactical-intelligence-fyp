"""Single GPU worker: runs queued analyses one at a time, each in its own
`python -m app.engine` process, and mirrors its progress.json into the DB."""
from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone

from . import config
from .db import now_iso, store

RETENTION_DAYS = float(os.environ.get("VTA_RETENTION_DAYS", "14"))
# An open website sends a heartbeat for its unfinished analyses every 20 s; a job
# nobody has watched for this long (tab / browser closed) is cancelled so it
# does not block the queue for everyone else.
ABANDON_SECONDS = float(os.environ.get("VTA_ABANDON_SECONDS", "300"))


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=config.NO_WINDOW)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="analysis-worker")
        self._q: "queue.Queue[str]" = queue.Queue()
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self.current: str | None = None
        self._proc: subprocess.Popen | None = None
        self._seen: dict[str, float] = {}

    # ------------------------------------------------------------ public API
    def submit(self, job_id: str) -> None:
        with self._lock:
            self._pending.append(job_id)
            self._seen[job_id] = time.time()
        self._q.put(job_id)

    def position(self, job_id: str) -> int | None:
        with self._lock:
            return self._pending.index(job_id) + 1 if job_id in self._pending else None

    def queue_length(self) -> int:
        with self._lock:
            return len(self._pending) + (1 if self.current else 0)

    def touch(self, job_ids) -> None:
        """Heartbeat from an open browser tab: these analyses are still wanted."""
        now = time.time()
        with self._lock:
            for j in job_ids:
                if j in self._pending or j == self.current:
                    self._seen[j] = now

    def active_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending) + ([self.current] if self.current else [])

    def cancel(self, job_id: str, message: str = "Cancelled by user") -> None:
        with self._lock:
            if job_id in self._pending:
                self._pending.remove(job_id)
        if self.current == job_id:
            (config.JOBS_DIR / job_id / "CANCEL").touch()
            if self._proc is not None:
                _kill_tree(self._proc)
        with self._lock:
            self._seen.pop(job_id, None)
        store.update(job_id, {"status": "cancelled", "stage": "cancelled",
                              "message": message, "finished_at": now_iso()})

    def recover(self) -> None:
        """Re-queue work interrupted by a server restart (detection resumes from its checkpoint)."""
        for job in store.find_status(["queued", "processing"]):
            store.update(job["id"], {"status": "queued", "stage": "queued",
                                     "message": "Waiting in queue (restarted)"})
            self.submit(job["id"])

    # ------------------------------------------------------------ loop
    def run(self) -> None:
        threading.Thread(target=self._janitor, daemon=True, name="janitor").start()
        threading.Thread(target=self._watchdog, daemon=True, name="watchdog").start()
        while True:
            job_id = self._q.get()
            with self._lock:
                if job_id not in self._pending:
                    continue                     # cancelled while waiting
                self._pending.remove(job_id)
                self.current = job_id
            try:
                self._run_job(job_id)
            except Exception as exc:  # noqa: BLE001 - never let the worker thread die
                store.update(job_id, {"status": "failed", "message": f"Worker error: {exc}",
                                      "finished_at": now_iso()})
            finally:
                self.current, self._proc = None, None

    def _run_job(self, job_id: str) -> None:
        job_dir = config.JOBS_DIR / job_id
        if not (job_dir / "params.json").exists():
            store.update(job_id, {"status": "failed", "message": "Uploaded video is missing."})
            return
        (job_dir / "CANCEL").unlink(missing_ok=True)
        store.update(job_id, {"status": "processing", "stage": "starting", "progress": 0.5,
                              "message": "Starting analysis", "started_at": now_iso()})
        env = dict(os.environ, PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT="4",
                   PYTHONPATH=str(config.BACKEND_DIR))
        log = open(job_dir / "engine.log", "a", encoding="utf-8")
        kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": config.NO_WINDOW}
        self._proc = subprocess.Popen([config.PYTHON_EXE, "-u", "-m", "app.engine", str(job_dir)],
                                      cwd=str(config.BACKEND_DIR), stdout=log, stderr=subprocess.STDOUT,
                                      env=env, **kw)
        prog_path = job_dir / "progress.json"
        last = None
        while self._proc.poll() is None:
            time.sleep(1.0)
            last = self._sync_progress(job_id, prog_path, last)
        log.close()
        self._sync_progress(job_id, prog_path, last)
        if (job_dir / "CANCEL").exists():
            return                                # cancel() already set the status
        rc = self._proc.returncode
        if rc == 0 and (job_dir / "result.json").exists():
            result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
            store.update(job_id, {"status": "done", "stage": "done", "progress": 100,
                                  "message": "Analysis complete", "result": result,
                                  "finished_at": now_iso()})
            src = job_dir / json.loads((job_dir / "params.json").read_text())["video_file"]
            src.unlink(missing_ok=True)           # the original upload is no longer needed
        else:
            msg = "Analysis failed."
            try:
                msg = json.loads(prog_path.read_text(encoding="utf-8")).get("message") or msg
            except (OSError, json.JSONDecodeError):
                pass
            store.update(job_id, {"status": "failed", "stage": "failed", "message": msg,
                                  "finished_at": now_iso()})

    @staticmethod
    def _sync_progress(job_id, prog_path, last):
        try:
            raw = prog_path.read_text(encoding="utf-8")
        except OSError:
            return last
        if raw == last:
            return last
        try:
            p = json.loads(raw)
        except json.JSONDecodeError:
            return last
        if p.get("stage") not in ("done", "failed"):
            store.update(job_id, {"stage": p.get("stage"), "progress": p.get("progress", 0),
                                  "message": p.get("message", "")})
        return raw

    def _watchdog(self) -> None:
        """Cancel analyses whose website was closed (no heartbeat for ABANDON_SECONDS)."""
        while True:
            time.sleep(15)
            now = time.time()
            for j in self.active_ids():
                with self._lock:
                    seen = self._seen.setdefault(j, now)
                if now - seen > ABANDON_SECONDS:
                    self.cancel(j, "Cancelled automatically: the website was closed before the analysis finished")

    def _janitor(self) -> None:
        """Delete analyses (files + DB rows) older than VTA_RETENTION_DAYS and stale uploads."""
        while True:
            cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
            for d in config.JOBS_DIR.iterdir():
                if not d.is_dir() or d.name == self.current:
                    continue
                job = store.get(d.name)
                created = job.get("created_at") if job else None
                try:
                    old = created is None or datetime.fromisoformat(created) < cutoff
                except ValueError:
                    old = False
                if old and (job is None or job.get("status") not in ("queued", "processing")):
                    shutil.rmtree(d, ignore_errors=True)
                    store.delete(d.name)
            for d in config.UPLOAD_DIR.iterdir():
                if d.is_dir() and time.time() - d.stat().st_mtime > 24 * 3600:
                    shutil.rmtree(d, ignore_errors=True)
            time.sleep(3600)


worker = Worker()
