"""Analysis-job storage.

MongoDB is the primary store (collection `analyses`). If the server cannot be
reached at start-up the app falls back to a JSON file in the storage folder so
the website still works on a machine without MongoDB; /api/health reports
which backend is live.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from . import config


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MongoStore:
    kind = "mongodb"

    def __init__(self, uri: str, db_name: str):
        from pymongo import DESCENDING, MongoClient

        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._client.admin.command("ping")
        self._col = self._client[db_name]["analyses"]
        self._col.create_index("id", unique=True)
        self._col.create_index([("created_at", DESCENDING)])
        self._desc = DESCENDING

    def insert(self, doc: dict) -> None:
        self._col.insert_one(dict(doc))

    def update(self, job_id: str, fields: dict) -> None:
        self._col.update_one({"id": job_id}, {"$set": fields})

    def get(self, job_id: str) -> dict | None:
        return self._col.find_one({"id": job_id}, {"_id": 0})

    def list(self, limit: int = 50) -> list[dict]:
        cur = self._col.find({}, {"_id": 0, "result.events": 0, "result.tactics": 0})
        return list(cur.sort("created_at", self._desc).limit(limit))

    def delete(self, job_id: str) -> None:
        self._col.delete_one({"id": job_id})

    def find_status(self, statuses: list[str]) -> list[dict]:
        return list(self._col.find({"status": {"$in": statuses}}, {"_id": 0}).sort("created_at", 1))


class JsonStore:
    kind = "json-file (MongoDB unreachable)"

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self._docs: dict[str, dict] = {}
        if path.exists():
            try:
                self._docs = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._docs = {}

    def _flush(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._docs, default=str), encoding="utf-8")
        tmp.replace(self._path)

    def insert(self, doc: dict) -> None:
        with self._lock:
            self._docs[doc["id"]] = dict(doc)
            self._flush()

    def update(self, job_id: str, fields: dict) -> None:
        with self._lock:
            if job_id in self._docs:
                self._docs[job_id].update(fields)
                self._flush()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            d = self._docs.get(job_id)
            return json.loads(json.dumps(d, default=str)) if d else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            docs = sorted(self._docs.values(), key=lambda d: d.get("created_at", ""), reverse=True)
            out = []
            for d in docs[:limit]:
                d = json.loads(json.dumps(d, default=str))
                if d.get("result"):
                    d["result"].pop("events", None)
                    d["result"].pop("tactics", None)
                out.append(d)
            return out

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._docs.pop(job_id, None)
            self._flush()

    def find_status(self, statuses: list[str]) -> list[dict]:
        with self._lock:
            docs = [d for d in self._docs.values() if d.get("status") in statuses]
            return sorted(docs, key=lambda d: d.get("created_at", ""))


def _connect() -> Any:
    try:
        store = MongoStore(config.MONGODB_URI, config.MONGODB_DB)
        print(f"[db] connected to MongoDB ({config.MONGODB_DB})", flush=True)
        return store
    except Exception as exc:  # noqa: BLE001 - any connection failure -> fallback
        print(f"[db] MongoDB unavailable ({exc.__class__.__name__}); using JSON file store", flush=True)
        return JsonStore(config.STORAGE_DIR / "analyses.json")


store = _connect()
