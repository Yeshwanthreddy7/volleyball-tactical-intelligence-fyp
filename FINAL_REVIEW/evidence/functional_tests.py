"""Functional tests of the website backend (FastAPI), run in-process with TestClient.

Uses a throw-away storage folder, so real analyses are not touched.
Run:  python FINAL_REVIEW/evidence/functional_tests.py
Out:  FINAL_REVIEW/evidence/functional_tests.json
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1] / "finalyearproject" / "webapp" / "backend"
tmp = tempfile.mkdtemp(prefix="vta_ft_")
os.environ["VTA_STORAGE_DIR"] = tmp
os.environ["MONGODB_URI"] = "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=500"   # force the JSON store
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app                   # noqa: E402

c = TestClient(app)
H = {"X-Client-Id": "functional-test-client-0001"}
results = []


def check(tc, name, action, resp, expect):
    ok = resp.status_code == expect
    results.append({"id": tc, "case": name, "action": action, "expected": expect,
                    "got": resp.status_code, "detail": (resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else None),
                    "pass": ok})
    print(tc, name, resp.status_code, "PASS" if ok else "FAIL")


check("TC-01", "Server health", "GET /api/health", c.get("/api/health"), 200)
check("TC-02", "Wrong file type", "start upload of notes.pdf",
      c.post("/api/uploads", json={"filename": "notes.pdf", "size": 1000}, headers=H), 415)
check("TC-03", "File too large", "start upload of a 5 GB .mp4",
      c.post("/api/uploads", json={"filename": "match.mp4", "size": 5 * 2**30}, headers=H), 413)
check("TC-04", "Missing browser id", "start upload without X-Client-Id",
      c.post("/api/uploads", json={"filename": "match.mp4", "size": 1000}), 400)
r = c.post("/api/uploads", json={"filename": "fake.mp4", "size": 64}, headers=H)
check("TC-05", "Valid upload start", "start upload of a 64-byte .mp4", r, 200)
uid = r.json()["upload_id"]
check("TC-06", "Renamed non-video file", "send text bytes named fake.mp4",
      c.put(f"/api/uploads/{uid}/chunks/0", content=b"this is not a video, only text bytes!!" + b"." * 26, headers=H), 415)
check("TC-07", "Unknown analysis id", "GET /api/analyses/<random id>",
      c.get("/api/analyses/" + "0" * 32, headers=H), 404)
check("TC-08", "Path-traversal attempt", "GET a file named ../../config.py",
      c.get("/api/analyses/" + "0" * 32 + "/files/..%2F..%2Fconfig.py", headers=H), 404)

(HERE / "functional_tests.json").write_text(json.dumps(results, indent=1))
print(sum(r["pass"] for r in results), "of", len(results), "passed")
