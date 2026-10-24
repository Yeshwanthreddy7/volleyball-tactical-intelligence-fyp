"""Volleyball Tactical Analysis - Windows desktop app.

Opens the website in its own app window (Edge WebView2 via pywebview) and,
if it is not already running, starts everything it needs on this PC:
MongoDB (portable copy in %USERPROFILE%\\mongodb) and the FastAPI server with
the GPU analysis engine. Closing the window stops the server it started.

Run:  pythonw webapp\\desktop\\desktop_app.py     (or the desktop shortcut)
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEBAPP = HERE.parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("VTA_PORT", "80"))
# *.localhost always resolves to this PC, so the app gets a readable address
URL = f"http://volleyballtactics.localhost{"" if PORT == 80 else f":{PORT}"}/"

os.environ.setdefault("VTA_STORAGE_DIR", str(Path.home() / "vta_storage"))
os.environ.setdefault("VTA_DEMOS_DIR", str(WEBAPP / "storage" / "demos"))
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def port_open(port: int, host: str = HOST) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_mongo() -> None:
    if port_open(27017):
        return
    # installed app: <install>\mongodb\mongod.exe (data in <install>\data);
    # developer PC: portable copy in %USERPROFILE%\mongodb
    install_root = WEBAPP.parents[2]
    home = Path.home() / "mongodb"
    candidates = [(install_root / "mongodb" / "mongod.exe", install_root / "data")]
    if home.exists():
        candidates += [(p, home) for p in home.glob("mongodb-win32-*/bin/mongod.exe")]
    found = next(((exe, data) for exe, data in candidates if exe.exists()), None)
    if found is None:
        return                      # the app falls back to its JSON file store
    exe, data = found
    dbpath = data / "db" if exe.parent.name == "mongodb" else data / "data"
    dbpath.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    try:
        subprocess.Popen([str(exe), "--dbpath", str(dbpath), "--bind_ip", "127.0.0.1",
                          "--port", "27017", "--wiredTigerCacheSizeGB", "0.25",
                          "--logpath", str(data / "mongod.log")], creationflags=flags)
    except OSError:
        return                      # e.g. missing Visual C++ runtime -> JSON file store
    for _ in range(40):
        if port_open(27017):
            return
        time.sleep(0.25)


def start_server():
    """Run uvicorn in a background thread; returns the Server so it can be stopped."""
    sys.path.insert(0, str(WEBAPP / "backend"))
    os.chdir(WEBAPP / "backend")
    import uvicorn

    server = uvicorn.Server(uvicorn.Config("app.main:app", host=HOST, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/api/health", timeout=1)
            return server
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(f"The analysis server did not start (port {PORT}).")


def main() -> None:
    import webview

    server = None
    if not port_open(PORT):          # reuse a server started by run_website.bat
        start_mongo()
        server = start_server()
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window("Volleyball Tactical Analysis", URL, width=1440, height=920,
                          min_size=(900, 600), background_color="#0B1220", text_select=True,
                          maximized=True)
    webview.start(private_mode=False, storage_path=str(Path.home() / "vta_storage" / "webview"))
    if server is not None:
        # an analysis still running or queued finishes in the background
        # (results appear under "My analyses" next time) before the server stops
        from app.worker import worker
        while worker.queue_length() > 0:
            time.sleep(5)
        server.should_exit = True
        time.sleep(1.0)


if __name__ == "__main__":
    main()
