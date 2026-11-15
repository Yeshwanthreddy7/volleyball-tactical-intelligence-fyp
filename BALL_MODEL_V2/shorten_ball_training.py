"""Wait for the running ball-v2 training to finish epoch 15 and save last.pt,
stop it, cap the run at 20 epochs in the checkpoint, and resume it."""
import os, subprocess, sys, time
from pathlib import Path

PID = 30800
TARGET_EPOCHS = 20
W = Path(os.environ["USERPROFILE"]) / "vta_build" / "ball_v2" / "runs" / "ball_v2"
CSV, LAST = W / "results.csv", W / "weights" / "last.pt"
REPO = Path(r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject")
LOG = REPO / "BALL_MODEL_V2" / "train_resume.log"

def rows():
    return len(CSV.read_text().strip().splitlines()) - 1

start_mtime = LAST.stat().st_mtime
print("waiting for epoch 15 to be saved...", flush=True)
while not (rows() >= 15 and LAST.stat().st_mtime > start_mtime):
    time.sleep(15)
time.sleep(30)  # let the save finish
print("epoch", rows(), "saved; stopping PID", PID, flush=True)
subprocess.run(["taskkill", "/PID", str(PID), "/T", "/F"], capture_output=True)
time.sleep(10)

import shutil
shutil.copy2(LAST, LAST.with_name("last_epoch15_backup.pt"))  # untouched copy, in case anything goes wrong

import torch
ck = torch.load(LAST, map_location="cpu", weights_only=False)
print("checkpoint epoch", ck.get("epoch"), flush=True)
ck["train_args"]["epochs"] = TARGET_EPOCHS
for key in ("model", "ema"):
    m = ck.get(key)
    if m is not None and isinstance(getattr(m, "args", None), dict):
        m.args["epochs"] = TARGET_EPOCHS
torch.save(ck, LAST)
print("epochs capped at", TARGET_EPOCHS, "- resuming", flush=True)

with open(LOG, "w") as log:
    p = subprocess.Popen([sys.executable, "BALL_MODEL_V2/train_ball_v2.py", "--resume"], cwd=REPO,
                         stdout=log, stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    p.wait()
print("training script exit code", p.returncode, flush=True)
print((REPO / "BALL_MODEL_V2" / "results.json").read_text() if (REPO / "BALL_MODEL_V2" / "results.json").exists() else "no results.json")
