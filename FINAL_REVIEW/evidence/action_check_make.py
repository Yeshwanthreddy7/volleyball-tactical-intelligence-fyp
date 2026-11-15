"""Opponent-action check, step 1 (added 2026-11-15).

The SERVE / SPIKE / BLOCK events come from the rule-based touch detector
(OUTPUTS/bicausal_render.py) on the 8 rendered 5-minute segments in
NEW_OUTPUTS/videos/*_opponent_actions.csv. This script draws a random sample of
events (up to 12 SERVE, 18 SPIKE and every BLOCK), and for each one saves a strip
of 3 frames (0.3 s before, at, 0.3 s after the touch) zoomed on the player the
rule named, so a person can mark each event right or wrong.
Out: action_check/sheet_*.jpg + action_check/events.json
"""
import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject")
OUT = Path(__file__).resolve().parent / "action_check"
OUT.mkdir(exist_ok=True)
random.seed(11)

rows = []
for f in sorted((ROOT / "NEW_OUTPUTS" / "videos").glob("*_opponent_actions.csv")):
    slug = f.name.split("_team_tactics")[0]
    for _, r in pd.read_csv(f).iterrows():
        rows.append({"slug": slug, "time": r["video_time"], "frame": int(r["frame"]),
                     "action": r["action"], "actor": int(r["actor_track_id"])})
pick = []
for act, k in (("SERVE", 12), ("SPIKE", 18), ("BLOCK", 99)):
    pool = [r for r in rows if r["action"] == act]
    pick += random.sample(pool, min(k, len(pool)))
random.shuffle(pick)

caches = {}
strips = []
for i, e in enumerate(pick):
    if e["slug"] not in caches:
        with open(ROOT / "PAPER_OUTPUT" / "cache" / f"cache_{e['slug']}.pkl", "rb") as fh:
            caches[e["slug"]] = pickle.load(fh)
    c = caches[e["slug"]]
    fps = c["fps"]
    start = int(round(c["start"] * fps)) if c["start"] < 10000 else int(c["start"])
    people = c["frames"][e["frame"]]["people"]
    box = next((p["box"] for p in people if p["id"] == e["actor"]), None)
    cap = cv2.VideoCapture(str(ROOT / "fyp" / "Datasets" / Path(c["video"]).name))
    tiles = []
    for off in (-0.3, 0.0, 0.3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + e["frame"] + int(off * fps))
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((720, 1280, 3), np.uint8)
        h, w = fr.shape[:2]
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            bh = max(box[3] - box[1], 40)
            cv2.rectangle(fr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
        else:
            cx, cy, bh = w / 2, h / 3, h / 4
        half = int(bh * 1.6)
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        crop = fr[y0:int(min(cy + half, h)), x0:int(min(cx + half, w))]
        tiles.append(cv2.resize(crop, (300, 300)))
    cap.release()
    strip = np.hstack(tiles)
    cv2.rectangle(strip, (0, 0), (190, 30), (0, 0, 0), -1)
    cv2.putText(strip, f"{i} {e['action']}", (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    strips.append(strip)
    e["id"] = i
    e["actor_found"] = box is not None

for s in range(0, len(strips), 8):
    part = strips[s:s + 8]
    rows_ = [np.hstack(part[j:j + 2]) if j + 1 < len(part) else np.hstack([part[j], np.zeros_like(part[j])])
             for j in range(0, len(part), 2)]
    cv2.imwrite(str(OUT / f"sheet_{s // 8}.jpg"), np.vstack(rows_))
(OUT / "events.json").write_text(json.dumps(pick, indent=1))
print(len(pick), "events", {a: sum(e["action"] == a for e in pick) for a in ("SERVE", "SPIKE", "BLOCK")})
