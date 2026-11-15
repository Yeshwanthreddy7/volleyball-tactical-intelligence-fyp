"""Remove static false positives from the pseudo-labelled ball set.

Frames are sampled seconds apart, so a real ball almost never sits on the same
spot in many of them. A label whose centre falls in the same 1 %-of-frame cell
as labels in >= MIN_REPEAT other frames of the same video is a fixed object
(scoreboard logo, light, sponsor ball graphic): the frame is deleted, because
keeping it as "no ball" or as "ball" would both teach the wrong thing.

Run after make_ball_dataset.py:  python BALL_MODEL_V2/clean_static_labels.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

DATA = Path(os.environ["USERPROFILE"]) / "vta_build" / "ball_v2" / "data"
CELL = 0.01
MIN_REPEAT = 5

labels = defaultdict(list)                     # video -> [(split, stem, cell)]
for split in ("train", "val"):
    for f in (DATA / "labels" / split).glob("*.txt"):
        line = f.read_text().strip()
        if not line:
            continue
        _, x, y, *_ = map(float, line.split())
        video = f.stem.rsplit("_", 1)[0]
        labels[video].append((split, f.stem, (round(x / CELL), round(y / CELL))))

removed = {}
for video, rows in labels.items():
    count = defaultdict(int)
    for *_, c in rows:                          # count each cell plus its 8 neighbours
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                count[(c[0] + dx, c[1] + dy)] += 1
    bad = [(s, stem) for s, stem, c in rows if count[c] > MIN_REPEAT]
    for s, stem in bad:
        (DATA / "labels" / s / f"{stem}.txt").unlink()
        (DATA / "images" / s / f"{stem}.jpg").unlink()
    removed[video] = {"labelled": len(rows), "removed_static": len(bad)}
    print(video, removed[video])
(DATA / "static_removed.json").write_text(json.dumps(removed, indent=1))
for split in ("train", "val"):
    print(split, len(list((DATA / "images" / split).glob("*.jpg"))), "images")
