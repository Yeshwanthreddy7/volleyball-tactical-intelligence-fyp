"""Upload checks: the file must be a real video, and the video a volleyball match.

Two levels:

  * `sniff_container(head)` - the first bytes of the upload must be a known video
    container (MP4 / MOV / M4V, AVI, Matroska / WebM). A text file or an image
    renamed to .mp4 is rejected as soon as its first chunk arrives.
  * `check_volleyball(video, start, end)` - frames sampled across the selected
    part of the video. A frame counts as volleyball when the project's court
    detector (BASEFRAMES/detect_court_corners.py) fits the 9 x 18 m court lines
    AND at least MIN_PLAYERS people (YOLO11n-pose) stand inside that court.
    The video passes when at least MIN_FRAME_SHARE of the readable frames do.

Measured on 5 frames of each of the 11 dataset videos (end-court broadcast view):
every frame of 9 videos and at least 4 of 5 of the other 2 pass, with 5-14 players
inside the court. A test pattern, a screen recording and still photos of a street
and of two people score 0 players inside a court on every frame.

Runs in its own process (`python -m app.validate <video> <start> <end>`, prints
one JSON line) so the detector models never load into the web server.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_PLAYERS = 4
MIN_FRAME_SHARE = 0.375          # 3 of 8 sampled frames: replays / close-ups are allowed
N_FRAMES = 8
IMGSZ = 960

NOT_VOLLEYBALL = ("This video does not look like a volleyball match. The analysis needs a volleyball "
                  "match filmed from behind one baseline (end-court view), with the court lines and the "
                  "players visible.")


# ISO base media files that are not video: HEIC / AVIF photos, M4A / M4B / M4P audio
NON_VIDEO_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1",
                    b"avif", b"avis", b"M4A ", b"M4B ", b"M4P "}


def sniff_container(head: bytes) -> bool:
    """True when the first bytes are a video container this site accepts."""
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return head[8:12] not in NON_VIDEO_BRANDS
    if len(head) >= 12 and head[4:8] in (b"moov", b"mdat", b"free", b"wide", b"skip"):
        return True                                  # ISO base media: MP4 / MOV / M4V
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True
    return head[:4] == b"\x1a\x45\xdf\xa3"           # EBML: Matroska / WebM


def check_volleyball(video: Path, start: float = 0.0, end: float | None = None) -> dict:
    import cv2
    import numpy as np

    from . import config
    sys.path.insert(0, str(config.REPO_ROOT / "BASEFRAMES"))
    import detect_court_corners as dcc
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"ok": False, "reason": "The file could not be opened as a video."}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    a = int(start * fps)
    b = min(n, int(end * fps)) if end else n
    if b - a < 1:
        return {"ok": False, "reason": "No readable video frames were found in the selected part."}
    pose = YOLO(str(config.REPO_ROOT / "OUTPUTS" / "yolo11n-pose.pt"))

    frames = []
    for i in np.linspace(a + 0.05 * (b - a), a + 0.95 * (b - a), N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        players = 0
        try:
            court = dcc.detect(img)
        except Exception:                            # noqa: BLE001 - no court lines found
            court = None
        if court is not None:
            boxes = pose(img, imgsz=IMGSZ, conf=0.3, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
            if len(boxes):
                feet = np.float32([[[(x1 + x2) / 2, y2] for x1, y1, x2, y2 in boxes]])
                xy = cv2.perspectiveTransform(feet, np.linalg.inv(court["H"]))[0]
                # court is 9 x 18 m; allow the free zone around it
                players = int(((xy[:, 0] > -2) & (xy[:, 0] < 11) & (xy[:, 1] > -3) & (xy[:, 1] < 21)).sum())
        frames.append({"frame": int(i), "court": court is not None, "players_on_court": players})
    cap.release()
    if not frames:
        return {"ok": False, "reason": "No readable video frames were found in the selected part."}
    good = sum(f["court"] and f["players_on_court"] >= MIN_PLAYERS for f in frames)
    ok = good >= MIN_FRAME_SHARE * len(frames)
    return {"ok": ok, "reason": None if ok else NOT_VOLLEYBALL,
            "volleyball_frames": good, "sampled_frames": len(frames), "frames": frames}


def main():
    video, start = Path(sys.argv[1]), float(sys.argv[2])
    end = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("", "None") else None
    print(json.dumps(check_volleyball(video, start, end)), flush=True)


if __name__ == "__main__":
    main()
