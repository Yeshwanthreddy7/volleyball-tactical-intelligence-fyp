"""
two_stream_dataset.py -- real-data loader for TwoStreamClassifier.

Wires BASEFRAMES into branch 2 for real, replacing the random-tensor stand-in
used by smoke_test.py. For every labelled clip it pairs up the THREE things
the sketch's branch 2 needs:

  1. training_csv_v4/<Class>__<video>_clip_<NNN>.csv
       -> pose sequence, via the SAME features.build_model_sequence() pipeline
          train_mamba.py uses (train/serve parity preserved). (B, 29, 18)

  2. fyp/dataset/<Class>/<video>_clip_<NNN>.mp4
       -> the actual 30-frame, 30fps, re-encoded clip video that
          extract_clips.py cut from the source video (fyp/Datasets/<video>.*).

  3. BASEFRAMES/base_frame/<video>.png
       -> that source video's empty-court base frame (built by
          BASEFRAMES/extract_base_frames.py -- see base-frames memory).

Branch-2 input is built as:  motion_t = resize(gray(frame_t)) - resize(gray(base_frame))
for t = 1..30, i.e. exactly the "Base frame - each clip - 30" step in the sketch.

All three files are matched purely by filename (class prefix, video stem, clip
number) -- no new CSV/index needed, the existing repo layout already carries
enough information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent          # finalyearproject/
_FYP_DIR = _REPO_ROOT / "fyp"
if str(_FYP_DIR) not in sys.path:
    sys.path.insert(0, str(_FYP_DIR))

from mamba_model import FEATURE_COLS, LABEL_NAMES              # noqa: E402
from features import build_model_sequence                       # noqa: E402

DEFAULT_CSV_DIR = _REPO_ROOT / "training_csv_v4"
DEFAULT_CLIP_DIR = _REPO_ROOT / "fyp" / "dataset"
DEFAULT_BASE_FRAME_DIR = _REPO_ROOT / "BASEFRAMES" / "base_frame"

FRAMES_PER_CLIP = 30    # matches extract_clips.py's "-t 1.000 -r 30"
POSE_SEQ_LEN = 29       # matches label_clips.py's TRAIN_END
IMG_SIZE = 64            # motion-frame resolution fed to the conv stem


def _parse_csv_name(csv_name: str) -> tuple[str, str, str]:
    """'Coordinated_Attack__videoplayback (3)_clip_045.csv' ->
    ('Coordinated_Attack', 'videoplayback (3)', '045')"""
    stem = csv_name[:-4] if csv_name.lower().endswith(".csv") else csv_name
    cls, rest = stem.split("__", 1)
    video, num = rest.rsplit("_clip_", 1)
    return cls, video, num


def _read_clip_frames(mp4_path: Path, n_frames: int = FRAMES_PER_CLIP) -> list[np.ndarray]:
    """Read up to n_frames BGR frames from a clip video, in order."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames: list[np.ndarray] = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames


def _to_gray_resized(frame_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


class TwoStreamVolleyballDataset(Dataset):
    """Real-data pose + base-frame-diffed-motion dataset.

    Returns
    -------
    pose_seq      : (29, 18) float32   -- branch 1 input (build_model_sequence output)
    motion_frames : (30, 1, IMG_SIZE, IMG_SIZE) float32, roughly in [-1, 1]
                    -- branch 2 input (frame_t - base_frame, per clip)
    label_idx     : int64
    """

    def __init__(
        self,
        csv_dir: Path | str = DEFAULT_CSV_DIR,
        clip_dir: Path | str = DEFAULT_CLIP_DIR,
        base_frame_dir: Path | str = DEFAULT_BASE_FRAME_DIR,
        img_size: int = IMG_SIZE,
        limit: int | None = None,
    ) -> None:
        self.csv_dir = Path(csv_dir)
        self.clip_dir = Path(clip_dir)
        self.base_frame_dir = Path(base_frame_dir)
        self.img_size = img_size
        self.label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}

        self._base_frame_cache: dict[str, np.ndarray] = {}  # video stem -> gray, resized

        self.samples: list[tuple[Path, Path, str, int]] = []  # csv, mp4, video_stem, label_idx
        self.skipped: list[tuple[str, str]] = []               # (csv_name, reason)

        csv_files = sorted(p for p in self.csv_dir.glob("*.csv"))
        for csv_path in csv_files:
            try:
                cls, video, num = _parse_csv_name(csv_path.name)
            except ValueError:
                self.skipped.append((csv_path.name, "unparseable filename"))
                continue

            mp4_path = self.clip_dir / cls / f"{video}_clip_{num}.mp4"
            base_frame_path = self.base_frame_dir / f"{video}.png"

            if not mp4_path.exists():
                self.skipped.append((csv_path.name, f"no clip video: {mp4_path}"))
                continue
            if not base_frame_path.exists():
                self.skipped.append((csv_path.name, f"no base frame: {base_frame_path}"))
                continue

            label_str = cls.replace("_", " ")
            if label_str not in self.label_to_idx:
                self.skipped.append((csv_path.name, f"unknown label '{label_str}'"))
                continue

            self.samples.append((csv_path, mp4_path, video, self.label_to_idx[label_str]))
            if limit is not None and len(self.samples) >= limit:
                break

    # ------------------------------------------------------------------

    def _get_base_frame_gray(self, video_stem: str) -> np.ndarray:
        cached = self._base_frame_cache.get(video_stem)
        if cached is not None:
            return cached
        path = self.base_frame_dir / f"{video_stem}.png"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"could not read base frame: {path}")
        gray = _to_gray_resized(bgr, self.img_size)
        self._base_frame_cache[video_stem] = gray
        return gray

    def _build_pose(self, csv_path: Path) -> np.ndarray:
        import pandas as pd

        df = pd.read_csv(csv_path)
        raw = df[FEATURE_COLS].values.astype(np.float32)  # (T, 14) raw soup
        return build_model_sequence(raw, target_len=POSE_SEQ_LEN)  # (29, 18)

    def _build_motion(self, mp4_path: Path, video_stem: str) -> np.ndarray:
        base_gray = self._get_base_frame_gray(video_stem)          # (S, S)
        frames = _read_clip_frames(mp4_path, FRAMES_PER_CLIP)
        motion = np.zeros((FRAMES_PER_CLIP, 1, self.img_size, self.img_size), dtype=np.float32)
        for t, frame in enumerate(frames):
            gray = _to_gray_resized(frame, self.img_size)
            diff = (gray - base_gray) / 128.0                       # roughly [-1, 1]
            motion[t, 0] = diff
        # short clip (rare, e.g. a truncated read): repeat the last motion frame
        for t in range(len(frames), FRAMES_PER_CLIP):
            motion[t, 0] = motion[max(len(frames) - 1, 0), 0]
        return motion

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        csv_path, mp4_path, video_stem, label_idx = self.samples[idx]
        pose_seq = self._build_pose(csv_path)
        motion = self._build_motion(mp4_path, video_stem)
        return torch.from_numpy(pose_seq), torch.from_numpy(motion), label_idx
