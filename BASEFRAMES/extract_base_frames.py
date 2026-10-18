"""
Extract an empty-court "base frame" for every source video in the dataset.

For each video we want a single still image that shows only the court and its
surroundings, with no players (and ideally no moving people) visible.

Method
------
1. Sample a few hundred frames spread across the whole video and reduce each one
   to a small grayscale signature (cheap).
2. Broadcast footage is a mix of shots: the main court camera, replays, crowd,
   bench, scoreboard, etc. Cluster the signatures with KMeans and keep only the
   largest cluster -- that is almost always the main court camera.
3. Inside that cluster, keep the frames whose framing is closest to the cluster
   centre (this trims most of the pan / zoom spread).
4. Re-read those frames at full resolution and take the per-pixel median.
   Players move between positions from frame to frame, so the median keeps the
   static court and drops the people.

Output: BASEFRAMES/base_frame/<video-name>.png  (one PNG per video)

Run:
    python BASEFRAMES/extract_base_frames.py
    python BASEFRAMES/extract_base_frames.py --videos-dir fyp/Datasets --sample 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - sklearn is a repo dependency
    KMeans = None

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

# Default location of the 11 source videos, relative to the repo root.
DEFAULT_VIDEOS_DIR = Path("fyp") / "Datasets"


def list_videos(videos_dir: Path) -> list[Path]:
    vids = [p for p in sorted(videos_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    return vids


def frame_signature(frame: np.ndarray, size: int = 32) -> np.ndarray:
    """Small, illumination-normalised grayscale descriptor of a frame."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    if n > 1e-6:
        g /= n
    return g.reshape(-1)


def read_frames(cap: cv2.VideoCapture, indices, sequential: bool):
    """Yield (index, frame) for the wanted indices.

    seek mode  : jump straight to each index with CAP_PROP_POS_FRAMES. Fast, and
                 fine for healthy files.
    sequential : walk the stream frame by frame with grab() and only decode the
                 wanted ones. Needed for partially corrupt files, where seeking
                 lands in the damaged region and the decoder keeps handing back
                 the same stale frame.
    """
    want = sorted(set(int(i) for i in indices))
    if not sequential:
        for fi in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok and frame is not None:
                yield fi, frame
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    want_set = set(want)
    last = want[-1]
    pos = 0
    misses = 0
    while pos <= last:
        ok = cap.grab()
        if not ok:
            misses += 1
            if misses > 300:
                break
            pos += 1
            continue
        misses = 0
        if pos in want_set:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                yield pos, frame
        pos += 1


def dedupe_by_signature(pairs):
    """Drop frames whose small signature exactly repeats (corrupt-stream echo)."""
    out = []
    seen: set[bytes] = set()
    for fi, frame in pairs:
        key = np.round(frame_signature(frame, size=16), 3).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((fi, frame))
    return out


def pick_court_frames(
    cap: cv2.VideoCapture,
    total: int,
    n_sample: int,
    n_clusters: int,
    n_keep: int,
    sequential: bool = False,
    seed: int = 0,
) -> list[int]:
    """Return frame indices that belong to the dominant (court-camera) shot."""
    sample_idx = np.linspace(0, max(total - 1, 0), num=min(n_sample, total), dtype=int)
    sample_idx = np.unique(sample_idx)

    n_requested = len(sample_idx)
    pairs = dedupe_by_signature(read_frames(cap, sample_idx, sequential))

    sigs = [frame_signature(f) for _, f in pairs]
    kept_idx = np.array([fi for fi, _ in pairs])

    if not sigs:
        raise RuntimeError("could not read any frames")

    # Signal to the caller: distinct frames returned vs frames asked for. A
    # damaged file loses ground here (failed reads + repeated stale frames) and
    # is worth a sequential retry.
    pick_court_frames.unique_ratio = len(sigs) / max(n_requested, 1)  # type: ignore[attr-defined]

    X = np.stack(sigs)

    def spread_over_time(cand_idx: np.ndarray, keep: int) -> list[int]:
        """Pick `keep` frames spread evenly across the candidates' timeline.

        Temporal spread matters: if we grab a tight run of adjacent frames we
        might land inside one rally / time-out where players barely move, and
        the median would then keep them. Spreading across the whole match means
        every player stands in many different spots, so the median drops them.
        """
        cand_idx = np.sort(np.unique(cand_idx))
        if len(cand_idx) <= keep:
            return cand_idx.tolist()
        pick = np.linspace(0, len(cand_idx) - 1, num=keep, dtype=int)
        return cand_idx[np.unique(pick)].tolist()

    if KMeans is None or len(X) <= n_clusters:
        # Fallback: drop the frames least like the median view, spread the rest.
        centre = np.median(X, axis=0)
        d = np.linalg.norm(X - centre, axis=1)
        inliers = kept_idx[d <= np.quantile(d, 0.75)]
        return spread_over_time(inliers, n_keep)

    k = min(n_clusters, len(X))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    # Largest cluster == dominant shot (the main court camera).
    counts = np.bincount(labels, minlength=k)
    main = int(np.argmax(counts))
    mask = labels == main
    Xm = X[mask]
    idx_m = kept_idx[mask]

    # Drop the loosest 25% (replay / cutaway leakage), then spread what is left
    # across the match timeline before feeding it to the median.
    centre = km.cluster_centers_[main]
    d = np.linalg.norm(Xm - centre, axis=1)
    inliers = idx_m[d <= np.quantile(d, 0.75)]
    if len(inliers) < min(n_keep, len(idx_m)) // 2:
        inliers = idx_m
    return spread_over_time(inliers, n_keep)


def median_base_frame(
    cap: cv2.VideoCapture, frame_indices: list[int], sequential: bool = False
) -> np.ndarray:
    pairs = dedupe_by_signature(read_frames(cap, frame_indices, sequential))
    stack = [f for _, f in pairs]
    if not stack:
        raise RuntimeError("no frames read for median")
    # Uniform shape guard (framing changes should not, but be safe).
    h = min(f.shape[0] for f in stack)
    w = min(f.shape[1] for f in stack)
    stack = [f[:h, :w] for f in stack]
    arr = np.stack(stack, axis=0)
    med = np.median(arr, axis=0).astype(np.uint8)
    return med


def process_video(
    path: Path,
    out_dir: Path,
    n_sample: int,
    n_clusters: int,
    n_keep: int,
    overwrite: bool,
    sequential: str = "auto",
) -> None:
    out_path = out_dir / (path.stem + ".png")
    if out_path.exists() and not overwrite:
        print(f"  skip (exists): {out_path.name}")
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {path.name}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {path.name}: {w}x{h}, {total} frames")

    t0 = time.time()
    seq = sequential == "always"
    court_idx = pick_court_frames(cap, total, n_sample, n_clusters, n_keep, sequential=seq)
    ratio = getattr(pick_court_frames, "unique_ratio", 1.0)

    if sequential == "auto" and not seq and ratio < 0.8:
        # Too many duplicate / failed reads on the seek pass -> damaged file.
        print(f"    seek pass looks damaged (unique ratio {ratio:.2f}); retrying sequentially")
        cap.release()
        cap = cv2.VideoCapture(str(path))
        seq = True
        court_idx = pick_court_frames(cap, total, n_sample, n_clusters, n_keep, sequential=True)

    print(
        f"    dominant-shot frames selected: {len(court_idx)}"
        f"  (sequential={seq}, {time.time()-t0:.1f}s)"
    )

    med = median_base_frame(cap, court_idx, sequential=seq)
    cap.release()

    cv2.imwrite(str(out_path), med)
    print(f"    wrote {out_path}  ({time.time()-t0:.1f}s total)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="folder holding the source videos (default: <repo>/fyp/Datasets)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder for base frames (default: <this script>/base_frame)",
    )
    ap.add_argument("--sample", type=int, default=400, help="frames to sample per video for shot clustering")
    ap.add_argument("--clusters", type=int, default=8, help="KMeans clusters over frame signatures")
    ap.add_argument("--keep", type=int, default=120, help="frames fed into the per-pixel median")
    ap.add_argument("--overwrite", action="store_true", help="rewrite base frames that already exist")
    ap.add_argument(
        "--sequential",
        choices=["auto", "always", "never"],
        default="auto",
        help="frame reading: auto = seek, retry sequentially if the file looks damaged; "
        "always = sequential (slow, for corrupt files); never = seek only",
    )
    ap.add_argument("--only", default=None, help="process only videos whose name contains this substring")
    args = ap.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    videos_dir = args.videos_dir or (repo_root / DEFAULT_VIDEOS_DIR)
    out_dir = args.out_dir or (script_dir / "base_frame")
    videos_dir = videos_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not videos_dir.is_dir():
        print(f"videos dir not found: {videos_dir}")
        return 1

    vids = list_videos(videos_dir)
    if args.only:
        vids = [v for v in vids if args.only.lower() in v.name.lower()]
    if not vids:
        print(f"no videos found in {videos_dir}")
        return 1

    print(f"videos dir : {videos_dir}")
    print(f"out dir    : {out_dir}")
    print(f"found {len(vids)} videos\n")

    for i, v in enumerate(vids, 1):
        print(f"[{i}/{len(vids)}]")
        try:
            process_video(
                v, out_dir, args.sample, args.clusters, args.keep, args.overwrite, args.sequential
            )
        except Exception as e:  # keep going on the rest
            print(f"  FAILED {v.name}: {e}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
