import os
import subprocess
import shutil
from datetime import timedelta

import pandas as pd


CANONICAL_CLASSES = {
    "coordinated_attack": "Coordinated_Attack",
    "coordinated_defense": "Coordinated_Defense",
    "coordinated_defence": "Coordinated_Defense",
    "delayed_support": "Delayed_Support",
    "spacing_breakdown": "Spacing_Breakdown",
}


def _parse_time_to_seconds(anchor_time_str: str) -> float:
    """Accept SS(.s), MM:SS(.s) or HH:MM:SS(.s)."""
    parts = anchor_time_str.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid time format: {anchor_time_str}")

    # Left-pad missing fields with zeros: "18:07.5" -> ["0", "18", "07.5"].
    parts = ["0"] * (3 - len(parts)) + parts
    hh = int(parts[0])
    mm = int(parts[1])
    sec = float(parts[2])
    return hh * 3600 + mm * 60 + sec


def _format_seconds_for_ffmpeg(total_seconds: float) -> str:
    total_seconds = max(0.0, total_seconds)
    td = timedelta(seconds=total_seconds)
    total = td.total_seconds()
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _normalize_tactic(raw_tactic: str) -> str:
    key = raw_tactic.strip().lower().replace(" ", "_")
    return CANONICAL_CLASSES.get(key, raw_tactic.strip())


def _resolve_ffmpeg_executable() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def build_dataset_pipeline(
    labels_path: str = "labels.csv",
    video_dir: str | None = None,
    out_dir: str | None = None,
    fps: int = 30,
) -> None:
    print("Initializing Dataset Extraction Pipeline...\n")

    ffmpeg_exe = _resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        print(
            "Error: ffmpeg is not available. Install ffmpeg or run `pip install imageio-ffmpeg`."
        )
        return

    labels_path = os.path.abspath(labels_path)
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found.")
        return

    root_dir = os.path.dirname(labels_path)
    # video_file entries resolve against --video-dir (default: the labels.csv
    # folder); clips are written under --out-dir/<Class>/ (default:
    # <labels.csv folder>/dataset).
    video_root = os.path.abspath(video_dir) if video_dir else root_dir
    dataset_dir = os.path.abspath(out_dir) if out_dir else os.path.join(root_dir, "dataset")

    df = pd.read_csv(labels_path)
    required_cols = {"video_file", "anchor_time", "tactic_class"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: labels.csv is missing columns: {sorted(missing)}")
        return

    for class_name in sorted(set(CANONICAL_CLASSES.values())):
        os.makedirs(os.path.join(dataset_dir, class_name), exist_ok=True)

    print(f"Verified folder structure in {dataset_dir}")

    success_count = 0
    skip_count = 0

    for index, row in df.iterrows():
        raw_video_file = str(row["video_file"]).strip()
        anchor_time = str(row["anchor_time"]).strip()
        raw_tactic = str(row["tactic_class"]).strip()
        tactic = _normalize_tactic(raw_tactic)

        if tactic not in CANONICAL_CLASSES.values():
            print(f"Warning: Unknown tactic '{raw_tactic}' on row {index + 1}. Skipping.")
            skip_count += 1
            continue

        if os.path.isabs(raw_video_file):
            video_path = raw_video_file
        else:
            video_path = os.path.join(video_root, raw_video_file)
            if not os.path.exists(video_path):
                # tolerate a bare filename when video_file also carries a
                # sub-path (or vice versa)
                alt = os.path.join(video_root, os.path.basename(raw_video_file))
                if os.path.exists(alt):
                    video_path = alt

        if not os.path.exists(video_path):
            print(f"Warning: video not found '{video_path}' for row {index + 1}. Skipping.")
            skip_count += 1
            continue

        try:
            start_seconds = _parse_time_to_seconds(anchor_time)
        except ValueError as exc:
            print(f"Warning: {exc} on row {index + 1}. Skipping.")
            skip_count += 1
            continue

        if tactic == "Coordinated_Defense":
            start_seconds -= 0.5

        start_str = _format_seconds_for_ffmpeg(start_seconds)
        clip_name = f"{os.path.splitext(os.path.basename(raw_video_file))[0]}_clip_{index + 1:03d}.mp4"
        clip_path = os.path.join(dataset_dir, tactic, clip_name)

        # Re-encode, do NOT stream-copy (-c copy): a copy cut whose -ss lands
        # off a keyframe produces a container with no valid moov atom -
        # cv2.VideoCapture then cannot open it at all (AUDIT_FINDINGS.md D8,
        # 4 clips lost this way). libx264 -crf 18 is visually lossless for a
        # 1 s analysis clip and always writes a seekable file.
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss",
            start_str,
            "-i",
            video_path,
            "-t",
            "00:00:01.000",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-an",
            # Normalise every clip to the same frame rate. Source videos mix
            # 30 and 50 fps; without this the fixed 29-frame training window
            # spans 0.97 s on one video and 0.58 s on another, and per-frame
            # speed features drift by source - a leak the LOVO split exists
            # to expose.
            "-r",
            str(fps),
            clip_path,
        ]

        print(f"[{index + 1}/{len(df)}] {tactic} at {start_str} -> {clip_path}")
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode == 0:
            success_count += 1
        else:
            print(f"Warning: ffmpeg failed on row {index + 1}.")
            skip_count += 1

    print(f"\nPipeline complete. Extracted: {success_count}, Skipped: {skip_count}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Cut 1-second labelled clips from source videos using a "
                    "manual labels.csv (video_file, anchor_time, tactic_class). "
                    "anchor_time may be SS, MM:SS or HH:MM:SS (fractional "
                    "seconds ok).",
    )
    ap.add_argument("labels_csv", nargs="?", default="labels.csv",
                    help="Path to labels.csv (default: ./labels.csv).")
    ap.add_argument("--video-dir", default=None,
                    help="Folder holding the source videos named in video_file "
                         "(default: the labels.csv folder).")
    ap.add_argument("--out-dir", default=None,
                    help="Where the <Class>/ clip folders are written "
                         "(default: <labels.csv folder>/dataset).")
    ap.add_argument("--fps", type=int, default=30,
                    help="Normalise every clip to this frame rate (default 30). "
                         "Source videos mix 30/50 fps; a fixed rate keeps the "
                         "29-frame training window a constant real duration.")
    args = ap.parse_args()
    build_dataset_pipeline(args.labels_csv, args.video_dir, args.out_dir, args.fps)