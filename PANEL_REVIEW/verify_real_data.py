"""
verify_real_data.py -- proves two_stream_dataset.py + two_stream_model.py work
together on REAL clips (pose CSV + clip video + BASEFRAMES base frame), not
just the random tensors in smoke_test.py.

1. Builds the dataset index over training_csv_v4 / fyp/dataset / BASEFRAMES,
   reports how many clips matched all three files vs were skipped and why.
2. Pulls a real batch, runs it through TwoStreamClassifier end to end.
3. Saves a visual proof image (raw frame | base frame | motion diff) for one
   sample so the base-frame subtraction can be eyeballed, not just trusted.
"""

import numpy as np
import cv2
import torch

from two_stream_dataset import TwoStreamVolleyballDataset, IMG_SIZE
from two_stream_model import TwoStreamClassifier


def main() -> None:
    print("=== Building dataset index (training_csv_v4 x fyp/dataset x BASEFRAMES) ===")
    ds = TwoStreamVolleyballDataset()
    print(f"  matched samples : {len(ds)}")
    print(f"  skipped         : {len(ds.skipped)}")

    if ds.skipped:
        reasons: dict[str, int] = {}
        for _, reason in ds.skipped:
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print("  skip reasons:")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason}: {n}")

    # per-class coverage
    counts: dict[int, int] = {}
    for _, _, _, label_idx in ds.samples:
        counts[label_idx] = counts.get(label_idx, 0) + 1
    from mamba_model import LABEL_NAMES
    print("  per-class matched counts:")
    for idx, name in enumerate(LABEL_NAMES):
        print(f"    [{idx}] {name:20s}: {counts.get(idx, 0)}")

    assert len(ds) > 0, "no real samples matched -- check repo paths"

    print("\n=== Pulling a real batch (4 samples) through the dataset ===")
    batch_pose, batch_motion, batch_label = [], [], []
    sample_idxs = np.linspace(0, len(ds) - 1, num=4, dtype=int)
    for i in sample_idxs:
        pose, motion, label = ds[int(i)]
        batch_pose.append(pose)
        batch_motion.append(motion)
        batch_label.append(label)
        csv_path, mp4_path, video_stem, _ = ds.samples[int(i)]
        print(f"  [{i}] {csv_path.name}  <-  {mp4_path.name}  (base: {video_stem}.png)"
              f"  pose {tuple(pose.shape)}  motion {tuple(motion.shape)}")

    pose_batch = torch.stack(batch_pose)        # (4, 29, 18)
    motion_batch = torch.stack(batch_motion)    # (4, 30, 1, 64, 64)
    label_batch = torch.tensor(batch_label)

    print("\n=== Forward pass through TwoStreamClassifier ===")
    model = TwoStreamClassifier(num_classes=len(__import__("mamba_model").LABEL_NAMES))
    logits = model(pose_batch, motion_batch)
    assert logits.shape == (4, len(__import__("mamba_model").LABEL_NAMES))
    print(f"  logits shape: {tuple(logits.shape)}  OK")
    loss = torch.nn.functional.cross_entropy(logits, label_batch)
    print(f"  cross-entropy loss on real batch: {loss.item():.4f}")
    loss.backward()
    print("  backward() OK on real data")

    print("\n=== Saving visual proof of the base-frame diff (one sample) ===")
    csv_path, mp4_path, video_stem, label_idx = ds.samples[int(sample_idxs[0])]
    cap = cv2.VideoCapture(str(mp4_path))
    ok, raw_frame = cap.read()
    cap.release()
    assert ok, f"could not read first frame of {mp4_path}"

    base_bgr = cv2.imread(str(ds.base_frame_dir / f"{video_stem}.png"))
    base_resized = cv2.resize(base_bgr, (raw_frame.shape[1], raw_frame.shape[0]))

    raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(base_resized, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(raw_gray, base_gray)
    diff_vis = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)

    h, w = raw_frame.shape[:2]
    strip = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    strip[:, :w] = raw_frame
    strip[:, w + 10:2 * w + 10] = base_resized
    strip[:, 2 * w + 20:] = diff_vis
    for x, label in ((10, "raw frame"), (w + 20, "base frame"), (2 * w + 30, "|diff| (motion signal)")):
        cv2.putText(strip, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    out_path = "sample_motion_preview.png"
    cv2.imwrite(out_path, strip)
    print(f"  wrote {out_path}  (clip: {mp4_path}, label: {label_idx})")

    print("\nALL REAL-DATA CHECKS PASSED")


if __name__ == "__main__":
    main()
