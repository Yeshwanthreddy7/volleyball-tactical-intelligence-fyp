# PANEL_REVIEW

Outputs generated from the whiteboard sketch (WhatsApp photo, 2026-09-12) of the
two-stream Mamba + Vision-Mamba architecture.

| File | What it is |
|---|---|
| `architecture_diagram.svg` | Clean, redrawn version of the hand-sketch (open in a browser or VS Code). |
| `architecture_spec.md` | Full written spec: what the sketch shows, base-frame's role, mapping to existing repo code, new-module design, open params. |
| `two_stream_model.py` | Runnable PyTorch implementation — `PoseMambaBranch`, `VisionMambaBranch`, `CrossAttentionFusion`, `TwoStreamClassifier`. Reuses `MambaBlock` from `finalyearproject/fyp/mamba_model.py`. |
| `smoke_test.py` / `smoke_test_output.txt` | Shape + gradient-flow check with random tensors (no real data). All pass, 205,029 total params. |
| `two_stream_dataset.py` | **Real-data loader.** Matches every `training_csv_v4/*.csv` clip to its actual `fyp/dataset/<Class>/*.mp4` video and its source video's `BASEFRAMES/base_frame/*.png`, purely by filename. Builds branch-1 pose input via the existing `features.build_model_sequence()` and branch-2 motion input as `frame_t - base_frame` for all 30 frames/clip. |
| `verify_real_data.py` / `verify_real_data_output.txt` | Runs the real loader end to end through `TwoStreamClassifier` (forward + backward). **945/945 clips matched, 0 skipped.** |
| `sample_motion_preview.png` | Visual proof: raw frame \| base frame \| `\|diff\|` motion signal, side by side, for one real clip — confirms the subtraction actually isolates players. |

## Quick facts

- Branch 1 (pose, 29x18 -> Mamba) is **not new** — it's the existing
  `MambaClassifier` encoder in `finalyearproject/fyp/mamba_model.py`
  (already trained: `dl_models_v4/mamba_v4.pt`).
- Branch 2 (Vision Mamba) and the Cross Attention / Feature Fusion stage are
  **new**, built here to match the sketch.
- Base frame's role: subtraction anchor for branch 2
  (`frame_t - base_frame = motion_t`), not a direct model input. Uses the
  per-video base frames already built by `BASEFRAMES/extract_base_frames.py`.
- Real-data pipeline is wired and verified: every clip in `training_csv_v4`
  (945 of 945) resolves to its actual `fyp/dataset/<Class>/*.mp4` clip video
  and its source video's base frame in `BASEFRAMES/base_frame/`. No synthetic
  data needed.
- Status: architecture + real-data loader both verified end to end (forward +
  backward pass on a real batch). **Not trained yet** — no training loop /
  optimizer wired up in this folder (that's the natural next step).

## Run it yourself

```
cd PANEL_REVIEW
python smoke_test.py          # random-tensor shape/gradient check
python verify_real_data.py    # real clips + base frames, end to end
```
