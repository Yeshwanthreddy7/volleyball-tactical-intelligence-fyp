# Two-Stream Mamba + Vision-Mamba Architecture — Spec

Transcribed from the whiteboard sketch (WhatsApp photo, 2026-09-12). Translates the
hand-drawn diagram into an exact, buildable spec. Ground truth for numbers: the sketch
itself and the existing `fyp/mamba_model.py` implementation (branch 1 already exists
in the repo under a different name — see "Mapping to existing code" below).

## What the sketch shows

Two parallel branches, both keyed on "2 pos" (2-person / pairwise player context),
feeding a fusion stage:

```
Branch 1:  2 pos --29x, 18 feature--> Mamba model --> Feature Extraction --.
                                                                             \
                                                                              Cross Attention --> Feature Fusion
                                                                             /
Branch 2:  2 pos --player movements--> Base frame - each clip - 30 frames  /
                                        --> Vision Mamba --> Feature Extraction
```

- **Branch 1** — numeric branch: 29 timesteps x 18 hand-crafted features per clip,
  encoded by a Mamba (selective-SSM) sequence model.
- **Branch 2** — visual branch: for each clip, 30 frames are diffed against that
  video's **base frame** (empty-court reference, see [[base-frames]] memory /
  `BASEFRAMES/extract_base_frames.py`) to isolate player-movement signal from the
  static court/background, then encoded by a **Vision Mamba** (frame-sequence SSM).
- Both branches produce a **Feature Extraction** output (a hidden sequence, not just
  a single vector).
- The two hidden sequences go through **Cross Attention** (each branch attends to the
  other), then **Feature Fusion** combines the attended representations into one
  vector for classification.

## Role of the base frame in this diagram

The base frame is **not** a model input by itself. It's the subtraction anchor for
branch 2's preprocessing:

```
frame_t  -  base_frame  =  motion_t      (player-only signal, court removed)
```

Wired for real in `two_stream_dataset.py`: every clip in `training_csv_v4/` is
matched by filename to its actual `fyp/dataset/<Class>/*.mp4` clip video and
its source video's base frame (`BASEFRAMES/base_frame/<video>.png`), grayscale
+ resized to 64x64 on both sides before the subtraction. Verified against all
945 real clips (`verify_real_data.py` — 0 skipped) and visually confirmed in
`sample_motion_preview.png` (players stand out, court/crowd mostly cancel).

`motion_t` for `t = 1..30` is what actually enters the Vision Mamba encoder. This
reuses the existing base-frame artifacts already built per source video
(`BASEFRAMES/base_frame/*.png`, one per video, built by per-pixel median across
~600 sampled frames — see `base-frames` memory).

## Mapping to existing code

| Sketch element                          | Existing repo code                                   | Status |
|------------------------------------------|--------------------------------------------------------|--------|
| "2 pos, 29x, 18 feature -> Mamba model"  | `fyp/mamba_model.py: MambaClassifier` (`input_dim=18`, `seq_len=29`) | **Already built, already trained** (`dl_models_v4/mamba_v4.pt`) |
| "Base frame - each clip - 30" diff       | `BASEFRAMES/extract_base_frames.py` (base frame only; the per-clip 30-frame diff step is new) | Base frame exists; diff step new |
| "Vision Mamba"                          | none                                                    | **New — built here** |
| "Feature fusion" / "Cross attention"    | none                                                    | **New — built here** |

So branch 1 is not reinvented: this design reuses `MambaBlock` from
`fyp/mamba_model.py` as the encoder for both streams (same SSM building block,
one applied to 18-dim pose vectors, one applied to visual motion-patch embeddings) —
consistent with the repo's existing pattern of swappable, contract-compatible
temporal encoders (`create_temporal_model`, spec in `classifier-framing` memory).

## New module: `two_stream_model.py`

- `VisionMambaBranch` — small conv stem patchifies each `motion_t` frame into a
  token, `d_model`-dim per frame, then stacks `MambaBlock`s along the 30-frame time
  axis (Vision-Mamba-style: SSM temporal encoder over per-frame visual tokens,
  no attention, no CUDA-specific mamba-ssm kernel needed — matches the pure-PyTorch
  reimplementation style already used for branch 1).
- `PoseMambaBranch` — thin wrapper around the existing `MambaBlock` stack, input
  `(B, 29, 18)`, returns the full hidden sequence `(B, 29, d_model)` (not pooled —
  needed as cross-attention keys/values).
- `CrossAttentionFusion` — bidirectional multi-head cross attention
  (pose queries attend to vision keys/values, vision queries attend to pose
  keys/values), each attended stream mean-pooled, concatenated, and fused through a
  linear + GELU projection back to `d_model` — the "Feature fusion" blob.
- `TwoStreamClassifier` — wraps both branches + fusion + the same
  `Dropout -> Linear -> GELU -> Dropout -> Linear` head pattern used by every
  classifier in `mamba_model.py`, so the output contract
  (`logits: (B, num_classes)`) matches the rest of the repo.

## Open params (not in the sketch, defaulted for now — confirm before training)

- Frame size fed to the conv stem: defaulted to 64x64 grayscale motion patches.
- `d_model`: defaulted to 64, matching `MambaClassifier`'s default.
- Cross-attention heads: defaulted to 4.
- No training loop / data loader included yet — this is the architecture module
  only, verified with a shape-correctness smoke test (`smoke_test.py`), not trained.
