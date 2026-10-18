# FINALE_OUTPUTS: tactic videos of videoplayback (7) for 13 models

- **Video:** `fyp/Datasets/videoplayback (7).mp4` (Paris 2024, Japan vs Argentina).
- **Segment:** 03:00 to 08:00, which is 5 minutes (9,000 frames at 30 fps). This is the part of the match where predictions from all 13 models are saved.
- **Models:** 9 machine-learning (ML) and 4 deep-learning (DL) tactical classifiers. All were trained on the hand-labelled clips in `fyp/dataset/`.
- **Predictions:** each model predicts one of four tactics for every 1-second window:
  - Coordinated Attack
  - Coordinated Defense
  - Delayed Support
  - Spacing Breakdown

## Folder contents

| Path | Contents |
|---|---|
| `videos/NN_<Model>_<ML/DL>.mp4` | 13 annotated videos (1680x784, H.264). NN is the model's rank by test accuracy. |
| `accuracy/model_accuracy.csv` | Accuracy of all 13 models, plus each model's class split on this segment |
| `accuracy/model_accuracy_chart.png` | Chart of test accuracy and macro-F1 |
| `ball_track_vp7.npz` | Ball position for every frame (`pos`, `size`, and `src`: 0 = not visible, 1 = detected, 2 = bridged) |
| `code/` | The scripts that made this folder |

## What each video shows

**Players**
- Every player has a bounding box.
- Japan (far side) is red and Argentina (near side) is blue.
- The team is decided by jersey colour (Japan red, Argentina black), voted over time per tracked player. The white / light-blue liberos use their side of the net instead.
- Officials, coaches, photographers and the floor crew are filtered out.

**Team graphs**
- Each team has its own graph in its own colour: red lines connect the Japan players and blue lines connect the Argentina players.
- A node sits on each player.
- Edges are the minimum spanning tree of the team plus each player's 2 nearest teammates. Distances are measured in metres on the court, not in pixels.

**Ball**
- The ball has a yellow bounding box and a trail.
- It is tracked offline over the whole segment (see "Ball tracking").

**Tactic (this is the only thing that changes between the 13 videos)**
- The model's call for the Argentina (near) team for the current second is shown as a coloured banner and a tinted outline around the team:
  - orange: Coordinated Attack
  - green: Coordinated Defense
  - pink: Delayed Support
  - purple: Spacing Breakdown
- The side panel shows:
  - the model name and whether it is ML or DL
  - its rank
  - its test accuracy and macro-F1
  - its confidence for this second
  - the 13-model majority vote for this second
  - the class definitions
- The bottom strip is a 5-minute timeline of all the model's calls, with a playhead.

## Accuracy

Accuracy is measured with **leave-one-video-out (LOVO)** testing. Each model is trained on 7 source videos and tested on the 8th, repeated for all 8 videos (945 labelled clips). The test clips always come from a match the model never saw. Guessing the most common class every time scores 39.0%.

| Rank | Model | Type | Test accuracy | Test macro-F1 | Train accuracy |
|---|---|---|---|---|---|
| 1 | HistGradientBoosting | ML | **66.8%** | 0.510 | 100.0% |
| 2 | InceptionTime | DL | 64.9% | **0.559** | 79.7% |
| 3 | ExtraTrees | ML | 64.9% | 0.447 | 100.0% |
| 4 | GradientBoosting | ML | 64.8% | 0.489 | 100.0% |
| 5 | RBF-SVM | ML | 64.2% | 0.496 | 100.0% |
| 6 | RandomForest | ML | 63.9% | 0.415 | 100.0% |
| 7 | MLP | ML | 61.1% | 0.465 | 100.0% |
| 8 | k-NN | ML | 60.0% | 0.436 | 100.0% |
| 9 | Linear SVM | ML | 59.2% | 0.470 | 91.4% |
| 10 | LogisticRegression | ML | 58.5% | 0.490 | 81.1% |
| 11 | TCN | DL | 49.0% | 0.426 | 68.1% |
| 12 | BiLSTM+Attention | DL | 48.9% | 0.411 | 63.8% |
| 13 | Mamba (SSM) | DL | 47.1% | 0.399 | 71.5% |

InceptionTime and ExtraTrees tie at 64.9% accuracy. The tie is broken by macro-F1.

**How to read these numbers**
- Coordinated Attack and Coordinated Defense are recognised well, with per-class F1 of about 0.72–0.79 for the best models.
- Delayed Support and Spacing Breakdown are weak, with F1 of about 0.2–0.4. Each of these classes has only about 110 training clips.
- Most ML models score 100% on their own training clips, so the test columns are the ones to trust.
- **There is no accuracy on videoplayback (7) itself.** No one hand-labelled this video, so its calls cannot be scored. `model_accuracy.csv` reports how often each model agrees with the 13-model majority on this segment. That agreement measures consistency between models, not correctness.

## Ball tracking

The volleyball detector (`fyp/volleyball_best.pt`) runs on every frame at a low confidence threshold (0.02). A low threshold is needed because fast, motion-blurred balls score only 0.05–0.10. The track is then built offline in five steps:

1. **Fixed false positives are removed.** These are spots that fire in thousands of frames: the Olympic logo on the net banner, spots in the crowd and the scoreboard.
2. **The remaining detections are linked into short tracklets** that move consistently from frame to frame. Tracklets that jitter like noise are dropped, because real ball flight is smooth.
3. **Dynamic programming picks the best chain of tracklets** over the 5 minutes. Consecutive tracklets must be reachable at real ball speed, so the marker can never jump to a false detection.
4. **Gaps of up to 1.2 s are bridged** with a smooth curve that matches the ball's speed and direction at both ends.
5. **When the ball really cannot be seen for longer, the marker disappears.** This happens between points, or when the ball leaves the top of the frame. The marker never freezes in place.

## Limits

- **The tactic calls come from the saved outputs of the trained models** in `vp7_3to8_results/vp7_3to8_all_models.csv`. The ML models are in `ml_models_all/*.joblib`; the DL models are in `dl_models_v4/*.pt` and `mamba_v4_*.pt`.
- **The calls apply to the near team (Argentina)**, which is the side the models were trained to read.
- **The HistGradientBoosting deployment gate is not applied.** It would mark 20 of 309 windows as "Unclassified", but the video shows the raw call.
- **Some players can be missed briefly.** This happens when a player is hidden behind another at the net, or is chasing the ball just outside the side lines.

## Re-running

Run these commands from `finalyearproject/OUTPUTS/`. The `code/` folder holds copies of the same scripts.

```bash
# 1. Player detection + pose + tracking (GPU, about 15 min)
python bicausal_extract.py "../fyp/Datasets/videoplayback (7).mp4" --start 180 --seconds 300 --out cache_vp7.pkl

# 2. Ball detection at low confidence (GPU, about 3 min)
python ball_redetect.py cache_vp7.pkl --conf 0.02

# 3. The 13 videos, all written in one pass (CPU, about 55 min)
python final_render.py cache_vp7.pkl --out-dir ../FINALE_OUTPUTS/videos

# 4. Accuracy table and chart
python make_accuracy_report.py
```
