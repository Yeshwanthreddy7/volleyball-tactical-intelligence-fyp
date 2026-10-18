# FINAL_OUTPUT: annotated videos of videoplayback (7) for 13 tactical models

**Source:** `fyp/Datasets/videoplayback (7).mp4` (Paris 2024, Japan vs Argentina)
**Segment:** 03:00 to 08:00, which is 5 minutes (9,000 frames at 30 fps). This is the same window as the saved 13-model predictions.
**Models:** 13 tactical classifiers, 9 machine-learning (ML) and 4 deep-learning (DL). All were trained on the hand-labelled clips in `fyp/dataset/`:

- Coordinated_Attack: 376 clips
- Coordinated_Defense: 368 clips
- Delayed_Support: 109 clips
- Spacing_Breakdown: 118 clips

## Folder contents

| Path | Contents |
|---|---|
| `videos/*.mp4` | 13 annotated videos, one per model, 1760x832 H.264 |
| `videos/videoplayback(7)_bicausal.events.csv` | Every detected ball touch: time, action, team, player track id and both teams' formations |
| `accuracy/model_accuracy.csv` | Accuracy table for all 13 models, sorted by test accuracy |
| `accuracy/model_accuracy_chart.png` | Chart of test accuracy and macro-F1 |
| `code/` | Scripts that produced everything in this folder (see "Re-running") |

## What each video shows

The video layout follows Fig. 1(b) of Zhang et al., *Bi-Causal: Group Activity Recognition via Bidirectional Causality*, CVPR 2024.

**Main frame: player relations (R)**
- Far team (Japan) is drawn in red and the near team (Argentina) in blue. Each player has a dot on the body and a ring at the feet.
- Thick white arrows form the team's relation graph. This graph is a minimum spanning tree built from player positions on the court, in metres. For the team touching the ball, the arrows point away from the player on the ball, which shows the interaction driving the team shape (I->R).
- Thin lines connect every pair of teammates.
- Each team's formation is shown in its corner, for example "3-Front 3-Back". Front row means within 3.5 m of the net.

**Main frame: human-object interaction (I)**
- The ball is circled and has a short trail behind it.
- A star marks the player touching the ball, and a dashed line joins the ball to that player's hand.
- A curved yellow arrow shows the ball's flow from the previous touch to the current one.
- The action name appears in a large banner and in a small tag above the player: SERVE, RECEPTION, PASS, SET, SPIKE, BLOCK, DIG or FREE BALL.

**Right panel**
- The panel copies the paper's t1/t2/t3 layout for the last three touches.
- Each row pairs a relation snapshot (R) with a close-up of the player and ball (I).
- Red arrows mark the R->I causal effect and blue arrows mark the I->R causal effect.

**Bottom strip: the model's tactic call**
- The model name and whether it is ML or DL.
- The model's held-out test accuracy and macro-F1.
- The model's call for the near team in the current 1-second window, with its confidence. The near team is also outlined in the tactic colour.
- The 13-model majority vote for the same window.
- A 5-minute timeline of the model's calls, with a playhead:
  - red: Attack
  - blue: Defense
  - amber: Delayed Support
  - purple: Spacing Breakdown

## Accuracy

Accuracy is measured with **leave-one-video-out (LOVO)** testing. Each model trains on 7 source videos and is tested on the 8th, repeated for all 8 videos (945 clips in total). Every test clip therefore comes from a match the model never saw.

| Rank | Model | Type | Test accuracy | Test macro-F1 | Train accuracy |
|---|---|---|---|---|---|
| 1 | HistGradientBoosting | ML | **66.8%** | 0.510 | 100.0% |
| 2 | ExtraTrees | ML | 64.9% | 0.447 | 100.0% |
| 3 | InceptionTime | DL | 64.9% | **0.559** | 79.7% |
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

For comparison, always guessing the most common class (Coordinated Attack) scores 39.0%.

**How to read these numbers**
- **Accuracy differs by class.** Coordinated Attack and Coordinated Defense are recognised well, with per-class F1 of about 0.72–0.79 for HistGradientBoosting and InceptionTime. Delayed Support and Spacing Breakdown are weak, with F1 of about 0.2–0.4. Those two classes have only about 110 clips each and are subtle to label. Per-class F1 values are in `accuracy/model_accuracy.csv` for the 4 DL models and HistGradientBoosting; the saved results do not include them for the other 8 ML models.
- **Most ML models overfit.** They score 100% on their own training clips, so the test columns are the ones to trust.
- **There is no accuracy on videoplayback (7).** Nobody hand-labelled this video. For vp(7), the CSV reports each model's class split and how often it agrees with the 13-model majority vote. That agreement measures consistency between models, not correctness.

## Limits

- **Actions are rule-based.** The action names come from a detected sudden change in the ball's path next to a player, the volleyball touch order and a pose check for overhand contact. No trained action model is used.
  - In these 5 minutes: 141 touches (SET 61, SPIKE 30, PASS 19, DIG 13, FREE BALL 11, SERVE 4, RECEPTION 2, BLOCK 1).
  - SET is over-called.
  - SERVE and RECEPTION are under-called, because the serve is often out of frame or the ball is not detected.
- **The ball is detected in 65% of frames.** A touch hidden from the camera can be missed.
- **Team colour can flip briefly** when a player jumps at the net or the player tracker switches ids.
- **The tactic calls apply to the near team** (the bottom team in the image). This matches how the models were trained.
- **The HistGradientBoosting deployment gate is not applied.** It would mark 20 of 309 windows as "Unclassified", but the video shows the model's raw call for those windows.
- **The court model is fixed.** It uses court points measured by hand for this fixed camera. A replay guard pauses the overlays if the view changes, but no camera change occurs in this segment.

## Re-running

Run these commands from the `OUTPUTS/` folder. The `code/` folder here holds copies of the same scripts.

```bash
# 1. Detection (GPU, about 15 min): YOLO11m-pose + BoT-SORT for players, fyp/volleyball_best.pt for the ball
python bicausal_extract.py "../fyp/Datasets/videoplayback (7).mp4" --start 180 --seconds 300 --out cache_vp7.pkl

# 2. The 13 annotated videos (CPU, about 50 min, all 13 written in one pass)
python render_13_models.py cache_vp7.pkl --out-dir ../FINAL_OUTPUT/videos

# 3. Accuracy table and chart
python make_accuracy_report.py
```

The per-window predictions of the 13 models come from `vp7_3to8_results/vp7_3to8_all_models.csv`. That file holds the saved outputs of the trained models:

- ML models: `ml_models_all/*.joblib`
- DL models: `dl_models_v4/*.pt` and `mamba_v4_*.pt`
