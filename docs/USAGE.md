# Usage Guide

How to use the project once it is installed ([INSTALLATION.md](INSTALLATION.md)). All commands run from the repository root.

- [1. Analyse a match video](#1-analyse-a-match-video)
- [2. Understand the outputs](#2-understand-the-outputs)
- [3. Use a different model](#3-use-a-different-model)
- [4. Use the classifier from Python](#4-use-the-classifier-from-python)
- [5. Use the web application](#5-use-the-web-application)
- [6. Train and evaluate](#6-train-and-evaluate)
- [7. Reproduce the reported numbers](#7-reproduce-the-reported-numbers)

---

## 1. Analyse a match video

```bash
python fyp/pipeline.py "match.mp4" tactical_ml_v6.joblib \
  --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 \
  --team-split colour --tracker botsort --auto-court --court-coords linear \
  --output-video annotated.mp4 --output-csv predictions.csv
```

Useful options:

| Option | Effect |
|---|---|
| `--start-frame N --max-frames M` | Analyse only part of the video |
| `--team-side bottom\|top\|all` | Which team is analysed (default: near / bottom) |
| `--strict-render` | Cleaner video: stable jersey slots 1–6, court-gated boxes, ball hidden after a few missed frames |
| `--labels-csv fyp/labels.csv` | Show the human labels in the banner instead of the classifier (ground-truth mode) |
| `--dump-tracks tracks.npz` | Save tracks so `rebanner.py` can re-render with other models without re-detecting |
| `--skip-preflight` | Continue when both teams wear similar kits (teams then split by the net line) |

> The feature-shaping flags (`--yolo-model`, `--ball-model`, `--imgsz`, `--team-split`, `--auto-court`, `--court-coords`, `--tracker`) must match the flags the model was trained with. The values above match every model in this repository. A mismatch stops the run.

---

## 2. Understand the outputs

| Output | Contents |
|---|---|
| `annotated.mp4` | Player boxes with stable IDs, ball trail, court outline, tactic banner with confidence |
| `predictions.csv` | One row per 1-second window: frame range, tactic, confidence, entropy, anomaly flag |
| Console report | Share of time in each tactic and an overall coordination score |

Tactic classes:

| Class | Meaning |
|---|---|
| Coordinated Attack | Front-court attackers approach fast while base players hold near the net |
| Coordinated Defense | The team shifts as one rigid unit |
| Delayed Support | The nearest player reacts late after ball contact |
| Spacing Breakdown | The formation is too spread out or too clustered |
| Transition | Dead ball, too few players visible, or out of play |

A banner marked **TACTICAL DEVIATION** means confidence is below 0.5: the window does not clearly match any trained pattern.

---

## 3. Use a different model

Any model file can replace `tactical_ml_v6.joblib`:

| Location | Models |
|---|---|
| `ml_models_v6/` | 12 scikit-learn models (HGB, LightGBM, XGBoost, CatBoost, SVMs, forests, kNN, MLP, logistic regression) |
| `mamba_v4_inceptiontime.pt`, `_tcn.pt`, `_bilstm.pt`, `_mamba.pt` | Deep sequence models |
| `ALL_13_MODELS/` | 9 ML + 4 DL models in one folder |
| `BINARY_ATTACK_DEFENSE/` | 2-class Attack / Defense models (InceptionTime: 89.5% LOVO) |

To compare models on the same video without repeating detection:

```bash
python fyp/pipeline.py "match.mp4" tactical_ml_v6.joblib <same flags> --dump-tracks tracks.npz --frame-csv frames.csv
python fyp/rebanner.py --help
```

---

## 4. Use the classifier from Python

```python
import sys
import numpy as np
sys.path.insert(0, "fyp")
from interfaces import create_temporal_classifier

model = create_temporal_classifier("tactical_ml_v6.joblib")   # or any .joblib / .pt
window = np.zeros((29, 18), dtype=np.float32)                  # one 1-second feature window
result = model.classify(window)
print(result.label, result.confidence, result.probs)
```

---

## 5. Use the web application

Start it with `webapp\run_website.bat` (Windows) or `uvicorn` (see [INSTALLATION.md §6](INSTALLATION.md#6-install-the-web-application)), or use the hosted version at **https://volleyballanalyzer.dpdns.org**.

1. Open **Analyze**, choose a video (up to 500 MB) and a time range.
2. Choose the processing mode and start. Progress shows each stage.
3. When it finishes, the **Results** page offers:
   - the annotated video in several qualities (original, 720p, 480p, 360p),
   - a team tactical report and an opponent SERVE / SPIKE / BLOCK report (TXT, CSV, JSON),
   - short clips for every detected tactic.
4. **My analyses** lists past jobs. Its history code restores them on another device.

The portal pages (Overview, Method, Results, Models, Demos, Docs, Team) read live from the repository files.

---

## 6. Train and evaluate

The complete runbook, with the reason for every flag, is [TRAINING.md](TRAINING.md). Short version:

```bash
# 1. Cut 1-second clips from the labelled videos
python fyp/extract_clips.py fyp/labels.csv --video-dir fyp/Datasets --out-dir fyp/dataset

# 2. Extract 29 x 18 feature CSVs (GPU recommended)
python fyp/prepare_training_data.py fyp/dataset --output-dir training_csv_v5_full \
  --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 \
  --team-split colour --auto-court --court-coords linear --tracker botsort --clean-output

# 3. Train and evaluate the classical zoo (leave-one-video-out)
python fyp/ml_classifier.py training_csv_v5_full --output-dir ml_results_v6 --save-model tactical_ml_v6.joblib

# 4. Train and evaluate the deep models
python fyp/train_deep_lovo.py training_csv_v5_full --out-dir deep_results_v6
```

Labels file format (`fyp/labels.csv`):

```csv
video_file,anchor_time,tactic_class
usa.mp4,01:23.5,Coordinated_Attack
```

`tactic_class` must be one of `Coordinated_Attack`, `Coordinated_Defense`, `Delayed_Support`, `Spacing_Breakdown`.

---

## 7. Reproduce the reported numbers

```bash
python reproduce.py
```

The script re-runs the evaluation from the committed feature CSVs, regenerates metrics and figures in `results/latest/`, and exits with a non-zero code if any reported property does not hold.
