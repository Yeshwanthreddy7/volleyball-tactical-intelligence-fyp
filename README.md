# Volleyball Tactical Intelligence

**Automated team-tactic analysis from a single broadcast camera.**

This project takes raw volleyball match video and produces a second-by-second tactical reading of each team. It detects and tracks every player and the ball, projects them onto the court, separates the two teams, and builds a formation descriptor. A learned classifier then labels every 1-second window with the team's collective tactic.

Final Year Project, Group 56.

<p align="center">
  <img src="FINALE_OUTPUTS/accuracy/model_accuracy_chart.png" width="780" alt="Cross-video accuracy of the 13 tactical classifiers">
</p>

---

## Highlights

- **End-to-end pipeline** from video to annotated MP4, per-window CSV and coordination report. It uses YOLOv11 detection, BoT-SORT tracking, automatic court calibration and jersey-colour team separation.
- **Human-labelled dataset** of 948 one-second clips from 8 international matches, in 4 tactical classes.
- **25+ trained models.** There are 12 scikit-learn models (including XGBoost, LightGBM and CatBoost) and 4 deep sequence models (InceptionTime, TCN, BiLSTM+Attention and Mamba SSM). A 2-class Attack/Defense variant was also trained.
- **Honest evaluation.** Every headline number is leave-one-video-out (LOVO): the test match is never seen during training.
- **Full-stack web app** (FastAPI + React + MongoDB). You upload a match, a GPU worker analyses it, and you get the annotated video back.
- **Reproducible.** `python reproduce.py` regenerates the metrics and figures, and exits non-zero if any reported property fails.

## Results at a glance

All numbers are LOVO on 948 clips from 8 source videos. The majority-class baseline scores 39.0% accuracy.

### 4-class tactic classifier

| Model | Type | Accuracy | Macro-F1 | Cohen κ |
|---|---|---|---|---|
| **HistGradientBoosting** (deployed, `tactical_ml_v6.joblib`) | ML | **67.0%** | 0.510 | 0.498 |
| LightGBM | ML | 67.0% | 0.507 | 0.500 |
| XGBoost | ML | 66.7% | 0.519 | 0.498 |
| CatBoost | ML | 64.7% | **0.520** | 0.478 |
| RBF-SVM | ML | 64.6% | 0.500 | 0.466 |
| InceptionTime | DL | 61.9% | 0.515 | – |
| TCN | DL | 51.2% | 0.441 | – |
| Mamba SSM | DL | 46.1% | 0.390 | – |
| BiLSTM + Attention | DL | 45.9% | 0.398 | – |
| Majority class | baseline | 39.0% | 0.140 | 0.000 |
| Hand-coded rule engine | baseline | 10.1% | 0.090 | 0.005 |

HistGradientBoosting was chosen as the deployed model **before** the results were seen. It was not picked as the best of many runs. The Mamba SSM is kept as a comparison arm.

Per-class accuracy shows where the difficulty is:

| Class | Clips | LOVO accuracy |
|---|---|---|
| Coordinated Attack | 370 | 69% |
| Coordinated Defense | 355 | 69% |
| Delayed Support | 108 | 38% |
| Spacing Breakdown | 115 | 38% |

### 2-class Attack / Defense classifier

The two minority classes are sparse and subjective. When the task is reduced to the two well-defined classes, accuracy rises sharply (723 clips, majority baseline 51%):

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| **InceptionTime** | **89.5%** | 0.895 | 0.957 |
| TCN | 88.0% | 0.879 | 0.950 |
| BiLSTM + Attention | 87.8% | 0.878 | 0.942 |
| Mamba SSM | 84.8% | 0.847 | 0.924 |
| Best ML (RBF-SVM) | 83.7% | 0.837 | 0.921 |

A weighted ensemble of InceptionTime, BiLSTM and TCN reaches **0.906 LOVO macro-F1**. It annotated a full unseen 45-minute match (`v7_tactical/`), with HMM smoothing and scoreboard-based rally-break detection.

Full tables: [FINALE_OUTPUTS/README.md](FINALE_OUTPUTS/README.md), [BINARY_ATTACK_DEFENSE/README.md](BINARY_ATTACK_DEFENSE/README.md), `ml_results_v6/`, `deep_results_v6/`.

---

## Tactical classes

| # | Class | What it means |
|---|---|---|
| 1 | Coordinated Attack | Front-court attackers approach fast while base players hold near the net |
| 2 | Coordinated Defense | The team shifts as one rigid unit |
| 3 | Delayed Support | The nearest player reacts late after ball contact |
| 4 | Spacing Breakdown | The formation is too spread out or too clustered |
| 0 | Transition | Dead ball, too few players visible, or out of play |

---

## How it works

```
broadcast video (1280x720, 30 fps)
   │
   ├─ Preflight gate      checks the player population, aborts on bad input     fyp/preflight.py
   ▼
Detection                players: YOLOv11 (COCO)  ·  ball: fine-tuned YOLOv11   fyp/interfaces.py
   ▼                     both at imgsz 1280 (the ball is ~15 px)
Court calibration        auto court-corner detection, camera-motion              fyp/court.py
   ▼                     compensation, in-court mask                             BASEFRAMES/
Tracking                 BoT-SORT + persistent player slots across occlusion     fyp/tracking.py, identity.py
   ▼
Team separation          jersey-colour k-means, labelled by median court side    fyp/teams.py
   ▼
Features                 18 permutation-invariant channels per frame             fyp/features.py
   ▼                     (centroid, spread, hull, spacing, speeds, sync, ball)
Windowing                29-frame (≈1 s) windows → (29 × 18) tensor
   ▼
Classifier               scikit-learn (.joblib)  or  PyTorch (.pt)               fyp/ml_classifier.py
   ▼                     same interface, chosen by file extension                fyp/mamba_model.py
Output                   annotated MP4 · per-window CSV · coordination report
```

<p align="center">
  <img src="figures_workflow/workflow_hgb.png" width="780" alt="Workflow of the deployed HistGradientBoosting model">
</p>

### Key design decisions

- **Two detectors.** Stock YOLOv11 finds 14–23 players per frame. A small fine-tune collapsed to 0.3 players per frame on unseen footage. The fine-tune is much better for the ball: 67% recall against 16% for the stock model. So players come from stock weights and the ball from the fine-tune.
- **Resolution matters.** With the same weights, ball recall goes from 15% at 640 px to 77% at 1280 px.
- **Teams by colour and geometry.** Colour is reliable from frame to frame but carries no team meaning. Court side is reliable on average but ambiguous at the net. Each colour cluster gets its team from its members' median court side. `k = 4` clusters, because each team also has a libero in a different colour. Result: 10.7 → 5.8 players per frame (a team has 6).
- **Scale-invariant features.** Without a calibrated homography, absolute distances leak camera identity. Dimensionless ratios and per-clip z-scored shapes lower within-video scores but raise cross-video scores. That trade is the one that matters.
- **Permutation invariance.** A formation is a *set* of players, so the descriptor does not depend on player order.

The full mathematical formulation is in [METHODOLOGY.md](METHODOLOGY.md).

---

## Web application

`webapp/` is a complete site that runs the research pipeline as a service.

| Part | Stack |
|---|---|
| Frontend | React 18 + Vite (documentation portal, upload and results UI) |
| API | FastAPI + Uvicorn |
| Database | MongoDB 7 (falls back to a JSON file if MongoDB is down) |
| Engine | `OUTPUTS/bicausal_extract.py` + `NEW_OUTPUTS/code/new_render.py` + `tactical_ml_v6.joblib` |
| Worker | One GPU job at a time; the queue survives restarts |
| Access | Cloudflare Tunnel, PWA install, Windows desktop app and installer |

Quick start on Windows:

```bash
pip install -r webapp/backend/requirements.txt
webapp\run_website.bat          # then open http://localhost:8000
```

Docker, public link and installer instructions are in [webapp/DEPLOY.md](webapp/DEPLOY.md). Analysis needs an NVIDIA GPU: on an RTX 3050 laptop, 5 minutes of video takes about 35 minutes.

---

## Getting started

### 1. Install

```bash
git clone https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp.git
cd volleyball-tactical-intelligence-fyp
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r fyp/requirements.txt
```

For GPU support, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Analyse a match

```bash
python fyp/pipeline.py "match.mp4" tactical_ml_v6.joblib \
  --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 \
  --team-split colour --tracker botsort --auto-court --court-coords linear \
  --output-video annotated.mp4 --output-csv predictions.csv
```

Any model in `ml_models_v6/`, `ALL_13_MODELS/` or `BINARY_ATTACK_DEFENSE/` can replace `tactical_ml_v6.joblib`. To load one in Python:

```python
import sys; sys.path.insert(0, "fyp")
from interfaces import create_temporal_classifier
model = create_temporal_classifier("ml_models_v6/hist_gradient_boosting.joblib")  # or a .pt file
```

### 3. Reproduce the metrics

```bash
python reproduce.py
```

### 4. Train from scratch

> **Important:** the feature flags (`--yolo-model`, `--imgsz`, `--team-split`, `--auto-court`, `--court-coords`, `--tracker`) must be identical in extraction and inference. If they differ, the model fails silently. See [TRAINING.md](TRAINING.md).

```bash
bash run_shards.sh 4                                                   # extract features from the clips
python fyp/ml_classifier.py training_csv_v5_full --features invariant \
  --save-model tactical_ml_v6.joblib                                   # train and evaluate the ML zoo (LOVO)
python fyp/train_deep_lovo.py training_csv_v5_full --out-dir deep_results_v6   # deep models (InceptionTime, TCN, BiLSTM, Mamba)
```

GPU notebooks: `kaggle_retrain_after_detector_fix.ipynb` and `colab_train_volleyball_yolov11.ipynb` (ball detector fine-tune).

### 5. Run the tests

```bash
python -m pytest fyp/tests -q
```

Many tests replay real failures the project hit (half-court homography, slot starvation, detector blindness, crowd-captured colour clusters). Most run without a GPU.

---

## Repository map

| Path | Contents |
|---|---|
| `fyp/` | Core library: pipeline, detection, court, tracking, teams, features, classifiers, training, tests |
| `webapp/` | FastAPI + React + MongoDB website, desktop app and Windows installer |
| `reproduce.py` | One command to regenerate the metrics and figures |
| `training_csv*/` | Extracted per-clip feature CSVs (the training data) |
| `ml_models_v6/`, `tactical_ml_v6.joblib` | Latest 12 scikit-learn models and the deployed model |
| `mamba_v4_*.pt` | Deep models (InceptionTime, TCN, BiLSTM, Mamba) |
| `ALL_13_MODELS/` | One-stop copy of 13 classifiers with a loader example |
| `BINARY_ATTACK_DEFENSE/` | 2-class Attack/Defense models and results |
| `ml_results_v*/`, `deep_results_v*/` | Metrics JSON and training logs per run |
| `FINALE_OUTPUTS/` | 13-model comparison on an unseen match, with code and accuracy |
| `v7_tactical/` | Full-match Attack/Defense annotation with HMM smoothing |
| `NEW_OUTPUTS/` | Team-graph renderer with per-team tactics and opponent actions |
| `PAPER_OUTPUT/`, `OUTPUTS/` | Bi-causal relation/action renderer, ball tracking |
| `BASEFRAMES/` | Empty-court base frames and automatic court-corner detection |
| `figures_atlas/`, `figures_workflow/`, `model_comparison/` | Data atlas and result figures |
| `volleyball-detections/`, `volleyball_tracking_annotation/` | Detection and tracking annotation sets |
| `METHODOLOGY.md` | Formal method specification |
| `TRAINING.md` | Ordered training runbook |
| `Technical_Review_and_QA.md` | Audit log: every bug, its root cause and its fix |
| `DEFENSE_MASTER.md`, `GLOSSARY_MASTER.md`, `PANEL_REVIEW/` | Viva preparation and review material |
| `README_ORIGINAL.md` | Earlier README with the first 3-video study and full method discussion |

### Not in this repository

GitHub's file-size limits keep these out of the repository:

- **Match videos and rendered outputs** (`*.mp4`, about 37 GB)
- **Feature caches** (`PAPER_OUTPUT/cache/`, `PAPER_OUTPUT/features/`). Your scripts can make them again.
- **`extra_trees.joblib`** in `ml_models_v5/` and `ml_models_v6/` (over 100 MB each)
- **Stock YOLO weights.** Ultralytics downloads them automatically.

---

## Limitations

1. **Single annotator.** The labels are human, and they were checked to match the clip folders with zero mismatches. But all labels come from one person, so the true ceiling is unknown. `fyp/annotate.py` supports a blind second-annotator pass to measure it.
2. **Minority classes are weak.** Delayed Support and Spacing Breakdown have about 110 clips each and reach about 38% accuracy.
3. **Few source videos.** With 8 matches, cross-video drift between cameras is still the main error source (see `fyp/drift.py`).
4. **Distances are relative** unless a per-camera homography is available. The linear fallback keeps ratios, not metres.
5. **Train/test gap.** Tree models fit their training data perfectly. Only the LOVO columns are meaningful.

## Future work

- More source matches, from different broadcasters and camera angles
- A second annotator, to measure inter-annotator agreement as the performance ceiling
- Per-frame keyframe homography, to remove camera-motion drift
- Pose features (`fyp/pose.py`), to give the minority classes more signal

---

## References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
- Ismail Fawaz, H. et al. (2020). *InceptionTime: Finding AlexNet for Time Series Classification.* DMKD
- Bai, S., Kolter, J. Z. & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271
- Aharon, N. et al. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* arXiv:2206.14651
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV
- Jocher, G. et al. *Ultralytics YOLO.* https://docs.ultralytics.com
