# Volleyball Tactical Analysis — Project Documentation

**Automated team-tactic analysis of volleyball from a single broadcast camera**

Final Year Project · Group 56 · Department of Computer Science and Engineering

| Item | Details |
|---|---|
| **Team** | Keshavramprakash D (CB.SC.U4CSE23425) · K. Surya Sekhar (CB.SC.U4CSE23420) · G. Bharath Chandra (CB.SC.U4CSE23415) · V. Punith Sai (CB.SC.U4CSE23768) · U. Yeshwanth Kumar Reddy (CB.SC.U4CS23453) |
| **Guide** | Dr. Divya Singh, Assistant Professor, Department of CSE |
| **Website** | https://volleyanalyzer.vercel.app (portal) · http://volleyballtactics.localhost (full app with upload, on the project laptop) |
| **Source code** | https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp |

---

## 1. Abstract

Coaches need fast, objective feedback on how a team coordinates on court, but manual video notation is slow and subjective, and most sports analytics tools describe individual actions or the ball path rather than the *collective* behaviour of the six players. This project converts raw broadcast volleyball video into a second-by-second tactical reading of a team.

The system detects and tracks every player (YOLO11 pose + BoT-SORT) and the ball (a volleyball-specific fine-tuned YOLO model at 1280 px), calibrates the court automatically from its painted lines, separates the two teams, and builds an 18-channel, permutation-invariant formation descriptor for every frame. A learned classifier labels every one-second window of the near-side team as **Coordinated Attack, Coordinated Defense, Delayed Support or Spacing Breakdown**, and rule-based ball-touch analysis time-stamps every opponent **SERVE, SPIKE and BLOCK**.

On 948 human-labelled clips from 8 international matches, the deployed HistGradientBoosting model reaches **67.0% leave-one-video-out (LOVO) accuracy and 0.510 macro-F1**, against a 39.0% majority baseline; a two-class Attack/Defense variant reaches **89.5%** (InceptionTime). The pipeline is delivered as a full-stack web application (React + FastAPI + MongoDB), a Windows desktop app and installer, and a public project portal.

---

## 2. Problem statement

Given a broadcast video of a volleyball match filmed from behind one baseline, automatically:

1. find and follow every player and the ball,
2. decide, for each second of live play, which collective tactic the near-side team is executing,
3. report when the opponent serves, spikes or blocks,
4. return an annotated video and timestamped reports that a coach can use directly.

**Why it is hard**

- The ball is only 15–20 px wide in 720p video and is often motion-blurred.
- At the net both teams stand within a metre of each other, so court position alone cannot assign team membership.
- Broadcast video contains replays, close-ups and camera cuts that break the court model.
- Tracker identities switch during occlusions, so per-player features must survive identity swaps.
- There is a single uncalibrated camera; every video has a different scale and viewpoint.

---

## 3. Objectives

1. Detect and track all players and the ball reliably from broadcast footage.
2. Map players onto real court coordinates using an automatically detected court homography.
3. Label ball actions with exact timestamps from ball-touch events, touch order and pose.
4. Classify whole-team tactics into four classes with a learned, honestly evaluated model.
5. Deliver coach-ready outputs: an annotated MP4 that plays everywhere and two separate reports.
6. Compare classical and deep models fairly (12 ML + 4 deep sequence models, LOVO validation).
7. Package everything as an easy-to-use website, desktop app and installer.

---

## 4. Tactical classes

| # | Class | Definition |
|---|---|---|
| 1 | **Coordinated Attack** | Front-court attackers approach fast while base players hold near the net |
| 2 | **Coordinated Defense** | The team shifts as one rigid unit |
| 3 | **Delayed Support** | The nearest player reacts late after ball contact |
| 4 | **Spacing Breakdown** | Structural failure: the formation is too spread out or too clustered |
| 0 | Transition / no call | Dead ball, replay, too few players visible, or out of play |

Opponent (far-side team) actions reported: **SERVE, SPIKE, BLOCK**.

---

## 5. Dataset

| Item | Value |
|---|---|
| Source | 11 broadcast match videos (8 used for the labelled training set) |
| Labelled clips | **948** one-second clips (29 frames) |
| Class counts | Coordinated Attack 370 · Coordinated Defense 355 · Delayed Support 108 · Spacing Breakdown 115 |
| Feature files | `training_csv_v5_full/` (per-clip CSV: ball + six player positions per frame) |
| Detector data | `volleyball-detections/`, `volleyball_tracking_annotation/` (ball fine-tune, tracking annotations) |
| Court data | `BASEFRAMES/base_frame/` (empty-court frame per video) + `corner_court_detection/labels/` |

Labels are human-made and were checked against the clip folders with zero mismatches. Delayed Support and Spacing Breakdown are minority classes, so **macro-F1** (equal weight per class) is the headline metric.

---

## 6. System architecture

```
Broadcast video (≤ 500 MB, MP4/MOV/AVI/MKV/WEBM)
   │
   ├─ 1. Court calibration    empty-court base frame → painted lines → homography
   │                          BASEFRAMES/extract_base_frames.py, detect_court_corners.py
   ├─ 2. Detection + tracking YOLO11m-pose (17 keypoints) + BoT-SORT; volleyball_best.pt (ball) @1280 px
   │                          OUTPUTS/bicausal_extract.py
   ├─ 3. Players on court     feet → court metres; near / far team by court side; 6 players per side
   ├─ 4. Ball + touches       static-ball suppression, tracklets, direction changes next to a player
   ├─ 5. Actions              volleyball touch order + pose → SERVE / RECEPTION / SET / SPIKE / BLOCK / DIG …
   │                          OUTPUTS/bicausal_render.py (label_touches)
   ├─ 6. Team tactics         29-sample windows, stride 1/3 s → (29 × 18) perm-invariant descriptor
   │                          → HistGradientBoosting (tactical_ml_v6.joblib), class-balanced, smoothed
   │                          NEW_OUTPUTS/code/team_tactics.py, fyp/features.py, fyp/ml_classifier.py
   ├─ 7. Rendering            relation graph + hull in tactic colour (near team), opponent action labels,
   │                          probability panel, tactic timeline       NEW_OUTPUTS/code/new_render.py
   └─ 8. Delivery             H.264 MP4 (Ultra/720p/480p/360p, original audio), tactical report,
                              opponent report, per-tactic clips          webapp/backend/app/engine.py
```

### 6.1 Key design decisions

- **Two detectors.** Stock YOLO finds 14–23 people per frame; a small fine-tune collapsed to 0.3 players per frame on unseen footage, but the fine-tune is far better for the ball (67% recall vs 16%). Players come from stock weights, the ball from the fine-tune.
- **Resolution matters.** With the same weights, ball recall rises from 15% at 640 px to 77% at 1280 px.
- **Teams by colour and geometry.** Colour is reliable frame to frame but carries no team meaning; court side is reliable on average but ambiguous at the net. Each colour cluster takes its team from its members' median court side (10.7 → 5.8 players per frame).
- **Scale-invariant features.** Without calibration, absolute distances leak camera identity. Dimensionless ratios and per-clip z-scored temporal shapes lower within-video scores but raise cross-video scores — the trade that matters.
- **Permutation invariance.** A formation is a *set* of players; the descriptor does not depend on player order, and identities are recovered before features are built.

---

## 7. Models and evaluation

All numbers are **leave-one-video-out (LOVO)**: each match is held out in turn, so the model is always tested on a match it has never seen.

### 7.1 Four-class tactic classifier

| Model | Type | Accuracy | Macro-F1 | Cohen κ |
|---|---|---|---|---|
| **HistGradientBoosting** (deployed) | ML | **67.0%** | 0.510 | 0.498 |
| LightGBM | ML | 67.0% | 0.507 | 0.500 |
| XGBoost | ML | 66.7% | 0.519 | 0.498 |
| CatBoost | ML | 64.7% | **0.520** | 0.478 |
| RBF-SVM | ML | 64.6% | 0.500 | 0.466 |
| InceptionTime | DL | 61.9% | 0.515 | – |
| TCN | DL | 51.2% | 0.441 | – |
| Mamba SSM | DL | 46.1% | 0.390 | – |
| BiLSTM + Attention | DL | 45.9% | 0.398 | – |
| Majority class | baseline | 39.0% | 0.140 | 0.000 |

Per class: Coordinated Attack 69%, Coordinated Defense 69%, Delayed Support 38%, Spacing Breakdown 38%.

HistGradientBoosting was chosen as the deployed model because it has the best LOVO accuracy among models within about 0.01 macro-F1 of each other, handles missing players natively and needs no feature scaling.

### 7.2 Two-class Attack / Defense classifier

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| **InceptionTime** | **89.5%** | 0.895 | 0.957 |
| TCN | 88.0% | 0.879 | 0.950 |
| BiLSTM + Attention | 87.8% | 0.878 | 0.942 |
| Mamba SSM | 84.8% | 0.847 | 0.924 |
| Best ML (RBF-SVM) | 83.7% | 0.837 | 0.921 |

Full tables: `ml_results_v6/`, `deep_results_v6/`, `FINALE_OUTPUTS/README.md`, `BINARY_ATTACK_DEFENSE/README.md`. `python reproduce.py` regenerates every reported number.

---

## 8. Web application

### 8.1 Technology

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite, responsive (phone / tablet / laptop), installable as an app (PWA) |
| API | FastAPI + Uvicorn (Python) |
| Database | MongoDB 7 (JSON-file fallback if MongoDB is unavailable) |
| Worker | One GPU analysis at a time; queue survives restarts; progress and ETA |
| Engine | The project's own research code (see §6), run as an isolated process per job |
| Video | ffmpeg / H.264 + AAC, fast start; Ultra / 720p / 480p / 360p renditions |

### 8.2 Pages

| Page | Contents |
|---|---|
| Home | Problem statement, objectives, the four tactics, headline numbers |
| Analyze video | Upload up to 500 MB (chunked, resumable), choose processing mode (Accurate 1280 px / Fast 960 px), optional time range |
| Analysis result | Annotated video player with quality menu, tactic timeline, tactics table, **Team tactical report** and **Opponent action report** (Notepad viewer, clickable timestamps, TXT/CSV/JSON download), court calibration |
| My analyses | Every analysis uploaded from this browser with status and summary |
| Methodology | Pipeline steps, design decisions, diagrams |
| Results | Model comparison, class distribution, per-class F1, and the user's latest analysis |
| Progress | Project timeline |
| Documentation | This document and all project write-ups |
| Demos | Sample annotated videos and **per-tactic clips cut from the user's own analyses** |
| Team | Team members and guide |

### 8.3 Outputs of one analysis

- `annotated_ultra.mp4` (+ `_high`, `_medium`, `_low`): near team with relation graph and tactic-coloured hull, tactic banner, opponent SERVE/SPIKE/BLOCK labels, ball trail, probability side panel.
- `tactical_report.txt/.csv/.json`: every tactic segment with start, end, length and confidence, plus a summary per tactic.
- `opponent_report.txt/.csv/.json`: every opponent serve, spike and block with its timestamp and player id.
- `clips/<tactic>/*.mp4`: short clips of each tactic segment, shown on the Demos page.

### 8.4 API (selection)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Server, database, GPU and model status |
| POST | `/api/uploads` | Start a chunked upload |
| PUT | `/api/uploads/{id}/chunks/{n}` | Upload one 32 MB chunk |
| POST | `/api/analyses` | Queue an analysis |
| GET | `/api/analyses`, `/api/analyses/{id}` | List / read analyses and results |
| GET | `/api/analyses/{id}/files/{name}` | Videos and reports |
| GET | `/api/project/*` | Metrics, docs, figures, demos for the portal |

Interactive API docs: `/api/docs-ui`.

---

## 9. Deployment

| Form | How |
|---|---|
| **Full app on the project laptop** | `webapp\run_website.bat` → http://volleyballtactics.localhost |
| **Desktop app** | "Volleyball Tactical Analysis" icon (starts MongoDB and the server itself) |
| **Installer for other PCs** | `VolleyballTacticalAnalysis-Setup.exe` (Inno Setup; downloads Python + CUDA/CPU PyTorch during install) |
| **Public portal (24/7)** | Vercel: https://volleyanalyzer.vercel.app (also volleyballtactics / volleytactix / volleyballtacticalanalysis `.vercel.app`) |
| **Public uploads** | `webapp\go_live.ps1` connects the Vercel site to the laptop GPU through a tunnel while the laptop is on |
| **Docker / cloud GPU** | `webapp/Dockerfile`, `webapp/docker-compose.yml` (web + MongoDB + Cloudflare Tunnel) |

Step-by-step instructions: `webapp/DEPLOY.md`.

---

## 10. How to run

```bash
# full website on Windows (laptop with NVIDIA GPU)
cd finalyearproject\webapp
run_website.bat                      # then open http://volleyballtactics.localhost
powershell -ExecutionPolicy Bypass -File check_website.ps1   # health check

# research pipeline only
python fyp/pipeline.py "match.mp4" tactical_ml_v6.joblib --yolo-model yolo11n.pt \
  --ball-model fyp/volleyball_best.pt --imgsz 1280 --team-split colour --tracker botsort \
  --auto-court --court-coords linear --output-video annotated.mp4 --output-csv predictions.csv

# reproduce all metrics / run tests
python reproduce.py
python -m pytest fyp/tests -q
```

Processing speed on an RTX 3050 laptop GPU: about 7 minutes per minute of 30 fps video (detection + rendering).

---

## 11. Testing

- `fyp/tests/`: unit and regression tests, many replaying real failures (half-court homography, slot starvation, detector blindness, crowd-captured colour clusters).
- `reproduce.py`: asserts every reported metric property and exits non-zero on failure.
- Web app: end-to-end tests of chunked upload → GPU analysis → annotated MP4 + reports, through the local site and through the public tunnel; `check_website.ps1` checks pages, API, database, GPU and models.

---

## 12. Limitations

1. **Single annotator** — the true agreement ceiling is unknown.
2. **Minority classes are weak** — Delayed Support and Spacing Breakdown reach about 38%.
3. **Few source matches** — cross-camera drift remains the main error source.
4. **Near-side team only** — the tactic model was trained on the team at the bottom of the picture; teams change ends between sets, so analyse the matching time range.
5. **GPU required** for practical speed; the public portal cannot analyse video by itself.

## 13. Future work

- More matches from different broadcasters and camera angles; grow the minority classes.
- A second annotator to measure inter-annotator agreement.
- Per-frame homography to remove camera-motion drift; pose features for the minority classes.
- Far-side team tactics and a cloud GPU deployment for always-on analysis.

---

## 14. References

1. Jocher, G. et al. *Ultralytics YOLO.* https://docs.ultralytics.com
2. Aharon, N. et al. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* arXiv:2206.14651
3. Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV
4. Zhang et al. (2024). *Bi-Causal: Group Activity Recognition via Bidirectional Causality.* CVPR
5. Ismail Fawaz, H. et al. (2020). *InceptionTime: Finding AlexNet for Time Series Classification.* DMKD
6. Bai, S., Kolter, J. Z. & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271
7. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
8. Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd ed.
