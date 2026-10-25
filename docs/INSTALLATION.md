# Installation Guide

This guide takes you from a clean machine to a working install of the research pipeline and the web application. Each part ends with a check that proves it works.

- [1. System requirements](#1-system-requirements)
- [2. Get the code](#2-get-the-code)
- [3. Install the research pipeline](#3-install-the-research-pipeline)
- [4. Verify the install](#4-verify-the-install)
- [5. Model weights and data](#5-model-weights-and-data)
- [6. Install the web application](#6-install-the-web-application)
- [7. Run with Docker](#7-run-with-docker)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. System requirements

| Component | Minimum | Tested with |
|---|---|---|
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Windows 11, Ubuntu 22.04 (Docker) |
| Python | 3.10 | 3.11 (Docker), 3.12.7 (development) |
| RAM | 8 GB | 8 GB |
| GPU | Optional for training and tests; recommended for video analysis | NVIDIA RTX 3050 Laptop, CUDA 12.1 |
| Disk | 2 GB for the repository and dependencies | — |
| Node.js | 18+ (only to build the web frontend) | 20 (Docker), 24 |
| Git | 2.30+ | 2.50 |

Without a GPU, everything works, but video analysis is roughly 10× slower. Unit tests, training on the extracted CSVs and the classifiers all run on a CPU.

---

## 2. Get the code

```bash
git clone https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp.git
cd volleyball-tactical-intelligence-fyp
```

The clone is about 500 MB, because the trained models and the extracted training data are committed. For a faster clone that skips history, use `git clone --depth 1 ...`.

---

## 3. Install the research pipeline

### Option A: `venv` (recommended)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda create -n volleyball python=3.11 -y
conda activate volleyball
pip install -r requirements.txt
```

### GPU support (NVIDIA)

`pip install torch` installs a CPU build on Windows. For CUDA, reinstall PyTorch from the official index **after** the step above:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

Check that the GPU is visible:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### What gets installed

| Package | Used for |
|---|---|
| `ultralytics` | YOLO11 player/ball detection and the BoT-SORT tracker |
| `torch` | Deep sequence models (InceptionTime, TCN, BiLSTM, Mamba) |
| `opencv-python`, `numpy`, `scipy`, `pandas` | Video I/O, court geometry, features |
| `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `joblib` | Tactical classifier zoo and model files |
| `supervision`, `lapx` | ByteTrack backend and the assignment solver BoT-SORT needs |
| `pytest` | Test suite |
| `matplotlib` | Figures and the data atlas |

---

## 4. Verify the install

### 4.1 Run the test suite

```bash
python -m pytest fyp/tests -q
```

Expected result: **257 passed** in about 20 seconds. No GPU or video is needed.

### 4.2 Load the deployed model

```bash
python -c "import sys, numpy as np; sys.path.insert(0, 'fyp'); from interfaces import create_temporal_classifier; m = create_temporal_classifier('tactical_ml_v6.joblib'); print(m.classify(np.random.rand(29, 18).astype('float32')).label)"
```

Expected result: one tactic name, for example `Coordinated Attack`. The input is random, so the label itself has no meaning; the check proves the model file, scikit-learn and the feature interface agree.

### 4.3 Check the pipeline command line

```bash
python fyp/pipeline.py --help
```

Expected result: the usage text with flags such as `--auto-court`, `--team-split` and `--output-video`.

---

## 5. Model weights and data

### Included in the repository

| File / folder | What it is |
|---|---|
| `tactical_ml_v6.joblib` | Deployed 4-class classifier (HistGradientBoosting) |
| `ml_models_v6/` | All 12 scikit-learn models from the latest run (except `extra_trees.joblib`, which is over GitHub's 100 MB limit) |
| `mamba_v4_*.pt` | Deep models: InceptionTime, TCN, BiLSTM + Attention, Mamba SSM |
| `ALL_13_MODELS/`, `BINARY_ATTACK_DEFENSE/` | 13-model bundle and the 2-class Attack/Defense models |
| `fyp/volleyball_best.pt` | Fine-tuned YOLO ball detector |
| `training_csv_v4/`, `training_csv_v5_full/` | Extracted per-clip features (the training data) |
| `fyp/labels.csv` | Human labels: video, timestamp and tactic for every clip |

### Downloaded automatically

Ultralytics downloads the stock weights (`yolo11n.pt`, `yolo11m-pose.pt`) the first time a script asks for them. This needs internet access once.

### Not included

The source match videos (about 37 GB) and rendered outputs are not in the repository, because of GitHub's file-size limits. To analyse your own footage, use any broadcast-style volleyball video filmed from behind one baseline, 720p or higher. To retrain from scratch, place the videos in `fyp/Datasets/` and follow [TRAINING.md](TRAINING.md).

---

## 6. Install the web application

The web application is a React frontend served by a FastAPI backend. MongoDB is optional: when it is not reachable, the app stores its data in a JSON file.

### 6.1 Backend

```bash
pip install -r webapp/backend/requirements.txt
```

### 6.2 Frontend

```bash
cd webapp/frontend
npm install --include=dev
npm run build
cd ../..
```

### 6.3 Start the site

**Windows (one command)**

```bat
webapp\run_website.bat
```

The script starts MongoDB if a portable copy is found, builds the frontend if needed, and serves the site at `http://volleyballtactics.localhost` (port 80) or `http://localhost:8000` when port 80 is busy.

**Any OS (manual)**

```bash
cd webapp/backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`. Check the API with:

```bash
curl http://localhost:8000/api/health
```

### 6.4 Settings

Copy `webapp/.env.example` to `webapp/.env` and edit as needed.

| Variable | Default | Meaning |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `VTA_STORAGE_DIR` | `webapp/storage` (`run_website.bat` uses `%USERPROFILE%\vta_storage`) | Where uploads and results are written (keep it outside OneDrive) |
| `VTA_MAX_UPLOAD_MB` | `500` | Largest accepted upload |
| `VTA_TACTICAL_MODEL` | `tactical_ml_v6.joblib` | Classifier the analysis engine uses |
| `VTA_RETENTION_DAYS` | `14` | Days before an analysis is deleted |
| `VTA_PORT` | `80`, else `8000` | Port for `run_website.bat` |

Full hosting options (desktop app, Windows installer, public link, cloud VM) are in [webapp/DEPLOY.md](../webapp/DEPLOY.md).

---

## 7. Run with Docker

Build from the repository root:

```bash
docker build -f webapp/Dockerfile -t vta-web .
docker compose -f webapp/docker-compose.yml up -d
```

The GPU image needs the NVIDIA Container Toolkit on the host. For a CPU-only cloud VM, use `webapp/azure/docker-compose.yml`, which also starts MongoDB and a Caddy reverse proxy with automatic HTTPS.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `lap` fails to build on Windows | The old `lap` package has no Windows wheel | `pip install lapx` (already in `fyp/requirements.txt`) |
| `torch.cuda.is_available()` is `False` | CPU build of PyTorch installed | Reinstall from the CUDA index (see [GPU support](#gpu-support-nvidia)) |
| `check_provenance` aborts the pipeline | Inference flags differ from the flags the model was trained with | Use the exact flags in the README: `--imgsz 1280 --team-split colour --tracker botsort --auto-court --court-coords linear` |
| `numpy.dtype size changed` when loading a `.joblib` | NumPy 2 with models pickled under NumPy 1.26 | `pip install "numpy<2"` and `pip install "opencv-python<4.11"` |
| Analysis runs out of memory | 1080p video with several jobs on an 8 GB machine | Run one job at a time, or downscale the video to 720p |
| Team colour preflight reports `degenerate` | Both teams wear similar dark kits | Add `--skip-preflight`; teams are then split by the net line |
| Frontend shows `OFFLINE` | Backend not running, or an old service worker is cached | Start the backend, then hard-refresh the page (Ctrl+Shift+R) |
| `ffmpeg` not found | `extract_clips.py` calls ffmpeg directly | Install ffmpeg and add it to `PATH` (the web app bundles its own via `imageio-ffmpeg`) |

Still stuck? Open an issue with the full command, the error and the output of `python --version` and `pip freeze`.
