# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Hugging Face Space ([yeshwanthkumarreddy/volleyball-tactical-analysis](https://huggingface.co/spaces/yeshwanthkumarreddy/volleyball-tactical-analysis)): free static Space (`webapp/hf_static_space/`) that shows the live website.
- Website walkthrough video (`docs/assets/ui_walkthrough.mp4`): screen recording of an upload, the live analysis and the results; linked from the README and shown first on the website's Demos page.
- Always-on Azure analysis server: shared CPU Docker image (`webapp/hfspace/`) and one-command deploy script (`webapp/azure/deploy_azure.ps1`).
- Team photos on the web app's Team page; cloud-aware upload page.
- Result files for the 2-class models (`BINARY_ATTACK_DEFENSE/results/`) and the full-match run (`v7_tactical/results/`).

### Fixed
- G. Bharath Chandra's roll number (CB.SC.U4CSE23415).
- `.gitignore` rule `results/` hid every results folder; it now ignores only the root `/results/` written by `reproduce.py`.

## [1.0.0] - 2026-10-25

First public release of the complete project.

### Added
- Installation guide, usage guide, architecture guide and documentation index in `docs/`.
- MIT licence, `CITATION.cff`, contributing guide, issue and pull-request templates.
- Continuous integration: the unit-test suite runs on every push and pull request.
- Per-tactic clip export and clip download route in the web app.
- Live-upload routing from the static portal to a running backend; history codes to restore analyses on another device.
- Cloud deployment configurations: Azure VM (Docker Compose + MongoDB + Caddy HTTPS) and Hugging Face Spaces (CPU Dockerfile).
- Full project documentation (`docs/PROJECT_DOCUMENTATION.md`).

### Changed
- Repository layout: long-form documents moved to `docs/`, training logs moved to `logs/`.
- README rewritten with a demo, results, quick start and documentation map.
- XGBoost, LightGBM and CatBoost declared in `fyp/requirements.txt`.

### Removed
- Two stray debug files (`fyp/_synctest.py`, `fyp/error.txt`) and an empty render log.

## [0.3.0] - 2026-10-24

### Added
- Full-stack web application (`webapp/`): FastAPI API, React portal, MongoDB storage with JSON fallback, single GPU worker, desktop app, Windows installer and PWA support.
- `NEW_OUTPUTS/` renderer: near-team relation graph with four tactics and opponent SERVE / SPIKE / BLOCK timeline.
- `PAPER_OUTPUT/` paper-feature renderer.
- v5 / v6 model runs: 12 scikit-learn models (adds XGBoost, LightGBM, CatBoost) on 948 clips from 8 matches; deployed `tactical_ml_v6.joblib`.

## [0.2.0] - 2026-10-20

### Changed
- Bi-Causal extraction and rendering scripts (`OUTPUTS/bicausal_extract.py`, `bicausal_render.py`).

## [0.1.0] - 2026-10-19

### Added
- Research pipeline (`fyp/`): dual YOLO detection, BoT-SORT tracking, automatic court calibration, colour team separation, 18-channel permutation-invariant features, scikit-learn and PyTorch classifiers behind one interface.
- Human-labelled dataset index (`fyp/labels.csv`) and extracted feature CSVs (`training_csv*/`).
- Deep sequence models: InceptionTime, TCN, BiLSTM + Attention, Mamba SSM, with leave-one-video-out evaluation.
- 2-class Attack/Defense models (`BINARY_ATTACK_DEFENSE/`) and the 13-model comparison (`ALL_13_MODELS/`, `FINALE_OUTPUTS/`).
- Empty-court base frames and automatic court-corner detection (`BASEFRAMES/`).
- Data atlas, workflow figures, `reproduce.py` and a 257-test suite.

[1.0.0]: https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp/releases/tag/v1.0.0
[0.3.0]: https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp/compare/a7966274...9526694b
[0.2.0]: https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp/compare/cbb09372...a7966274
[0.1.0]: https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp/commit/cbb09372
