# Data Atlas — figures_atlas/

Regenerate everything with:

```bash
python make_data_atlas.py    # writes the 27 PNGs below
python build_atlas.py        # rebuilds figures_atlas/data_atlas.html with them embedded
```

`data_atlas.html` is a single self-contained page (figures embedded as base64) —
open it in a browser, or see the published Artifact.

Each figure is built directly from the raw repo artefacts (per-clip tracking CSVs,
`vp7_frames.csv`, the model-comparison JSON/MD, the deep training histories) and is
tagged with the reviewer audience it speaks to.

| # | folder | audience | reads |
|---|--------|----------|-------|
| 1 | `1_dataset/` | all | corpus size, class balance, per-video label shift, feature-space separability |
| 2 | `2_cv/` | computer vision | detection funnel, ball recall vs resolution, detector AP, court tracks, broadcast reliability, tracking filter |
| 3 | `3_volleyball/` | volleyball / domain | biomechanics by class, role kinematics, tactical fingerprint, Attack-vs-Defense, rally timeline, match budget |
| 4 | `4_ml/` | classical ML | k-fold vs LOVO, train vs test, per-class F1, confusion, channel importance, calibration, minority-class supply |
| 5 | `5_dl/` | deep learning | training curves, params vs F1, deep-vs-baseline bar, seed sensitivity, 13-model agreement |

## Corpus versions are not interchangeable

- **v1** `training_csv/` — 285 clips, pre-detector-fix (4.40 players/frame)
- **v3** `training_csv_v3/` — 375 clips, fixed pipeline, 3 source videos — the benchmark corpus
- **v4** `training_csv_v4/` — 945 clips, 6+ source videos, minority classes ~doubled

Every figure title names the corpus it was computed on.
