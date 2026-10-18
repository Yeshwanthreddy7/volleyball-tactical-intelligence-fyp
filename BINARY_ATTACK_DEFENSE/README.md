# BINARY_ATTACK_DEFENSE: 13 models trained on Coordinated Attack vs Coordinated Defense only

## Data

The training data is every usable clip in `fyp/dataset/Coordinated_Attack` and `fyp/dataset/Coordinated_Defense`: **723 clips** (Attack 369, Defense 354), taken from 8 source videos.

The dataset folders hold 744 clips, but **21 of them are empty files**: 261 bytes each, with no video stream. All 21 come from `videoplayback (2)`, and all are timed at 22:27 or later. The source file `fyp/Datasets/videoplayback (2).mp4` is truncated, and its video stops at 22:01. Re-cutting the 21 clips from that source also produced empty files. The only way to recover them is a complete copy of that video.

## Models

| Folder | Contents |
|---|---|
| `ML_models/` | 9 scikit-learn models (`.joblib`) |
| `DL_models/` | 4 PyTorch models |

For each DL model:
- `<arch>.pt` is the servable checkpoint.
- `<arch>_seed0..2.pt` are the 3 ensemble seeds.

The ML models use the project's 132 permutation-invariant features and the same 9 pipelines as the 4-class run. The DL models use the (29 frames x 18 channels) sequence.

Load any model with `fyp/interfaces.create_temporal_classifier(path)`. All 13 were verified to load and to output the two classes.

## Accuracy

Accuracy is measured with **leave-one-video-out (LOVO)** testing. Each model is tested on each of the 8 videos after training on the other 7. Always guessing "Attack" would score 51%.

| Rank | Model | Type | Test accuracy | Macro-F1 | ROC-AUC | Train accuracy |
|---|---|---|---|---|---|---|
| 1 | InceptionTime | DL | **89.5%** | 0.895 | 0.957 | - |
| 2 | TCN | DL | 88.0% | 0.879 | 0.950 | - |
| 3 | BiLSTM+Attention | DL | 87.8% | 0.878 | 0.942 | - |
| 4 | Mamba (SSM) | DL | 84.8% | 0.847 | 0.924 | - |
| 5 | RBF-SVM | ML | 83.7% | 0.837 | 0.921 | 100% |
| 6 | MLP | ML | 83.5% | 0.835 | 0.907 | 100% |
| 7 | HistGradientBoosting | ML | 83.1% | 0.831 | 0.918 | 100% |
| 8 | LogisticRegression | ML | 83.1% | 0.831 | 0.892 | 93.9% |
| 9 | GradientBoosting | ML | 83.0% | 0.830 | 0.912 | 100% |
| 10 | ExtraTrees | ML | 82.4% | 0.824 | 0.918 | 100% |
| 11 | k-NN | ML | 81.3% | 0.813 | 0.873 | 100% |
| 12 | Linear SVM | ML | 81.2% | 0.812 | 0.870 | 94.5% |
| 13 | RandomForest | ML | 80.9% | 0.809 | 0.914 | 100% |

**How to read these numbers**
- The weakest held-out video for every model is `videoplayback (1)`. Per-fold numbers are in `results/`.
- The DL models were trained with augmentation (mirroring, jitter and mixup) and a 3-seed ensemble. The ML models were trained with no augmentation.
- The DL results have no train-accuracy number, because the training script does not record one.
- The old 4-class models (Attack, Defense, Delayed Support, Spacing Breakdown) reached at best 66.8% accuracy.

## Files in results/

- `binary_model_accuracy.csv`
- `binary_model_accuracy_chart.png`
- `ml_binary_results.json`
- `lovo_<arch>.json`
- `oof_<arch>.npz`
- training logs

## Re-running (from `finalyearproject/`)

```bash
python BINARY_ATTACK_DEFENSE/code/train_binary_ml.py      # 9 ML models, about 2 min
bash   BINARY_ATTACK_DEFENSE/code/run_dl.sh               # 4 DL models, about 50 min on GPU (Mamba takes about 35 of that)
python BINARY_ATTACK_DEFENSE/code/make_binary_report.py   # accuracy table and chart
```
