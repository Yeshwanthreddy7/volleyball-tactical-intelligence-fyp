# ALL_13_MODELS

This folder holds copies of all 13 trained tactical classifiers. The originals are still in their usual places, so no script breaks.

Every model predicts one of four tactics:
- Coordinated Attack
- Coordinated Defense
- Delayed Support
- Spacing Breakdown

## ML_models (9, scikit-learn, copied from `ml_models_all/`)

| File | Test accuracy (LOVO) |
|---|---|
| hist_gradient_boosting.joblib | 66.8% (deployed model, same as `tactical_ml_v4.joblib`) |
| extra_trees.joblib | 64.9% |
| gradient_boosting.joblib | 64.8% |
| rbf_svm.joblib | 64.2% |
| random_forest.joblib | 63.9% |
| mlp.joblib | 61.1% |
| knn.joblib | 60.0% |
| linear_svm.joblib | 59.2% |
| logistic_regression.joblib | 58.5% |

## DL_models (4, PyTorch)

These are copied from `mamba_v4_<arch>.pt` in the project root. They are the final checkpoints that `deep_results_v4/deep_results.json` reports.

| File | Test accuracy (LOVO) |
|---|---|
| inceptiontime.pt | 64.9% |
| tcn.pt | 49.0% |
| bilstm.pt | 48.9% |
| mamba.pt | 47.1% |

`DL_models/alternate_run_80epochs/` holds copies of `dl_models_v4/*.pt`. These come from a separate 80-epoch training run and are different files from the ones above. The saved results do not include accuracy numbers for this run.

## Loading any model

```python
import sys; sys.path.insert(0, "../fyp")
from interfaces import create_temporal_classifier
model = create_temporal_classifier("ML_models/hist_gradient_boosting.joblib")   # or DL_models/inceptiontime.pt
```

All 17 files were checked with this loader on 2026-09-29, and all of them load.
