"""Held-out (leave-one-video-out) per-class accuracy of the deployed tactical
model, scored with the same prior-corrected decision rule the videos use.

Each fold refits a clone of the tactical_ml_v6.joblib pipeline on every video
except one and predicts that video's clips. Per-class accuracy = recall
(share of the clips of that class the model called correctly).

Out: NEW_OUTPUTS/lovo_per_class_accuracy.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fyp"))
import ml_classifier                                   # noqa: E402

bundle = joblib.load(ROOT / "tactical_ml_v6.joblib")
classes = list(bundle["class_names"])
(X, y, groups, files), skipped = ml_classifier.load_dataset(
    str(ROOT / "training_csv_v5_full"), mode=bundle.get("feature_mode", "invariant"))
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f"{len(y)} clips, {len(skipped)} skipped, videos {sorted(set(groups))}")

pred = np.full(len(y), -1)
for g in sorted(set(groups)):
    te = groups == g
    m = clone(bundle["pipeline"]).fit(X[~te], y[~te])
    P = m.predict_proba(X[te])
    prior = np.bincount(y[~te], minlength=4) / (~te).sum()      # training-fold prior
    pred[te] = (P / prior).argmax(1)
    print(f"  fold {g}: {np.mean(pred[te] == y[te]):.3f}")

out = {"n_clips": int(len(y)), "overall_accuracy": float(np.mean(pred == y)),
       "per_class": {}}
for c, name in enumerate(classes):
    k = y == c
    out["per_class"][name] = {"n": int(k.sum()), "accuracy": float(np.mean(pred[k] == c))}
cm = np.zeros((4, 4), int)
for t, p in zip(y, pred):
    cm[t, p] += 1
out["confusion_rows_true_cols_pred"] = cm.tolist()
print(json.dumps(out, indent=1))
(ROOT / "NEW_OUTPUTS" / "lovo_per_class_accuracy.json").write_text(json.dumps(out, indent=1))
