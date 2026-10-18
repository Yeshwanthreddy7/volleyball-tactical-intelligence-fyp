"""Train the 9 ML models on Coordinated Attack vs Coordinated Defense only.

Data: the 723 Attack/Defense clips of fyp/dataset (369 Attack, 354 Defense),
as tracked-coordinate CSVs in v7_tactical/csv_geom (geometric team split, the
same clips the 4 DL models use via v7_tactical/cache/binary_cache.npz).

Features and models are the project's own: ml_classifier.load_dataset
(132 permutation-invariant features) and ml_classifier.build_models (the same
9 pipelines as the 4-class run).

Evaluation: 8-fold leave-one-video-out (LOVO) - every clip is predicted by a
model that never saw its source video. Train accuracy = fit on all clips,
scored on the same clips (shows over-fitting).

Deployment: each model is refit on all 723 clips, probability-calibrated, and
saved as a .joblib bundle that fyp/interfaces.create_temporal_classifier loads.

Usage (from finalyearproject/):
  python BINARY_ATTACK_DEFENSE/code/train_binary_ml.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "fyp"))

import joblib                                                     # noqa: E402
from sklearn.base import clone                                    # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)

import ml_classifier as mc                                        # noqa: E402
from features import FEATURE_VERSION                              # noqa: E402

CSV_DIR = os.path.join(ROOT, "v7_tactical", "csv_geom")
OUT = os.path.join(ROOT, "BINARY_ATTACK_DEFENSE")
CLASSES = ["Coordinated Attack", "Coordinated Defense"]          # index 0, 1 (= ml_classifier order)


def report(y, pred, p_attack):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "attack_f1": round(float(f1_score(y, pred, pos_label=0)), 4),
        "defense_f1": round(float(f1_score(y, pred, pos_label=1)), 4),
        "roc_auc": round(float(roc_auc_score(y == 0, p_attack)), 4),
        "confusion_rows_true_A_D": cm.tolist(),
    }


def main():
    (X, y, groups, files), skipped = mc.load_dataset(CSV_DIR, mode="invariant")
    keep = y <= 1
    X, y, groups = X[keep], y[keep], groups[keep]
    print(f"clips={len(y)}  attack={int((y == 0).sum())}  defense={int((y == 1).sum())}  "
          f"features={X.shape[1]}  videos={sorted(set(groups))}  skipped={len(skipped)}")

    results = {}
    t0 = time.time()
    for name, pipe in mc.build_models(seed=0).items():
        oof = np.zeros((len(y), 2))
        folds = []
        for vid in sorted(set(groups)):
            te = groups == vid
            m = clone(pipe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[~te], y[~te])
            oof[te] = m.predict_proba(X[te])
            pr = oof[te].argmax(1)
            folds.append({"video": vid, "n": int(te.sum()),
                          "accuracy": round(float(accuracy_score(y[te], pr)), 4)})
        pooled = report(y, oof.argmax(1), oof[:, 0])

        full = clone(pipe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full.fit(X, y)
        train_acc = float(accuracy_score(y, full.predict(X)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal, _ = mc.calibrate(clone(pipe).fit(X, y), X, y, seed=0)
        bundle = {
            "pipeline": cal, "class_names": CLASSES,
            "feature_names": mc.names_for("invariant"),
            "aggregations": list(mc.AGGREGATIONS), "feature_mode": "invariant",
            "feature_version": FEATURE_VERSION, "model_name": name,
            "train_class_counts": [int((y == 0).sum()), int((y == 1).sum())],
            "extraction_provenance": None, "task": "binary Attack vs Defense",
        }
        joblib.dump(bundle, os.path.join(OUT, "ML_models", f"{name}.joblib"))
        results[name] = {"lovo": pooled, "train_accuracy": round(train_acc, 4), "folds": folds}
        print(f"  {name:24s} LOVO acc {pooled['accuracy']:.3f}  macroF1 {pooled['macro_f1']:.3f}  "
              f"AUC {pooled['roc_auc']:.3f}  train acc {train_acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "results", "ml_binary_results.json"), "w") as fh:
        json.dump({"protocol": "8-fold leave-one-video-out, pooled out-of-fold",
                   "data": CSV_DIR, "n_clips": int(len(y)), "classes": CLASSES,
                   "models": results}, fh, indent=2)
    print("done")


if __name__ == "__main__":
    main()
