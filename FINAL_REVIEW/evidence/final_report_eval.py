"""Evidence for final_report.pdf.

Re-runs leave-one-video-out (LOVO) on the 945 clips of the 8 labelled source
videos (labels.csv), i.e. the SAME 8 folds the deep models use, and stores
per-clip predictions so that:
  * the deployed decision rule (calibrated HGB, prior-corrected) is scored,
  * calibration is measured with ECE and Brier score,
  * models are compared with McNemar's exact test and a video-level bootstrap.

Run from anywhere:  python final_report_eval.py
Out: evidence/final_report_eval.json (next to this script)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject")
sys.path.insert(0, str(ROOT / "fyp"))
import ml_classifier  # noqa: E402

OUT = Path(__file__).resolve().parent / "final_report_eval.json"
rng = np.random.default_rng(0)

bundle = joblib.load(ROOT / "tactical_ml_v6.joblib")
classes = list(bundle["class_names"])
(X, y, groups, files), _ = ml_classifier.load_dataset(str(ROOT / "training_csv_v5_full"), mode="invariant")
groups = np.array(["usa" if g == "?" else g for g in groups])
keep = groups != "external"                      # the 3 clips not in labels.csv
X, y, groups = X[keep], y[keep], groups[keep]
vids = sorted(set(groups))
print(len(y), "clips", {v: int((groups == v).sum()) for v in vids})


def metrics(pred):
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "cohen_kappa": cohen_kappa_score(y, pred),
            "recall": {c: float(np.mean(pred[y == i] == i)) for i, c in enumerate(classes)}}


def lovo_proba(est):
    P = np.zeros((len(y), 4))
    prior = np.zeros((len(y), 4))
    for v in vids:
        te = groups == v
        m = clone(est).fit(X[~te], y[~te])
        P[te] = m.predict_proba(X[te])
        prior[te] = np.bincount(y[~te], minlength=4) / (~te).sum()
    return P, prior


def ece(P, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        k = (conf > lo) & (conf <= lo + 1 / bins)
        if k.any():
            e += k.mean() * abs(np.mean(pred[k] == y[k]) - conf[k].mean())
    return float(e)


def brier(P):
    return float(np.mean(np.sum((P - np.eye(4)[y]) ** 2, 1)))


def boot_ci(pred, n=2000):
    """Video-level (cluster) bootstrap: resample whole videos with replacement."""
    idx = {v: np.where(groups == v)[0] for v in vids}
    acc, f1 = [], []
    for _ in range(n):
        s = np.concatenate([idx[v] for v in rng.choice(vids, len(vids))])
        acc.append(np.mean(pred[s] == y[s]))
        f1.append(f1_score(y[s], pred[s], average="macro", labels=range(4)))
    return {"accuracy_95ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "macro_f1_95ci": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]}


def mcnemar(pa, pb):
    a_ok, b_ok = pa == y, pb == y
    b, c = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"hgb_right_other_wrong": b, "hgb_wrong_other_right": c, "p_value": float(p)}


res = {"n_clips": int(len(y)), "videos": {v: int((groups == v).sum()) for v in vids}, "folds": len(vids)}

# deployed pipeline: calibrated HGB
t = time.time()
P, prior = lovo_proba(bundle["pipeline"])
arg = P.argmax(1)
pc = (P / prior).argmax(1)
res["deployed_calibrated_hgb"] = {
    "argmax": metrics(arg), "prior_corrected": metrics(pc),
    "argmax_ci": boot_ci(arg), "prior_corrected_ci": boot_ci(pc),
    "ece": ece(P), "brier": brier(P), "mean_confidence": float(P.max(1).mean())}
print("deployed done", round(time.time() - t), "s")

# model zoo, uncalibrated, same folds
models = ml_classifier.build_models()
preds = {}
for name in ["hist_gradient_boosting", "lightgbm", "xgboost", "catboost", "gradient_boosting",
             "rbf_svm", "extra_trees", "random_forest", "mlp", "logistic_regression"]:
    t = time.time()
    Pm, _ = lovo_proba(models[name])
    preds[name] = Pm.argmax(1)
    r = metrics(preds[name])
    r.update(boot_ci(preds[name]))
    if name == "hist_gradient_boosting":
        r["ece"], r["brier"], r["mean_confidence"] = ece(Pm), brier(Pm), float(Pm.max(1).mean())
    res.setdefault("models", {})[name] = r
    print(name, round(r["accuracy"], 3), round(r["macro_f1"], 3), round(time.time() - t), "s")
res["mcnemar_vs_hgb"] = {n: mcnemar(preds["hist_gradient_boosting"], p)
                         for n, p in preds.items() if n != "hist_gradient_boosting"}
res["mcnemar_deployed_rule_vs_argmax"] = mcnemar(arg, pc)
OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT)
