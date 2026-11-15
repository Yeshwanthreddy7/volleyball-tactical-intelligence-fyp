"""Evidence added after the external review (2026-11-15).

Answers the reviewer's ML points with numbers from the SAME 945 clips / 8 videos
and the SAME leave-one-video-out (LOVO) folds as final_report_eval.py:

  1. confusion matrix, per-class precision / recall / F1, macro-F1, weighted-F1
     for the deployed model (calibrated HGB + prior correction) and plain HGB
  2. fold-wise (per held-out video) accuracy and macro-F1, HGB vs InceptionTime,
     with a paired Wilcoxon test over the 8 folds
  3. window-length study: the model trained on the first N frames (N = 8..29)
  4. per-class median of the key team measurements (data-backed class criteria)

Run:  python review_response_eval.py      Out: review_response_eval.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject")
sys.path.insert(0, str(ROOT / "fyp"))
import ml_classifier as M  # noqa: E402
from features import MODEL_FEATURE_COLS, build_model_sequence  # noqa: E402

OUT = Path(__file__).resolve().parent / "review_response_eval.json"
CSV_DIR = ROOT / "training_csv_v5_full"
bundle = joblib.load(ROOT / "tactical_ml_v6.joblib")
classes = list(bundle["class_names"])

# raw clips, kept so the window study can cut them
import pandas as pd  # noqa: E402
from train_utils import video_key  # noqa: E402

raws, y, groups = [], [], []
for f in sorted(CSV_DIR.glob("*.csv")):
    df = pd.read_csv(f)
    lab = str(df["target_label"].iloc[0]).strip()
    if lab not in M.CLASS_NAMES:
        continue
    g = video_key(f.name)
    g = "usa" if g == "?" else g
    if g == "external":
        continue
    raws.append(df.drop(columns=[c for c in ("frame_id", "target_label") if c in df]).to_numpy(float))
    y.append(M.CLASS_NAMES.index(lab))
    groups.append(g)
y, groups = np.array(y), np.array(groups)
vids = sorted(set(groups))
print(len(y), "clips", {v: int((groups == v).sum()) for v in vids})


def feats(n):
    if n == 29:      # full clip: exactly the feature path used in training (ml_classifier.load_dataset)
        return np.vstack([M.extract(r, "invariant") for r in raws])
    return np.vstack([M.invariant_from_model_sequence(build_model_sequence(r[:n], target_len=n)) for r in raws])


def lovo(est, X):
    P = np.zeros((len(y), 4))
    prior = np.zeros((len(y), 4))
    for v in vids:
        te = groups == v
        m = clone(est).fit(X[~te], y[~te])
        P[te] = m.predict_proba(X[te])
        prior[te] = np.bincount(y[~te], minlength=4) / (~te).sum()
    return P, prior


def report(pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    folds = {}
    for v in vids:
        k = groups == v
        folds[v] = {"n": int(k.sum()), "accuracy": float(accuracy_score(y[k], pred[k])),
                    "macro_f1": float(f1_score(y[k], pred[k], average="macro", labels=range(4), zero_division=0))}
    accs = [d["accuracy"] for d in folds.values()]
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "weighted_f1": float(f1_score(y, pred, average="weighted")),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(classes)},
            "confusion_rows_true_cols_pred": confusion_matrix(y, pred, labels=range(4)).tolist(),
            "folds": folds, "fold_accuracy_mean": float(np.mean(accs)), "fold_accuracy_sd": float(np.std(accs, ddof=1)),
            "fold_accuracy_min": float(np.min(accs)), "fold_accuracy_max": float(np.max(accs))}


res = {"n_clips": int(len(y)), "videos": {v: int((groups == v).sum()) for v in vids}, "class_order": classes}

t = time.time()
X29 = feats(29)
P, prior = lovo(bundle["pipeline"], X29)
res["deployed_prior_corrected"] = report((P / prior).argmax(1))
res["deployed_argmax"] = report(P.argmax(1))
print("deployed", round(time.time() - t), "s")

hgb = M.build_models()["hist_gradient_boosting"]
Ph, _ = lovo(hgb, X29)
res["hgb_plain"] = report(Ph.argmax(1))

# fold-wise vs InceptionTime (macro-F1 per fold from deep_results_v6)
deep = json.load(open(ROOT / "deep_results_v6" / "deep_results.json"))["architectures"]
cmp = {}
for arch in deep:
    fd = {f["video"]: f["macro_f1_4"] for f in deep[arch]["folds"]}
    a = [res["deployed_prior_corrected"]["folds"][v]["macro_f1"] for v in vids]
    b = [fd[v] for v in vids]
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    cmp[arch] = {"fold_macro_f1": dict(zip(vids, b)), "pooled_accuracy": deep[arch]["pooled_lovo_accuracy"],
                 "pooled_macro_f1": deep[arch]["pooled_lovo_macro_f1_4class"],
                 "hgb_better_folds": int(sum(x > z for x, z in zip(a, b))), "wilcoxon_p": p}
res["deep_vs_deployed_foldwise"] = cmp

# window-length study (plain HGB, argmax, so only N changes)
win = {}
for n in (8, 12, 16, 20, 24, 29):
    t = time.time()
    Pn, _ = lovo(hgb, feats(n))
    pr = Pn.argmax(1)
    win[n] = {"seconds": round(n / 30, 2), "accuracy": float(accuracy_score(y, pr)),
              "macro_f1": float(f1_score(y, pr, average="macro"))}
    print("window", n, win[n], round(time.time() - t), "s")
res["window_study_hgb"] = win

# per-class medians of the main team measurements (data-backed class criteria)
seqs = [build_model_sequence(r, target_len=29) for r in raws]
ch = {c: i for i, c in enumerate(MODEL_FEATURE_COLS)}
crit = {}
for i, c in enumerate(classes):
    S = [s for s, t in zip(seqs, y) if t == i]
    def med(fn):
        return float(np.nanmedian([fn(s) for s in S]))
    crit[c] = {
        "team_spread_cm (nn_dist_mean)": med(lambda s: np.nanmean(s[:, ch["nn_dist_mean"]])),
        "court_area_m2 (hull_area)": med(lambda s: np.nanmean(s[:, ch["hull_area"]]) / 1e4),
        "fast_two_speed_cm_per_frame": med(lambda s: np.nanmean(s[:, ch["speed_top2"]])),
        "other_four_speed_cm_per_frame": med(lambda s: np.nanmean(s[:, ch["speed_bot4"]])),
        "sync_0to1": med(lambda s: np.nanmean(s[:, ch["sync_inst"]])),
        "ball_to_team_centre_cm": med(lambda s: np.nanmean(s[:, ch["ball_to_centroid"]])),
        "team_centre_y_cm": med(lambda s: np.nanmean(s[:, ch["centroid_y"]])),
    }
res["class_medians"] = crit
OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT)
