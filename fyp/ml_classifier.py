"""
ml_classifier.py - learned tactical classifier (scikit-learn), replacing the
hand-coded threshold engine as the DECISION mechanism.

LABEL PROVENANCE - READ THIS FIRST, IT IS THE MOST-MISUNDERSTOOD PART
---------------------------------------------------------------------
**The labels are HUMAN annotations, not rule-engine output.** An earlier
revision of this file asserted the opposite and built a whole three-tier
"break the circularity" argument on it. That assertion was wrong. The verified
chain (`fyp/extract_clips.py`) is:

    labels.csv   <- HUMAN: (video_file, anchor_time, tactic_class)
      -> ffmpeg cuts a 1-second clip at each annotated timestamp
      -> clip written into dataset/<tactic_class>/
      -> prepare_training_data.py reads the label from THAT FOLDER NAME
      -> training_csv/*.csv `target_label`

Checked directly: all 159 resolvable rows of `labels.csv` land in the folder
their `tactic_class` names, with ZERO mismatches.

`label_clips.py` - the hand-coded rule engine - is a SEPARATE, alternative
labelling path. It adds a `target_label` column to 41-frame CSVs and never
creates or populates `dataset/<class>/`. It is not in the training chain.

Consequences, and they are large:

  * The "circular training" objection ("the model just learns your own rules")
    does not apply. The model is trained on human tactical judgement.
  * The rule engine becomes an INDEPENDENT COMPETING PREDICTOR of those human
    labels, seeing exactly the same 29-frame window the model sees. That makes
    `rule_engine_baseline()` the single most important number in this file:
    it is the honest answer to "does the learned model beat the heuristic?".
  * The correct baseline ladder is therefore:
        majority class  <  hand-coded rule engine  <  learned model
    and all three must be reported together.

What is still an open limitation (see sec 16 of the technical review): dataset
size, class imbalance, and the number of distinct source videos - NOT label
validity. `--gold` remains available for a second independent annotator, which
would let inter-annotator agreement be quoted as the true performance ceiling.

WHY CLASSICAL ML AND NOT ONLY THE MAMBA SSM
-------------------------------------------
With 285-508 clips and 16 examples in the smallest class, a 4-layer selective
state-space model is heavily over-parameterised - the project already measured
train_acc 1.00 vs val_acc 0.66. Gradient-boosted trees and SVMs are the correct
model class for this sample size, they train in seconds (so leave-one-video-out
is cheap), and they expose permutation importance, which turns "the model said
Coordinated Attack" into "because top-2 player speed rose while spacing
collapsed" - the only form a volleyball expert can actually argue with. The
Mamba model stays in the project as the deep-learning comparison arm; this
module is the evidence that a simpler learner is not being ignored.

Usage
-----
    python fyp/ml_classifier.py training_csv --output-dir ml_results
    python fyp/ml_classifier.py training_csv --gold gold_labels.csv
    python fyp/ml_classifier.py training_csv --save-model tactical_rf.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from features import (  # noqa: E402
    MODEL_FEATURE_COLS,
    build_model_sequence,
)
from train_utils import video_key  # noqa: E402

SEQ_LEN = 29
CLASS_NAMES = [
    "Coordinated Attack",
    "Coordinated Defense",
    "Delayed Support",
    "Spacing Breakdown",
]


# ---------------------------------------------------------------------------
# 1. Sequence -> fixed-length feature vector
# ---------------------------------------------------------------------------

# Each of the 18 permutation-invariant channels is summarised by these six
# statistics. The choice is deliberate rather than exhaustive:
#   mean/std   - level and variability (a formation's spread and its steadiness)
#   min/max    - the extremes that define an event (peak speed IS the spike)
#   slope      - direction of travel over the window; this is the channel the
#                rule engine cannot see at all, because it reads scalars, not
#                trends. "Spacing widening" and "spacing already wide" are
#                different tactical situations and only slope separates them.
#   iqr        - robust spread, unaffected by a single tracking glitch
AGGREGATIONS = ("mean", "std", "min", "max", "slope", "iqr")


def _slope(x: np.ndarray) -> float:
    """Least-squares slope per frame, NaN-aware. 0.0 when under-determined."""
    t = np.arange(len(x), dtype=float)
    m = np.isfinite(x)
    if m.sum() < 2:
        return 0.0
    t, y = t[m], x[m]
    t = t - t.mean()
    denom = float((t * t).sum())
    if denom <= 0:
        return 0.0
    return float((t * (y - y.mean())).sum() / denom)


def sequence_features(seq_raw: np.ndarray) -> np.ndarray:
    """
    (29, 14) raw clip rows -> (18 * 6,) feature vector.

    Routes through `features.build_model_sequence`, so this shares ONE feature
    contract with the Mamba model, `pipeline.py` and `infer_mamba.py`. A model
    trained here can therefore be served live without a second feature path -
    the train/serve mismatch class that cost this project two debugging cycles
    (L6, sec 12.8, sec 13.7).
    """
    return absolute_from_model_sequence(
        build_model_sequence(np.asarray(seq_raw, dtype=float),
                             target_len=SEQ_LEN))


def absolute_from_model_sequence(seq: np.ndarray) -> np.ndarray:
    """Aggregate an already-built (29, 18) model sequence.

    Split out from `sequence_features` because `pipeline.py` hands the
    classifier the MODEL sequence, not the raw 14-column rows - so training and
    live serving must share this half of the path, not just the earlier half.
    """
    out = []
    with warnings.catch_warnings():
        # All-NaN channels are expected (a clip where the ball was never seen);
        # they become NaN here and are imputed inside the sklearn Pipeline.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for c in range(seq.shape[1]):
            ch = seq[:, c]
            out.extend([
                np.nanmean(ch),
                np.nanstd(ch),
                np.nanmin(ch),
                np.nanmax(ch),
                _slope(ch),
                np.nanpercentile(ch, 75) - np.nanpercentile(ch, 25),
            ])
    return np.asarray(out, dtype=float)


def feature_names() -> list[str]:
    return [f"{col}__{agg}" for col in MODEL_FEATURE_COLS for agg in AGGREGATIONS]


# ---------------------------------------------------------------------------
# 1b. Scale-invariant features - the fix for cross-video collapse
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# Measured on the 285-clip set, the absolute feature block scores k-fold
# macro-F1 0.52 but leave-one-video-out macro-F1 0.27 with Cohen's kappa 0.08 -
# i.e. essentially no agreement beyond chance once the camera changes.
#
# The cause is dimensional, not statistical. Without a per-camera homography the
# pipeline falls back to a LINEAR pixel->court scaling, so "cm" means something
# different in every video: a 300 cm nearest-neighbour distance in
# videoplayback (1) and in videoplayback (4) are different physical distances.
# Channels like centroid_x, hull_area, max_pair_dist and ball_x are therefore
# camera-identity features. A learner given them will happily use them to
# recognise WHICH VIDEO it is looking at - exactly the per-video signature
# reported in sec 12.6 - and that skill transfers to no new match.
#
# The fix is to feed the model only quantities that survive an unknown scale
# factor:
#   * RATIOS of two lengths (dimensionless by construction), and
#   * the SHAPE of each channel over time, via per-clip z-scoring, which removes
#     that clip's own level and scale and keeps only how the signal moved.
#
# This is also the tactically correct representation. "The formation is 340 cm
# wide" is a camera measurement; "the formation is twice as wide as its tightest
# pair, and widening" is a volleyball observation.
_EPS = 1e-6
_CH = {c: i for i, c in enumerate(MODEL_FEATURE_COLS)}

# (name, numerator channel, denominator channel) - each is a length/length or
# speed/speed ratio, so any per-camera scale factor cancels.
RATIO_FEATURES = [
    ("compactness",      "nn_dist_mean",     "max_pair_dist"),
    ("tightest_pair",    "nn_dist_min",      "nn_dist_mean"),
    ("loosest_pair",     "nn_dist_max",      "nn_dist_mean"),
    ("spread_aspect",    "spread_x",         "spread_y"),
    ("ball_offset",      "ball_to_centroid", "max_pair_dist"),
    ("speed_ratio",      "speed_top2",       "speed_bot4"),
    ("speed_diff_share", "speed_diff",       "speed_top2"),
]

# Already dimensionless: a count, a cosine similarity and a 0/1 flag.
DIMENSIONLESS_CHANNELS = ("n_present", "sync_inst", "ball_present")

# Aggregations applied to z-scored channels. mean and std are omitted because
# z-scoring fixes them at 0 and 1 - keeping them would add two constant columns.
SHAPE_AGGREGATIONS = ("min", "max", "slope", "iqr")


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.zeros_like(x)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < _EPS:
        return np.zeros_like(x)
    return (x - mu) / sd


def _agg(ch: np.ndarray, which) -> list[float]:
    fns = {
        "mean": lambda a: np.nanmean(a),
        "std": lambda a: np.nanstd(a),
        "min": lambda a: np.nanmin(a),
        "max": lambda a: np.nanmax(a),
        "slope": _slope,
        "iqr": lambda a: np.nanpercentile(a, 75) - np.nanpercentile(a, 25),
    }
    return [float(fns[w](ch)) for w in which]


def invariant_sequence_features(seq_raw: np.ndarray) -> np.ndarray:
    """(29, 14) raw rows -> scale-invariant feature vector.

    Three blocks: dimensionless ratios, already-dimensionless channels, and the
    temporal shape of every channel after per-clip z-scoring.
    """
    return invariant_from_model_sequence(
        build_model_sequence(np.asarray(seq_raw, dtype=float),
                             target_len=SEQ_LEN))


def invariant_from_model_sequence(seq: np.ndarray) -> np.ndarray:
    """Scale-invariant aggregation of an already-built (29, 18) sequence."""
    out: list[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for _name, num, den in RATIO_FEATURES:
            a = seq[:, _CH[num]]
            b = seq[:, _CH[den]]
            r = a / (np.abs(b) + _EPS)
            r = np.clip(r, -50.0, 50.0)     # a near-zero denominator must not
            out.extend(_agg(r, AGGREGATIONS))   # become an outlier that dominates

        for col in DIMENSIONLESS_CHANNELS:
            out.extend(_agg(seq[:, _CH[col]], AGGREGATIONS))

        for c in range(seq.shape[1]):
            out.extend(_agg(_zscore(seq[:, c]), SHAPE_AGGREGATIONS))

    return np.asarray(out, dtype=float)


def invariant_feature_names() -> list[str]:
    names = [f"{n}__{a}" for (n, _, _) in RATIO_FEATURES for a in AGGREGATIONS]
    names += [f"{c}__{a}" for c in DIMENSIONLESS_CHANNELS for a in AGGREGATIONS]
    names += [f"{c}__z_{a}" for c in MODEL_FEATURE_COLS for a in SHAPE_AGGREGATIONS]
    return names


def feature_groups(mode: str) -> dict[str, list[int]]:
    """Column indices grouped by source channel.

    Permutation importance over SINGLE columns is meaningless here: the six
    statistics of one channel are strongly correlated, so shuffling one leaves
    the model five substitutes and the measured drop is ~0 (observed:
    every importance below 0.0005). Shuffling a whole channel at once asks the
    question that has an answer - and the answer is one a volleyball expert can
    argue with.
    """
    names = feature_names() if mode == "absolute" else invariant_feature_names()
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        base = n.split("__")[0]
        groups.setdefault(base, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# 2. Dataset loading
# ---------------------------------------------------------------------------

def extract(seq_raw: np.ndarray, mode: str = "invariant") -> np.ndarray:
    """Feature extractor selected by mode: 'absolute' or 'invariant'. ('both'
    is still accepted for loading a legacy .joblib, but is no longer a
    training choice - S4.)"""
    return features_from_model_sequence(
        build_model_sequence(np.asarray(seq_raw, dtype=float),
                             target_len=SEQ_LEN),
        mode,
    )


def features_from_model_sequence(seq: np.ndarray,
                                 mode: str = "invariant") -> np.ndarray:
    """
    (29, 18) model sequence -> feature vector, in the SAME order used at fit
    time.

    This is the one function both training and live inference call, so a
    trained .joblib cannot silently receive differently-ordered features than it
    was fitted on. Guarding this is not paranoia - a train/serve feature
    mismatch has already cost this project two debugging cycles (L6, sec 12.8).
    """
    seq = np.asarray(seq, dtype=float)
    if mode == "absolute":
        return absolute_from_model_sequence(seq)
    if mode == "invariant":
        return invariant_from_model_sequence(seq)
    if mode == "both":
        return np.concatenate([absolute_from_model_sequence(seq),
                               invariant_from_model_sequence(seq)])
    raise ValueError(f"unknown feature mode {mode!r}")


def names_for(mode: str) -> list[str]:
    if mode == "absolute":
        return feature_names()
    if mode == "invariant":
        return invariant_feature_names()
    return feature_names() + invariant_feature_names()


def load_dataset(csv_dir: str, label_map: dict | None = None,
                 mode: str = "invariant"):
    """
    Read every labelled clip CSV in `csv_dir`.

    Returns (X, y, groups, files):
      X      (n, 108) float
      y      (n,)     int  index into CLASS_NAMES
      groups (n,)     str  source-video key, for leave-one-video-out
      files  (n,)     str  basename, so predictions stay traceable to a clip
    """
    import pandas as pd

    X, y, groups, files = [], [], [], []
    skipped = []

    for fname in sorted(os.listdir(csv_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception as exc:               # noqa: BLE001
            skipped.append((fname, f"unreadable: {exc}"))
            continue

        if "target_label" not in df.columns:
            skipped.append((fname, "no target_label column"))
            continue

        label = str(df["target_label"].iloc[0]).strip()
        if label_map:
            label = label_map.get(label, label)
        if label not in CLASS_NAMES:
            skipped.append((fname, f"label not in the 4 classes: {label!r}"))
            continue

        drop = [c for c in ("frame_id", "target_label") if c in df.columns]
        raw = df.drop(columns=drop).to_numpy(dtype=float)
        if raw.shape[0] < 2:
            skipped.append((fname, "fewer than 2 frames"))
            continue

        X.append(extract(raw, mode))
        y.append(CLASS_NAMES.index(label))
        groups.append(video_key(fname))
        files.append(fname)

    if not X:
        raise SystemExit(f"No usable labelled CSVs found in {csv_dir!r}")

    return (np.vstack(X), np.asarray(y, dtype=int),
            np.asarray(groups), np.asarray(files)), skipped


# ---------------------------------------------------------------------------
# 3. Model zoo
# ---------------------------------------------------------------------------

def build_models(seed: int = 0) -> dict:
    """
    Candidate classifiers, each wrapped in a Pipeline that owns its own
    imputation and (where needed) scaling.

    Putting the preprocessing INSIDE the pipeline is not a style choice: with
    cross-validation, fitting a scaler or imputer on the full dataset before
    splitting leaks test-fold statistics into training and inflates every score.
    This is the single most common way a CV number becomes indefensible.

    `class_weight='balanced'` everywhere it exists, because Delayed Support has
    16 examples against Coordinated Defense's 134 - a 8:1 imbalance that an
    unweighted learner answers by never predicting the minority class at all.
    """
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier,
    )
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def scaled(clf):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])

    def reduced(clf):
        # S4 (AUDIT_FINDINGS.md): 132 features over ~150 rows/fold with no
        # dimensionality reduction. Distance- and kernel-based learners (kNN,
        # RBF-SVM, MLP) degrade badly in that regime - the both-mode winner
        # was kNN in 240 dimensions. PCA(0.95) inside the Pipeline decorrelates
        # and drops the noise dimensions, and being a step it REFITS on each
        # CV fold's training rows only (no leakage). Tree ensembles below keep
        # the raw features on purpose: each split already selects one feature,
        # so they self-regularise, and PCA would destroy the channel semantics
        # the permutation-importance analysis depends on.
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=seed)),
            ("clf", clf),
        ])

    def bare(clf):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])

    return {
        "logistic_regression": scaled(LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed)),
        "linear_svm": scaled(SVC(
            kernel="linear", probability=True, class_weight="balanced",
            random_state=seed)),
        "rbf_svm": reduced(SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=seed)),
        "knn": reduced(KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "mlp": reduced(MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=3000, alpha=1e-2,
            random_state=seed)),
        "random_forest": bare(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
        "extra_trees": bare(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
        "gradient_boosting": bare(GradientBoostingClassifier(
            random_state=seed)),
        "hist_gradient_boosting": bare(HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=seed)),
    }


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(y_true, y_pred, n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """S2 (AUDIT_FINDINGS.md): a single point estimate of macro-F1 on ~285
    clips says nothing about how much of it is sampling noise. Clip-level
    bootstrap: resample the (y_true, y_pred) pairs with replacement `n_boot`
    times, recompute macro-F1 each time, report the [alpha/2, 1-alpha/2]
    percentile interval. Clip-level (not frame-level) because the clip is the
    independent unit - frames within a clip are not.
    """
    from sklearn.metrics import f1_score, cohen_kappa_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    f1s, kappas = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        f1s[b] = f1_score(y_true[idx], y_pred[idx], average="macro",
                          zero_division=0)
        kappas[b] = cohen_kappa_score(y_true[idx], y_pred[idx])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "macro_f1": {
            "point": round(float(f1_score(y_true, y_pred, average="macro",
                                          zero_division=0)), 4),
            "ci_low": round(float(np.percentile(f1s, lo)), 4),
            "ci_high": round(float(np.percentile(f1s, hi)), 4),
        },
        "cohen_kappa": {
            "point": round(float(cohen_kappa_score(y_true, y_pred)), 4),
            "ci_low": round(float(np.percentile(kappas, lo)), 4),
            "ci_high": round(float(np.percentile(kappas, hi)), 4),
        },
        "n_boot": int(n_boot), "n_clips": int(n),
        "confidence": round(1 - alpha, 2),
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """S2: paired comparison of two classifiers on the SAME clips. McNemar's
    test on the discordant pairs (a right / b wrong vs a wrong / b right) -
    the correct test for 'is classifier A significantly better than B', since
    an unpaired comparison of two macro-F1s ignores that they saw identical
    inputs. Exact binomial p-value (no chi-square approximation - the
    discordant counts here are small).
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    a_only = int(np.sum(a_ok & ~b_ok))
    b_only = int(np.sum(~a_ok & b_ok))
    n_disc = a_only + b_only
    p = 1.0 if n_disc == 0 else float(
        binomtest(min(a_only, b_only), n_disc, 0.5).pvalue)
    return {
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "n_discordant": n_disc,
        "p_value": round(p, 4),
        "significant_at_0.05": bool(p < 0.05),
        "favours": ("a" if a_only > b_only else
                    "b" if b_only > a_only else "tie"),
    }


def _scores(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def _ranking_scores(y_true, proba, n_classes: int) -> dict:
    """
    One-vs-rest AUC-ROC and AUC-PR, macro-averaged across classes plus
    per-class, computed on POOLED out-of-fold predictions (consistent with how
    `_scores` already treats `cross_val_predict` output - pooling avoids the
    degenerate per-fold case where a fold's test set has zero examples of the
    16-example Delayed Support class).

    WHY THIS EXISTS (senior-data-scientist checklist: "Always report AUC-PR
    alongside AUC-ROC for imbalanced datasets")
    -----------------------------------------------------------------------
    Accuracy, macro-F1 and kappa - everything reported before this function
    existed - are all HARD-LABEL metrics: they only see the argmax of the
    softmax. On this 8.4:1-imbalanced dataset the argmax for Delayed Support
    and Spacing Breakdown is very often wrong (measured recall 0.06 and 0.02
    respectively, sec 14.4), which makes it look like the model has learned
    NOTHING about those two classes. AUC-PR asks a different, more forgiving
    question: even where the argmax is wrong, does the model's probability for
    the true class still rank higher than it would by chance? A macro-F1 near
    zero with an AUC-PR clearly above the class's prevalence means the
    probabilities carry usable signal that a smarter decision rule (e.g. a
    per-class threshold, or a human reviewing the top-k candidates) could
    still exploit - a materially different and more useful finding than "the
    model has no signal for this class" would be.

    AUC-PR is preferred over AUC-ROC as the headline for the minority classes
    specifically because ROC is insensitive to class imbalance (a classifier
    can look excellent on ROC while being useless in practice when negatives
    vastly outnumber positives); both are still reported since ROC-AUC is the
    more familiar reference point.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize

    labels = list(range(n_classes))
    y_bin = label_binarize(y_true, classes=labels)

    out = {"roc_auc_macro": None, "pr_auc_macro": None,
          "per_class_pr_auc": [None] * n_classes,
          "per_class_roc_auc": [None] * n_classes}
    with warnings.catch_warnings():
        # A class entirely absent from y_true in this pool is expected on tiny
        # synthetic test inputs and handled below via the ValueError branch;
        # sklearn also emits an UndefinedMetric/User warning for the same
        # condition, which would otherwise spam every reproduce.py run.
        warnings.simplefilter("ignore")
        try:
            out["roc_auc_macro"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                             labels=labels))
            per_class_roc = roc_auc_score(y_true, proba, multi_class="ovr",
                                          average=None, labels=labels)
            out["per_class_roc_auc"] = [float(v) for v in per_class_roc]
        except ValueError:
            pass  # a class entirely absent from y_true in this pool - skip
        try:
            out["pr_auc_macro"] = float(
                average_precision_score(y_bin, proba, average="macro"))
            per_class_pr = average_precision_score(y_bin, proba, average=None)
            out["per_class_pr_auc"] = [float(v) for v in np.atleast_1d(per_class_pr)]
        except ValueError:
            pass
    return out


def _overfit_gap(model, X, y, cv, groups=None, seed: int = 0) -> dict:
    """
    Train-score minus held-out-score, macro-F1, for the SAME cv splitter used
    for the headline numbers - PLUS (S5) the per-fold held-out macro-F1 and
    balanced-accuracy, so a mean ± SD and a per-fold table can sit next to
    the pooled `cross_val_predict` headline.

    WHY THE GAP EXISTS (senior-data-scientist checklist: "Check overfit_gap >
    0.05 as a warning sign of overfitting")
    ------------------------------------------------------------------------
    This project already knows the Mamba SSM overfits badly (train_acc 1.00 vs
    val_acc 0.66, README/METHODOLOGY sec 5) and knows the sklearn models show a
    LOVO-vs-k-fold protocol gap (sec 14.3). Neither of those is the same
    measurement as this one: this asks, WITHIN a single fold, how much better
    does a model fit the rows it trained on than the rows it did not - the
    classic overfitting diagnostic, independent of which protocol produced the
    fold.

    WHY THE PER-FOLD BREAKDOWN (S5, AUDIT_FINDINGS.md)
    -------------------------------------------------
    The headline macro-F1/kappa come from POOLED out-of-fold predictions
    (`cross_val_predict`); this table was previously mean-of-per-fold - two
    aggregations printed side by side as if comparable. Per-fold macro-F1 is
    noisy when a fold's test set has few or zero examples of a rare class
    (why the pooled number stays the headline), and the SD makes that
    visible instead of hiding it. One `cross_validate` call feeds both.
    """
    from sklearn.model_selection import cross_validate

    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cvres = cross_validate(
            model, X, y, cv=cv, groups=groups, scoring=scoring,
            return_train_score=True, n_jobs=1,
        )
    train_mean = float(np.mean(cvres["train_macro_f1"]))
    test_mean = float(np.mean(cvres["test_macro_f1"]))
    out = {
        "train_macro_f1": round(train_mean, 4),
        "test_macro_f1": round(test_mean, 4),
        "overfit_gap": round(train_mean - test_mean, 4),
        "overfitting": bool(train_mean - test_mean > 0.05),
    }
    for key in scoring:
        vals = [float(v) for v in cvres[f"test_{key}"]]
        out[f"{key}_per_fold"] = [round(v, 4) for v in vals]
        out[f"{key}_fold_mean"] = round(float(np.mean(vals)), 4)
        out[f"{key}_fold_std"] = round(float(np.std(vals)), 4)
    return out


def cross_validate_models(X, y, groups, seed: int = 0, n_splits: int = 5):
    """
    Two protocols, because they answer two different questions and only
    reporting the friendlier one is how evaluation chapters get torn apart:

      STRATIFIED K-FOLD - clips from one video land on both sides of the split.
      Measures WITHIN-video generalisation. It is the number comparable to the
      project's existing 63.6%, and it is optimistic.

      LEAVE-ONE-VIDEO-OUT - every clip of a held-out video is unseen. Measures
      CROSS-video generalisation, which is what "will this work on next week's
      match" actually means. It is the honest number, and it will be lower.

    The GAP between them is itself a finding (sec 12.6: the model can learn which
    camera it is looking at). Report both, with the gap explained.

    Each model is additionally scored on two axes the accuracy/F1/kappa metrics
    above cannot see (senior-data-scientist checklist, see `_ranking_scores` and
    `_overfit_gap` docstrings): probability-ranking quality (AUC-ROC / AUC-PR),
    and within-fold overfitting (train vs held-out score gap).
    """
    from sklearn.model_selection import (
        StratifiedKFold, LeaveOneGroupOut, cross_val_predict,
    )

    results = {}
    models = build_models(seed)
    n_classes = len(np.unique(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    logo = LeaveOneGroupOut()

    for name, model in models.items():
        row = {"model": name}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_k = cross_val_predict(model, X, y, cv=skf, n_jobs=1)
            for k, v in _scores(y, pred_k).items():
                row[f"kfold_{k}"] = v

            if hasattr(build_models(seed)[name], "predict_proba"):
                proba_k = cross_val_predict(model, X, y, cv=skf, n_jobs=1,
                                            method="predict_proba")
                for k, v in _ranking_scores(y, proba_k, n_classes).items():
                    row[f"kfold_{k}"] = v

            gap_k = _overfit_gap(model, X, y, skf, seed=seed)
            row["kfold_train_macro_f1"] = gap_k["train_macro_f1"]
            row["kfold_overfit_gap"] = gap_k["overfit_gap"]
            row["kfold_overfitting"] = gap_k["overfitting"]
            for k in ("macro_f1_per_fold", "macro_f1_fold_mean",
                      "macro_f1_fold_std", "balanced_accuracy_per_fold",
                      "balanced_accuracy_fold_mean", "balanced_accuracy_fold_std"):
                row[f"kfold_{k}"] = gap_k[k]

            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                pred_l = cross_val_predict(model, X, y, cv=logo, groups=groups,
                                           n_jobs=1)
                for k, v in _scores(y, pred_l).items():
                    row[f"lovo_{k}"] = v

                if hasattr(build_models(seed)[name], "predict_proba"):
                    proba_l = cross_val_predict(model, X, y, cv=logo,
                                                groups=groups, n_jobs=1,
                                                method="predict_proba")
                    for k, v in _ranking_scores(y, proba_l, n_classes).items():
                        row[f"lovo_{k}"] = v
                    # retained so the decision rule can be varied without refit
                    row["_proba_lovo"] = proba_l

                gap_l = _overfit_gap(model, X, y, logo, groups=groups, seed=seed)
                row["lovo_train_macro_f1"] = gap_l["train_macro_f1"]
                row["lovo_overfit_gap"] = gap_l["overfit_gap"]
                row["lovo_overfitting"] = gap_l["overfitting"]
                for k in ("macro_f1_per_fold", "macro_f1_fold_mean",
                          "macro_f1_fold_std", "balanced_accuracy_per_fold",
                          "balanced_accuracy_fold_mean",
                          "balanced_accuracy_fold_std"):
                    row[f"lovo_{k}"] = gap_l[k]

                row["_pred_lovo"] = pred_l
            row["_pred_kfold"] = pred_k
        results[name] = row

    return results


def calibrate(model, X, y, seed: int = 0, method: str = "sigmoid", cv: int = 5):
    """
    Wrap a fitted-able pipeline in cross-validated probability calibration.

    WHY THIS IS NOT OPTIONAL HERE
    -----------------------------
    Gradient-boosted trees fitted on 285 clips emit probabilities of 1.000 on
    almost every window - measured live: conf=1.000, entropy=0.00 bits on four
    of five sequences. That silently disables two features this project already
    depends on:

      * the anomaly / tactical-deviation flag, which fires on low confidence and
        therefore never fires at all;
      * the Shannon-entropy channel in the per-sequence CSV and the HUD, which
        the report presents as a "how unusual is this play" cue.

    An uncalibrated 1.000 is not a claim of certainty, it is an artefact of
    ensembles voting unanimously on training-set-like inputs. Calibration maps
    scores back onto frequencies that mean what the report says they mean.

    `sigmoid` (Platt) rather than `isotonic`: isotonic is non-parametric and
    needs far more data per class than the 16 examples in Delayed Support, where
    it would overfit the calibration curve itself.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold

    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    smallest = int(counts[counts > 0].min())
    folds = max(2, min(cv, smallest))
    if smallest < 2:
        return model, {"calibrated": False,
                       "reason": f"smallest class has {smallest} sample(s)"}

    cal = CalibratedClassifierCV(
        model, method=method,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.fit(X, y)
    return cal, {"calibrated": True, "method": method, "cv_folds": folds}


def confidence_profile(model, X) -> dict:
    """Summarise how confident a model is - the check that catches a model
    reporting 1.000 on everything."""
    p = model.predict_proba(X)
    conf = p.max(axis=1)
    ent = np.array([shannon_entropy(row) for row in p])
    return {
        "mean_confidence": float(conf.mean()),
        "median_confidence": float(np.median(conf)),
        "frac_confidence_above_0.99": float((conf > 0.99).mean()),
        "mean_entropy_bits": float(ent.mean()),
    }


def shannon_entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def reliability_curve(y_true, proba, n_bins: int = 10) -> dict:
    """S8 (AUDIT_FINDINGS.md): calibration efficacy was only ever measured
    IN-SAMPLE (`confidence_profile` on the same X the calibrator was fit on).
    This computes a reliability curve on whatever probabilities are passed -
    intended for the POOLED out-of-fold LOVO probabilities, where "the model
    says 0.9, is it right 90% of the time" is an honest cross-video question.

    Returns per-bin (mean predicted confidence, empirical accuracy, count)
    plus the Expected Calibration Error (ECE) - the count-weighted mean gap
    between the two, the single number to quote.
    """
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, n = [], 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        c = int(m.sum())
        if c == 0:
            continue
        mean_conf = float(conf[m].mean())
        acc = float(correct[m].mean())
        bins.append({"bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
                     "mean_confidence": round(mean_conf, 4),
                     "empirical_accuracy": round(acc, 4), "count": c})
        ece += (c / n) * abs(mean_conf - acc)
    return {"bins": bins, "ece": round(float(ece), 4), "n": int(n)}


def prior_corrected_predict(proba: np.ndarray, class_counts: np.ndarray
                            ) -> np.ndarray:
    """
    Decision rule for imbalanced classes: argmax of p(k|x) / p(k) instead of
    argmax of p(k|x).

    WHY, AND WHY THIS RATHER THAN A TUNED THRESHOLD
    -----------------------------------------------
    `class_weight="balanced"` reweights the LOSS during fitting, but the final
    decision is still a plain argmax over the posterior, and with an 8.4:1
    imbalance the posterior for a 16-example class rarely wins that argmax
    outright. Measured consequence: Delayed Support recall 0.06 and Spacing
    Breakdown recall 0.02 - the model almost never *names* those classes even
    when it ranks them plausibly (see the AUC-PR figures, which sit above their
    class prevalence).

    Dividing the posterior by the training prior converts it to the posterior
    that a balanced training set would have produced:

        p_bal(k|x)  ∝  p(k|x) / p(k)

    This is the standard prior-correction / balanced-posterior rule. Two
    properties make it the right first choice here:

      * It is PARAMETER-FREE given the class counts, so unlike a tuned
        per-class threshold it cannot overfit the evaluation data - which
        matters enormously when the smallest class has 16 examples spread over
        4 folds and any tuned threshold would be fitted on ~4 points.
      * It trades majority-class recall for minority-class recall in a way that
        is predictable and explainable to a panel, rather than an opaque
        search result.

    It will usually LOWER raw accuracy and RAISE macro-F1 / balanced accuracy.
    That is the intended trade, and it is the correct trade whenever the
    minority classes are the ones the application cares about - a coach is not
    helped by a system that never says "Delayed Support".
    """
    proba = np.asarray(proba, dtype=float)
    counts = np.asarray(class_counts, dtype=float)
    prior = counts / max(counts.sum(), 1.0)
    prior = np.clip(prior, 1e-9, None)
    return np.argmax(proba / prior[None, :], axis=1)


def _prior_corrected_predict_per_fold(y, proba, groups) -> np.ndarray:
    """S6 (AUDIT_FINDINGS.md): `prior_corrected_predict` divides by
    `np.bincount(y)` over ALL rows - in a LOVO evaluation that prior includes
    the held-out video, a leak. This applies the correction group by group,
    each held-out group's rows divided by the prior of the OTHER groups only
    (its actual training prior), so the reported number has no leak.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_classes = proba.shape[1]
    pred = np.empty(len(y), dtype=int)
    for g in np.unique(groups):
        test = groups == g
        train_prior = np.bincount(y[~test], minlength=n_classes)
        pred[test] = prior_corrected_predict(proba[test], train_prior)
    return pred


def compare_decision_rules(y, proba, class_counts, groups=None) -> dict:
    """Score plain argmax against the prior-corrected rule on the same
    out-of-fold probabilities. No refitting, so the comparison is free and
    isolates the decision rule from the model.

    If `groups` is given, the prior-corrected rule uses each fold's TRAINING
    prior (S6, no leak) instead of the global `class_counts`; the global
    version is still returned as `prior_corrected_global` for comparison.
    """
    from sklearn.metrics import recall_score

    def _row(pred):
        s = _scores(y, pred)
        s["per_class_recall"] = [
            float(v) for v in recall_score(
                y, pred, labels=list(range(len(class_counts))),
                average=None, zero_division=0)
        ]
        return s

    out = {"argmax": _row(np.argmax(proba, axis=1))}
    if groups is not None:
        out["prior_corrected"] = _row(
            _prior_corrected_predict_per_fold(y, proba, groups))
        out["prior_corrected_global"] = _row(
            prior_corrected_predict(proba, class_counts))
        out["_prior"] = "per-fold training prior (no leak)"
    else:
        out["prior_corrected"] = _row(
            prior_corrected_predict(proba, class_counts))
        out["_prior"] = "global (includes held-out fold - illustrative only)"
    return out


def majority_baseline(y) -> dict:
    """The number every reported score must beat to mean anything."""
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    pred = np.full_like(y, int(counts.argmax()))
    return _scores(y, pred)


def rule_engine_baseline(csv_dir: str, files, y_true) -> dict:
    """
    Run the hand-coded rule engine (`label_clips.label_clip`) on exactly the
    window the ML model sees, and score it against the same ground truth.

    WHY THIS IS THE CORRECT BASELINE - AND WHY AN EARLIER VERSION OF THIS
    FUNCTION WAS WRONG
    --------------------------------------------------------------------
    This function previously returned None, on the stated grounds that "the
    rule engine PRODUCED y, so its agreement with y is 100% by construction".
    **That was factually incorrect and the error mattered**, because it is the
    exact assumption behind the "circular training" objection.

    Verified label provenance (`fyp/extract_clips.py`):
      labels.csv (HUMAN annotation: video_file, anchor_time, tactic_class)
        -> ffmpeg cuts a 1-second clip per annotated timestamp
        -> the clip is written into dataset/<tactic_class>/
        -> prepare_training_data.py takes the label from THAT FOLDER NAME
        -> training_csv/*.csv `target_label`

    `label_clips.py` never touches that chain: it is a separate, alternative
    labelling path that adds a `target_label` column to 41-frame CSVs, and it
    does not create or populate `dataset/<class>/`. Checked directly: all 159
    resolvable rows of labels.csv match their clip's folder exactly, with zero
    mismatches.

    So `y` is HUMAN ground truth, the rule engine is an independent competing
    predictor of it, and the comparison is both meaningful and fair - the rules
    and the model receive identical input and are scored against identical
    targets. This is the number that answers "does the learned model add
    anything over the hand-coded heuristic?".

    Returns None only if `label_clips` cannot be imported or no clip is usable.
    """
    import pandas as pd

    try:
        from label_clips import label_clip
    except Exception:                                    # noqa: BLE001
        return None

    preds, keep = [], []
    n_abstain = 0
    for i, fname in enumerate(files):
        path = os.path.join(csv_dir, str(fname))
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            obs = df[df["frame_id"] <= SEQ_LEN].copy() if "frame_id" in df else df
            if obs.empty:
                continue
            label = label_clip(obs)
        except Exception:                                # noqa: BLE001
            continue
        if label not in CLASS_NAMES:
            # The rules decline to commit ("Unclassified"). Counted, and scored
            # as a miss rather than silently dropped - a baseline that abstains
            # on hard cases must not be credited for the ones it skipped.
            n_abstain += 1
            preds.append(-1)
        else:
            preds.append(CLASS_NAMES.index(label))
        keep.append(i)

    if not keep:
        return None

    y_t = np.asarray(y_true)[np.asarray(keep, dtype=int)]
    y_p = np.asarray(preds, dtype=int)

    with warnings.catch_warnings():
        # -1 is the abstention sentinel and is deliberately absent from y_true;
        # sklearn warns about the unseen class, which is expected here.
        warnings.simplefilter("ignore")
        scored = _scores(y_t, y_p)
    scored["n_clips"] = int(len(y_t))
    scored["n_abstained"] = int(n_abstain)
    scored["abstain_rate"] = round(n_abstain / max(len(y_t), 1), 4)
    # underscore-prefixed: per-clip rule predictions + the row indices they
    # cover, so a caller can pair them against the model for McNemar (S2).
    scored["_pred"] = y_p
    scored["_keep_idx"] = np.asarray(keep, dtype=int)
    return scored


def gold_clip_names(gold_csv: str) -> set:
    """Bare `clip` column of a gold CSV, for excluding those rows from a
    model's training set before scoring against them (R2/Q2: the model
    passed to `evaluate_against_gold` must never have been fitted on the
    clips it is then scored against, or the comparison is in-sample, not
    held-out). No label/class validation here on purpose - this is only used
    to build an exclusion mask, so a name that doesn't end up matching any
    training row is harmless.
    """
    import pandas as pd

    gold = pd.read_csv(gold_csv)
    if "clip" not in gold.columns:
        return set()
    return {str(c).strip() for c in gold["clip"]}


def evaluate_against_gold(model, csv_dir: str, gold_csv: str,
                          mode: str = "invariant") -> dict:
    """
    TIER 2 - the number that answers the examiner.

    `gold_csv` is produced by `fyp/annotate.py`: columns `clip,label` where
    `label` is a human judgement. Both the rule engine's label (already in the
    clip CSV) and the model's prediction are scored against it.

    If the model beats the rules against human judgement, it has denoised its
    teacher rather than merely copied it - and that is a defensible,
    literature-backed claim (weak supervision). If it does not, say so: the
    honest negative result is worth more than an unfalsifiable positive one.
    """
    import pandas as pd

    gold = pd.read_csv(gold_csv)
    if not {"clip", "label"}.issubset(gold.columns):
        raise SystemExit("gold CSV needs columns: clip,label")

    y_h, y_rule, feats, used = [], [], [], []
    for _, r in gold.iterrows():
        human = str(r["label"]).strip()
        if human not in CLASS_NAMES:
            continue
        path = os.path.join(csv_dir, str(r["clip"]).strip())
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "target_label" not in df.columns:
            continue
        rule = str(df["target_label"].iloc[0]).strip()
        if rule not in CLASS_NAMES:
            continue
        drop = [c for c in ("frame_id", "target_label") if c in df.columns]
        # MUST use the same feature mode the model was fitted with; using the
        # default here silently feeds a 132-column model a 108-column vector.
        feats.append(extract(df.drop(columns=drop).to_numpy(dtype=float), mode))
        y_h.append(CLASS_NAMES.index(human))
        y_rule.append(CLASS_NAMES.index(rule))
        used.append(str(r["clip"]).strip())

    if not feats:
        raise SystemExit("No gold rows matched a clip CSV with a rule label.")

    y_h = np.asarray(y_h)
    y_rule = np.asarray(y_rule)
    y_model = model.predict(np.vstack(feats))

    return {
        "n_gold": int(len(y_h)),
        "rules_vs_human": _scores(y_h, y_rule),
        "model_vs_human": _scores(y_h, y_model),
        "model_vs_rules": _scores(y_rule, y_model),
        "clips": used,
    }


def grouped_permutation_importance(model_name, X, y, groups, mode: str,
                                   seed: int = 0, n_repeats: int = 5,
                                   top: int = 20):
    """
    Channel-level importance, measured OUT OF SAMPLE.

    Two corrections over the textbook call, both of which changed the answer on
    this dataset:

    1. SHUFFLE THE CHANNEL, NOT THE COLUMN. The six statistics of a channel are
       strongly correlated, so removing one leaves five substitutes and the
       measured drop collapses to ~0 - observed here, where every single-column
       importance came out below 0.0005 and the ranking was pure noise. The
       block permutation removes the information rather than one encoding of it.

    2. MEASURE ON HELD-OUT DATA. Importance computed on the data the model was
       fitted to is close to meaningless for a high-capacity learner: fitted on
       all 285 clips the winning model scores in-sample macro-F1 1.000, i.e. it
       has memorised, and it can answer from any surviving column no matter what
       is shuffled. Every channel then looks unimportant. Here the model is
       refitted per leave-one-video-out fold and permuted on the held-out video,
       so the number means "how much CROSS-VIDEO performance depended on this
       measurement" - which is the only version worth showing an expert.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import LeaveOneGroupOut

    rng = np.random.default_rng(seed)
    chan_cols = feature_groups(mode)
    logo = LeaveOneGroupOut()

    base_scores: list[float] = []
    drops: dict[str, list[float]] = {c: [] for c in chan_cols}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in logo.split(X, y, groups):
            mdl = build_models(seed)[model_name]
            mdl.fit(X[tr], y[tr])
            base = f1_score(y[te], mdl.predict(X[te]),
                            average="macro", zero_division=0)
            base_scores.append(base)

            for channel, cols in chan_cols.items():
                for _ in range(n_repeats):
                    Xp = X[te].copy()
                    Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
                    drops[channel].append(
                        base - f1_score(y[te], mdl.predict(Xp),
                                        average="macro", zero_division=0))

    out = [{
        "channel": c,
        "n_columns": len(chan_cols[c]),
        "importance_mean": float(np.mean(v)),
        "importance_std": float(np.std(v)),
    } for c, v in drops.items()]
    out.sort(key=lambda d: -d["importance_mean"])

    return {
        "baseline_macro_f1": float(np.mean(base_scores)),
        "protocol": "leave-one-video-out, permuted on the held-out video",
        "channels": out[:top],
    }


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------

def confusion_table(y_true, y_pred) -> str:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    short = [c[:14] for c in CLASS_NAMES]
    w = max(len(s) for s in short) + 2

    lines = [" " * w + "".join(f"{s[:9]:>11}" for s in short) + "   <- predicted"]
    for i, row in enumerate(cm):
        total = row.sum()
        recall = row[i] / total if total else 0.0
        lines.append(f"{short[i]:<{w}}"
                     + "".join(f"{v:>11d}" for v in row)
                     + f"   recall {row[i]}/{total} = {recall:.2f}")
    return "\n".join(lines)


def per_class_report(y_true, y_pred) -> str:
    from sklearn.metrics import classification_report
    return classification_report(
        y_true, y_pred, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0, digits=3,
    )



def _run_ablation(args) -> int:
    """Same models, same folds, the two feature blocks - isolates the effect
    of the representation from the effect of the learner. ('both'/union was
    dropped, S4.)"""
    print("\n" + "=" * 74)
    print("FEATURE-BLOCK ABLATION")
    print("=" * 74)
    print("\nAbsolute features are court-cm aggregates. Without a per-camera")
    print("homography those centimetres are not comparable between videos, so a")
    print("learner can use them to identify the CAMERA rather than the tactic.")
    print("Invariant features are dimensionless ratios plus per-clip z-scored")
    print("temporal shape, both of which survive an unknown scale factor.")
    print("\nIf LOVO improves under 'invariant', the per-video signature reported")
    print("in sec 12.6 was substantially a units problem, not a data-size problem.\n")

    rows = []
    for mode in ("absolute", "invariant"):
        (X, y, groups, _f), _s = load_dataset(args.csv_dir, mode=mode)
        res = cross_validate_models(X, y, groups, seed=args.seed,
                                    n_splits=args.folds)
        best = max(res, key=lambda k: res[k].get("lovo_macro_f1", -1))
        rows.append((mode, X.shape[1], best,
                     res[best].get("kfold_macro_f1", float("nan")),
                     res[best].get("lovo_macro_f1", float("nan")),
                     res[best].get("lovo_accuracy", float("nan")),
                     res[best].get("lovo_cohen_kappa", float("nan"))))
        print(f"  {mode:<10} n_feat={X.shape[1]:<4} best={best:<24} "
              f"kfold_F1={rows[-1][3]:.3f}  LOVO_F1={rows[-1][4]:.3f}  "
              f"kappa={rows[-1][6]:+.3f}")

    print("\n" + "-" * 74)
    hdr = f"{'features':<12}{'n':>5}{'best model':>26}{'kfoldF1':>9}{'lovoF1':>9}{'kappa':>8}"
    print(hdr)
    print("-" * len(hdr))
    for mode, n, best, kf, lf, la, kp in rows:
        print(f"{mode:<12}{n:>5}{best:>26}{kf:>9.3f}{lf:>9.3f}{kp:>8.3f}")

    a = dict((r[0], r[4]) for r in rows)
    delta = a["invariant"] - a["absolute"]
    print(f"\ninvariant - absolute, LOVO macro-F1: {delta:+.3f}")
    if delta > 0.02:
        print("  -> Scale-invariant features generalise better across cameras.")
        print("     Report this ablation: it converts 'our model does not transfer'")
        print("     into 'our model transfers once the features are dimensionless'.")
    else:
        print("  -> No material gain. The cross-video gap is NOT mainly a units")
        print("     problem on this data; report that honestly and look to label")
        print("     quality and sample size instead.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Learned tactical classifier (scikit-learn).")
    ap.add_argument("csv_dir", help="Directory of labelled clip CSVs.")
    ap.add_argument("--output-dir", default="ml_results")
    ap.add_argument("--gold", default=None,
                    help="Human-labelled CSV (clip,label) for the Tier-2 test.")
    ap.add_argument("--save-model", default=None,
                    help="Path to persist the winning pipeline (.joblib).")
    ap.add_argument("--export-all-models", default=None, metavar="DIR",
                    help="Fit, calibrate and save EVERY model in the zoo as its "
                         "own serveable <name>.joblib in DIR (same bundle format "
                         "as --save-model). Lets pipeline.py render an annotated "
                         "video from each classifier for a side-by-side "
                         "comparison. These are NOT the pre-declared headline - "
                         "quote only the --declared-model's number.")
    ap.add_argument("--features", choices=("absolute", "invariant"),
                    default="invariant",
                    help="Feature block. 'absolute' = raw court-cm aggregates "
                         "(camera-dependent under the linear fallback); "
                         "'invariant' = dimensionless ratios + per-clip "
                         "z-scored temporal shape (default). ('both' was "
                         "dropped - S4: its only winner was kNN in 240 "
                         "dimensions over ~150 rows, indefensible.)")
    ap.add_argument("--ablate-features", action="store_true",
                    help="Run both feature blocks (absolute, invariant) and "
                         "print the comparison - the ablation an examiner "
                         "will ask for.")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip probability calibration. Not recommended: the "
                         "tree ensembles emit 1.000 confidence on nearly every "
                         "window, which disables the entropy/anomaly channel.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--select-by", default="lovo_macro_f1",
                    help="Metric used to RANK the model zoo in the exploratory "
                         "table (default lovo_macro_f1 - the honest one).")
    ap.add_argument("--declared-model", default="hist_gradient_boosting",
                    help="S1 (AUDIT_FINDINGS.md): the model whose numbers are "
                         "the HEADLINE and which gets saved / gold-scored - "
                         "pre-declared, not picked as max-of-9 (that is the "
                         "winner's curse). Default is the deployed model. Pass "
                         "'auto' to fall back to the old select-by-best "
                         "behaviour for exploration only.")
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 74)
    print("LEARNED TACTICAL CLASSIFIER")
    print("=" * 74)

    if args.ablate_features:
        return _run_ablation(args)

    (X, y, groups, files), skipped = load_dataset(args.csv_dir,
                                                  mode=args.features)
    print(f"\nLoaded {len(X)} clips, {X.shape[1]} features "
          f"(mode={args.features})")
    if skipped:
        print(f"  skipped {len(skipped)} files "
              f"(first: {skipped[0][0]} - {skipped[0][1]})")

    print("\nClass distribution:")
    for i, c in enumerate(CLASS_NAMES):
        n = int((y == i).sum())
        print(f"  {c:<22}: {n:>4}  ({100*n/len(y):>4.1f}%)")
    print("\nSource videos (leave-one-video-out folds):")
    for g in np.unique(groups):
        print(f"  {g:<10}: {int((groups == g).sum()):>4} clips")

    base = majority_baseline(y)
    rule_base = rule_engine_baseline(args.csv_dir, files, y)

    print("\n" + "-" * 74)
    print("BASELINE LADDER - every model below must clear these")
    print("-" * 74)
    print(f"  {'baseline':<38}{'accuracy':>10}{'macro-F1':>10}{'kappa':>9}")
    print(f"  {'majority class (no information)':<38}"
          f"{base['accuracy']:>10.3f}{base['macro_f1']:>10.3f}"
          f"{base['cohen_kappa']:>9.3f}")
    if rule_base:
        print(f"  {'hand-coded rule engine':<38}"
              f"{rule_base['accuracy']:>10.3f}{rule_base['macro_f1']:>10.3f}"
              f"{rule_base['cohen_kappa']:>9.3f}")
        print(f"    (rules abstained on {rule_base['n_abstained']}/"
              f"{rule_base['n_clips']} clips = "
              f"{100*rule_base['abstain_rate']:.0f}%, scored as misses)")
        print("\n  The rule-engine row is the one that matters: the labels are")
        print("  HUMAN annotations (labels.csv -> clip folder -> target_label,")
        print("  verified 159/159 exact), so the rule engine is an independent")
        print("  competing predictor seeing the SAME window as the model, not")
        print("  the source of the labels. Beating it is the evidence that the")
        print("  learned model adds something over the hand-coded heuristic.")

    print("\n" + "=" * 74)
    print("CROSS-VALIDATION  (k-fold = optimistic, LOVO = honest)")
    print("=" * 74)
    results = cross_validate_models(X, y, groups, seed=args.seed,
                                    n_splits=args.folds)

    hdr = (f"{'model':<24}{'kf_acc':>8}{'kf_F1':>8}"
           f"{'lovo_acc':>10}{'lovo_F1':>9}{'lovo_bal':>10}{'kappa':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in sorted(results.items(),
                          key=lambda kv: -kv[1].get(args.select_by, -1)):
        print(f"{name:<24}"
              f"{r.get('kfold_accuracy', float('nan')):>8.3f}"
              f"{r.get('kfold_macro_f1', float('nan')):>8.3f}"
              f"{r.get('lovo_accuracy', float('nan')):>10.3f}"
              f"{r.get('lovo_macro_f1', float('nan')):>9.3f}"
              f"{r.get('lovo_balanced_accuracy', float('nan')):>10.3f}"
              f"{r.get('lovo_cohen_kappa', float('nan')):>8.3f}")

    print("\nProbability-ranking quality (AUC-ROC / AUC-PR, one-vs-rest macro) "
          "and within-fold overfitting (LOVO protocol):")
    hdr2 = (f"{'model':<24}{'roc_auc':>9}{'pr_auc':>9}"
           f"{'train_F1':>10}{'test_F1':>9}{'gap':>8}{'flag':>7}")
    print(hdr2)
    print("-" * len(hdr2))
    for name, r in sorted(results.items(),
                          key=lambda kv: -kv[1].get(args.select_by, -1)):
        roc = r.get("lovo_roc_auc_macro")
        pr = r.get("lovo_pr_auc_macro")
        gap = r.get("lovo_overfit_gap")
        flag = "OVERFIT" if r.get("lovo_overfitting") else ""
        print(f"{name:<24}"
              f"{roc if roc is not None else float('nan'):>9.3f}"
              f"{pr if pr is not None else float('nan'):>9.3f}"
              f"{r.get('lovo_train_macro_f1', float('nan')):>10.3f}"
              f"{r.get('lovo_macro_f1', float('nan')):>9.3f}"
              f"{gap if gap is not None else float('nan'):>8.3f}"
              f"{flag:>7}")
    print("  AUC-PR > class prevalence means the probabilities carry usable")
    print("  signal even for a class whose hard-label recall is near zero.")
    print("  overfit_gap > 0.05 (flagged OVERFIT) means the model fits its own")
    print("  training rows much better than held-out ones, independent of the")
    print("  LOVO-vs-k-fold protocol gap reported above.")

    ranked_first = max(results, key=lambda k: results[k].get(args.select_by, -1))
    if args.declared_model == "auto" or args.declared_model not in results:
        if args.declared_model not in ("auto",) and args.declared_model not in results:
            print(f"\n[WARN] --declared-model {args.declared_model!r} not in the "
                  f"zoo; falling back to select-by-best.")
        best_name = ranked_first
        print(f"\nWinner by {args.select_by} (EXPLORATORY - winner's curse "
              f"applies): {best_name}")
    else:
        best_name = args.declared_model
        print(f"\nDeclared model (S1 - headline number, pre-registered not "
              f"picked): {best_name}")
        if ranked_first != best_name:
            rf = results[ranked_first].get(args.select_by, float("nan"))
            bn = results[best_name].get(args.select_by, float("nan"))
            print(f"  (the zoo's best on {args.select_by} was {ranked_first} at "
                  f"{rf:.3f} vs the declared model's {bn:.3f} - reported for "
                  f"transparency; the declared model's number is the one quoted)")
    best = results[best_name]

    pred_key = "_pred_lovo" if "_pred_lovo" in best else "_pred_kfold"
    y_pred = best[pred_key]
    proto = "leave-one-video-out" if pred_key == "_pred_lovo" else "k-fold"

    print(f"\nConfusion matrix ({proto}, {best_name}):")
    print(confusion_table(y, y_pred))
    print(f"\nPer-class report ({proto}):")
    print(per_class_report(y, y_pred))

    # ---- S5: pooled vs per-fold, so the fold-to-fold variance is visible ---
    fold_pref = "lovo" if pred_key == "_pred_lovo" else "kfold"
    pf = best.get(f"{fold_pref}_macro_f1_per_fold")
    if pf:
        pooled = best.get(f"{fold_pref}_macro_f1", float("nan"))
        mean = best.get(f"{fold_pref}_macro_f1_fold_mean", float("nan"))
        sd = best.get(f"{fold_pref}_macro_f1_fold_std", float("nan"))
        print(f"\nMacro-F1 aggregation ({proto}, {best_name}):")
        print(f"  pooled out-of-fold : {pooled:.3f}   "
              f"(headline - robust when a fold has few rare-class examples)")
        print(f"  mean +/- SD of folds: {mean:.3f} +/- {sd:.3f}")
        print(f"  per fold           : "
              + "  ".join(f"{v:.3f}" for v in pf))
        bpf = best.get(f"{fold_pref}_balanced_accuracy_per_fold")
        if bpf:
            print(f"  balanced-acc/fold  : "
                  + "  ".join(f"{v:.3f}" for v in bpf))

    # ---- decision rule: plain argmax vs prior-corrected -------------------
    decision = None
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    proba_key = "_proba_lovo" if "_proba_lovo" in best else None
    if proba_key:
        # S6: pass groups so the correction uses each fold's TRAINING prior,
        # not the global count (which would include the held-out video).
        decision = compare_decision_rules(y, best[proba_key], counts,
                                          groups=groups)
        print("\n" + "-" * 74)
        print(f"DECISION RULE - argmax vs prior-corrected  [prior: "
              f"{decision['_prior']}]")
        print("-" * 74)
        hdr = (f"  {'rule':<18}{'acc':>8}{'macroF1':>9}{'balAcc':>9}{'kappa':>8}"
               + "".join(f"{c.split()[-1][:7]:>9}" for c in CLASS_NAMES))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for rule in ("argmax", "prior_corrected"):
            s = decision[rule]
            print(f"  {rule:<18}{s['accuracy']:>8.3f}{s['macro_f1']:>9.3f}"
                  f"{s['balanced_accuracy']:>9.3f}{s['cohen_kappa']:>8.3f}"
                  + "".join(f"{r:>9.2f}" for r in s["per_class_recall"]))
        print("  (last 4 columns = per-class RECALL, in CLASS_NAMES order)")
        d_f1 = (decision["prior_corrected"]["macro_f1"]
                - decision["argmax"]["macro_f1"])
        print(f"\n  prior correction changes macro-F1 by {d_f1:+.3f} and is "
              "parameter-free,\n  so unlike a tuned threshold it cannot overfit "
              "a 16-example class.")

    print("Fitting the winner on all data for importance + deployment...")
    model = build_models(args.seed)[best_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    raw_profile = confidence_profile(model, X)
    cal_info = {"calibrated": False, "reason": "--no-calibrate"}
    if not args.no_calibrate:
        model, cal_info = calibrate(model, X, y, seed=args.seed)
    cal_profile = confidence_profile(model, X)

    print("\nProbability calibration:")
    print(f"  {cal_info}")
    print(f"  {'':<22}{'uncalibrated':>14}{'calibrated':>13}")
    for k in ("mean_confidence", "frac_confidence_above_0.99",
              "mean_entropy_bits"):
        print(f"  {k:<22}{raw_profile[k]:>14.3f}{cal_profile[k]:>13.3f}")
    if raw_profile["frac_confidence_above_0.99"] > 0.5 >= cal_profile["frac_confidence_above_0.99"]:
        print("  -> Calibration restored a usable confidence range, so the")
        print("     entropy channel and the anomaly flag work again.")

    print("\nChannel importance (macro-F1 lost when the whole channel is "
          "shuffled):")
    imp = grouped_permutation_importance(best_name, X, y, groups,
                                          args.features, seed=args.seed)
    print(f"  protocol: {imp['protocol']}")
    print(f"  held-out baseline macro-F1: {imp['baseline_macro_f1']:.3f}")
    for d in imp["channels"][:15]:
        bar = "#" * max(int(d["importance_mean"] * 120), 0)
        print(f"  {d['channel']:<22} {d['importance_mean']:+.4f} "
              f"+/- {d['importance_std']:.4f}  {bar}")

    gold_report = None
    if args.gold:
        print("\n" + "=" * 74)
        print("TIER 2 - scored against HUMAN labels")
        print("=" * 74)

        # R2/Q2 fix: `model` above was fitted on ALL of X/y for deployment,
        # which includes the very clips gold_csv scores it against - an
        # in-sample number dressed as a held-out one. Fit a SEPARATE model
        # with those rows excluded, purely for this comparison; the
        # deployment `model` used everywhere else is untouched.
        gold_names = gold_clip_names(args.gold)
        gold_mask = np.isin(files, list(gold_names)) if gold_names else \
            np.zeros(len(files), dtype=bool)
        n_excluded = int(gold_mask.sum())
        if n_excluded:
            print(f"  fitting a held-out model for this comparison "
                  f"(excluding {n_excluded}/{len(files)} clips that overlap "
                  f"gold_csv - the deployment model above saw them)")
            gold_model = build_models(args.seed)[best_name]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gold_model.fit(X[~gold_mask], y[~gold_mask])
            if not args.no_calibrate:
                gold_model, _ = calibrate(gold_model, X[~gold_mask],
                                          y[~gold_mask], seed=args.seed)
        else:
            gold_model = model
            print("  WARNING: no gold_csv clip name matched a training row - "
                  "could not confirm held-out status. Scoring the all-data "
                  "deployment model; verify gold_csv's `clip` column matches "
                  "csv_dir basenames.")

        gold_report = evaluate_against_gold(gold_model, args.csv_dir, args.gold,
                                            mode=args.features)
        gold_report["model_held_out_of_fit"] = bool(n_excluded)
        gold_report["n_excluded_from_fit"] = n_excluded
        r_h = gold_report["rules_vs_human"]["macro_f1"]
        m_h = gold_report["model_vs_human"]["macro_f1"]
        print(f"\n  gold clips                : {gold_report['n_gold']}")
        print(f"  RULE ENGINE vs human macro-F1 : {r_h:.3f}")
        print(f"  MODEL       vs human macro-F1 : {m_h:.3f}")
        print(f"  model vs rules  macro-F1      : "
              f"{gold_report['model_vs_rules']['macro_f1']:.3f}")
        if m_h > r_h:
            print(f"\n  -> The model outscores its own teacher by "
                  f"{m_h - r_h:+.3f} macro-F1 against human judgement.\n"
                  f"     It has denoised the rules rather than copied them - the\n"
                  f"     standard weak-supervision result, and the answer to\n"
                  f"     'isn't your ML model just your rules?'")
        else:
            print(f"\n  -> The model does NOT beat the rules against humans "
                  f"({m_h - r_h:+.3f}).\n"
                  f"     Report this honestly: on this data the learned model is\n"
                  f"     a faster, differentiable restatement of the heuristic,\n"
                  f"     and Tier 3 (training ON human labels) is required.")
    else:
        print("\n" + "-" * 74)
        print("LABEL PROVENANCE")
        print("-" * 74)
        print("These scores are measured against HUMAN tactical annotations")
        print("(labels.csv -> clip folder -> target_label; verified 159/159).")
        print("They are NOT agreement-with-a-rule-engine: label_clips.py is a")
        print("separate path that never wrote these labels, which is why it")
        print("appears above as an independent competing baseline.")
        print("")
        print("Remaining honest caveat: single annotator. A second independent")
        print("annotator would let inter-annotator agreement be quoted as the")
        print("true ceiling for every number here:")
        print("    python fyp/annotate.py dataset/dataset --out gold2.csv")
        print("    python fyp/ml_classifier.py <csv_dir> --gold gold2.csv")

    summary = {
        "n_clips": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)},
        "majority_baseline": base,
        "rule_engine_baseline": rule_base,
        "decision_rules": decision,
        "winner": best_name,
        "model_selection": ("declared (S1: pre-registered, not max-of-9)"
                            if args.declared_model not in ("auto",)
                            and args.declared_model in results
                            else f"max of {len(results)} by {args.select_by} "
                                 "(winner's curse applies - exploratory)"),
        "selected_by": args.select_by,
        "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                   for k, v in results.items()},
        "channel_importance": imp,
        "feature_mode": args.features,
        "gold": gold_report,
        "calibration": cal_info,
        "confidence_uncalibrated": raw_profile,
        "confidence_calibrated": cal_profile,
        "label_provenance": (
            "Labels are human tactical annotations, verified: labels.csv "
            "(video_file, anchor_time, tactic_class) -> ffmpeg clip -> "
            "dataset/<tactic_class>/ -> target_label, all resolvable rows "
            "folder-matched, zero mismatches. label_clips.py (the hand-coded "
            "rule engine) never wrote these labels; it runs above as an "
            "independent competing baseline, not the label source. See "
            "README section 4.1 and Technical_Review_and_QA.md section 16."
        ),
    }
    out_json = os.path.join(args.output_dir, "ml_results.json")

    def _jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_jsonable)
    print(f"\nSummary written to {out_json}")

    if args.save_model or args.export_all_models:
        try:
            import joblib
        except ImportError:
            print("[WARNING] joblib not installed; model(s) not saved.",
                  file=sys.stderr)
        else:
            # C2 (AUDIT_FINDINGS.md): carry the extraction-time provenance
            # manifest (imgsz / ball model / team split / court coords /
            # tracker - written by prepare_training_data.py) into the bundle,
            # so interfaces.py can compare it against the LIVE serving run's
            # flags at load time instead of only checking feature_version
            # (which only catches a tensor SHAPE change, not a CONTENT one).
            # Older CSV directories predate the manifest; None is a real,
            # honest "unknown", not an assumed match.
            extraction_provenance = None
            manifest_path = os.path.join(args.csv_dir, "extraction_provenance.json")
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    extraction_provenance = json.load(fh)

            def _bundle(fitted_pipeline, model_name: str) -> dict:
                return {
                    "pipeline": fitted_pipeline,
                    "class_names": CLASS_NAMES,
                    "feature_names": names_for(args.features),
                    "aggregations": list(AGGREGATIONS),
                    "feature_mode": args.features,
                    "feature_version": __import__("features").FEATURE_VERSION,
                    "model_name": model_name,
                    # S6 (AUDIT_FINDINGS.md): the training-set class counts, so a
                    # downstream consumer CAN apply the parameter-free
                    # prior-corrected decision rule (argmax of p(k|x)/p(k)).
                    # Serving still ships plain argmax (interfaces.py) - the rule
                    # helps minority recall but changing the deployed decision
                    # rule should be validated on the re-extracted data first.
                    "train_class_counts": [int((y == i).sum())
                                           for i in range(len(CLASS_NAMES))],
                    "extraction_provenance": extraction_provenance,
                }

            if extraction_provenance is None:
                print(f"[WARNING] No extraction_provenance.json found in "
                      f"'{args.csv_dir}' - bundle(s) carry no CV-flag "
                      "provenance; interfaces.py cannot check them at serve "
                      "time.", file=sys.stderr)

            if args.save_model:
                # `model` is the winner, already fit-on-all + calibrated above.
                joblib.dump(_bundle(model, best_name), args.save_model)
                print(f"Model saved to {args.save_model}")
                print("  Serve it live with:")
                print(f"    python fyp/pipeline.py video.mp4 {args.save_model} …")

            if args.export_all_models:
                os.makedirs(args.export_all_models, exist_ok=True)
                print(f"\nExporting every zoo model to '{args.export_all_models}/' "
                      "(fit-on-all + calibrated, one .joblib each):")
                for name, pipe in build_models(args.seed).items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipe.fit(X, y)
                        if not args.no_calibrate:
                            pipe, _ = calibrate(pipe, X, y, seed=args.seed)
                    path = os.path.join(args.export_all_models, f"{name}.joblib")
                    joblib.dump(_bundle(pipe, name), path)
                    lovo = results.get(name, {}).get("lovo_accuracy")
                    tag = f"  (LOVO acc {lovo:.3f})" if lovo is not None else ""
                    print(f"  {name:24s} -> {path}{tag}")
                print("  Render each:  for m in "
                      f"{args.export_all_models}/*.joblib; do python fyp/pipeline.py "
                      "VIDEO.mp4 \"$m\" --strict-render --auto-court "
                      "--output-video \"annot_$(basename \"$m\" .joblib).mp4\"; done")

    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
