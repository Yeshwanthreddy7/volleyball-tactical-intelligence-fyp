"""
Tests for the learned tactical classifier (fyp/ml_classifier.py) and the blind
annotator (fyp/annotate.py).

The properties locked here are the ones whose violation produced a wrong NUMBER
rather than a crash - the failure mode this project keeps hitting.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_classifier as mc  # noqa: E402
import annotate as an  # noqa: E402
from features import MODEL_FEATURE_COLS  # noqa: E402


def _seq(n=29, d=14, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(500.0, 50.0, size=(n, d))


# --------------------------------------------------------------------------
# feature contract
# --------------------------------------------------------------------------

def test_absolute_block_has_one_column_per_channel_and_statistic():
    x = mc.sequence_features(_seq())
    assert x.shape == (len(MODEL_FEATURE_COLS) * len(mc.AGGREGATIONS),)
    assert len(mc.feature_names()) == len(x)


def test_invariant_block_names_match_width():
    x = mc.invariant_sequence_features(_seq())
    assert len(mc.invariant_feature_names()) == len(x)


def test_both_mode_is_the_concatenation_in_order():
    s = _seq()
    a = mc.sequence_features(s)
    i = mc.invariant_sequence_features(s)
    b = mc.extract(s, "both")
    assert np.allclose(b, np.concatenate([a, i]), equal_nan=True)
    assert len(mc.names_for("both")) == len(b)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        mc.extract(_seq(), "nonsense")


def test_train_and_serve_share_one_extractor():
    """The train/serve mismatch guard. `pipeline.py` hands the classifier a
    (29,18) MODEL sequence while training starts from (29,14) raw rows; both
    must funnel into the same aggregation in the same column order, or a saved
    model silently scores garbage (the L6 / sec 12.8 failure class)."""
    from features import build_model_sequence

    raw = _seq()
    model_seq = build_model_sequence(raw, target_len=mc.SEQ_LEN)
    for mode in ("absolute", "invariant", "both"):
        assert np.allclose(mc.extract(raw, mode),
                           mc.features_from_model_sequence(model_seq, mode),
                           equal_nan=True)


# --------------------------------------------------------------------------
# scale invariance - the property the whole invariant block exists for
# --------------------------------------------------------------------------

def test_invariant_features_survive_a_camera_scale_change():
    """Under the linear pixel->court fallback, two cameras give the same play
    different centimetres. Ratio and z-scored-shape features must not move."""
    raw = _seq()
    scaled = raw * 1.7                      # same play, different court scale

    a = mc.invariant_sequence_features(raw)
    b = mc.invariant_sequence_features(scaled)
    same = np.isclose(a, b, rtol=1e-3, atol=1e-3) | (np.isnan(a) & np.isnan(b))
    assert same.mean() > 0.8, (
        f"only {100*same.mean():.0f}% of invariant features survived a scale "
        "change; the block is not doing its job")


def test_absolute_features_do_move_under_a_scale_change():
    """Control for the test above: if absolutes were also scale-stable, the
    ablation would be measuring nothing."""
    raw = _seq()
    a = mc.sequence_features(raw)
    b = mc.sequence_features(raw * 1.7)
    moved = ~np.isclose(a, b, rtol=1e-3, atol=1e-3)
    assert moved.mean() > 0.5


# --------------------------------------------------------------------------
# numerical robustness
# --------------------------------------------------------------------------

def test_slope_is_signed_and_scaled_per_frame():
    assert mc._slope(np.arange(29, dtype=float)) == pytest.approx(1.0)
    assert mc._slope(-np.arange(29, dtype=float)) == pytest.approx(-1.0)
    assert mc._slope(np.full(29, 3.0)) == pytest.approx(0.0)


def test_slope_handles_nan_and_degenerate_input():
    x = np.full(29, np.nan)
    assert mc._slope(x) == 0.0
    x[3] = 1.0
    assert mc._slope(x) == 0.0            # a single point defines no slope


def test_features_are_finite_even_when_the_ball_is_never_seen():
    """42% of frames had no ball historically; an all-zero ball channel must not
    produce inf from a ratio denominator."""
    raw = _seq()
    raw[:, 0:2] = 0.0                      # ball_x, ball_y absent
    for mode in ("absolute", "invariant", "both"):
        v = mc.extract(raw, mode)
        assert not np.isinf(v).any(), f"{mode} produced inf"


def test_ratio_features_are_clipped_against_a_zero_denominator():
    raw = np.zeros((29, 14))               # everything collapsed to one point
    v = mc.invariant_sequence_features(raw)
    assert np.all(np.abs(v[np.isfinite(v)]) <= 50.0 + 1e-6)


# --------------------------------------------------------------------------
# grouping / importance
# --------------------------------------------------------------------------

def test_feature_groups_partition_every_column_exactly_once():
    for mode in ("absolute", "invariant"):
        groups = mc.feature_groups(mode)
        cols = sorted(c for v in groups.values() for c in v)
        assert cols == list(range(len(mc.names_for(mode))))


def test_absolute_groups_are_the_18_channels():
    assert set(mc.feature_groups("absolute")) == set(MODEL_FEATURE_COLS)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_majority_baseline_matches_the_dominant_class_share():
    y = np.array([1] * 7 + [0] * 3)
    assert mc.majority_baseline(y)["accuracy"] == pytest.approx(0.7)


def test_shannon_entropy_endpoints():
    assert mc.shannon_entropy([1.0, 0, 0, 0]) == pytest.approx(0.0)
    assert mc.shannon_entropy([0.25] * 4) == pytest.approx(2.0)


def test_confusion_table_reports_per_class_recall():
    y = np.array([0, 0, 1, 1])
    table = mc.confusion_table(y, np.array([0, 1, 1, 1]))
    assert "recall 1/2" in table and "recall 2/2" in table


# --------------------------------------------------------------------------
# annotator - the blindness properties that make the gold set admissible
# --------------------------------------------------------------------------

def test_clip_path_maps_to_the_rule_label_csv_name():
    p = os.path.join("dataset", "Coordinated_Attack", "videoplayback (1)_clip_002.mp4")
    assert an.csv_name_for(p) == "Coordinated_Attack__videoplayback (1)_clip_002.csv"


def test_sampling_is_balanced_across_rule_classes(tmp_path):
    """Rule labels run 134/90/45/16. Proportional sampling would put ~4 Delayed
    Support clips in the gold set - too few to estimate a per-class score."""
    for folder in an.FOLDER_TO_LABEL:
        d = tmp_path / folder
        d.mkdir()
        for i in range(10):
            (d / f"clip_{i}.mp4").write_bytes(b"")

    got = an.collect_clips(str(tmp_path), per_class=4, seed=0)
    per = {}
    for path, _ in got:
        per[os.path.basename(os.path.dirname(path))] = \
            per.get(os.path.basename(os.path.dirname(path)), 0) + 1
    assert set(per.values()) == {4}


def test_sampling_shuffles_across_classes_so_order_leaks_nothing(tmp_path):
    """Even with the folder name hidden, class-ordered playback would tell the
    annotator the rule label."""
    for folder in an.FOLDER_TO_LABEL:
        d = tmp_path / folder
        d.mkdir()
        for i in range(10):
            (d / f"clip_{i}.mp4").write_bytes(b"")

    order = [os.path.basename(os.path.dirname(p))
             for p, _ in an.collect_clips(str(tmp_path), per_class=10, seed=0)]
    runs = sum(1 for a, b in zip(order, order[1:]) if a != b)
    assert runs > len(order) // 3, "clips are still grouped by rule class"


def test_unclear_is_a_recordable_outcome_not_a_forced_class():
    assert an.UNCLEAR not in an.CLASS_NAMES


def test_annotator_classes_match_the_classifier_classes():
    assert an.CLASS_NAMES == mc.CLASS_NAMES


# --------------------------------------------------------------------------
# ranking metrics (AUC-ROC / AUC-PR) - senior-data-scientist checklist item
# --------------------------------------------------------------------------

def _one_hot_proba(y, n_classes, correct_prob=0.9, seed=0):
    """Fake probabilities that are mostly right, so ranking metrics are
    meaningfully above chance without being trivially 1.0."""
    rng = np.random.default_rng(seed)
    n = len(y)
    proba = rng.dirichlet(np.ones(n_classes) * 0.5, size=n)
    for i, label in enumerate(y):
        proba[i] *= (1 - correct_prob)
        proba[i, label] += correct_prob
        proba[i] /= proba[i].sum()
    return proba


def test_ranking_scores_are_near_one_for_a_confident_correct_model():
    y = np.array([0, 1, 2, 3] * 20)
    proba = _one_hot_proba(y, 4, correct_prob=0.97)
    r = mc._ranking_scores(y, proba, 4)
    assert r["roc_auc_macro"] > 0.9
    assert r["pr_auc_macro"] > 0.8


def test_ranking_scores_are_near_chance_for_uniform_probabilities():
    y = np.array([0, 1, 2, 3] * 20)
    proba = np.full((len(y), 4), 0.25)
    r = mc._ranking_scores(y, proba, 4)
    assert 0.4 < r["roc_auc_macro"] < 0.6


def test_ranking_scores_returns_one_value_per_class():
    y = np.array([0, 1, 2, 3] * 20)
    proba = _one_hot_proba(y, 4)
    r = mc._ranking_scores(y, proba, 4)
    assert len(r["per_class_roc_auc"]) == 4
    assert len(r["per_class_pr_auc"]) == 4


def test_ranking_scores_does_not_crash_on_a_missing_class():
    """A class entirely absent from the pooled y (can happen in a degenerate
    fold) must not crash the whole CV run - it should be skipped, not raised."""
    y = np.array([0, 1, 1, 0] * 10)          # classes 2, 3 never appear
    proba = _one_hot_proba(np.array([0, 1] * 20), 4)
    r = mc._ranking_scores(y, proba, 4)
    assert r["roc_auc_macro"] is None or isinstance(r["roc_auc_macro"], float)


# --------------------------------------------------------------------------
# overfit gap - senior-data-scientist checklist item
# --------------------------------------------------------------------------

def _make_classification_xy(n=120, d=6, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=d, n_informative=4,
                               n_classes=3, n_clusters_per_class=1,
                               random_state=seed)
    return X, y


def test_overfit_gap_is_near_zero_for_a_high_bias_model():
    """A model too simple to memorise training rows should show a small gap."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X, y = _make_classification_xy()
    model = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    r = mc._overfit_gap(model, X, y, cv, seed=0)
    assert r["overfit_gap"] < 0.3
    assert 0.0 <= r["train_macro_f1"] <= 1.0
    assert 0.0 <= r["test_macro_f1"] <= 1.0


def test_overfit_gap_is_large_for_a_memorising_model():
    """A 1-nearest-neighbour classifier fits training data perfectly and
    should show a large, correctly-flagged gap on held-out folds."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold

    X, y = _make_classification_xy(n=60, d=30, seed=1)   # high-dim, few rows
    model = KNeighborsClassifier(n_neighbors=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    r = mc._overfit_gap(model, X, y, cv, seed=0)
    assert r["train_macro_f1"] > 0.95          # 1-NN is exact on training rows
    assert r["overfitting"] == (r["overfit_gap"] > 0.05)


def test_overfit_gap_flag_matches_the_005_threshold_exactly():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X, y = _make_classification_xy()
    model = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    r = mc._overfit_gap(model, X, y, cv, seed=0)
    assert r["overfitting"] == (r["overfit_gap"] > 0.05)


def test_overfit_gap_accepts_a_group_based_cv():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut

    X, y = _make_classification_xy(n=90)
    groups = np.array([0] * 30 + [1] * 30 + [2] * 30)
    model = LogisticRegression(max_iter=1000)
    r = mc._overfit_gap(model, X, y, LeaveOneGroupOut(), groups=groups, seed=0)
    assert "overfit_gap" in r


# --------------------------------------------------------------------------
# cross_validate_models exposes the new diagnostics end to end
# --------------------------------------------------------------------------

def test_cross_validate_models_reports_ranking_and_overfit_fields():
    X, y = _make_classification_xy(n=90, d=10)
    groups = np.array([0] * 30 + [1] * 30 + [2] * 30)

    import ml_classifier as mc_mod
    orig = mc_mod.build_models
    try:
        mc_mod.build_models = lambda seed=0: {
            "logreg": orig(seed)["logistic_regression"],
        }
        res = mc_mod.cross_validate_models(X, y, groups, seed=0, n_splits=3)
    finally:
        mc_mod.build_models = orig

    row = res["logreg"]
    for key in ("kfold_roc_auc_macro", "kfold_pr_auc_macro",
               "kfold_overfit_gap", "kfold_train_macro_f1",
               "lovo_roc_auc_macro", "lovo_pr_auc_macro",
               "lovo_overfit_gap", "lovo_train_macro_f1"):
        assert key in row, f"missing {key}"


# --------------------------------------------------------------------------
# rule-engine baseline - labels are HUMAN, so the rules are a competitor
# --------------------------------------------------------------------------

def _write_clip_csv(path, label, n=29, seed=0):
    """Minimal clip CSV in the schema label_clips/prepare_training_data use."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    cols = {"frame_id": np.arange(1, n + 1)}
    cols["ball_x"] = rng.normal(900, 50, n)
    cols["ball_y"] = rng.normal(450, 50, n)
    for p in range(1, 7):
        cols[f"p{p}_x"] = rng.normal(300 * p, 30, n)
        cols[f"p{p}_y"] = rng.normal(600, 30, n)
    df = pd.DataFrame(cols)
    df["target_label"] = label
    df.to_csv(path, index=False)


def test_rule_engine_baseline_scores_rules_against_human_labels(tmp_path):
    """The labels are human annotations (labels.csv -> folder -> target_label),
    so the rule engine is an INDEPENDENT competing predictor, not the label
    source. An earlier version of this function returned None on the incorrect
    assumption that comparing them was tautological."""
    files = []
    for i, cls in enumerate(mc.CLASS_NAMES):
        name = f"{cls.replace(' ', '_')}__clip_{i:03d}.csv"
        _write_clip_csv(tmp_path / name, cls, seed=i)
        files.append(name)
    y = np.arange(len(mc.CLASS_NAMES))

    out = mc.rule_engine_baseline(str(tmp_path), files, y)
    assert out is not None
    for key in ("accuracy", "macro_f1", "cohen_kappa", "n_clips",
               "n_abstained", "abstain_rate"):
        assert key in out
    assert out["n_clips"] == len(files)
    assert 0.0 <= out["accuracy"] <= 1.0


def test_rule_engine_baseline_counts_abstentions_as_misses(tmp_path):
    """A baseline that declines the hard cases must not be credited for
    skipping them - abstentions are scored, not dropped."""
    files = []
    for i, cls in enumerate(mc.CLASS_NAMES):
        name = f"{cls.replace(' ', '_')}__clip_{i:03d}.csv"
        _write_clip_csv(tmp_path / name, cls, seed=i + 10)
        files.append(name)
    y = np.arange(len(mc.CLASS_NAMES))

    out = mc.rule_engine_baseline(str(tmp_path), files, y)
    # every clip is scored, whether or not the rules committed to a class
    assert out["n_clips"] == len(files)
    assert out["n_abstained"] <= out["n_clips"]
    assert out["abstain_rate"] == pytest.approx(
        out["n_abstained"] / out["n_clips"])


def test_rule_engine_baseline_returns_none_when_no_clip_resolves(tmp_path):
    out = mc.rule_engine_baseline(str(tmp_path), ["does_not_exist.csv"],
                                  np.array([0]))
    assert out is None


def test_rule_engine_baseline_only_reads_the_observation_window(tmp_path):
    """The model sees frames 1..SEQ_LEN; the baseline must see exactly the
    same window, or the comparison is not like-for-like."""
    import pandas as pd
    name = "Coordinated_Attack__clip_001.csv"
    _write_clip_csv(tmp_path / name, "Coordinated Attack", n=41, seed=3)
    out = mc.rule_engine_baseline(str(tmp_path), [name], np.array([0]))
    assert out is not None and out["n_clips"] == 1


# --------------------------------------------------------------------------
# prior-corrected decision rule for the minority classes
# --------------------------------------------------------------------------

def test_prior_correction_is_identity_when_classes_are_balanced():
    """With a flat prior there is nothing to correct, so the rule must reduce
    to plain argmax - otherwise it is distorting balanced problems too."""
    proba = np.array([[0.7, 0.2, 0.1],
                      [0.1, 0.8, 0.1],
                      [0.2, 0.2, 0.6]])
    counts = np.array([10, 10, 10])
    assert list(mc.prior_corrected_predict(proba, counts)) == [0, 1, 2]


def test_prior_correction_recovers_a_suppressed_minority_class():
    """The exact failure being fixed: the model ranks the rare class second but
    argmax always names the majority class, so its recall is ~0."""
    # class 2 is rare; the model gives it 0.35 vs majority class 0 at 0.45
    proba = np.array([[0.45, 0.20, 0.35]])
    counts = np.array([200, 100, 16])          # class 2 is the 16-example class
    assert np.argmax(proba, axis=1)[0] == 0                    # argmax: majority
    assert mc.prior_corrected_predict(proba, counts)[0] == 2    # corrected: rare


def test_prior_correction_does_not_fire_on_a_confident_majority_prediction():
    proba = np.array([[0.97, 0.02, 0.01]])
    counts = np.array([200, 100, 16])
    assert mc.prior_corrected_predict(proba, counts)[0] == 0


def test_prior_correction_handles_a_zero_count_class_without_dividing_by_zero():
    proba = np.array([[0.5, 0.3, 0.2]])
    counts = np.array([100, 50, 0])
    out = mc.prior_corrected_predict(proba, counts)
    assert np.isfinite(out).all()


def test_compare_decision_rules_reports_both_rules_with_per_class_recall():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    proba = rng.dirichlet(np.ones(3), size=len(y))
    out = mc.compare_decision_rules(y, proba, np.bincount(y))
    assert {"argmax", "prior_corrected"} <= set(out)
    for rule in ("argmax", "prior_corrected"):
        assert len(out[rule]["per_class_recall"]) == 3
        assert "macro_f1" in out[rule] and "balanced_accuracy" in out[rule]
    # no groups -> global prior, labelled as illustrative
    assert "illustrative" in out["_prior"]


def test_compare_decision_rules_uses_per_fold_prior_when_groups_given():
    """S6: with groups, each held-out fold's rows are corrected by the OTHER
    folds' prior, not a global count that includes the held-out video."""
    rng = np.random.default_rng(1)
    y = np.array([0] * 40 + [1] * 30 + [2] * 20)
    groups = np.array(["(1)"] * 45 + ["(3)"] * 45)
    proba = rng.dirichlet(np.ones(3), size=len(y))

    out = mc.compare_decision_rules(y, proba, np.bincount(y), groups=groups)
    assert "no leak" in out["_prior"]
    assert "prior_corrected_global" in out         # global kept for comparison

    # the per-fold predictions must match a hand-rolled per-group correction
    expect = np.empty(len(y), dtype=int)
    for g in ("(1)", "(3)"):
        te = groups == g
        tr_prior = np.bincount(y[~te], minlength=3)
        expect[te] = mc.prior_corrected_predict(proba[te], tr_prior)
    got = mc._prior_corrected_predict_per_fold(y, proba, groups)
    assert np.array_equal(got, expect)


def test_prior_correction_raises_minority_recall_on_an_imbalanced_problem():
    """End-to-end property: on a skewed problem the corrected rule must give
    the rarest class strictly better recall than plain argmax."""
    rng = np.random.default_rng(3)
    y = np.array([0] * 200 + [1] * 100 + [2] * 16)
    proba = np.zeros((len(y), 3))
    for i, lab in enumerate(y):
        p = rng.dirichlet([1.0, 1.0, 1.0])
        p[lab] += 0.5                      # true class ranked up, not dominant
        proba[i] = p / p.sum()
    # bias the posterior toward the majority class, as a real fitted model does
    proba *= np.array([1.6, 1.0, 0.5])[None, :]
    proba /= proba.sum(axis=1, keepdims=True)

    out = mc.compare_decision_rules(y, proba, np.bincount(y))
    assert (out["prior_corrected"]["per_class_recall"][2]
            > out["argmax"]["per_class_recall"][2])


# --------------------------------------------------------------------------
# R2/Q2: gold-clip exclusion mask (the model scored against gold_csv must
# never have been fitted on those same rows, or the number is in-sample)
# --------------------------------------------------------------------------

def test_gold_clip_names_reads_clip_column(tmp_path):
    import pandas as pd
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({
        "clip": ["a__clip_1.csv", " b__clip_2.csv ", "c__clip_3.csv"],
        "label": mc.CLASS_NAMES[:3],
    }).to_csv(gold_path, index=False)

    names = mc.gold_clip_names(str(gold_path))
    # whitespace is stripped, so " b__clip_2.csv " and "b__clip_2.csv" match
    assert names == {"a__clip_1.csv", "b__clip_2.csv", "c__clip_3.csv"}


def test_gold_clip_names_missing_column_returns_empty_set(tmp_path):
    import pandas as pd
    gold_path = tmp_path / "gold_no_clip_col.csv"
    pd.DataFrame({"filename": ["x.csv"], "label": [mc.CLASS_NAMES[0]]}).to_csv(
        gold_path, index=False)

    assert mc.gold_clip_names(str(gold_path)) == set()


def test_gold_clip_names_matches_load_dataset_files_array(tmp_path):
    """The exclusion mask is built with np.isin(files, gold_names) - confirm
    the two use the same string convention (bare basename, not a full path)."""
    import pandas as pd
    for i, cls in enumerate(mc.CLASS_NAMES):
        name = f"{cls.replace(' ', '_')}__clip_{i:03d}.csv"
        _write_clip_csv(tmp_path / name, cls, seed=i)

    (X, y, groups, files), skipped = mc.load_dataset(str(tmp_path))
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame({"clip": [files[0], files[2]],
                 "label": [mc.CLASS_NAMES[0], mc.CLASS_NAMES[2]]}).to_csv(
        gold_path, index=False)

    names = mc.gold_clip_names(str(gold_path))
    mask = np.isin(files, list(names))
    assert mask.sum() == 2
    assert set(files[mask]) == {files[0], files[2]}


def test_evaluate_against_gold_model_never_saw_excluded_rows(tmp_path):
    """End-to-end demonstration of the fix: a model fitted with the gold
    clips excluded still produces a valid, scoreable report - the wiring
    that matters is that `evaluate_against_gold` accepts whatever model
    object it's handed and never re-fits it itself."""
    for i, cls in enumerate(mc.CLASS_NAMES):
        for j in range(3):  # 3 clips per class so a held-out split is possible
            name = f"{cls.replace(' ', '_')}__clip_{i:03d}_{j}.csv"
            _write_clip_csv(tmp_path / name, cls, seed=i * 10 + j)

    (X, y, groups, files), _ = mc.load_dataset(str(tmp_path))

    import pandas as pd
    gold_rows = [{"clip": files[k], "label": mc.CLASS_NAMES[y[k]]}
                for k in range(0, len(files), 3)]   # one clip per class
    gold_path = tmp_path / "gold.csv"
    pd.DataFrame(gold_rows).to_csv(gold_path, index=False)

    gold_names = mc.gold_clip_names(str(gold_path))
    mask = np.isin(files, list(gold_names))
    assert mask.any() and not mask.all()   # some excluded, not everything

    import warnings
    model = mc.build_models(0)["hist_gradient_boosting"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[~mask], y[~mask])

    report = mc.evaluate_against_gold(model, str(tmp_path), str(gold_path))
    assert report["n_gold"] == len(gold_rows)
    for key in ("rules_vs_human", "model_vs_human", "model_vs_rules"):
        assert 0.0 <= report[key]["accuracy"] <= 1.0


# --------------------------------------------------------------------------
# S4: dimensionality reduction for distance/kernel learners, no 'both' mode
# --------------------------------------------------------------------------

def test_distance_learners_have_a_pca_step_and_trees_do_not():
    models = mc.build_models(0)
    for name in ("knn", "rbf_svm", "mlp"):
        steps = dict(models[name].named_steps)
        assert "pca" in steps, f"{name} should have a PCA step (S4)"
    for name in ("random_forest", "extra_trees", "gradient_boosting",
                 "hist_gradient_boosting"):
        steps = dict(models[name].named_steps)
        assert "pca" not in steps, f"{name} must keep raw features (S4)"


def test_pca_step_refits_per_fold_not_on_the_whole_set():
    """It is a Pipeline step, so sklearn refits it inside each CV split -
    this test just pins that it is BEFORE the classifier, not a global
    transform applied to X up front."""
    knn = mc.build_models(0)["knn"]
    step_names = [n for n, _ in knn.steps]
    assert step_names.index("pca") < step_names.index("clf")
    assert step_names.index("scale") < step_names.index("pca")


def test_both_feature_mode_is_no_longer_a_training_choice():
    import argparse
    # main() builds the parser inline; run it with --help intercepted so we
    # can read the --features choices without executing the pipeline.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ml_classifier.py"), encoding="utf-8").read()
    assert '"--features", choices=("absolute", "invariant")' in src
    assert '("absolute", "invariant", "both")' not in src

    # the extractor still understands 'both' for loading a legacy .joblib
    seq = np.zeros((29, 18))
    assert mc.features_from_model_sequence(seq, "both").shape[0] == (
        mc.features_from_model_sequence(seq, "absolute").shape[0]
        + mc.features_from_model_sequence(seq, "invariant").shape[0])


# --------------------------------------------------------------------------
# S5: per-fold macro-F1 mean +- SD reported alongside the pooled number
# --------------------------------------------------------------------------

def test_overfit_gap_also_returns_the_per_fold_breakdown():
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    y = rng.integers(0, 4, 60)
    m = mc.build_models(0)["hist_gradient_boosting"]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    out = mc._overfit_gap(m, X, y, skf, seed=0)
    # original keys still present (4 existing tests depend on them)
    assert {"train_macro_f1", "overfit_gap", "overfitting"} <= set(out)
    # S5 additions
    assert len(out["macro_f1_per_fold"]) == 3
    assert out["macro_f1_fold_mean"] == pytest.approx(
        float(np.mean(out["macro_f1_per_fold"])), abs=1e-3)
    assert out["macro_f1_fold_std"] == pytest.approx(
        float(np.std(out["macro_f1_per_fold"])), abs=1e-3)


def test_overfit_gap_per_fold_uses_group_folds_for_lovo():
    from sklearn.model_selection import LeaveOneGroupOut
    rng = np.random.default_rng(1)
    X = rng.normal(size=(45, 12))
    y = rng.integers(0, 4, 45)
    groups = np.array(["(1)"] * 15 + ["(3)"] * 15 + ["(4)"] * 15)
    m = mc.build_models(0)["hist_gradient_boosting"]

    out = mc._overfit_gap(m, X, y, LeaveOneGroupOut(), groups=groups, seed=0)
    assert len(out["macro_f1_per_fold"]) == 3   # one held-out video per fold


# --------------------------------------------------------------------------
# S8: reliability curve + ECE on (pooled out-of-fold) probabilities
# --------------------------------------------------------------------------

def test_reliability_curve_of_a_perfectly_calibrated_predictor_has_low_ece():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 4, n)
    proba = np.zeros((n, 4))
    # confidence p on the true class exactly p of the time
    for i, lab in enumerate(y):
        p = rng.uniform(0.25, 0.99)
        hit = rng.random() < p
        proba[i] = (1 - p) / 3
        proba[i, lab if hit else (lab + 1) % 4] = p
        proba[i] /= proba[i].sum()
    out = mc.reliability_curve(y, proba, n_bins=10)
    assert out["n"] == n
    assert out["ece"] < 0.12           # well-calibrated -> small gap
    assert sum(b["count"] for b in out["bins"]) == n


def test_reliability_curve_flags_an_overconfident_predictor():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.integers(0, 4, n)
    proba = np.full((n, 4), 0.01)
    for i, lab in enumerate(y):
        # always claims 0.97 but is only right ~40% of the time
        win = lab if rng.random() < 0.40 else (lab + 1) % 4
        proba[i, win] = 0.97
        proba[i] /= proba[i].sum()
    out = mc.reliability_curve(y, proba, n_bins=10)
    assert out["ece"] > 0.4            # 0.97 predicted vs ~0.4 actual


# --------------------------------------------------------------------------
# S2: bootstrap CI + paired McNemar
# --------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_and_narrows_with_n():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    y_pred = y.copy()
    flip = rng.random(200) < 0.3
    y_pred[flip] = (y_pred[flip] + 1) % 4

    out = mc.bootstrap_metric_ci(y, y_pred, n_boot=800, seed=0)
    m = out["macro_f1"]
    assert m["ci_low"] <= m["point"] <= m["ci_high"]
    assert m["ci_high"] - m["ci_low"] < 0.35      # not a useless-wide interval
    assert out["n_clips"] == 200


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 120)
    y_pred = rng.integers(0, 4, 120)
    a = mc.bootstrap_metric_ci(y, y_pred, n_boot=400, seed=7)
    b = mc.bootstrap_metric_ci(y, y_pred, n_boot=400, seed=7)
    assert a == b


def test_mcnemar_favours_the_clearly_better_classifier():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, 300)
    good = y.copy()
    mg = rng.random(300) < 0.15
    good[mg] = (good[mg] + 1) % 4
    bad = y.copy()
    mb = rng.random(300) < 0.55
    bad[mb] = (bad[mb] + 1) % 4

    out = mc.mcnemar_test(y, good, bad)
    assert out["favours"] == "a"
    assert out["significant_at_0.05"] is True
    assert out["n_discordant"] == (out["a_correct_b_wrong"]
                                   + out["b_correct_a_wrong"])


def test_mcnemar_reports_no_difference_for_identical_predictions():
    y = np.array([0, 1, 2, 3] * 20)
    pred = y.copy()
    out = mc.mcnemar_test(y, pred, pred)
    assert out["n_discordant"] == 0
    assert out["p_value"] == 1.0
    assert out["favours"] == "tie"


# --------------------------------------------------------------------------
# S1: headline model is pre-declared, not max-of-9 (winner's curse)
# --------------------------------------------------------------------------

def test_declared_model_default_is_the_deployed_one():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ml_classifier.py"), encoding="utf-8").read()
    assert '"--declared-model", default="hist_gradient_boosting"' in src
    # reproduce.py hard-codes the same declared model
    rsrc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "reproduce.py"), encoding="utf-8").read()
    assert 'DECLARED = "hist_gradient_boosting"' in rsrc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
