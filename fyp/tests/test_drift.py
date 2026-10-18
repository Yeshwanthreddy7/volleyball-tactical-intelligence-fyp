"""
Tests for fyp/drift.py - direct quantification of the per-video signature
(sec 12.6 / 14.3), which previously rested only on the indirect
LOVO-vs-k-fold performance gap.

Pure numpy/scipy; no video, no weights, no torch.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift import (  # noqa: E402
    compare, compare_all_pairs, ReferenceDriftMonitor, _psi,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# PSI
# --------------------------------------------------------------------------

def test_psi_is_near_zero_for_identical_distributions():
    r = _rng()
    a = r.normal(0, 1, 2000)
    b = r.normal(0, 1, 2000)
    assert _psi(a, b) < 0.05


def test_psi_is_large_for_a_shifted_distribution():
    r = _rng()
    a = r.normal(0, 1, 2000)
    b = r.normal(5, 1, 2000)          # 5-sigma shift
    assert _psi(a, b) > 0.25


def test_psi_increases_monotonically_with_shift_magnitude():
    r = _rng()
    ref = r.normal(0, 1, 2000)
    small = _psi(ref, r.normal(0.5, 1, 2000))
    large = _psi(ref, r.normal(3.0, 1, 2000))
    assert small < large


def test_psi_handles_tiny_samples_without_crashing():
    assert np.isnan(_psi(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])))


def test_psi_handles_a_constant_reference():
    out = _psi(np.full(50, 3.0), np.random.default_rng(0).normal(3, 1, 50))
    assert np.isfinite(out)


# --------------------------------------------------------------------------
# compare() - the direct per-feature drift measurement
# --------------------------------------------------------------------------

def test_compare_flags_a_shifted_feature_and_not_a_stable_one():
    """The core property this module exists for: given one feature that truly
    differs between two 'videos' and one that does not, the shifted one must
    score higher on both KS and PSI."""
    r = _rng()
    n = 300
    stable_a = r.normal(0, 1, n)
    stable_b = r.normal(0, 1, n)
    shifted_a = r.normal(0, 1, n)
    shifted_b = r.normal(4, 1, n)         # camera-scale-like shift

    ref = np.column_stack([stable_a, shifted_a])
    oth = np.column_stack([stable_b, shifted_b])
    report = compare(ref, oth, ["stable_feature", "shifted_feature"],
                     pair=("vidA", "vidB"))

    by_name = {c.feature: c for c in report.channels}
    assert by_name["shifted_feature"].psi > by_name["stable_feature"].psi
    assert by_name["shifted_feature"].shift_level in ("moderate", "major")
    assert by_name["stable_feature"].shift_level == "none"


def test_compare_reports_the_requested_pair_label():
    ref = _rng().normal(0, 1, (100, 1))
    oth = _rng(1).normal(0, 1, (100, 1))
    report = compare(ref, oth, ["x"], pair=("(1)", "(3)"))
    assert report.pair == ("(1)", "(3)")


def test_compare_rejects_mismatched_feature_name_count():
    ref = _rng().normal(0, 1, (50, 3))
    oth = _rng(1).normal(0, 1, (50, 3))
    with pytest.raises(ValueError):
        compare(ref, oth, ["only", "two"])


def test_compare_top_orders_by_psi_descending():
    r = _rng()
    n = 200
    ref = np.column_stack([r.normal(0, 1, n), r.normal(0, 1, n), r.normal(0, 1, n)])
    oth = np.column_stack([r.normal(0, 1, n), r.normal(2, 1, n), r.normal(6, 1, n)])
    report = compare(ref, oth, ["f0", "f1", "f2"])
    top = report.top(3)
    assert [c.psi for c in top] == sorted([c.psi for c in top], reverse=True)
    assert top[0].feature == "f2"          # the most-shifted feature


def test_mean_psi_and_major_shift_count_are_consistent():
    r = _rng()
    n = 200
    ref = np.column_stack([r.normal(0, 1, n) for _ in range(4)])
    oth = np.column_stack([r.normal(0, 1, n), r.normal(0, 1, n),
                           r.normal(5, 1, n), r.normal(5, 1, n)])
    report = compare(ref, oth, ["a", "b", "c", "d"])
    assert report.n_major_shift == 2
    assert np.isfinite(report.mean_psi)


# --------------------------------------------------------------------------
# compare_all_pairs()
# --------------------------------------------------------------------------

def test_compare_all_pairs_covers_every_ordered_pair():
    groups = {"(1)": _rng(0).normal(0, 1, (60, 2)),
              "(3)": _rng(1).normal(0, 1, (60, 2)),
              "(4)": _rng(2).normal(3, 1, (60, 2))}
    reports = compare_all_pairs(groups, ["a", "b"])
    pairs = {r.pair for r in reports}
    assert len(pairs) == 6            # 3 groups -> 3*2 ordered pairs
    assert ("(1)", "(3)") in pairs and ("(3)", "(1)") in pairs


def test_compare_all_pairs_psi_is_asymmetric_by_construction():
    """PSI bins on the REFERENCE's own quantiles, so A->B need not equal B->A -
    this is documented behaviour, not a bug, and the live monitor relies on it
    (reference = training videos, other = the new match)."""
    groups = {"A": _rng(0).normal(0, 1, (400, 1)),
              "B": _rng(1).normal(0, 3, (400, 1))}   # very different spread
    reports = {r.pair: r for r in compare_all_pairs(groups, ["x"])}
    ab = reports[("A", "B")].channels[0].psi
    ba = reports[("B", "A")].channels[0].psi
    assert ab != pytest.approx(ba, rel=0.2)


# --------------------------------------------------------------------------
# ReferenceDriftMonitor - the live-scoring path
# --------------------------------------------------------------------------

def test_monitor_scores_in_distribution_clip_as_low_drift():
    r = _rng()
    X = r.normal(0, 1, (200, 4))
    mon = ReferenceDriftMonitor.fit(X, ["a", "b", "c", "d"])
    typical = np.array([0.1, -0.2, 0.05, 0.0])
    out = mon.score(typical)
    assert out["mean_abs_z"] < 1.0


def test_monitor_scores_out_of_distribution_clip_as_high_drift():
    r = _rng()
    X = r.normal(0, 1, (200, 4))
    mon = ReferenceDriftMonitor.fit(X, ["a", "b", "c", "d"])
    outlier = np.array([10.0, -12.0, 9.0, 11.0])
    out = mon.score(outlier)
    assert out["mean_abs_z"] > 5.0


def test_monitor_top_offenders_names_the_worst_channel():
    r = _rng()
    X = r.normal(0, 1, (200, 3))
    mon = ReferenceDriftMonitor.fit(X, ["calm", "calm2", "wild"])
    x = np.array([0.1, -0.1, 20.0])
    out = mon.score(x)
    assert out["top_offenders"][0]["feature"] == "wild"


def test_monitor_handles_a_constant_reference_channel():
    X = np.column_stack([np.full(100, 5.0), np.random.default_rng(0).normal(0, 1, 100)])
    mon = ReferenceDriftMonitor.fit(X, ["constant", "varying"])
    out = mon.score(np.array([5.0, 0.0]))
    assert np.isfinite(out["mean_abs_z"])


def test_monitor_handles_nan_in_the_scored_clip():
    X = np.random.default_rng(0).normal(0, 1, (100, 3))
    mon = ReferenceDriftMonitor.fit(X, ["a", "b", "c"])
    out = mon.score(np.array([0.1, np.nan, 0.2]))
    assert np.isfinite(out["mean_abs_z"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
