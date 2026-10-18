"""M11 (AUDIT_FINDINGS.md): order_corners must produce a simple (non-bowtie)
polygon or refuse; manual --court-corners must fail loud on a degenerate
input; an auto-refresh quad that jumps too far is a bad detection, not
camera motion, and must be rejected. cv2 + numpy only."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import court as court_mod  # noqa: E402
from court import CourtCalibrator, order_corners  # noqa: E402


# --- order_corners -------------------------------------------------------

def test_order_corners_puts_top_left_first_on_a_clean_quad():
    # given deliberately out of order (BR, BL, TL, TR)
    pts = np.array([[900, 680], [120, 690], [300, 290], [920, 285]], dtype=np.float32)
    out = order_corners(pts)
    assert out is not None
    # slot 0 = TL = smallest (x+y)
    assert int(np.argmin(out.sum(axis=1))) == 0
    # area preserved (same 4 points, just reordered)
    import cv2
    assert cv2.contourArea(out) == pytest.approx(
        cv2.contourArea(cv2.convexHull(pts)), rel=0.02)


def test_order_corners_returns_a_simple_polygon_for_any_permutation():
    base = np.array([[300, 290], [920, 285], [900, 680], [120, 690]], dtype=np.float32)
    import cv2
    hull_area = cv2.contourArea(cv2.convexHull(base))
    rng = np.random.default_rng(0)
    for _ in range(12):
        perm = base[rng.permutation(4)]
        out = order_corners(perm)
        assert out is not None
        # a bowtie ordering has |shoelace area| < hull area; a simple one matches
        assert cv2.contourArea(out.astype(np.float32)) == pytest.approx(
            hull_area, rel=0.02)


def test_order_corners_rejects_a_collinear_quad():
    collinear = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=np.float32)
    assert order_corners(collinear) is None


# --- manual corners fail loud ------------------------------------------

def test_manual_corners_raise_on_a_degenerate_input():
    with pytest.raises(ValueError):
        CourtCalibrator(1280, 720,
                        manual_corners=[[0, 0], [10, 10], [20, 20], [30, 30]])


def test_manual_corners_accept_a_valid_quad():
    cal = CourtCalibrator(1280, 720,
                          manual_corners=[[100, 100], [1100, 100],
                                          [1100, 600], [100, 600]])
    assert cal.corners is not None
    assert cal.H is not None


# --- auto-refresh displacement bound ----------------------------------

def _auto_cal_with_baseline():
    cal = CourtCalibrator(1280, 720, auto=True, refresh_every=2)
    cal.set_auto_quad(np.array([[296, 293], [925, 285], [1165, 696], [117, 685]],
                               dtype=np.float32))
    return cal


def test_auto_refresh_rejects_a_far_jump(monkeypatch):
    cal = _auto_cal_with_baseline()
    before = cal.corners.copy()
    far = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype=np.float32)
    monkeypatch.setattr(court_mod, "detect_court_quad", lambda frame: far)
    fails0 = cal.n_auto_fail
    cal._frame_idx = 0            # so the next update lands on idx 1 (refresh)
    cal.update(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert cal.n_auto_fail == fails0 + 1
    np.testing.assert_array_equal(cal.corners, before)   # kept previous


def test_auto_refresh_accepts_a_small_plausible_drift(monkeypatch):
    cal = _auto_cal_with_baseline()
    drift = cal.corners + np.array([8.0, -6.0], dtype=np.float32)
    monkeypatch.setattr(court_mod, "detect_court_quad", lambda frame: drift)
    ok0 = cal.n_auto_success
    cal._frame_idx = 0
    cal.update(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert cal.n_auto_success == ok0 + 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
