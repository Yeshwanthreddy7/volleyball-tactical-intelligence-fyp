"""Regression tests for M12 (AUDIT_FINDINGS.md): CameraMotionEstimator must
reject an implausible frame-to-frame affine instead of applying it - CMC
warps CourtCalibrator.corners UNCONDITIONALLY on whatever estimate() returns,
so one bad estimate (a scene cut, a low-texture frame) corrupts the
calibration for every subsequent frame, not just the current one.

Builds affines directly rather than driving real optical flow, so this is
fast and deterministic: cv2 + numpy only.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from court import CameraMotionEstimator  # noqa: E402

cv2 = pytest.importorskip("cv2")

W, H = 1280, 720


def _affine(scale=1.0, rot_deg=0.0, tx=0.0, ty=0.0):
    """A 2x3 similarity-transform affine, the same shape
    estimateAffinePartial2D returns."""
    th = np.radians(rot_deg)
    a, c = scale * np.cos(th), scale * np.sin(th)
    b, d = -scale * np.sin(th), scale * np.cos(th)
    return np.array([[a, b, tx], [c, d, ty]], dtype=np.float32)


def test_healthy_small_pan_is_plausible():
    cme = CameraMotionEstimator()
    A = _affine(scale=1.01, rot_deg=0.3, tx=8, ty=2)
    assert cme._is_plausible(A, W, H, inlier_ratio=0.9)


def test_wild_scale_jump_is_rejected():
    cme = CameraMotionEstimator()
    A = _affine(scale=1.8, rot_deg=0, tx=0, ty=0)
    assert not cme._is_plausible(A, W, H, inlier_ratio=0.9)


def test_wild_rotation_jump_is_rejected():
    cme = CameraMotionEstimator()
    A = _affine(scale=1.0, rot_deg=25, tx=0, ty=0)
    assert not cme._is_plausible(A, W, H, inlier_ratio=0.9)


def test_wild_translation_jump_is_rejected():
    cme = CameraMotionEstimator()
    A = _affine(scale=1.0, rot_deg=0, tx=400, ty=0)
    assert not cme._is_plausible(A, W, H, inlier_ratio=0.9)


def test_low_inlier_ratio_is_rejected_even_with_plausible_geometry():
    """A scene cut can still produce a geometrically-plausible-LOOKING
    affine from whichever points happened to survive RANSAC - the inlier
    ratio is what actually signals the match itself was unreliable."""
    cme = CameraMotionEstimator()
    A = _affine(scale=1.0, rot_deg=0.3, tx=5, ty=2)
    assert not cme._is_plausible(A, W, H, inlier_ratio=0.3)


def test_boundary_case_within_all_four_bounds_simultaneously_passes():
    """Confirms the four checks compose correctly - none is accidentally
    stricter than documented when combined with the others."""
    cme = CameraMotionEstimator()
    A = _affine(
        scale=1.0 + cme.MAX_SCALE_CHANGE - 0.01,
        rot_deg=cme.MAX_ROTATION_DEG - 0.1,
        tx=cme.MAX_TRANSLATION_FRAC * min(W, H) - 1.0,
        ty=0,
    )
    assert cme._is_plausible(A, W, H, inlier_ratio=cme.MIN_INLIER_RATIO + 0.01)


def test_estimate_returns_none_on_first_frame():
    """No previous frame yet - nothing to compare motion against."""
    cme = CameraMotionEstimator()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    assert cme.estimate(frame) is None


def test_estimate_returns_none_on_a_blank_second_frame():
    """A textureless frame yields no trackable features - must return None
    (keep previous calibration), not raise or fabricate a motion estimate."""
    cme = CameraMotionEstimator()
    blank = np.full((H, W, 3), 128, dtype=np.uint8)
    cme.estimate(blank)
    assert cme.estimate(blank) is None
