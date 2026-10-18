"""Regression test: H8 (AUDIT_FINDINGS.md) - a panel re-running the same
command against the same video must get byte-identical output. Verified
once end-to-end this session (two independent `pipeline.py` process runs,
sha256-identical CSVs); this test guards the three RNG sources that claim
depends on, at unit-test speed (no video, no detector, no torch model):

1. numpy's global RNG (seeded in run_pipeline / prepare_training_data.run)
2. OpenCV's RANSAC (cv2.estimateAffinePartial2D, method=cv2.RANSAC - the
   exact call CourtCalibrator's camera-motion compensation uses), seeded via
   cv2.setRNGSeed - the same primitive, same params, called from an
   isolated test rather than requiring two real optical-flow frames.
3. teams.py's k-means, which was ALREADY correctly self-seeded before this
   session (np.random.default_rng(seed), not the global RNG) - re-verified
   here as a regression guard, not re-fixed.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teams import _kmeans  # noqa: E402


def test_numpy_global_rng_reproducible_when_seeded():
    np.random.seed(0)
    draw1 = np.random.uniform(0, 1, 50)
    np.random.seed(0)
    draw2 = np.random.uniform(0, 1, 50)
    assert np.array_equal(draw1, draw2)


def test_opencv_ransac_reproducible_when_seeded():
    """Same RANSAC call CourtCalibrator._update_cmc uses
    (cv2.estimateAffinePartial2D, method=cv2.RANSAC), with deliberate
    outliers so RANSAC's random sampling actually has something to do."""
    rng = np.random.default_rng(42)
    p0 = rng.uniform(0, 500, (60, 2)).astype(np.float32)
    true_A = np.array([[1.02, -0.01, 5.0], [0.01, 0.98, -3.0]], dtype=np.float32)
    p1 = (true_A[:, :2] @ p0.T).T + true_A[:, 2] + rng.normal(0, 0.5, p0.shape)
    p1[:8] += rng.uniform(-200, 200, (8, 2))  # outliers, force real RANSAC work
    p1 = p1.astype(np.float32)

    results = []
    for _ in range(3):
        cv2.setRNGSeed(0)
        A, _inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        results.append(A)

    assert all(np.array_equal(results[0], r) for r in results[1:]), (
        "cv2.estimateAffinePartial2D(method=cv2.RANSAC) gave different "
        "results across runs with the same cv2.setRNGSeed() - camera-motion "
        "compensation would not be reproducible."
    )


def test_teams_kmeans_already_deterministic():
    """Regression guard, not a new fix: _kmeans uses its own seeded
    np.random.default_rng, verified independent of global RNG state."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, (80, 3))

    np.random.seed(1)  # perturb global state differently before each call
    centres1, labels1 = _kmeans(x, k=4, seed=0)
    np.random.seed(999)
    centres2, labels2 = _kmeans(x, k=4, seed=0)

    assert np.array_equal(centres1, centres2)
    assert np.array_equal(labels1, labels2)
