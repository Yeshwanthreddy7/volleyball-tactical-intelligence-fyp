"""
Tests for jersey-colour team separation (fyp/teams.py).

The failure this module fixes, measured on "videoplayback (4)" frame 11290:
the geometric foot-point-vs-net filter passed TEN players - both teams mixed -
into a six-slot identity system, so every downstream tactical feature described
a formation that never existed on court.

Numpy + cv2 only; no torch, no weights, no video.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teams import (  # noqa: E402
    TeamClassifier, TeamVoter, torso_descriptor, NEAR, FAR, UNKNOWN,
)

cv2 = pytest.importorskip("cv2")


def _frame_with_jerseys(colours, box_w=40, box_h=120, gap=20):
    """Synthetic frame: one solid-colour 'player' per entry, left to right.

    Returns (frame, xyxy). Colours are BGR tuples.
    """
    n = len(colours)
    w = n * (box_w + gap) + gap
    frame = np.full((box_h + 2 * gap, w, 3), 60, dtype=np.uint8)  # dark court
    boxes = []
    for i, bgr in enumerate(colours):
        x1 = gap + i * (box_w + gap)
        y1 = gap
        x2, y2 = x1 + box_w, y1 + box_h
        frame[y1:y2, x1:x2] = bgr
        boxes.append([x1, y1, x2, y2])
    return frame, np.array(boxes, dtype=float)


LIGHT_BLUE = (230, 170, 90)     # BGR - ARG-like
DARK_NAVY = (70, 40, 25)        # BGR - ITA-like
WHITE = (240, 240, 240)         # libero-like contrasting kit


# --------------------------------------------------------------------------
# descriptor
# --------------------------------------------------------------------------

def test_same_jersey_gives_near_identical_descriptors():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, LIGHT_BLUE])
    d = torso_descriptor(frame, boxes)
    assert np.allclose(d[0], d[1], atol=1e-6)


def test_different_jerseys_are_separated_in_descriptor_space():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY])
    d = torso_descriptor(frame, boxes)
    assert np.linalg.norm(d[0] - d[1]) > 0.2


def test_descriptor_samples_the_torso_not_the_whole_box():
    """Legs/background must not drive the colour: paint only the torso band."""
    frame, boxes = _frame_with_jerseys([DARK_NAVY])
    x1, y1, x2, y2 = boxes[0].astype(int)
    frame[y1:y2, x1:x2] = (60, 60, 60)                       # neutral body
    bh = y2 - y1
    frame[y1 + int(0.15 * bh):y1 + int(0.50 * bh), x1:x2] = LIGHT_BLUE
    d = torso_descriptor(frame, boxes)

    ref, ref_boxes = _frame_with_jerseys([LIGHT_BLUE])
    d_ref = torso_descriptor(ref, ref_boxes)
    assert np.linalg.norm(d[0] - d_ref[0]) < 0.15


def test_empty_input_returns_empty():
    frame, _ = _frame_with_jerseys([LIGHT_BLUE])
    assert torso_descriptor(frame, np.empty((0, 4))).shape == (0, 3)


def test_degenerate_box_does_not_crash():
    frame, _ = _frame_with_jerseys([LIGHT_BLUE])
    d = torso_descriptor(frame, np.array([[10.0, 10.0, 10.0, 10.0]]))
    assert d.shape == (1, 3) and np.all(np.isfinite(d))


# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------

def _run_warmup(clf, frame, boxes, ys, n_frames=60):
    for _ in range(n_frames):
        out = clf.update(frame, boxes, ys)
    return out


def test_classifier_learns_team_from_court_side_then_uses_colour():
    """The core contract: geometry teaches, colour decides."""
    frame, boxes = _frame_with_jerseys(
        [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY, DARK_NAVY, DARK_NAVY]
    )
    ys = [700, 650, 600, 200, 250, 300]        # blue near, navy far
    clf = TeamClassifier(n_clusters=2, warmup_frames=10, min_samples=12)
    _run_warmup(clf, frame, boxes, ys, n_frames=20)
    assert clf.ready

    labels = clf.predict(frame, boxes)
    assert list(labels) == [NEAR, NEAR, NEAR, FAR, FAR, FAR]


def test_player_at_the_net_is_assigned_by_colour_not_position():
    """The exact frame-11290 failure: a near-team player standing at the net.

    Geometry would call this player FAR (foot point on the wrong side of an
    ambiguous line); colour must keep them with their team-mates.
    """
    frame, boxes = _frame_with_jerseys(
        [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY, DARK_NAVY, DARK_NAVY]
    )
    ys = [700, 650, 600, 200, 250, 300]
    clf = TeamClassifier(n_clusters=2, warmup_frames=10, min_samples=12)
    _run_warmup(clf, frame, boxes, ys, n_frames=20)

    # Same six jerseys, but now the blue #3 has advanced to the net (y=455).
    labels = clf.predict(frame, boxes)
    assert labels[2] == NEAR, "blue player at the net must stay with the blues"


def test_libero_contrasting_jersey_attaches_to_its_own_team():
    """FIVB requires a contrasting libero kit, so a team is TWO colour
    populations. With k>2 the libero cluster is labelled by its own court-side
    statistics instead of being forced into the nearest opponent colour."""
    frame, boxes = _frame_with_jerseys(
        [LIGHT_BLUE, LIGHT_BLUE, WHITE, DARK_NAVY, DARK_NAVY]
    )
    ys = [700, 650, 720, 200, 250]           # white libero plays NEAR back court
    clf = TeamClassifier(n_clusters=3, warmup_frames=10, min_samples=12,
                          min_active_players=1)  # fixture uses 5 synthetic players
    _run_warmup(clf, frame, boxes, ys, n_frames=20)

    labels = clf.predict(frame, boxes)
    assert labels[2] == NEAR, "libero must attach to the near team, not the navy"
    assert list(labels[:2]) == [NEAR, NEAR]
    assert list(labels[3:]) == [FAR, FAR]


def test_update_returns_none_during_warmup():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(warmup_frames=50, min_samples=1000)
    assert clf.update(frame, boxes, [700, 200]) is None
    assert not clf.ready


def test_predict_before_fit_raises_rather_than_guessing():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE])
    clf = TeamClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(frame, boxes)


def test_fit_report_records_the_evidence():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=6,
                          min_active_players=1)  # fixture uses 3 synthetic players
    _run_warmup(clf, frame, boxes, [700, 650, 200], n_frames=10)
    rep = clf.fit_report
    assert rep["n_samples"] > 0
    assert rep["n_near_clusters"] >= 1 and rep["n_far_clusters"] >= 1
    assert all("median_court_y" in c for c in rep["clusters"])


def test_non_finite_court_y_is_not_used_for_training():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=4,
                          min_active_players=1)  # fixture uses 2 synthetic players
    for _ in range(10):
        clf.update(frame, boxes, [700.0, np.nan])
    # Only the finite sample trained the model, so nothing is labelled FAR.
    assert clf.ready
    assert set(clf.cluster_team.tolist()) <= {NEAR, FAR, UNKNOWN}


# --------------------------------------------------------------------------
# degenerate-fit guard
# --------------------------------------------------------------------------

def test_all_one_side_is_flagged_degenerate():
    """The real clip_002 failure: with no court mask active, 24.6 detections
    per frame were mostly CROWD. Their varied shirt colours captured every
    cluster, all cluster medians landed beyond the net, the whole clip was
    labelled FAR, and it extracted with ZERO players - silently."""
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY, WHITE])
    clf = TeamClassifier(n_clusters=3, warmup_frames=5, min_samples=6,
                          min_active_players=1)  # fixture uses 3 synthetic players
    _run_warmup(clf, frame, boxes, [100, 150, 200], n_frames=10)   # all FAR

    assert clf.ready
    assert clf.degenerate, "a fit that found no near-side team must be flagged"


def test_healthy_two_sided_fit_is_not_degenerate():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=6,
                          min_active_players=1)  # fixture uses 3 synthetic players
    _run_warmup(clf, frame, boxes, [700, 650, 200], n_frames=10)

    assert clf.ready and not clf.degenerate


def test_unfitted_classifier_reports_degenerate():
    """Callers check `degenerate` to decide whether to trust colour; an
    unfitted model must not read as trustworthy."""
    assert TeamClassifier().degenerate


# --------------------------------------------------------------------------
# H4 (AUDIT_FINDINGS.md): similar-kit colour separation
# --------------------------------------------------------------------------

def _run_noisy_warmup(clf, base_colours, positions, seed=0, n_frames=30, noise_std=4.0):
    """Like _run_warmup, but with realistic per-frame colour jitter (camera
    noise / lighting / compression) instead of exact-repeated pixels - an
    exact-repeated synthetic frame has ~zero within-cluster spread, which
    breaks distance-based separation measures in ways real video never
    does (every player's jersey looks slightly different frame to frame)."""
    rng = np.random.default_rng(seed)
    out = None
    for _ in range(n_frames):
        jittered = [
            tuple(int(c) for c in (np.asarray(col) + rng.normal(0, noise_std, 3)).clip(0, 255))
            for col in base_colours
        ]
        frame, boxes = _frame_with_jerseys(jittered)
        out = clf.update(frame, boxes, positions)
    return out


def test_similar_kits_both_sides_present_is_still_flagged_degenerate():
    """H4: the case the old guard missed. Both NEAR and FAR clusters are
    non-empty (the old "all one side" check alone would pass this), but the
    two kits are colour-indistinguishable, so colour is adding nothing over
    what position already said - must still be degenerate."""
    navy = (70, 40, 25)
    barely_different = (74, 44, 29)
    clf = TeamClassifier(n_clusters=4, warmup_frames=5, min_samples=8,
                          min_active_players=1)  # fixture uses 4 synthetic players
    _run_noisy_warmup(clf, [navy, barely_different, navy, barely_different],
                       [700, 650, 200, 150])
    assert clf.ready
    assert (clf.cluster_team == NEAR).any() and (clf.cluster_team == FAR).any(), \
        "test setup check: both sides must be non-empty for this to test H4 and not condition 1"
    assert clf.degenerate
    assert clf.fit_report["geometric_agreement"] < TeamClassifier.MIN_GEOMETRIC_AGREEMENT


def test_identical_kits_is_the_zero_signal_floor_case():
    """H4 floor case: every player the SAME colour (pure noise, zero real
    signal) - colour-derived agreement with position should be ~chance."""
    same = (70, 40, 25)
    clf = TeamClassifier(n_clusters=4, warmup_frames=5, min_samples=8,
                          min_active_players=1)  # fixture uses 4 synthetic players
    _run_noisy_warmup(clf, [same, same, same, same], [700, 650, 200, 150])
    assert clf.ready
    assert clf.degenerate
    assert clf.fit_report["geometric_agreement"] < 0.7  # well below chance+margin


def test_stark_contrast_kits_pass_the_same_noisy_pipeline():
    """Control for the two tests above: real colour contrast, same noise
    and positions, must NOT be flagged - confirms the guard discriminates
    rather than just always firing."""
    light_blue = (230, 170, 90)
    dark_navy = (70, 40, 25)
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=8,
                          min_active_players=1)  # fixture uses 4 synthetic players
    _run_noisy_warmup(clf, [light_blue, light_blue, dark_navy, dark_navy],
                       [700, 650, 200, 150])
    assert clf.ready
    assert not clf.degenerate


# --------------------------------------------------------------------------
# H5 (AUDIT_FINDINGS.md): activity-gated warm-up + periodic re-fit
# --------------------------------------------------------------------------

def test_low_population_frame_is_not_counted_as_warmup_evidence():
    """A dead-ball/bench/substitution frame shows far fewer people than a
    real rally - must not poison the ONE fit a frozen-forever classifier
    would make from it."""
    clf = TeamClassifier(n_clusters=2, warmup_frames=3, min_samples=2,
                          min_active_players=6)
    dead_frame, dead_boxes = _frame_with_jerseys([(230, 170, 90), (70, 40, 25)])  # 2 - below threshold
    clf.update(dead_frame, dead_boxes, [700.0, 200.0])
    assert len(clf._desc) == 0, "a below-threshold frame must contribute zero evidence"

    live_frame, live_boxes = _frame_with_jerseys([(230, 170, 90)] * 3 + [(70, 40, 25)] * 3)  # 6
    clf.update(live_frame, live_boxes, [700.0, 650.0, 600.0, 200.0, 250.0, 300.0])
    assert len(clf._desc) == 6, "a full-population frame must contribute all its evidence"


def test_periodic_refit_replaces_a_good_fit_with_fresh_evidence():
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=8,
                          min_active_players=1, refit_every=10)
    _run_noisy_warmup(clf, [(230, 170, 90), (230, 170, 90), (70, 40, 25), (70, 40, 25)],
                       [700, 650, 200, 150], n_frames=10)
    assert clf.ready
    first_centres = clf.centres.copy()

    rng = np.random.default_rng(5)
    for _ in range(10):  # exactly refit_every more frames
        jittered = [tuple(int(c) for c in (np.asarray(col) + rng.normal(0, 4, 3)).clip(0, 255))
                    for col in [(230, 170, 90), (230, 170, 90), (70, 40, 25), (70, 40, 25)]]
        frame, boxes = _frame_with_jerseys(jittered)
        clf.update(frame, boxes, [700, 650, 200, 150])

    assert not np.array_equal(first_centres, clf.centres), \
        "a re-fit on fresh evidence must actually replace the frozen-forever centres"
    assert not clf.degenerate


def test_bad_refit_window_rolls_back_instead_of_replacing_a_good_fit():
    """H5's own safety net: periodic re-fitting must not be able to make
    things WORSE than never re-fitting - a re-fit window that lands on
    degenerate evidence (the exact scenario H5 exists to recover FROM, if
    it happens again later in a render) must not destroy a working fit."""
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=8,
                          min_active_players=1, refit_every=10)
    _run_noisy_warmup(clf, [(230, 170, 90), (230, 170, 90), (70, 40, 25), (70, 40, 25)],
                       [700, 650, 200, 150], n_frames=10)
    assert clf.ready and not clf.degenerate
    good_centres = clf.centres.copy()

    # Deterministically-degenerate re-fit evidence: identical descriptor
    # regardless of near/far position -> geometric_agreement forced to
    # exactly chance (0.5), guaranteed degenerate - not left to a noise draw.
    rng = np.random.default_rng(2)
    fixed_point = np.array([0.3, 0.1, 0.5])
    clf._refit_desc = [fixed_point + rng.normal(0, 0.001, 3) for _ in range(20)]
    clf._refit_ys = [700.0, 650.0, 600.0, 620.0, 610.0] * 2 + [200.0, 150.0, 180.0, 160.0, 170.0] * 2
    clf._frames_since_fit = clf.refit_every

    frame, boxes = _frame_with_jerseys([(230, 170, 90), (230, 170, 90), (70, 40, 25), (70, 40, 25)])
    clf.update(frame, boxes, [700, 650, 200, 150])

    assert np.array_equal(good_centres, clf.centres), "a degenerate re-fit must be rolled back"
    assert not clf.degenerate, "the previously-good fit must remain in effect"
    assert "rejected_refit" in clf.fit_report
    assert clf.fit_report["geometric_agreement"] >= TeamClassifier.MIN_GEOMETRIC_AGREEMENT


# --------------------------------------------------------------------------
# per-track voting
# --------------------------------------------------------------------------

def test_vote_overrides_a_single_bad_frame():
    """One blurred/occluded frame must not flip a track that is right in the
    other N-1 - the residual error mode measured on real footage."""
    v = TeamVoter()
    ids = np.array([7])
    for _ in range(9):
        v.update(ids, np.array([NEAR]))
    out = v.update(ids, np.array([FAR]))      # the bad frame
    assert out[0] == NEAR


def test_vote_follows_a_sustained_majority():
    v = TeamVoter()
    ids = np.array([7])
    for _ in range(3):
        v.update(ids, np.array([NEAR]))
    for _ in range(20):
        out = v.update(ids, np.array([FAR]))
    assert out[0] == FAR


def test_votes_are_independent_per_track():
    v = TeamVoter()
    ids = np.array([1, 2])
    for _ in range(5):
        out = v.update(ids, np.array([NEAR, FAR]))
    assert out[0] == NEAR and out[1] == FAR


def test_forget_drops_dead_tracks_so_recycled_ids_start_clean():
    v = TeamVoter()
    for _ in range(10):
        v.update(np.array([3]), np.array([NEAR]))
    v.forget(np.array([]))                      # track 3 died
    out = v.update(np.array([3]), np.array([FAR]))   # id recycled by ByteTrack
    assert out[0] == FAR


def test_voter_handles_empty_frame():
    v = TeamVoter()
    assert len(v.update(np.array([]), np.array([]))) == 0


def test_m13_zero_evidence_track_is_unknown_not_near():
    """M13 (AUDIT_FINDINGS.md): a track only ever observed as UNKNOWN (e.g.
    every sighting landed in the net-margin band during colour warm-up)
    must not silently default to NEAR - that fills a slot with a track the
    system has zero real evidence belongs to "our team"."""
    v = TeamVoter()
    ids = np.array([9])
    out = v.update(ids, np.array([UNKNOWN]))
    assert out[0] == UNKNOWN
    # stays UNKNOWN across repeated UNKNOWN-only observations, not just frame 1
    for _ in range(5):
        out = v.update(ids, np.array([UNKNOWN]))
    assert out[0] == UNKNOWN


def test_m13_genuine_tie_is_unknown_not_defaulted_to_near():
    """A real 50/50 NEAR/FAR split is just as much "no evidence for either
    side" as zero evidence - must not silently favour NEAR either."""
    v = TeamVoter(decay=1.0)  # no decay, so exact vote counts are easy to reason about
    ids = np.array([11])
    out = v.update(ids, np.array([NEAR]))
    out = v.update(ids, np.array([FAR]))
    assert out[0] == UNKNOWN  # 1-1, genuine tie


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
