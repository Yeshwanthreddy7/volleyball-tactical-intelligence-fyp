"""
pipeline.py – End-to-end volleyball analytics pipeline.

Implements the complete processing chain from a raw match video to a
printed tactical coordination report:

  Match Video (.mp4)
        ↓
  Preflight Safety Check  (abort on degenerate input rather than report on it)
        ↓
  Dual Detection  (YOLOv11: stock weights for players, fine-tuned for ball)
        ↓
  Player Tracking   (BoT-SORT by default, ByteTrack fallback, via supervision)
        ↓
  Court Calibration  (homography from court corners, or documented linear fallback)
        ↓
  Team Assignment  (jersey-colour clustering → 6 own-team identity slots)
        ↓
  Feature Engineering  (18 permutation-invariant channels: formation, spacing,
                         motion, synchrony, ball relation)
        ↓
  Sequence Formation  (non-overlapping 29-frame windows)
        ↓
  Classifier  (scikit-learn model by default; Mamba SSM checkpoint as the
               deep-learning comparison arm — selected by checkpoint file
               extension, .joblib vs .pt)
        ↓
  Final Tactical Report

Usage
-----
  python pipeline.py <video> <checkpoint> [options]

Example
-------
  python pipeline.py match.mp4 mamba_checkpoint.pt \\
      --output-video annotated.mp4 \\
      --output-csv   predictions.csv

Court coordinate system
-----------------------
  Default: pixel coordinates are linearly scaled to 1800 × 900 cm
  (1800 cm wide, 900 cm tall, net at Y = 450 cm for top-down view).

  For accurate court mapping supply --court-corners with the four pixel
  coordinates of the court boundary corners (TL TR BR BL) and the script
  will compute a perspective homography.

  Example:
    --court-corners 42,18 1238,18 1238,702 42,702
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    HAS_ULTRALYTICS = False

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:  # pragma: no cover
    HAS_SUPERVISION = False

from mamba_model import FEATURE_COLS, LABEL_NAMES
from label_clips import LABEL_TO_INDEX, generate_report, label_clip
from features import (
    MODEL_FEATURE_COLS,
    build_model_sequence,
    interpolate_gaps,
    kinematic_features,
    raw_frame_to_arrays,
    recover_identity,
    sync_score as _id_sync_score,
    convex_hull_area,
)
from interfaces import (
    DEFAULT_IMGSZ,
    ClassificationResult,
    TorchTemporalClassifier,
    create_detector,
    create_temporal_classifier,
    shannon_entropy_bits,
)
from tracking import create_tracker
from identity import SlotManager
from strict_render import StrictRenderer
from teams import (
    TeamClassifier,
    TeamVoter,
    NEAR as TEAM_NEAR,
    FAR as TEAM_FAR,
    UNKNOWN as TEAM_UNKNOWN,
)
import preflight
from court import CourtCalibrator
from pose import PoseEstimator, PoseFrameSummary


# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

PERSON_CLASS_ID = 0        # COCO class 0 = person
SPORTS_BALL_CLASS_ID = 32  # COCO class 32 = sports ball (standard COCO model)


SEQ_LEN = 29               # frames per sequence (matches training window)
N_PLAYERS = 6              # players tracked per sequence

COURT_W_CM = 1800.0        # court width  (cm)
COURT_H_CM = 900.0         # court height (cm)
NET_Y_CM = COURT_H_CM / 2.0

# Visual style for annotated output video
TRACK_COLOR = (0, 255, 0)       # green – player bounding boxes (default / no label yet)
BALL_COLOR = (0, 165, 255)      # orange – ball marker
TEXT_COLOR = (255, 255, 255)    # white – overlaid text
LABEL_BG_COLOR = (30, 30, 30)  # dark – label banner background (default)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Label-specific box / banner colors (BGR) for tactical classification visualization
#   🟠 1 – Coordinated Attack  → orange
#   🔵 2 – Coordinated Defense → blue
#   🩵 3 – Delayed Support     → cyan
#   🔴 4 – Spacing Breakdown   → red
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "Coordinated Attack":  (0, 165, 255),   # BGR: orange
    "Coordinated Defense": (255, 0, 0),      # BGR: blue
    "Delayed Support":     (255, 255, 0),    # BGR: cyan
    "Spacing Breakdown":   (0, 0, 255),      # BGR: red
    "Unclassified":        (140, 140, 140),  # BGR: grey – noise sink (class 0)
    "No manual label":     (90, 90, 90),     # BGR: dim grey – --labels-csv gap
}
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# Number of historical ball positions kept for the trail overlay.
BALL_TRAIL_LEN = 45

# Maximum consecutive frames the BallTracker will predict ball position
# (constant-velocity model) without a new YOLO detection.
BALL_MAX_MISS_FRAMES = 12

# Simplified phase names displayed prominently in the annotation banner.
# Internal label name → user-facing display string.
DISPLAY_NAMES: dict[str, str] = {
    "Coordinated Attack":  "ATTACKING",
    "Coordinated Defense": "DEFENDING",
    "Delayed Support":     "DELAYED SUPPORT",
    "Spacing Breakdown":   "SPACING BREAKDOWN",
    "Unclassified":        "TRANSITION / DEAD BALL",
    "No manual label":     "NO MANUAL LABEL",
}


# ---------------------------------------------------------------------------
# The class-0 activity gate (spec §4B – the noise sink)
# ---------------------------------------------------------------------------

def _activity_gate(
    seq_raw: np.ndarray,
    min_mean_speed_cm: float = 2.0,
    min_players: float = 3.0,
) -> tuple[bool, dict[str, float]]:
    """
    Decide whether a 29-frame window is real play or dead-ball noise.

    Forcing celebrations / huddles / service prep into a tactical class would
    poison the model's gradients (and its report).  A window is routed to
    class 0 ("Unclassified / Transition") when EITHER:
      • mean per-player speed is below `min_mean_speed_cm` cm/frame
        (≈ 0.6 m/s at 30 fps: below deliberate walking pace), or
      • fewer than `min_players` players are visible on average (camera cut,
        replay, crowd shot – the 6-player matrix is not observable).

    Returns (is_dead_ball, gate_metrics).
    """
    players, _ = raw_frame_to_arrays(seq_raw)
    tracked = interpolate_gaps(recover_identity(players))

    n_present = float(np.mean([
        (~np.isnan(f).any(axis=1)).sum() for f in tracked
    ]))

    disp = np.diff(tracked, axis=0)
    speed = np.linalg.norm(disp, axis=-1)            # (T-1, K) cm/frame
    mean_speed = float(np.nanmean(speed)) if np.isfinite(speed).any() else 0.0

    gate = {
        "gate_mean_speed_cm_frame": round(mean_speed, 2),
        "gate_players_present": round(n_present, 2),
    }
    return (mean_speed < min_mean_speed_cm) or (n_present < min_players), gate


def _frame_hsv_hist(frame: np.ndarray) -> np.ndarray:
    """Normalised 2-D H-S histogram of a frame, for shot-boundary detection.

    H3 (AUDIT_FINDINGS.md): a hard cut inside a 29-frame window (replay,
    angle change, full-screen graphic) does not lower motion, so
    `_activity_gate` never catches it - the window is classified as if the
    two spliced shots were one rally. An HSV histogram is used rather than a
    raw pixel diff because it is invariant to fast pans and to a single
    bright flash (camera strobe) while still dropping sharply on a real cut.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
    return hist


def _is_scene_cut(
    prev_hist: np.ndarray | None,
    cur_hist: np.ndarray,
    ids_prev: set,
    ids_now: set,
    corr_threshold: float,
    id_churn_threshold: float,
) -> tuple[bool, float, float]:
    """True if the frame is a shot boundary. Returns (is_cut, hist_corr,
    id_carryover). Either half can fire: the histogram catches a cut to a
    visually different shot; the identity-churn half catches a cut between
    two visually similar court angles where the tracker still lost every
    track."""
    corr = 1.0 if prev_hist is None else float(
        cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_CORREL))
    if ids_prev and ids_now:
        carry = len(ids_prev & ids_now) / max(len(ids_now), 1)
    else:
        carry = 1.0  # no tracks either side -> churn signal not meaningful
    hist_cut = prev_hist is not None and corr < corr_threshold
    churn_cut = (id_churn_threshold > 0.0 and bool(ids_prev) and bool(ids_now)
                 and carry < id_churn_threshold)
    return (hist_cut or churn_cut), corr, carry


# ---------------------------------------------------------------------------
# Ball tracking (constant-velocity model with EMA velocity smoothing)
# ---------------------------------------------------------------------------

class BallTracker:
    """
    Lightweight ball tracker that bridges gaps in YOLO ball detections.

    YOLO (especially the generic COCO-weights model) frequently misses a fast-moving
    volleyball for several consecutive frames due to motion blur, small
    apparent size, or occlusion.  This class maintains an estimated ball state
    (position + velocity) and predicts the ball's position during missed
    frames using a constant-velocity kinematic model.

    Detection vs Prediction
    -----------------------
    • When YOLO finds the ball → state is updated with the measured centre
      and velocity is refined with an exponential moving average (EMA).
    • When YOLO misses the ball → position is advanced by the last velocity
      estimate.  After ``BALL_MAX_MISS_FRAMES`` consecutive misses the tracker
      gives up (returns None) so stale predictions don't persist indefinitely.

    Parameters
    ----------
    max_miss_frames : int
        Maximum consecutive frames the tracker will predict without a fresh
        YOLO detection.  Default: ``BALL_MAX_MISS_FRAMES`` (12 frames).
    velocity_alpha  : float
        EMA weight applied to each new velocity measurement.
        Higher values → more responsive but noisier.  Default: 0.55.

    Usage
    -----
    ::

        tracker = BallTracker()
        for frame in video_frames:
            detected_center = yolo_detect_ball(frame)   # np.ndarray or None
            ball_pos, is_predicted = tracker.update(detected_center)
            # ball_pos: (2,) pixel coords or None (ball lost)
            # is_predicted: True when position is an extrapolation
    """

    #: Reject a detected centre that would imply a single-frame jump larger
    #: than this (pixels). A volleyball tops out near ~35 m/s; on a 1280-wide
    #: broadcast frame (~18 m court) that is ~40 px/frame at 30 fps, so 180 px
    #: is a generous ceiling that still cuts the far-court false positives that
    #: make the marker teleport. Scaled up while the ball is missing (the true
    #: ball keeps moving under the miss).
    _MAX_JUMP_PX = 180.0

    def __init__(
        self,
        max_miss_frames: int = BALL_MAX_MISS_FRAMES,
        velocity_alpha: float = 0.55,
        max_jump_px: float = _MAX_JUMP_PX,
    ) -> None:
        self._max_miss = max_miss_frames
        self._alpha = velocity_alpha
        self._max_jump = float(max_jump_px)
        self._pos: np.ndarray | None = None   # last known pixel (x, y)
        self._vel: np.ndarray = np.zeros(2)   # estimated velocity (dx, dy) in pixels/frame
        self._miss: int = 0                   # consecutive missed-detection frames

    def update(
        self, detected_center: np.ndarray | None
    ) -> tuple[np.ndarray | None, bool]:
        """
        Advance the tracker by one frame.

        Parameters
        ----------
        detected_center : (2,) pixel position from YOLO, or None if not found.

        Returns
        -------
        position     : (2,) pixel position estimate, or None when ball is lost.
        is_predicted : True when the returned position is an extrapolation;
                       False when it comes directly from YOLO.
        """
        if detected_center is not None:
            detected_center = np.asarray(detected_center, dtype=float)
            # Trajectory gate: a detection that would teleport the ball across
            # the court in one frame is a false positive (a head, a line
            # marking, a second ball in the crowd). Reject it and coast on the
            # prediction instead - but only for a few frames, so a genuine new
            # rally that starts far away is still picked up.
            if self._pos is not None and self._miss < 4:
                jump = float(np.linalg.norm(detected_center - self._pos))
                if jump > self._max_jump * (1.0 + self._miss):
                    self._miss += 1
                    self._pos = self._pos + self._vel
                    return self._pos.copy(), True
            # Ball detected – update velocity with EMA and reset miss counter
            if self._pos is not None:
                raw_vel = detected_center - self._pos
                self._vel = self._alpha * raw_vel + (1.0 - self._alpha) * self._vel
            self._pos = detected_center.copy()
            self._miss = 0
            return self._pos.copy(), False

        # Ball not detected
        self._miss += 1
        if self._miss > self._max_miss or self._pos is None:
            # Prediction horizon exceeded or ball was never seen → give up
            return None, False

        # Predict next position using last known velocity
        self._pos = self._pos + self._vel
        return self._pos.copy(), True

    def reset(self) -> None:
        """Reset internal state (use between video clips)."""
        self._pos = None
        self._vel = np.zeros(2)
        self._miss = 0


# ---------------------------------------------------------------------------
# Side-of-net filtering (uses the live calibrator, spec §3)
# ---------------------------------------------------------------------------

_SIDE_FILTER_WARNED = False


def _safe_fps(cap, default: float = 30.0) -> float:
    """L18 (AUDIT_FINDINGS.md): cap.get(CAP_PROP_FPS) can return NaN for a
    malformed container, and `nan or default` does NOT fall back - NaN is
    truthy in Python, so the existing `... or 30.0` pattern silently let a
    NaN fps propagate into time_s, the HUD and the VideoWriter. Validate
    finite-and-positive explicitly instead of relying on truthiness."""
    v = cap.get(cv2.CAP_PROP_FPS)
    return v if (v and math.isfinite(v) and v > 0) else default


def _safe_frame_count(cap, default: int = 0) -> int:
    """L18: CAP_PROP_FRAME_COUNT can be negative or absurd on a broken
    container; `int(-1) or 0` does not catch a negative value (-1 is
    truthy). Clamp to a sane non-negative int."""
    v = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def _preflight_sample(cap, detector, calibrator, args, n_samples: int = 24) -> dict:
    """
    Measure the surviving player population at each stage on a sample of the
    frames that are about to be processed.

    Sampling is spread across the requested segment rather than taken from its
    first frames: broadcast footage opens on replays, graphics and crowd shots,
    and a contiguous sample from frame 0 would measure those instead of the
    rally.  Uses the geometric team split for the team-stage estimate because
    it is cheap and always available - which makes the estimate conservative
    when nothing else is running (geometry over-counts by passing both front
    rows).

    H7 (AUDIT_FINDINGS.md): when the run will actually use
    `--team-split colour`, the geometric estimate above is not what will
    run - it is a crude proxy that is EXPECTED to over-count (both front
    rows near the net), so a FATAL threshold tuned for it is not the right
    check for a colour run, and conversely a colour fit that is wrong but
    not degenerate (tracking the OPPONENT's six) would pass a geometric-only
    preflight cleanly, since a person count can't see WHICH six were
    counted. So: also fit a cheap batch `TeamClassifier` on these same
    sampled frames' evidence and report ITS population estimate and
    `degenerate` state - `assess()` prefers this one when it is available.
    """
    start = max(int(args.start_frame), 0)
    total = _safe_frame_count(cap)
    end = total if args.max_frames <= 0 else min(total, start + int(args.max_frames))
    if end <= start:
        end = start + n_samples
    idxs = np.linspace(start, max(end - 1, start), num=max(n_samples, 1)).astype(int)

    raw, masked, team = [], [], []
    ball_hits = 0
    n_read = 0

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        n_read += 1
        calibrator.update(frame)

        det = detector.predict(frame)
        raw.append(len(det.person_xyxy))
        if det.ball_center is not None:
            ball_hits += 1

        mx, mc = calibrator.filter_detections(det.person_xyxy, det.person_conf)
        masked.append(len(mx))

        tx, tc = _filter_players_by_team_side(
            mx, mc, calibrator, args.team_side, args.net_margin_cm
        )
        team.append(len(tx))

    colour_report: dict | None = None
    if args.team_split == "colour" and args.team_side != "all":
        # H7: deliberately NOT reusing the sparse whole-segment `idxs` above.
        # Measured directly: fitting on 24 samples spread across the whole
        # segment gave a false-negative "degenerate" verdict (0.667
        # agreement) on a clip that classifies cleanly via the real run's
        # dense 45-CONSECUTIVE-frame warm-up - sparse samples spanning a
        # wide time range are a harder, unrepresentative fitting scenario,
        # not evidence colour will actually fail. Read a fresh CONSECUTIVE
        # run starting at the real segment's start instead, matching
        # exactly what the live warm-up will see.
        colour_clf = TeamClassifier(n_clusters=args.team_clusters, warmup_frames=0,
                                     min_samples=8, min_active_players=1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(args.team_warmup_frames):
            ok, frame = cap.read()
            if not ok:
                break
            calibrator.update(frame)
            det = detector.predict(frame)
            mx, mc = calibrator.filter_detections(det.person_xyxy, det.person_conf)
            if len(mx):
                court_ys = [calibrator.pixel_to_court(float((x1 + x2) / 2.0), float(y2))[1]
                            for (x1, y1, x2, y2) in mx]
                colour_clf.update(frame, mx, court_ys)
        # Preflight is a one-shot sample, not a stream - force a fit on
        # whatever evidence accumulated rather than waiting for warmup_frames
        # sample-count bookkeeping meant for the live per-frame path.
        if colour_clf.centres is None and len(colour_clf._desc) >= 8:
            colour_clf._fit()
        if colour_clf.ready:
            # NEAR/FAR are already fixed by court-Y (>= net_y_cm is NEAR),
            # so near medians are always numerically above far medians for
            # a sane fit - no --team-side branching needed here.
            near_medians = [c["median_court_y"] for c in colour_clf.fit_report["clusters"]
                             if c["team"] == "near"]
            far_medians = [c["median_court_y"] for c in colour_clf.fit_report["clusters"]
                            if c["team"] == "far"]
            colour_report = {
                "degenerate": colour_clf.degenerate,
                "geometric_agreement": colour_clf.fit_report.get("geometric_agreement"),
                "near_far_court_y_gap": (
                    (min(near_medians) - max(far_medians))
                    if near_medians and far_medians else None
                ),
            }

    def _median(v):
        return float(np.median(v)) if v else 0.0

    return {
        "n_frames": n_read,
        "raw_person_median": _median(raw),
        "masked_person_median": _median(masked),
        "team_person_median": _median(team),
        "ball_recall": (ball_hits / n_read) if n_read else 0.0,
        "team_side": args.team_side,  # M9: lets assess() know "all" is expected-high
        "team_split": args.team_split,  # H7: lets assess() know whether colour_* applies
        "colour_report": colour_report,  # H7: None unless --team-split colour and a fit was reachable
    }


def _team_labels_geometric(
    court_ys, net_margin_cm: float, net_y_cm: float = NET_Y_CM,
) -> np.ndarray:
    """
    NEAR/FAR label per detection from the foot point alone.

    Used (a) as the colour model's warm-up fallback and (b) as the label source
    that tells the colour clusters which side they belong to.  Players within
    `net_margin_cm` of the net are genuinely ambiguous geometrically - that is
    the whole reason the colour model exists - so they are marked UNKNOWN here
    rather than guessed.

    H6 (AUDIT_FINDINGS.md): `net_y_cm` defaults to the frame-midpoint
    constant for backward compatibility, but the real call site passes
    `CourtCalibrator.net_y_cm_current()` - the frame-midpoint assumption is
    routinely wrong under --court-coords linear (measured on video (4): net
    pixel-row 289 -> court_y ~361, not 450), and every player this function
    labels feeds directly into which cluster teams.py's colour model is told
    is "our team" during warm-up.
    """
    out = []
    for y in np.asarray(court_ys, dtype=float).reshape(-1):
        if not np.isfinite(y):
            out.append(TEAM_UNKNOWN)
        elif y >= net_y_cm + net_margin_cm:
            out.append(TEAM_NEAR)
        elif y <= net_y_cm - net_margin_cm:
            out.append(TEAM_FAR)
        else:
            out.append(TEAM_UNKNOWN)
    return np.array(out, dtype=int)


def _filter_players_by_team_side(
    person_xyxy: np.ndarray,
    person_conf: np.ndarray,
    calibrator: CourtCalibrator,
    team_side: str,
    net_margin_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only player detections on the requested side of the net.

    team_side values:
      - "bottom": keep players with court_y >= NET_Y_CM + net_margin_cm
      - "top"   : keep players with court_y <= NET_Y_CM - net_margin_cm
      - "all"   : keep every detected player

    Positions are evaluated at the FOOT POINT (bottom-centre of the box) via
    the CURRENT homography, which follows the camera when CMC / auto-court is
    enabled.
    """
    if team_side == "all" or len(person_xyxy) == 0:
        return person_xyxy, person_conf

    # H6 (AUDIT_FINDINGS.md): derived from the detected quad's own top edge
    # when one exists, not the frame-midpoint assumption - see
    # CourtCalibrator.net_y_cm_current for why that assumption is wrong
    # under --court-coords linear.
    net_y = calibrator.net_y_cm_current()

    keep_indices: list[int] = []
    for i, (x1, y1, x2, y2) in enumerate(person_xyxy):
        px = float((x1 + x2) / 2.0)
        py = float(y2)                       # foot point, not centroid
        _, court_y = calibrator.pixel_to_court(px, py)

        if team_side == "bottom":
            on_side = court_y >= (net_y + net_margin_cm)
        else:  # team_side == "top"
            on_side = court_y <= (net_y - net_margin_cm)

        if on_side:
            keep_indices.append(i)

    if not keep_indices:
        global _SIDE_FILTER_WARNED
        if len(person_xyxy) >= 4 and not _SIDE_FILTER_WARNED:
            _SIDE_FILTER_WARNED = True
            print(
                "[WARNING] team-side filter removed ALL {} detections in a "
                "frame - court calibration and --team-side may disagree "
                "(shown once).".format(len(person_xyxy)),
                file=sys.stderr,
            )
        return (
            np.empty((0, 4), dtype=person_xyxy.dtype),
            np.empty((0,), dtype=person_conf.dtype),
        )

    idx = np.array(keep_indices, dtype=int)
    return person_xyxy[idx], person_conf[idx]


# ---------------------------------------------------------------------------
# Position extraction (Pose / Position Extraction step)
# ---------------------------------------------------------------------------

def _extract_positions(
    tracked_xyxy: np.ndarray,
    track_ids: np.ndarray,
    ball_center: np.ndarray | None,
    calibrator: CourtCalibrator,
    slot_map: "dict[int, int] | SlotManager",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ByteTrack detections to court coordinates with a STABLE, identity-
    consistent slot assignment.

    Loophole fixed (L1 – identity swap)
    -----------------------------------
    The previous implementation re-sorted every track by "age" on *every frame*
    and packed them into slots 0..5 in that order.  Because ages change as
    tracks appear/disappear, the player occupying a given slot kept changing,
    so the per-slot velocity / sync features were computed across DIFFERENT
    physical players – i.e. they were noise.

    This version keeps a persistent ``slot_map: {tracker_id -> slot}``.  Once a
    ByteTrack ID is given a slot it keeps it for the whole clip; a slot is only
    reused after its track has been gone.  Column ``p_i`` is therefore the same
    player across the entire sequence.

    Loophole fixed (L2 – (0,0) sentinel)  &  foot-point (distance bias)
    ------------------------------------------------------------------
    Absent players are returned as NaN (not (0,0), which is a real court
    corner), and player ground position is taken at the FEET (bottom-centre of
    the bounding box) rather than the box centroid, which removes a
    perspective-dependent position bias.

    Returns
    -------
    player_positions : (6, 2) court-cm coords; NaN rows for empty slots.
    ball_pos         : (2,) court-cm coords; NaN if ball unknown.
    """
    # Foot point (bottom-centre) per active track ID, in court cm.
    court_feet: dict[int, tuple[float, float]] = {}
    for tid, (x1, y1, x2, y2) in zip(track_ids, tracked_xyxy):
        cx, cy = calibrator.pixel_to_court((x1 + x2) / 2.0, y2)
        court_feet[int(tid)] = (cx, cy)  # foot = bottom-centre

    if isinstance(slot_map, SlotManager):
        # Online identity bridge (stage 1): a slot survives <=15-frame
        # occlusions in limbo, and a NEW tracker id appearing within a
        # court-distance gate of the vacated position inherits the slot -
        # column p_i stays the same physical player across tracker re-ids.
        # (Stage 2, recover_identity + interpolate_gaps in features.py,
        # still repairs anything that slips through, for train AND serve.)
        id_to_slot = slot_map.assign(court_feet)
    else:
        # Legacy dict behaviour (kept for backward compatibility).
        used_slots = {s for s in slot_map.values()}
        for tid in court_feet:
            if tid not in slot_map:
                free = next((s for s in range(N_PLAYERS) if s not in used_slots), None)
                if free is not None:
                    slot_map[tid] = free
                    used_slots.add(free)
        for tid in [t for t in slot_map if t not in court_feet]:
            del slot_map[tid]
        id_to_slot = dict(slot_map)

    player_positions = np.full((N_PLAYERS, 2), np.nan, dtype=float)
    for tid, (cx, cy) in court_feet.items():
        slot = id_to_slot.get(tid)
        if slot is None:
            continue
        player_positions[slot] = [cx, cy]

    # Ball
    if ball_center is not None:
        bx, by = calibrator.pixel_to_court(
            float(ball_center[0]), float(ball_center[1])
        )
        ball_pos = np.array([bx, by], dtype=float)
    else:
        ball_pos = np.array([np.nan, np.nan], dtype=float)

    return player_positions, ball_pos


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _compute_frame_features(
    player_positions: np.ndarray,  # (6, 2)
    ball_pos: np.ndarray,          # (2,)
) -> np.ndarray:
    """
    Build a 14-element feature row matching the model's FEATURE_COLS:
      [ball_x, ball_y, p1_x, p1_y, p2_x, p2_y, …, p6_x, p6_y]

    Parameters
    ----------
    player_positions : (6, 2) court cm coordinates
    ball_pos         : (2,) court cm coordinates

    Returns
    -------
    row : (14,) float32 feature vector
    """
    row = np.empty(len(FEATURE_COLS), dtype=np.float32)
    row[0] = ball_pos[0]
    row[1] = ball_pos[1]
    for i in range(N_PLAYERS):
        row[2 + i * 2] = player_positions[i, 0]
        row[2 + i * 2 + 1] = player_positions[i, 1]
    return row


def _sequence_metrics(seq: np.ndarray) -> dict[str, float]:
    """
    Compute descriptive metrics for a 29-frame raw sequence.

    All metrics are now computed on IDENTITY-CONSISTENT tracks (via
    features.recover_identity) and ignore missing (NaN) slots, so sync_score
    and centroid velocity are no longer corrupted by slot-swapping or by the
    legacy (0,0) sentinel.
    """
    players, ball = raw_frame_to_arrays(seq)       # (T,6,2)/(T,2), NaN = missing
    identity_fixed = recover_identity(players)     # fix identity (L1)
    tracked = interpolate_gaps(identity_fixed)     # bridge <=15-frame occlusion

    sync = _id_sync_score(tracked)

    # Mean nearest-neighbour spacing per frame (present players only).
    spacing_samples: list[float] = []
    for pos_frame in tracked:
        occ = pos_frame[~np.isnan(pos_frame).any(axis=1)]
        if len(occ) < 2:
            continue
        d = np.linalg.norm(occ[:, None, :] - occ[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        spacing_samples.extend(d.min(axis=-1).tolist())
    mean_spacing = float(np.mean(spacing_samples)) if spacing_samples else 0.0

    # Team centroid velocity using nan-aware per-frame centroid.
    centroids = np.array([
        np.nanmean(f, axis=0) if (~np.isnan(f).any(axis=1)).any() else [np.nan, np.nan]
        for f in tracked
    ])
    cvel = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    centroid_vel = float(np.nanmean(cvel)) if np.isfinite(cvel).any() else 0.0

    # Spatial variance (spec §1B): mean squared distance to the team centroid.
    var_samples: list[float] = []
    for f, c in zip(tracked, centroids):
        occ = f[~np.isnan(f).any(axis=1)]
        if len(occ) >= 2 and np.all(np.isfinite(c)):
            var_samples.append(float(np.mean(np.sum((occ - c) ** 2, axis=1))))
    var_spatial = float(np.mean(var_samples)) if var_samples else 0.0

    # Top-2 / bottom-4 speed differential (spec §1B), mean over the window.
    kin = kinematic_features(tracked)              # (T, 4)
    speed_top2 = float(kin[1:, 0].mean()) if len(kin) > 1 else 0.0
    speed_bot4 = float(kin[1:, 1].mean()) if len(kin) > 1 else 0.0

    # M17 (AUDIT_FINDINGS.md): per-window quality columns, so a downstream
    # consumer of the CSV can tell a clean window from a garbage one without
    # re-deriving it from the raw video. All four reuse arrays already
    # computed above - no extra passes over the sequence.
    ball_recall = float((~np.isnan(ball).any(axis=1)).mean())
    mean_present = float((~np.isnan(players).any(axis=-1)).sum(axis=1).mean())
    nan_frac = float(np.isnan(players).any(axis=-1).mean())
    # An "id bridge" is one contiguous occlusion run that WAS missing before
    # interpolate_gaps and is NOT missing after - count distinct runs, not
    # bridged frames, so one 10-frame gap counts as 1, not 10.
    was_missing = np.isnan(identity_fixed).any(axis=-1)   # (T, K)
    now_present = ~np.isnan(tracked).any(axis=-1)          # (T, K)
    bridged = was_missing & now_present
    id_bridges = 0
    for col in range(bridged.shape[1]):
        col_bool = bridged[:, col]
        if col_bool[0]:
            id_bridges += 1
        id_bridges += int((np.diff(col_bool.astype(int)) == 1).sum())

    return {
        "sync_score": round(sync, 4),
        "mean_spacing_cm": round(mean_spacing, 1),
        "centroid_vel_cm_per_frame": round(centroid_vel, 2),
        "var_spatial_cm2": round(var_spatial, 1),
        "speed_top2_cm_frame": round(speed_top2, 2),
        "speed_bot4_cm_frame": round(speed_bot4, 2),
        "speed_diff_cm_frame": round(speed_top2 - speed_bot4, 2),
        "window_ball_recall": round(ball_recall, 4),
        "window_mean_present": round(mean_present, 2),
        "window_nan_frac": round(nan_frac, 4),
        "window_id_bridges": id_bridges,
    }


# ---------------------------------------------------------------------------
# Sequence classification (temporal-model interface, spec §5B)
# ---------------------------------------------------------------------------

def _classify_sequence_rule_based(seq: np.ndarray) -> ClassificationResult:
    """
    Classify a 29-frame RAW sequence using the rule-based engine from
    label_clips (fallback when no trained checkpoint is supplied).

    "Unclassified" is a legitimate first-class outcome here (class 0, the
    noise sink) – the old behaviour of coercing it into "Spacing Breakdown"
    polluted the report with phantom structural failures.

    L21 (AUDIT_FINDINGS.md), investigated and narrowed: label_clips.py's
    rules (rule1-4) were themselves tuned against the (0,0)-means-missing
    convention (see rule2_delayed_support's own docstring) - this conversion
    is not a bug relative to ITS consumer, it is what that consumer expects.
    The real residual risk is narrower than "reintroduces the ambiguity":
    a window with a lot of genuinely missing data silently degrades this
    legacy fallback's reliability with no indication to the caller, since
    every rule was tuned assuming missing data is the rare case. That part
    is fixed below - not the sentinel choice itself, which would require
    re-tuning four hand-calibrated rule engines with no ground truth to
    validate against, for a fallback path nothing currently deployed uses
    (--checkpoint is always passed by every current script; this only runs
    on a bare `pipeline.py <video>` invocation with no model).
    """
    nan_frac = float(np.isnan(seq).mean())
    if nan_frac > 0.25:
        print(
            f"[WARNING] rule-based fallback classifying a window with "
            f"{nan_frac:.0%} missing values - label_clips.py's rules were "
            "tuned assuming missing data is rare; treat this label as "
            "low-confidence.", file=sys.stderr,
        )
    # The legacy rule engine expects the (0,0) sentinel for missing, not NaN.
    seq_legacy = np.nan_to_num(seq, nan=0.0)
    df = pd.DataFrame(seq_legacy, columns=FEATURE_COLS)
    df.insert(0, "frame_id", range(1, len(df) + 1))
    label = label_clip(df)
    return ClassificationResult(
        label=label,
        numeric_idx=LABEL_TO_INDEX.get(label, 0),
        confidence=1.0,
        entropy=0.0,
        entropy_norm=0.0,
        is_anomaly=False,
        probs={},
    )


def _unclassified_result(gate: dict[str, float]) -> ClassificationResult:
    """ClassificationResult for a window routed to the class-0 noise sink."""
    return ClassificationResult(
        label="Unclassified",
        numeric_idx=LABEL_TO_INDEX.get("Unclassified", 0),
        confidence=1.0,
        entropy=0.0,
        entropy_norm=0.0,
        is_anomaly=False,
        probs={},
    )


def _load_manual_labels(csv_path: str, video_path: str):
    """Build a ground-truth banner lookup from a manual labels.csv.

    The CSV carries (video_file, anchor_time, tactic_class) - the same format
    extract_clips.py consumes. Every row whose video_file matches *video_path*
    marks a 1.0 s window starting at anchor_time; Coordinated_Defense starts
    0.5 s earlier, matching extract_clips.py's own offset so the banner lines
    up with the clips that were actually cut.

    Returns (label_at, n_windows) where label_at(t_seconds) -> label string
    ("No manual label" for any time outside every labelled window).
    """
    canon = {
        "coordinated_attack": "Coordinated Attack",
        "coordinated_defense": "Coordinated Defense",
        "coordinated_defence": "Coordinated Defense",
        "delayed_support": "Delayed Support",
        "spacing_breakdown": "Spacing Breakdown",
    }
    df = pd.read_csv(csv_path)
    want = os.path.basename(str(video_path)).strip().lower()
    windows: list[tuple[float, float, str]] = []
    for _, row in df.iterrows():
        vf = os.path.basename(str(row["video_file"]).strip()).lower()
        if vf != want:
            continue
        label = canon.get(str(row["tactic_class"]).strip().lower().replace(" ", "_"))
        if label is None:
            continue
        parts = str(row["anchor_time"]).strip().split(":")
        if not 1 <= len(parts) <= 3:
            continue
        # Left-pad missing fields: "13.2" -> ["0","0","13.2"]; "00:13.2" ->
        # ["0","00","13.2"]. Matches extract_clips._parse_time_to_seconds so
        # a labels.csv in MM:SS.s form (the manual set) lines up here too.
        parts = ["0"] * (3 - len(parts)) + parts
        try:
            start = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except (ValueError, IndexError):
            continue
        if label == "Coordinated Defense":
            start -= 0.5
        windows.append((start, start + 1.0, label))
    windows.sort()

    def label_at(t_seconds: float) -> str:
        hit = "No manual label"
        for start, end, label in windows:
            if start <= t_seconds < end:
                hit = label  # later-starting window wins on overlap
        return hit

    return label_at, len(windows)


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    tracked_xyxy: np.ndarray,
    track_ids: np.ndarray,
    ball_center: np.ndarray | None,
    ball_is_predicted: bool,
    ball_trail: "deque[tuple[int, int, bool]]",
    current_label: str,
    current_conf: float,
    frame_idx: int,
    fps: float,
    calibrator: CourtCalibrator | None = None,
    hud_metrics: dict | None = None,
    current_entropy: float = 0.0,
    is_anomaly: bool = False,
    pose_summary: "PoseFrameSummary | None" = None,
) -> np.ndarray:
    """
    Draw tracking boxes, ball trail + marker, frame info, and current label.

    Ball trail
    ----------
    Up to BALL_TRAIL_LEN past ball positions are stored as (x, y, is_predicted)
    tuples and drawn as small fading circles:
      • Detected positions  : orange circles, fading from dim (oldest) to
                              bright (newest) – solid filled.
      • Predicted positions : grey circles, slightly smaller – indicates the
                              BallTracker extended the path between detections.

    Ball marker
    -----------
      • Detected  : white outer ring + orange filled centre + "BALL" label.
      • Predicted : grey ring (no fill) + "BALL~" label – signals estimation.

    Label banner
    ------------
    A two-line banner occupies the bottom of the frame:
      • Line 1 (large, centred) – simplified phase name: "ATTACKING" / "DEFENDING"
        / "DELAYED SUPPORT" / "SPACING BREAKDOWN".  Background colour matches
        the tactical label (orange / blue / cyan / red).
      • Line 2 (small, right-aligned) – technical label, numeric index, and
        classifier confidence.

    Before the first sequence is classified a "CLASSIFYING…" placeholder is
    shown so the viewer knows the system is warming up.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Resolve label-specific color
    box_color  = LABEL_COLORS.get(current_label, TRACK_COLOR)
    banner_bg  = LABEL_COLORS.get(current_label, LABEL_BG_COLOR)
    display_name = DISPLAY_NAMES.get(current_label, current_label) if current_label else ""

    # Court polygon overlay (dynamic homography) or net reference line.
    if calibrator is not None and calibrator.corners is not None:
        calibrator.draw(out)
    else:
        net_y_px = int(round(h * (NET_Y_CM / COURT_H_CM)))
        cv2.line(out, (0, net_y_px), (w, net_y_px), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(out, "NET", (10, max(20, net_y_px - 8)), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Ball trail (detected = orange fading, predicted = grey fading) ─────────
    trail_list = list(ball_trail)
    n_trail = len(trail_list)
    bc_b, bc_g, bc_r = BALL_COLOR   # BGR components of orange ball colour
    for i, (tx, ty, pred) in enumerate(trail_list):
        alpha = (i + 1) / max(n_trail, 1)       # 0 → oldest, 1 → newest
        radius = max(2, round(5 * alpha))
        if pred:
            # Predicted segment: grey, slightly smaller
            grey = int(160 * alpha)
            cv2.circle(out, (tx, ty), radius, (grey, grey, grey), -1, cv2.LINE_AA)
        else:
            # Detected segment: orange fading trail
            cv2.circle(out, (tx, ty), radius, (
                int(bc_b * alpha),
                int(bc_g * alpha),
                int(bc_r * alpha),
            ), -1, cv2.LINE_AA)

    # ── Player bounding boxes (colour reflects current tactical label) ─────────
    if len(tracked_xyxy) > 0:
        pose_tracks = pose_summary.per_track if pose_summary else {}
        for tid, (x1, y1, x2, y2) in zip(track_ids, tracked_xyxy):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            tag = f"P {tid}"
            info = pose_tracks.get(int(tid))
            if info is not None and info.get("airborne"):
                tag += f"  JUMP {info['vy_ms']:.1f} m/s"
            cv2.putText(
                out, tag, (x1, y1 - 6),
                FONT, 0.45, box_color, 1, cv2.LINE_AA,
            )

    # Large action-focus box highlighting the active team region.
    if current_label and len(tracked_xyxy) > 0:
        x1 = int(np.min(tracked_xyxy[:, 0]))
        y1 = int(np.min(tracked_xyxy[:, 1]))
        x2 = int(np.max(tracked_xyxy[:, 2]))
        y2 = int(np.max(tracked_xyxy[:, 3]))

        pad_x = max(24, int(0.06 * max(1, x2 - x1)))
        pad_y = max(20, int(0.08 * max(1, y2 - y1)))
        ax1 = max(0, x1 - pad_x)
        ay1 = max(0, y1 - pad_y)
        ax2 = min(w - 1, x2 + pad_x)
        ay2 = min(h - 1, y2 + pad_y)

        overlay_focus = out.copy()
        cv2.rectangle(overlay_focus, (ax1, ay1), (ax2, ay2), box_color, -1)
        cv2.addWeighted(overlay_focus, 0.12, out, 0.88, 0, out)
        cv2.rectangle(out, (ax1, ay1), (ax2, ay2), box_color, 3)

        action_text = f"ACTION FOCUS: {display_name}"
        (tw, th), _ = cv2.getTextSize(action_text, FONT, 0.6, 2)
        tx = ax1 + 10
        ty = max(24, ay1 - 10)
        cv2.rectangle(out, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
        cv2.putText(out, action_text, (tx, ty), FONT, 0.6, box_color, 2, cv2.LINE_AA)

    # ── Ball marker ─────────────────────────────────────────────────────────────
    if ball_center is not None:
        cx, cy = int(ball_center[0]), int(ball_center[1])
        if ball_is_predicted:
            # Predicted: grey ring only – no fill, label with tilde
            cv2.circle(out, (cx, cy), 11, (160, 160, 160), 2, cv2.LINE_AA)
            cv2.putText(out, "BALL~", (cx + 13, cy + 5),
                        FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        else:
            # Detected: white ring + orange fill
            cv2.circle(out, (cx, cy), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 8,  BALL_COLOR,      -1, cv2.LINE_AA)
            cv2.putText(out, "BALL", (cx + 13, cy + 5),
                        FONT, 0.5, BALL_COLOR, 1, cv2.LINE_AA)

    # ── Frame counter + timestamp ──────────────────────────────────────────────
    ts = frame_idx / fps
    cv2.putText(
        out, f"Frame {frame_idx:05d}  {ts:.2f}s",
        (10, 28), FONT, 0.65, TEXT_COLOR, 2, cv2.LINE_AA,
    )

    # ── HUD: live team metrics (spec §6.3) ─────────────────────────────────────
    hud_lines: list[str] = []
    if hud_metrics:
        cvel_cms = hud_metrics.get("centroid_vel_cm_per_frame", 0.0) * fps
        hud_lines += [
            f"sync      {hud_metrics.get('sync_score', 0.0):+.2f}",
            f"spacing   {hud_metrics.get('mean_spacing_cm', 0.0):.0f} cm",
            f"team vel  {cvel_cms:.0f} cm/s",
            f"var_spat  {hud_metrics.get('var_spatial_cm2', 0.0):.0f} cm2",
            f"spd t2/b4 {hud_metrics.get('speed_top2_cm_frame', 0.0):.1f}/"
            f"{hud_metrics.get('speed_bot4_cm_frame', 0.0):.1f}",
            f"entropy   {current_entropy:.2f} bits",
        ]
    if pose_summary is not None and pose_summary.per_track:
        hud_lines.append(f"max Vy    {pose_summary.max_vy_up:.1f} m/s")
        hud_lines.append(f"arm omega {pose_summary.max_arm_omega:.0f} rad/s")
    if is_anomaly:
        hud_lines.append("!! TACTICAL DEVIATION !!")

    if hud_lines:
        pad, lh = 8, 20
        hud_w = 240
        hud_h = pad * 2 + lh * len(hud_lines)
        x0 = w - hud_w - 10
        y0 = 10
        overlay_hud = out.copy()
        cv2.rectangle(overlay_hud, (x0, y0), (x0 + hud_w, y0 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_hud, 0.65, out, 0.35, 0, out)
        for i, line in enumerate(hud_lines):
            color = (0, 0, 255) if line.startswith("!!") else TEXT_COLOR
            cv2.putText(
                out, line, (x0 + pad, y0 + pad + lh * (i + 1) - 6),
                FONT, 0.5, color, 1, cv2.LINE_AA,
            )

    # ── Label banner at the bottom of the frame ────────────────────────────────
    big_scale, big_thick = 1.1, 2
    sm_scale,  sm_thick  = 0.55, 1
    (_, big_h), _ = cv2.getTextSize("Ag", FONT, big_scale, big_thick)
    (_, sm_h),  _ = cv2.getTextSize("Ag", FONT, sm_scale,  sm_thick)
    banner_h = big_h + sm_h + 24

    banner_y0 = h - banner_h - 4

    overlay = out.copy()
    cv2.rectangle(overlay, (0, banner_y0), (w, h), banner_bg, -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    if current_label:
        numeric_idx = LABEL_TO_INDEX.get(current_label, 0)

        (dn_w, _), _ = cv2.getTextSize(display_name, FONT, big_scale, big_thick)
        big_x = max(10, (w - dn_w) // 2)
        big_y = banner_y0 + big_h + 8
        cv2.putText(out, display_name, (big_x, big_y),
                    FONT, big_scale, TEXT_COLOR, big_thick, cv2.LINE_AA)

        sub_text = f"[{numeric_idx}] {current_label}   conf: {current_conf:.2f}"
        (sub_w, _), _ = cv2.getTextSize(sub_text, FONT, sm_scale, sm_thick)
        sub_x = max(10, w - sub_w - 10)
        sub_y = banner_y0 + big_h + sm_h + 18
        cv2.putText(out, sub_text, (sub_x, sub_y),
                    FONT, sm_scale, TEXT_COLOR, sm_thick, cv2.LINE_AA)

    else:
        warmup = "CLASSIFYING..."
        (ww, _), _ = cv2.getTextSize(warmup, FONT, big_scale, big_thick)
        cv2.putText(out, warmup, (max(10, (w - ww) // 2), banner_y0 + big_h + 8),
                    FONT, big_scale, TEXT_COLOR, big_thick, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _risky_serving_defaults(args: argparse.Namespace) -> list[str]:
    """R13: risky argparse defaults in effect for a serving run - the ones
    that are silently wrong for the deployed model when a flag is omitted."""
    out: list[str] = []
    if not args.ball_model:
        out.append("--ball-model not set (ball channel will be near-constant)")
    if not args.auto_court and args.court_corners is None:
        out.append("no --auto-court and no --court-corners (court mask off; "
                   "coaches/bench/refs not filtered)")
    if args.court_coords == "homography" and (args.auto_court
                                              or args.court_corners is not None):
        out.append("--court-coords homography (training CSVs were built with "
                   "linear; pass --court-coords linear to match)")
    if args.tracker == "bytetrack":
        out.append("--tracker bytetrack (deployed model was extracted with botsort)")
    return out


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full analytics pipeline on a single match video."""

    # ── Determinism (H8, AUDIT_FINDINGS.md) ──────────────────────────────────
    # teams.py's k-means is already seeded internally (np.random.default_rng,
    # not the global numpy RNG) and is independently deterministic - verified,
    # not assumed, by reading _kmeans() directly. What is NOT already
    # controlled is every OTHER source of randomness the run touches: the
    # global numpy/random RNGs (in case any dependency reaches for them), the
    # RANSAC calls inside CourtCalibrator's camera-motion compensation
    # (cv2.estimateAffinePartial2D, method=cv2.RANSAC), and torch (unused at
    # inference for the sklearn backend, but Mamba dropout/init would matter
    # if this pipeline ever trained rather than only served). Seeding all
    # four up front turns "no observed nondeterminism" into "determinism by
    # contract" - a panel re-running the same command against the same video
    # should get byte-identical output.
    random.seed(args.seed)
    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    torch.manual_seed(args.seed)

    # L22 (AUDIT_FINDINGS.md): a positive --max-frames below SEQ_LEN can
    # never produce one complete window - the run used to burn through
    # detector/tracker setup and the whole capped segment only to print one
    # passive INFO line and write no CSV. Fail fast instead. (0 means
    # "unlimited", not "zero frames" - not affected by this check.)
    if 0 < args.max_frames < SEQ_LEN:
        print(
            f"[ERROR] --max-frames {args.max_frames} is below SEQ_LEN "
            f"({SEQ_LEN}) - no window could ever complete, so nothing would "
            f"be classified. Use --max-frames 0 (whole video) or "
            f">= {SEQ_LEN}.", file=sys.stderr,
        )
        sys.exit(1)

    # M12 (AUDIT_FINDINGS.md): --cmc warps the stored court corners every
    # frame from an optical-flow affine; without --auto-court there is no
    # periodic re-detection to correct a bad warp, so any single bad
    # estimate (a scene cut, a low-texture frame) corrupts the calibration
    # PERMANENTLY for the rest of the run - not recoverable, and not
    # obviously wrong-looking in the report.
    if args.cmc and not args.auto_court:
        print(
            "[ERROR] --cmc requires --auto-court. Without periodic "
            "re-detection, a single bad camera-motion estimate (a scene "
            "cut, a low-texture frame) permanently corrupts the court "
            "calibration for the rest of the run, with no way to recover "
            "and no obvious sign in the output that it happened.",
            file=sys.stderr,
        )
        sys.exit(1)

    # R13 (AUDIT_FINDINGS.md): every default here is the WRONG choice for the
    # deployed model - it was trained with a ball model, auto-court on, linear
    # coords and BoT-SORT (see any extraction_provenance.json). A full script
    # passes all four; an ad-hoc `python pipeline.py video.mp4` silently uses
    # none of them. C2's check_provenance catches this hard IF the bundle
    # carries provenance, but the deployed .joblib predates that and only
    # warns. Make the risky defaults visible either way. Not a hard error:
    # a rule-only or --team-side all quick look is a legitimate use.
    for _r in _risky_serving_defaults(args):
        print(f"[WARN] risky default: {_r}", file=sys.stderr)

    # ── Dependency check ────────────────────────────────────────────────────
    if not HAS_ULTRALYTICS:
        print(
            "[ERROR] ultralytics is required.\n"
            "        Install with: pip install ultralytics>=8.0.0",
            file=sys.stderr,
        )
        sys.exit(1)
    if not HAS_SUPERVISION:
        print(
            "[ERROR] supervision is required.\n"
            "        Install with: pip install supervision>=0.18.0",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # ── Load models (behind the abstract interfaces, spec §5) ───────────────
    print(f"Loading detector model : {args.yolo_model}")
    if args.ball_model:
        print(f"Loading ball model     : {args.ball_model}")
    detector = create_detector(
        args.yolo_model,
        ball_weights=args.ball_model,
        ball_conf_threshold=args.ball_conf_threshold,
        conf_threshold=args.conf_threshold,
        person_class_id=args.person_class_id,
        ball_class_id=args.ball_class_id,
        imgsz=args.imgsz,
    )
    print(f"  Detector backend : {detector.name}")
    print(f"  Detector classes : {detector.class_names}")
    print(f"  Using person_id={detector.person_id}  ball_id={detector.ball_id}"
          + ("" if detector.ball_id is not None
             else "  (no ball class - ball overlay disabled)"))

    classifier: TorchTemporalClassifier | None = None
    if args.checkpoint:
        print(f"Loading temporal-model checkpoint : {args.checkpoint}")
        try:
            # Backend chosen from the file extension: .joblib -> scikit-learn
            # tactical model, anything else -> torch checkpoint.
            classifier = create_temporal_classifier(
                args.checkpoint,
                device=device,
                anomaly_threshold=args.anomaly_threshold,
            )
            print(f"  Temporal backend : {classifier.name}   Device: {device}")
        except FileNotFoundError:
            print(
                f"[WARNING] Checkpoint not found: {args.checkpoint}\n"
                "          Falling back to rule-based classification.",
                file=sys.stderr,
            )
    else:
        print("No checkpoint supplied – using rule-based classification.")

    # ── Ground-truth banner mode (--labels-csv) ────────────────────────────
    # The manual label active at each frame's timestamp replaces the
    # classifier's per-window verdict. Boxes / IDs / ball stay automated;
    # only the tactical banner becomes ground truth.
    gt_label_at = None
    if args.labels_csv:
        gt_label_at, n_gt = _load_manual_labels(args.labels_csv, args.video)
        print(f"Manual-label mode: {n_gt} labelled window(s) from "
              f"'{args.labels_csv}' for {os.path.basename(args.video)}; "
              "banner = ground truth, classifier disabled.")
        classifier = None

    # ── Pose estimator (17-keypoint biomechanics, spec §1A) ─────────────────
    pose: PoseEstimator | None = None
    if args.pose:
        print(f"Loading pose model : {args.pose_model}")
        pose = PoseEstimator(args.pose_model, fps=30.0)

    # ── Open video (need fps before tracker/calibrator construction) ────────
    print(f"\nOpening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = _safe_fps(cap)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = _safe_frame_count(cap)
    print(f"  {frame_w}×{frame_h}  {fps:.2f} FPS  ~{total_frames} frames")
    if pose is not None:
        pose._fps = fps

    # ── Tracker (paradigm switch + Kalman tuning, spec §2) ──────────────────
    tracker = create_tracker(
        args.tracker, fps=fps, q_scale=args.kalman_q_scale, with_reid=args.reid,
        allow_fallback=args.allow_tracker_fallback,
    )
    print(f"Tracker : {tracker.name}")

    if classifier is not None:
        run_provenance = {
            "imgsz": args.imgsz,
            "ball_model": os.path.basename(args.ball_model) if args.ball_model else None,
            "team_split": args.team_split,
            "court_coords": args.court_coords,
            "tracker": tracker.name,
        }
        classifier.check_provenance(
            run_provenance, strict=not args.allow_provenance_mismatch
        )

    # ── Court calibration (dynamic homography + CMC + mask, spec §3) ────────
    calibrator = CourtCalibrator(
        frame_w, frame_h,
        manual_corners=args.court_corners,
        auto=args.auto_court,
        cmc=args.cmc,
        refresh_every=args.court_refresh,
        force_linear=(args.court_coords == "linear"),
    )
    if args.court_corners:
        print("Court homography computed from supplied corner points"
              + (" (CMC keeps it aligned under camera motion)." if args.cmc else "."))
    elif args.auto_court:
        print(f"Automatic court detection every {args.court_refresh} frames"
              + (" + CMC between refreshes." if args.cmc else "."))
    else:
        print(
            f"No court corners supplied – pixel coords scaled to "
            f"{COURT_W_CM:.0f}×{COURT_H_CM:.0f} cm (court mask disabled)."
        )

    side_name = {
        "top": "TOP side of net",
        "bottom": "BOTTOM side of net",
        "all": "ALL players (no side filtering)",
    }[args.team_side]
    print(
        f"Team-side filter: {side_name} "
        f"(net margin: {args.net_margin_cm:.1f} cm)"
    )
    print(f"Class-0 gate: mean speed < {args.gate_speed_cm:.1f} cm/frame or "
          f"< {args.gate_min_players:.0f} players visible -> Unclassified")
    print(f"Scene-cut gate: HSV corr < {args.scene_cut_corr:.2f} or ID "
          f"carry-over < {args.scene_cut_id_churn:.2f} -> discard partial window")

    # ── Output video writer ──────────────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if args.output_video:
        writer = cv2.VideoWriter(args.output_video, FOURCC, fps, (frame_w, frame_h))
        if not writer.isOpened():
            print(
                f"[WARNING] Cannot create output video: {args.output_video}. "
                "Skipping video output.",
                file=sys.stderr,
            )
            writer = None
        else:
            print(f"  Output video: {args.output_video}")

    # ── State variables ──────────────────────────────────────────────────────
    seq_buffer: list[np.ndarray] = []   # accumulates (14,) raw rows → 29 frames
    frame_buffer: list[dict] = []       # per-frame records of current window
    slot_map = SlotManager()            # persistent slots + occlusion bridge (L1 fix)
    ball_tracker = BallTracker()        # constant-velocity ball tracker
    strict = (
        StrictRenderer(
            n_slots=N_PLAYERS,
            confirm_frames=args.strict_confirm_frames,
            ball_max_coast=args.ball_max_coast,
        )
        if args.strict_render else None
    )
    if strict is not None:
        print(f"Strict render: ON (confirm {args.strict_confirm_frames} frames, "
              f"ball coast <= {args.ball_max_coast})")
        if not args.auto_court and args.court_corners is None:
            print("[WARNING] --strict-render without --auto-court / --court-corners: "
                  "the court gate (referees, off-court players) will be inactive.",
                  file=sys.stderr)
    track_dump: "dict[str, list] | None" = (
        {"box": [], "ball": [], "court": []} if args.dump_tracks else None
    )
    # Jersey-colour team separation. Disabled for --team-side all (nothing to
    # separate) and in explicit --team-split geometric mode.
    team_clf: TeamClassifier | None = None
    team_voter = TeamVoter()
    if args.team_split == "colour" and args.team_side != "all":
        team_clf = TeamClassifier(
            n_clusters=args.team_clusters,
            warmup_frames=args.team_warmup_frames,
            net_y_cm=calibrator.net_y_cm_current(),  # H6: overwritten live each frame below anyway
        )
        print(f"Team split: jersey colour ({args.team_clusters} clusters, "
              f"{args.team_warmup_frames}-frame warm-up, per-track vote)")
    else:
        print("Team split: geometric (foot point vs net line)")
    ball_trail: deque[tuple[int, int, bool]] = deque(maxlen=BALL_TRAIL_LEN)
    label_counts: dict[str, int] = defaultdict(int)
    predictions: list[dict] = []
    frame_rows: list[dict] = []         # frame-level CSV (spec §6.2)
    anomaly_count = 0
    masked_out_total = 0
    ball_head_rejects = 0
    frames_before_lock = 0
    prev_hsv_hist: np.ndarray | None = None   # H3 scene-cut detector state
    prev_track_ids: set = set()
    scene_cut_count = 0
    frames_discarded_to_cuts = 0

    frame_idx = 0
    seq_idx = 0
    current_label = ""
    current_conf = 0.0
    current_entropy = 0.0
    current_anomaly = False
    current_metrics: dict = {}

    # ── PREFLIGHT: prove the population is sane before analysing anything ───
    if not args.skip_preflight:
        stats = _preflight_sample(
            cap, detector, calibrator, args,
            n_samples=args.preflight_frames,
        )
        verdict = preflight.assess(stats)
        print("\nPreflight check")
        print(preflight.summarise(stats))
        print(verdict.render())
        if verdict.fatal:
            print(
                "\n[ABORT] Preflight found a fatal problem. A tactical report "
                "built on this input would be meaningless, so the run is "
                "stopping instead of producing one.\n"
                "        Override with --skip-preflight if you are debugging.",
                file=sys.stderr,
            )
            cap.release()
            sys.exit(2)
        # Rewind: the sampler consumed frames from the capture.
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(args.start_frame, 0))
        # M16 (AUDIT_FINDINGS.md): _SIDE_FILTER_WARNED is a module-level
        # once-only flag. Preflight (above) calls the SAME filter function
        # on its own sampled frames and can trip it first, which would
        # silently mute the real run's own warning for a genuinely new
        # occurrence. Reset it here so preflight tripping it never spends
        # the real run's one-time budget.
        global _SIDE_FILTER_WARNED
        _SIDE_FILTER_WARNED = False
        calibrator = CourtCalibrator(
            frame_w, frame_h,
            manual_corners=args.court_corners,
            auto=args.auto_court,
            cmc=args.cmc,
            refresh_every=args.court_refresh,
            force_linear=(args.court_coords == "linear"),
        )

    print("\nProcessing frames…")

    # ── Main frame loop ──────────────────────────────────────────────────────
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        frame_idx = args.start_frame
        print(f"Seeking to start frame {args.start_frame}.")

    none_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames and (frame_idx - args.start_frame) > args.max_frames:
            print(f"Reached --max-frames limit ({args.max_frames}); stopping.")
            break
        if frame is None:
            # L19 (AUDIT_FINDINGS.md): some backends can return (ret=True,
            # frame=None) for one corrupted mid-stream frame without ending
            # the stream. Passing None into calibrator.update/detector.predict
            # crashes with an unhelpful low-level error - possibly after a
            # long render. Skip the one frame and keep going instead.
            none_frame_count += 1
            continue

        # ── STEP 1 – Frame Extraction + live calibration update ─────────────
        calibrator.update(frame)

        # ── STEP 2 – Player + ball detection (abstract detector, §5A) ───────
        det = detector.predict(frame)

        # Head-as-ball guard: the fine-tuned ball model occasionally fires on a
        # player's head (round, similar apparent size). If the ball centre sits
        # in the top ~22% of a detected person box and near its horizontal
        # centre, treat it as a head, not the ball, and drop it this frame -
        # the BallTracker then coasts on the last real trajectory.
        if det.ball_center is not None and len(det.person_xyxy):
            bx, by = float(det.ball_center[0]), float(det.ball_center[1])
            for x1, y1, x2, y2 in det.person_xyxy:
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                if (x1 + 0.15 * bw <= bx <= x2 - 0.15 * bw
                        and y1 - 0.05 * bh <= by <= y1 + 0.22 * bh):
                    det.ball_center = None
                    ball_head_rejects += 1
                    break

        # Court mask: drop referees / bench / crowd outside the lines (§3C)
        n_before = len(det.person_xyxy)
        person_xyxy, person_conf = calibrator.filter_detections(
            det.person_xyxy, det.person_conf
        )
        masked_out_total += n_before - len(person_xyxy)

        if team_clf is None:
            # Geometric team split (legacy): decided per frame, BEFORE tracking.
            person_xyxy, person_conf = _filter_players_by_team_side(
                person_xyxy, person_conf, calibrator,
                args.team_side, args.net_margin_cm,
            )
            det.person_xyxy, det.person_conf = person_xyxy, person_conf

            # ── STEP 3 – Player Tracking (ByteTrack / BoT-SORT, §2) ─────────
            tracked_xyxy, track_ids = tracker.update(det, frame)
            raw_track_ids = track_ids
        else:
            # Colour team split: track EVERY in-court person first, so the
            # per-frame colour label can be majority-voted over each track's
            # lifetime (teams.TeamVoter).  Tracking before filtering is also
            # better for the tracker: it sees a stable population instead of
            # detections blinking in and out as players cross the net line.
            det.person_xyxy, det.person_conf = person_xyxy, person_conf
            tracked_xyxy, track_ids = tracker.update(det, frame)
            raw_track_ids = track_ids   # H3: full tracker output, pre-team-filter

            court_ys = [
                calibrator.pixel_to_court(float((x1 + x2) / 2.0), float(y2))[1]
                for (x1, y1, x2, y2) in tracked_xyxy
            ]
            # H6 (AUDIT_FINDINGS.md): team_clf.net_y_cm was frozen at
            # construction time, before the calibrator had processed a
            # single frame (so before --auto-court could have locked a
            # quad). Keep it live instead - cheap (one float read), and
            # _fit() only actually consumes it once, ~team_warmup_frames
            # frames from now, by which point the court has almost always
            # locked.
            team_clf.net_y_cm = calibrator.net_y_cm_current()
            frame_labels = team_clf.update(frame, tracked_xyxy, court_ys)
            if frame_labels is None or team_clf.degenerate:
                # Either still warming up, or the fit put every cluster on one
                # side of the net (no court mask -> crowd dominates the
                # clusters). Both cases: geometry decides.
                frame_labels = _team_labels_geometric(
                    court_ys, args.net_margin_cm, calibrator.net_y_cm_current()
                )
            voted = team_voter.update(track_ids, frame_labels)
            team_voter.forget(track_ids)

            want = TEAM_NEAR if args.team_side == "bottom" else TEAM_FAR
            if args.team_side == "all":
                keep = np.ones(len(tracked_xyxy), dtype=bool)
            else:
                keep = voted == want
            tracked_xyxy = tracked_xyxy[keep]
            track_ids = track_ids[keep]

        # ── Ball tracking: bridge missed detections with predicted position ──
        tracked_ball, ball_is_predicted = ball_tracker.update(det.ball_center)

        # ── STEP 4 – Pose biomechanics (17 keypoints, §1A) ──────────────────
        pose_summary: PoseFrameSummary | None = None
        if pose is not None:
            pose_summary = pose.process(frame, tracked_xyxy, track_ids)

        # ── H3 – Scene-cut gate ────────────────────────────────────────────
        # A hard cut inside a window splices two shots into one "rally".
        # Detect it (HSV histogram break OR near-total track-ID turnover),
        # discard the partial window so no 29-frame sequence spans the cut,
        # and let the current frame start a fresh window. The activity gate
        # cannot do this: a cut raises motion, so the window sails through.
        cur_hsv_hist = _frame_hsv_hist(frame)
        ids_now = {int(i) for i in raw_track_ids}
        is_cut, hist_corr, id_carry = _is_scene_cut(
            prev_hsv_hist, cur_hsv_hist, prev_track_ids, ids_now,
            args.scene_cut_corr, args.scene_cut_id_churn,
        )
        prev_hsv_hist = cur_hsv_hist
        prev_track_ids = ids_now
        if is_cut and (seq_buffer or frame_buffer):
            scene_cut_count += 1
            frames_discarded_to_cuts += len(seq_buffer)
            print(f"  [scene cut] frame {frame_idx}: HSV corr {hist_corr:.2f}, "
                  f"ID carry-over {id_carry:.2f} - discarded "
                  f"{len(seq_buffer)}-frame partial window")
            seq_buffer.clear()
            frame_buffer = []

        # ── STEP 5 – Position extraction + feature engineering ──────────────
        player_positions, ball_pos = _extract_positions(
            tracked_xyxy, track_ids, tracked_ball, calibrator, slot_map
        )
        feature_row = _compute_frame_features(player_positions, ball_pos)
        court_locked = not (calibrator.auto and calibrator.corners is None)
        if not court_locked:
            # M15 (AUDIT_FINDINGS.md): before the first successful auto-quad,
            # CourtCalibrator.filter_detections is a documented no-op ("no
            # calibration polygon -> everything passes") - crowd/bench/
            # officials are indistinguishable from the analysed team, and
            # none of it is counted in masked_out_total. Don't let that
            # window into a feature window at all; still let detection/
            # tracking run above so identity has continuity once locked.
            frames_before_lock += 1
        else:
            seq_buffer.append(feature_row)

        # frame_buffer must stay in lockstep with seq_buffer: a pre-lock frame
        # that is withheld from seq_buffer must also be withheld here, or the
        # per-window frame CSV block below indexes `smooth[t]` (29 rows) with
        # t > 29 and raises IndexError (only reachable with --auto-court +
        # --frame-csv, when the court takes >0 frames to lock).
        if args.frame_csv and court_locked:
            frow: dict = {
                "frame": frame_idx,
                "time_s": round(frame_idx / fps, 3),
                "ball_px_x": round(float(tracked_ball[0]), 1) if tracked_ball is not None else np.nan,
                "ball_px_y": round(float(tracked_ball[1]), 1) if tracked_ball is not None else np.nan,
                "ball_predicted": bool(ball_is_predicted),
                "n_players_tracked": int(len(tracked_xyxy)),
            }
            for c, v in zip(FEATURE_COLS, feature_row):
                frow[f"raw_{c}"] = round(float(v), 1) if np.isfinite(v) else np.nan
            if pose_summary is not None:
                frow["pose_max_vy_up_ms"] = pose_summary.max_vy_up
                frow["pose_max_arm_omega"] = pose_summary.max_arm_omega
                frow["pose_n_airborne"] = pose_summary.n_airborne
            frame_buffer.append(frow)

        # ── STEP 6 – Sequence Formation → STEP 7 gate → STEP 8 classification
        if len(seq_buffer) == SEQ_LEN:
            seq_array = np.stack(seq_buffer)  # (29, 14) raw

            # Derived metrics (for output CSV + HUD)
            metrics = _sequence_metrics(seq_array)

            # Class-0 noise sink first (spec §4B): dead-ball windows never
            # reach the model, so they cannot pollute the tactical report.
            is_dead, gate_metrics = _activity_gate(
                seq_array, args.gate_speed_cm, args.gate_min_players
            )
            if is_dead and not args.no_gate:
                result = _unclassified_result(gate_metrics)
            elif classifier is not None:
                # The model consumes the corrected, identity-aware,
                # permutation-invariant representation – built by the same
                # function the trainer uses (train/serve parity).
                model_seq = build_model_sequence(seq_array, target_len=SEQ_LEN)
                result = classifier.classify(model_seq)
            else:
                result = _classify_sequence_rule_based(seq_array)

            if gt_label_at is not None:
                # Ground-truth banner mode: the manual label at this window's
                # midpoint overrides the verdict for the report / CSV too.
                gt = gt_label_at((frame_idx - SEQ_LEN // 2) / fps)
                result = ClassificationResult(
                    label=gt, numeric_idx=LABEL_TO_INDEX.get(gt, 0),
                    confidence=1.0, entropy=0.0, entropy_norm=0.0,
                    is_anomaly=False, probs={},
                )

            current_label = result.label
            current_conf = result.confidence
            current_entropy = result.entropy
            current_anomaly = result.is_anomaly
            current_metrics = metrics
            label_counts[result.label] += 1
            anomaly_count += int(result.is_anomaly)
            seq_idx += 1

            record = {
                "sequence": seq_idx,
                "frame_start": frame_idx - SEQ_LEN + 1,
                "frame_end": frame_idx,
                "label": result.label,
                "label_index": result.numeric_idx,
                "confidence": round(result.confidence, 4),
                "entropy_bits": result.entropy,
                "entropy_norm": result.entropy_norm,
                "anomaly": result.is_anomaly,
                "tracker": tracker.name,  # effective backend actually used this run
                **metrics,
                **gate_metrics,
                **{f"p_{n}": p for n, p in result.probs.items()},
            }
            predictions.append(record)

            # Frame-level CSV: attach smoothed real-world coordinates (the
            # identity-consistent, gap-interpolated tracks) + window verdict.
            if args.frame_csv and frame_buffer:
                players_raw, ball_raw = raw_frame_to_arrays(seq_array)
                smooth = interpolate_gaps(recover_identity(players_raw))
                ball_smooth = interpolate_gaps(ball_raw[:, None, :])[:, 0, :]
                model_seq_csv = build_model_sequence(seq_array, target_len=SEQ_LEN)
                for t, frow in enumerate(frame_buffer):
                    for k in range(N_PLAYERS):
                        sx, sy = smooth[t, k]
                        frow[f"smooth_p{k+1}_x"] = round(float(sx), 1) if np.isfinite(sx) else np.nan
                        frow[f"smooth_p{k+1}_y"] = round(float(sy), 1) if np.isfinite(sy) else np.nan
                    bx, by = ball_smooth[t]
                    frow["smooth_ball_x"] = round(float(bx), 1) if np.isfinite(bx) else np.nan
                    frow["smooth_ball_y"] = round(float(by), 1) if np.isfinite(by) else np.nan
                    for c, v in zip(MODEL_FEATURE_COLS, model_seq_csv[t]):
                        frow[f"feat_{c}"] = round(float(v), 3)
                    frow["sequence"] = seq_idx
                    frow["pred_label"] = result.label
                    frow["pred_label_index"] = result.numeric_idx
                    frow["pred_entropy_bits"] = result.entropy
                    frow["pred_anomaly"] = result.is_anomaly
                frame_rows.extend(frame_buffer)
                frame_buffer = []

            display = DISPLAY_NAMES.get(result.label, result.label)
            flag = "  [ANOMALY]" if result.is_anomaly else ""
            print(
                f"  Seq {seq_idx:4d}  "
                f"frames {frame_idx - SEQ_LEN + 1:5d}–{frame_idx:5d}  "
                f"→  [{result.numeric_idx}] {display:<22s}  "
                f"conf={result.confidence:.3f}  H={result.entropy:.2f}b  "
                f"sync={metrics['sync_score']:+.3f}  "
                f"spacing={metrics['mean_spacing_cm']:.0f}cm{flag}"
            )

            seq_buffer.clear()

        # Ground-truth banner mode: refresh the banner every frame (not only
        # every SEQ_LEN), so a frame between labelled windows correctly reads
        # "NO MANUAL LABEL" instead of holding the previous window's verdict.
        if gt_label_at is not None:
            current_label = gt_label_at(frame_idx / fps)
            current_conf = 1.0

        # ── Annotate and write output frame ─────────────────────────────────
        # Strict render is DISPLAY ONLY - it never feeds back into seq_buffer /
        # the classifier, which already consumed tracked_xyxy above.
        if strict is not None:
            draw_xyxy, draw_ids, draw_ball = strict.update(
                tracked_xyxy, track_ids, slot_map.id_to_slot, calibrator,
                tracked_ball, ball_is_predicted,
            )
            draw_ball_predicted = bool(ball_is_predicted) and draw_ball is not None
        else:
            draw_xyxy, draw_ids, draw_ball = tracked_xyxy, track_ids, tracked_ball
            draw_ball_predicted = ball_is_predicted

        if draw_ball is not None:
            ball_trail.append((int(draw_ball[0]), int(draw_ball[1]), draw_ball_predicted))

        # Optional per-frame render-state dump: lets rebanner.py re-render this
        # exact box/id/ball layout with a DIFFERENT model's banner in seconds,
        # instead of re-running detection + tracking per model.
        if track_dump is not None:
            for (x1, y1, x2, y2), jid in zip(np.asarray(draw_xyxy).reshape(-1, 4),
                                             np.asarray(draw_ids).reshape(-1)):
                track_dump["box"].append((frame_idx, x1, y1, x2, y2, int(jid)))
            if draw_ball is not None:
                track_dump["ball"].append(
                    (frame_idx, float(draw_ball[0]), float(draw_ball[1]),
                     int(bool(draw_ball_predicted))))
            if calibrator.corners is not None:
                track_dump["court"].append(
                    (frame_idx, *np.asarray(calibrator.corners,
                                            dtype=float).reshape(-1)))

        if writer is not None:
            annotated = _annotate_frame(
                frame, draw_xyxy, draw_ids, draw_ball, draw_ball_predicted,
                ball_trail, current_label, current_conf, frame_idx, fps,
                calibrator=calibrator,
                hud_metrics=current_metrics,
                current_entropy=current_entropy,
                is_anomaly=current_anomaly,
                pose_summary=pose_summary,
            )
            writer.write(annotated)

        if frame_idx % 300 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  … {frame_idx}/{total_frames} frames ({pct:.0f}%)")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nAnnotated video saved to '{args.output_video}'.")
        if strict is not None:
            print(strict.summary())

    if track_dump is not None:
        np.savez_compressed(
            args.dump_tracks,
            box=np.asarray(track_dump["box"], dtype=np.float32).reshape(-1, 6),
            ball=np.asarray(track_dump["ball"], dtype=np.float32).reshape(-1, 4),
            court=np.asarray(track_dump["court"], dtype=np.float32).reshape(-1, 9),
            fps=np.asarray([fps], dtype=np.float32),
            frame_wh=np.asarray([frame_w, frame_h], dtype=np.int32),
        )
        print(f"Render-state dump saved to '{args.dump_tracks}' "
              f"({len(track_dump['box'])} box rows) - re-banner with "
              f"fyp/rebanner.py.")

    # ── Save predictions CSVs ────────────────────────────────────────────────
    if predictions and args.output_csv:
        pd.DataFrame(predictions).to_csv(args.output_csv, index=False)
        print(f"Per-sequence predictions CSV saved to '{args.output_csv}'.")
    if frame_rows and args.frame_csv:
        pd.DataFrame(frame_rows).to_csv(args.frame_csv, index=False)
        print(f"Frame-level CSV saved to '{args.frame_csv}'.")
    if pose is not None and pose.events and args.pose_events_csv:
        pd.DataFrame(pose.events).to_csv(args.pose_events_csv, index=False)
        print(f"Pose events CSV saved to '{args.pose_events_csv}'.")

    # ── STEP 9 – Final Tactical Report ──────────────────────────────────────
    print(f"\nTracker used (effective, post-fallback) : {tracker.name}")
    print(f"Total frames processed : {frame_idx}")
    if none_frame_count:
        print(f"Corrupted mid-stream frames skipped (ret=True, frame=None) : "
              f"{none_frame_count}")
    print(f"Total sequences classified : {seq_idx}")
    print(f"Sequences flagged anomalous (conf < {args.anomaly_threshold}) : {anomaly_count}")
    if calibrator.corners is not None:
        print(f"Detections masked outside court : {masked_out_total}")
    if ball_head_rejects:
        print(f"Ball detections rejected as a head : {ball_head_rejects}")
    if frames_before_lock:
        print(f"Frames before first court lock (excluded from windows) : "
              f"{frames_before_lock}")
    if scene_cut_count:
        print(f"Scene cuts detected (partial windows discarded) : "
              f"{scene_cut_count}  ({frames_discarded_to_cuts} frames)")
    if calibrator.auto:
        print(f"Auto court detection : {calibrator.n_auto_success} ok / "
              f"{calibrator.n_auto_fail} kept-previous")
    if pose is not None:
        n_jumps = sum(1 for e in pose.events if e["event"] == "jump_takeoff")
        n_impacts = sum(1 for e in pose.events if e["event"] == "arm_swing_impact")
        print(f"Pose events : {n_jumps} jump take-offs, {n_impacts} arm-swing impacts")

    if gt_label_at is not None:
        print("\nManual-label windows rendered (per 29-frame sequence):")
        for lab, n in sorted(label_counts.items()):
            print(f"  {lab:<20s} {n}")
    elif label_counts:
        print(generate_report(dict(label_counts)))
    else:
        print("\n[INFO] No complete sequences were classified "
              f"(need at least {SEQ_LEN} frames).")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class _CornersAction(argparse.Action):
    """Parse four pixel corner coordinates passed as 'x0,y0 x1,y1 x2,y2 x3,y3'."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            pairs = [tuple(float(v) for v in s.split(",")) for s in values]
            if len(pairs) != 4 or any(len(p) != 2 for p in pairs):
                raise ValueError
        except ValueError:
            parser.error(
                "--court-corners expects exactly four x,y pairs "
                "(e.g. 42,18 1238,18 1238,702 42,702)"
            )
        setattr(namespace, self.dest, pairs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end volleyball analytics pipeline: "
            "dual YOLOv11 detection → BoT-SORT tracking → court projection "
            "→ tactical classification (scikit-learn or Mamba) → report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Path to the input match video (.mp4).")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help=(
            "Path to a trained Mamba checkpoint (.pt) produced by train_mamba.py. "
            "If omitted or the file is not found, rule-based classification is used."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11n.pt",
        help="Detector weights. Default yolo11n.pt (auto-downloads). Pass your "
             "fine-tuned best.pt to use the custom volleyball detector.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the pre-run population check. The check exists because every "
             "serious failure in this project was SILENT - the pipeline printed "
             "a complete report built on zero players. Skip only when debugging.",
    )
    parser.add_argument(
        "--preflight-frames",
        type=int,
        default=24,
        help="Frames sampled across the segment for the preflight check (default 24).",
    )
    parser.add_argument(
        "--team-split",
        choices=("colour", "geometric"),
        default="colour",
        help="How to pick which detections belong to the analysed team. "
             "'colour' (default) clusters jersey colour and votes per track - "
             "correct at the net, where both front rows stand within a metre of "
             "each other. 'geometric' is the legacy foot-point-vs-net rule, "
             "which passed 10 mixed-team players on the measured frame.",
    )
    parser.add_argument(
        "--team-clusters",
        type=int,
        default=4,
        help="Jersey colour clusters (default 4). More than 2 because FIVB "
             "rules make the libero wear a contrasting jersey, so each team is "
             "two colour populations.",
    )
    parser.add_argument(
        "--team-warmup-frames",
        type=int,
        default=45,
        help="Frames of geometric labelling used to teach the colour model "
             "which cluster is which team (default 45 = 1.5 s at 30 fps).",
    )
    parser.add_argument(
        "--ball-model",
        default=None,
        help="Separate weights used ONLY for the ball (e.g. fyp/volleyball_best.pt). "
             "Recommended: --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt. "
             "Stock COCO weights are domain-robust for players but weak on the ball; "
             "the volleyball fine-tune is the reverse. See §13 of the technical review.",
    )
    parser.add_argument(
        "--ball-conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold for the ball model (default: min(conf-threshold, 0.15)).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Detector inference resolution (default {DEFAULT_IMGSZ}). The ball is "
             "~15 px wide on a 720p frame, so the ultralytics 640 default destroys it: "
             "measured ball recall 15%% at 640 vs 77%% at 1280 on identical weights.",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=None,
        help="Override the person/player class id (default: auto-detect by name).",
    )
    parser.add_argument(
        "--ball-class-id",
        type=int,
        default=None,
        help="Override the ball class id (default: auto-detect by name; None if absent).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort"),
        default="bytetrack",
        help="Tracking paradigm: 'bytetrack' = IoU-only (fast); 'botsort' = "
             "motion + camera-motion compensation + optional appearance Re-ID "
             "(robust to dense player crossings).",
    )
    parser.add_argument(
        "--allow-tracker-fallback",
        action="store_true",
        help="If --tracker botsort cannot be constructed, silently fall back "
             "to ByteTrack instead of aborting. Off by default: a model "
             "trained with BoT-SORT flags served under a silent ByteTrack "
             "substitution is a train/serve mismatch (see tracking.py "
             "create_tracker). The effective tracker is always printed and "
             "written into the output CSV either way.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for numpy/random/OpenCV/torch RNGs at pipeline start. "
             "Same video + same flags + same seed -> byte-identical output "
             "(H8, AUDIT_FINDINGS.md).",
    )
    parser.add_argument(
        "--allow-provenance-mismatch",
        action="store_true",
        help="If the loaded classifier's extraction-time flags (imgsz / "
             "ball model / team split / court coords / tracker) do not "
             "match this run's flags, warn instead of aborting. Off by "
             "default: a model trained on data extracted one way and "
             "served under different flags produces predictions that do "
             "not mean what they claim to (C2, AUDIT_FINDINGS.md).",
    )
    parser.add_argument(
        "--kalman-q-scale",
        type=float,
        default=4.0,
        help="Multiplier on the tracker's Kalman process-noise Q (measurement "
             "noise R unchanged). >1 weights fresh detections over the "
             "constant-velocity prediction — needed for explosive volleyball "
             "kinetics. 1.0 = library default.",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable appearance Re-ID in BoT-SORT (identity survives full "
             "occlusion overlap; costs extra compute).",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Enable 17-keypoint pose biomechanics: hip vertical velocity "
             "(jump detection) and shoulder→wrist angular velocity (impact "
             "detection). Adds a pose-model inference per frame.",
    )
    parser.add_argument(
        "--pose-model",
        default="yolo11n-pose.pt",
        help="Ultralytics pose weights (17-keypoint COCO head).",
    )
    parser.add_argument(
        "--pose-events-csv",
        default="",
        help="Path to save detected jump/impact events (requires --pose).",
    )
    parser.add_argument(
        "--auto-court",
        action="store_true",
        help="Automatically detect the court quadrilateral every "
             "--court-refresh frames (colour+contour segmentation with sanity "
             "checks; a failed detection keeps the previous calibration).",
    )
    parser.add_argument(
        "--court-coords",
        choices=("homography", "linear"),
        default="homography",
        help="Coordinate mapping when a court quad exists. 'linear' keeps the "
             "TRAINING feature geometry (CSVs were extracted with linear "
             "coords) while the quad still masks out off-court people - use "
             "this when serving a model trained on linear-coordinate CSVs.",
    )
    parser.add_argument(
        "--cmc",
        action="store_true",
        help="Camera-motion compensation: sparse optical flow + RANSAC affine "
             "warps the court corners each frame so the homography follows "
             "camera pans/zooms.",
    )
    parser.add_argument(
        "--court-refresh",
        type=int,
        default=5,
        help="Auto court re-detection interval in frames (with --auto-court).",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.5,
        help="Sequences with max softmax probability below this are flagged "
             "'Anomaly / Tactical Deviation' (high-entropy, off-template play).",
    )
    parser.add_argument(
        "--gate-speed-cm",
        type=float,
        default=2.0,
        help="Class-0 gate: mean player speed (cm/frame) below which a window "
             "is routed to 'Unclassified / Transition'.",
    )
    parser.add_argument(
        "--gate-min-players",
        type=float,
        default=3.0,
        help="Class-0 gate: minimum mean number of visible players for a "
             "window to be classified tactically.",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable the class-0 activity gate (every window goes to the "
             "classifier).",
    )
    parser.add_argument(
        "--scene-cut-corr",
        type=float,
        default=0.5,
        help="H3 scene-cut gate: an HSV-histogram correlation below this "
             "between consecutive frames is treated as a hard cut (replay, "
             "angle change, full-screen graphic). The partial window buffer is "
             "discarded so no 29-frame window spans two shots. 1.0 disables.",
    )
    parser.add_argument(
        "--scene-cut-id-churn",
        type=float,
        default=0.2,
        help="H3 scene-cut gate: if the fraction of track IDs carried over "
             "from the previous frame drops below this (near-total identity "
             "turnover), treat it as a cut even if the histogram gate missed "
             "it. 0.0 disables this half.",
    )
    parser.add_argument(
        "--frame-csv",
        default="",
        help="Path for the frame-level CSV: raw + smoothed real-world "
             "coordinates, all engineered team features, predicted label and "
             "entropy per frame (skipped if empty).",
    )
    parser.add_argument(
        "--court-corners",
        nargs=4,
        metavar=("TL", "TR", "BR", "BL"),
        action=_CornersAction,
        default=None,
        help=(
            "Four pixel coordinates (x,y) of the court boundary corners "
            "in TL→TR→BR→BL order, for perspective homography. "
            "Example: --court-corners 42,18 1238,18 1238,702 42,702"
        ),
    )
    parser.add_argument(
        "--team-side",
        choices=("bottom", "top", "all"),
        default="bottom",
        help=(
            "Which side of the net to treat as our team for tracking and "
            "classification features. Use 'bottom' for players before the net "
            "(near-camera side in common broadcast view)."
        ),
    )
    parser.add_argument(
        "--net-margin-cm",
        type=float,
        default=30.0,
        help=(
            "Safety margin around the net line when filtering by side. "
            "Higher values reduce cross-side leakage near the net."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        default="",
        help="Ground-truth banner mode. Path to a manual labels.csv "
             "(video_file, anchor_time, tactic_class). The tactical banner is "
             "then driven by the manual label active at each frame's timestamp "
             "instead of the classifier (which is disabled). Detection, "
             "tracking, IDs and ball stay automated. Any checkpoint is ignored.",
    )
    parser.add_argument(
        "--output-video",
        default="",
        help="Path for the annotated output video (skipped if empty).",
    )
    parser.add_argument(
        "--strict-render",
        action="store_true",
        help="Display-only hardening of --output-video (strict_render.py): show "
             "stable jersey numbers 1-N from the identity SlotManager instead "
             "of raw tracker ids; drop any box whose feet leave the "
             "auto-detected court polygon (referees / bench / a player who "
             "steps off court get no box); only draw a slot after it has been "
             "held --strict-confirm-frames frames; hide the ball after "
             "--ball-max-coast undetected frames instead of freezing it. Does "
             "NOT change the features fed to the classifier, so the predictions "
             "CSV / banner are unaffected. Needs --auto-court for the court gate.",
    )
    parser.add_argument(
        "--strict-confirm-frames",
        type=int,
        default=5,
        help="--strict-render: consecutive frames a slot must be occupied "
             "before its box is drawn.",
    )
    parser.add_argument(
        "--dump-tracks",
        default="",
        help="Write a per-frame render-state .npz (drawn boxes + jersey "
             "numbers + ball + court quad). fyp/rebanner.py replays it with a "
             "different model's banner in seconds, so comparing N classifiers "
             "on one video costs ONE detection pass, not N.",
    )
    parser.add_argument(
        "--ball-max-coast",
        type=int,
        default=8,
        help="--strict-render: hide the ball marker after this many consecutive "
             "predicted (undetected) frames instead of drawing a frozen marker.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Path to save per-sequence predictions as a CSV (skipped if empty).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip to this frame index before processing (for rendering a segment).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process at most this many frames (0 = whole video). Use a few "
             "hundred to render a short annotated sample quickly.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(_parse_args())
