"""
court.py – Coordinate calibration, dynamic homography, camera-motion
compensation and the court mask (spec §3).

Why a static pixel→cm mapping is wrong for broadcast video
----------------------------------------------------------
Broadcast cameras pan, tilt and zoom.  With a fixed homography, pure camera
motion is indistinguishable from player motion: a 2° pan adds a phantom
~100 cm/s to every track.  This module keeps the calibration LIVE:

  1. FRAME OF REFERENCE.  All positions are expressed on a flat, normalised
     1800 cm × 900 cm court plane (both half-courts; net at y = 450 cm).
     A 3×3 homography H maps pixels (u, v) to that plane:

         [x', y', w']ᵀ = H · [u, v, 1]ᵀ ;   X = x'/w' ,  Y = y'/w'

     H is exact for points ON the floor plane – which is why player position
     is taken at the FEET (features.foot_point), never the box centroid.

  2. HOW THE CORNER VALUES ARE OBTAINED.
     (a) Manual: the operator supplies the 4 pixel corners of the court
         (TL TR BR BL) once, e.g. read off a paused frame.  Best accuracy.
     (b) Automatic (--auto-court): every `refresh_every` frames the court is
         segmented by colour + contour analysis and reduced to a convex
         quadrilateral (approxPolyDP over the dominant floor region, an
         efficient equivalent of intersecting probabilistic-Hough boundary
         lines).  Detections must pass sanity checks (convexity, area ratio,
         landscape aspect) before they replace the current corners – a failed
         detection NEVER destroys a working calibration.

  3. CAMERA MOTION COMPENSATION (--cmc).  Between refreshes, sparse optical
     flow (Shi–Tomasi corners + pyramidal Lucas-Kanade) is tracked from frame
     to frame and a partial-affine camera model is fitted with RANSAC (player
     motion is rejected as outliers because the majority of features sit on
     the static background).  The affine warp is applied to the stored court
     corners, and H is recomputed – so the pixel→cm mapping follows the pan.

  4. BACKGROUND FALSE-POSITIVE ELIMINATION.  A binary polygon mask built from
     the corners drops every detection whose FOOT POINT lies outside the
     court (+margin): referees, bench, coaches and crowd never reach the
     tracker, protecting the 6-player matrix.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

COURT_W_CM = 1800.0
COURT_H_CM = 900.0
NET_Y_CM = COURT_H_CM / 2.0

# Destination quad on the normalised court plane (TL TR BR BL).
_DST = np.array(
    [[0, 0], [COURT_W_CM, 0], [COURT_W_CM, COURT_H_CM], [0, COURT_H_CM]],
    dtype=np.float32,
)


def order_corners(pts: np.ndarray) -> np.ndarray | None:
    """Order 4 arbitrary points TL, TR, BR, BL.

    M11 (AUDIT_FINDINGS.md): the previous sum/diff heuristic assigned each
    of the 4 output slots independently by min/max of (x+y) and (y-x) - for
    a skewed or rotated quad, nothing guarantees those 4 independent picks
    trace a simple (non-self-intersecting) polygon; a "bowtie" ordering
    produces a wrong mask silently.

    Fix: sort the points by angle around their centroid first - this is a
    standard technique that is GUARANTEED to produce a simple polygon
    (consecutive-by-angle around an interior point cannot self-intersect,
    by construction). That fixes the CYCLIC order but not which point is
    "first"; rotate the cycle so the point with the smallest (x+y) - the
    same TL heuristic as before - is first, preserving the TL,TR,BR,BL
    semantic every caller (build_homography, the half-court dst, the mask)
    already depends on. A shoelace-area-vs-hull-area check is kept as a
    belt-and-braces assertion, not the primary defence.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    cyclic = pts[np.argsort(angles)]                       # simple polygon, order TBD
    start = int(np.argmin(cyclic.sum(axis=1)))              # smallest (x+y) = TL
    ordered = np.roll(cyclic, -start, axis=0)

    hull_area = cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))
    ordered_area = cv2.contourArea(ordered.astype(np.float32))
    if hull_area <= 0 or abs(ordered_area - hull_area) > 0.02 * hull_area:
        return None  # would have been a bowtie or degenerate quad; caller keeps previous
    return ordered


_HALF_DST = np.array(
    [[0, NET_Y_CM], [COURT_W_CM, NET_Y_CM],
     [COURT_W_CM, COURT_H_CM], [0, COURT_H_CM]], dtype=np.float32)


def build_homography(corners: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray | None:
    """3×3 homography mapping the 4 pixel corners (TL TR BR BL) to the court.

    dst=None maps to the FULL court plane (manual corner calibration).
    dst=_HALF_DST maps to the NEAR half (net at the quad's top edge) - used
    for auto-detected quads, because colour segmentation of broadcast footage
    finds the NEAR half-court (the far half is compressed and net-occluded).
    Mapping a half-court quad onto the full plane put the real net at y=0 and
    the attack zone at y<450, so the team-side filter deleted the entire
    front row (and in high formations, the whole team) as "opponents"."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    H, _ = cv2.findHomography(src, _DST if dst is None else dst)
    return H


def detect_court_quad(frame: np.ndarray) -> np.ndarray | None:
    """
    Segment the dominant playing surface and reduce it to a 4-corner convex
    quadrilateral.  Returns (4,2) corners in TL,TR,BR,BL order, or None.

    Method: sample the court colour from the lower-central region (where the
    floor almost always is in a broadcast frame), threshold in HSV, clean with
    morphology, take the largest contour and approximate its convex hull down
    to 4 vertices.  All rejects return None – caller keeps its previous
    calibration.
    """
    h, w = frame.shape[:2]
    scale = 480.0 / max(w, 1)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]

    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)

    # Sample floor colour from the lower-central patch.
    patch = hsv[int(sh * 0.55): int(sh * 0.9), int(sw * 0.3): int(sw * 0.7)]
    if patch.size == 0:
        return None
    med = np.median(patch.reshape(-1, 3), axis=0)
    lo = np.array([max(med[0] - 12, 0), max(med[1] - 70, 0), max(med[2] - 70, 0)])
    hi = np.array([min(med[0] + 12, 179), min(med[1] + 70, 255), min(med[2] + 70, 255)])
    mask = cv2.inRange(hsv, lo.astype(np.uint8), hi.astype(np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.08 * sh * sw or area > 0.95 * sh * sw:
        return None

    hull = cv2.convexHull(cnt)
    quad = None
    for eps_frac in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps_frac * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return None
    if not cv2.isContourConvex(quad.astype(np.int32)):
        return None

    ordered = order_corners(quad)
    if ordered is None:   # M11: bowtie/degenerate ordering - reject like any other bad quad
        return None
    corners = ordered / scale     # back to full resolution

    # Sanity: landscape shape (court is wider than tall on screen).
    width_px = np.linalg.norm(corners[1] - corners[0])
    height_px = np.linalg.norm(corners[3] - corners[0])
    if width_px < height_px:
        return None
    return corners


class CameraMotionEstimator:
    """Frame-to-frame partial-affine camera model from sparse optical flow."""

    #: M12 (AUDIT_FINDINGS.md): estimateAffinePartial2D returns a similarity
    #: transform (uniform scale + rotation + translation) with NO validation
    #: that it represents a plausible frame-to-frame camera motion. A scene
    #: cut or a low-texture frame can still pass its own >=12-inlier check
    #: while describing an implausible jump - and unlike a bad auto-quad
    #: detection (which is simply skipped, keeping the previous quad), this
    #: estimate is APPLIED by the caller (CourtCalibrator.update warps
    #: self.corners by it unconditionally), so one bad estimate corrupts the
    #: calibration for every subsequent frame, not just the current one.
    #: None of these bounds are empirically tuned against labelled bad/good
    #: motion data (none exists) - defensible, documented defaults for a
    #: BROADCAST camera between CONSECUTIVE frames (1/30s apart).
    MAX_SCALE_CHANGE = 0.15      # scale must be within [0.85, 1.15]
    MAX_ROTATION_DEG = 5.0       # real roll this fast in 1/30s would be extreme
    MAX_TRANSLATION_FRAC = 0.15  # of the smaller full-res frame dimension
    MIN_INLIER_RATIO = 0.6       # fraction of matched points RANSAC must accept

    def __init__(self, max_features: int = 400) -> None:
        self._prev_gray: np.ndarray | None = None
        self._max_features = max_features

    def reset(self) -> None:
        self._prev_gray = None

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a 2×3 affine A such that  p_curr ≈ A · [p_prev, 1]ᵀ,
        or None when motion cannot be estimated (first frame, no features,
        or the estimate fails the M12 plausibility bounds below)."""
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.5, fy=0.5)

        A = None
        if self._prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(
                self._prev_gray, maxCorners=self._max_features,
                qualityLevel=0.01, minDistance=12, blockSize=7,
            )
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, p0, None,
                    winSize=(21, 21), maxLevel=3,
                )
                if p1 is not None:
                    good = st.ravel() == 1
                    if good.sum() >= 12:
                        A_half, inliers = cv2.estimateAffinePartial2D(
                            p0[good], p1[good], method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                        )
                        if A_half is not None:
                            inlier_ratio = (
                                float(inliers.mean()) if inliers is not None else 0.0
                            )
                            # rescale translation from half-res to full-res
                            candidate = A_half.copy()
                            candidate[:, 2] *= 2.0
                            if self._is_plausible(
                                candidate, frame_w, frame_h, inlier_ratio
                            ):
                                A = candidate
        self._prev_gray = gray
        return A

    def _is_plausible(
        self, A: np.ndarray, frame_w: int, frame_h: int, inlier_ratio: float,
    ) -> bool:
        """M12: reject an affine that cannot be a real frame-to-frame motion,
        rather than applying it and corrupting the calibration silently."""
        if inlier_ratio < self.MIN_INLIER_RATIO:
            return False
        a, b, tx = A[0]
        c, d, ty = A[1]
        scale = float(np.hypot(a, c))  # partial-affine (similarity): sqrt(b*b+d*d) agrees
        if abs(scale - 1.0) > self.MAX_SCALE_CHANGE:
            return False
        rotation_deg = abs(float(np.degrees(np.arctan2(c, a))))
        if rotation_deg > self.MAX_ROTATION_DEG:
            return False
        translation = float(np.hypot(tx, ty))
        if translation > self.MAX_TRANSLATION_FRAC * min(frame_w, frame_h):
            return False
        return True


class CourtCalibrator:
    """
    Stateful pixel→court calibration for one video.

    Modes (all combinable):
      * manual corners             – fixed H (classic behaviour)
      * auto=True                  – periodic court re-detection every
                                     `refresh_every` frames
      * cmc=True                   – per-frame camera-motion warp of the
                                     corners between (re)detections
      * neither corners nor auto   – linear fallback scaling, mask disabled

    Call `update(frame)` once per frame BEFORE converting coordinates.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        manual_corners: np.ndarray | list | None = None,
        auto: bool = False,
        cmc: bool = False,
        refresh_every: int = 5,
        mask_margin_px: float = 12.0,
        force_linear: bool = False,
    ) -> None:
        # force_linear: use the detected quad ONLY as an in/out-of-court MASK
        # (drop coaches/refs/bench) while coordinates stay on the linear
        # pixel->court fallback. This matches the training-feature geometry
        # (CSVs were extracted with linear coords) while still cleaning the
        # detection population on full broadcast video.
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.auto = auto
        self.cmc = cmc
        self.refresh_every = max(int(refresh_every), 1)
        self.mask_margin_px = mask_margin_px
        self.force_linear = bool(force_linear)

        if manual_corners is not None:
            _ordered_manual = order_corners(np.asarray(manual_corners, dtype=np.float32))
            if _ordered_manual is None:
                # M11: fail loud. Silently falling back to uncalibrated here
                # would mean --court-corners is accepted, "works", and every
                # downstream distance is wrong with no indication why - the
                # exact silent-failure shape this project exists to remove.
                raise ValueError(
                    f"--court-corners {manual_corners} does not form a "
                    "simple (non-self-intersecting) quadrilateral - check "
                    "the 4 points are given in a consistent order around "
                    "the court, not crossed."
                )
            self.corners: np.ndarray | None = _ordered_manual
        else:
            self.corners = None
        # Manual corners describe the FULL court; auto-detected quads describe
        # the NEAR half (see build_homography docstring).
        self._dst: np.ndarray | None = None if manual_corners is not None else _HALF_DST
        self.H: np.ndarray | None = (
            build_homography(self.corners) if self.corners is not None else None
        )
        self._cme = CameraMotionEstimator() if cmc else None
        self._frame_idx = 0
        self.n_auto_success = 0
        self.n_auto_fail = 0

    # ------------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def update(self, frame: np.ndarray) -> None:
        """Advance calibration by one frame (CMC warp + periodic re-detect)."""
        self._frame_idx += 1

        # 1. Camera-motion compensation: warp stored corners by camera affine.
        if self._cme is not None:
            A = self._cme.estimate(frame)
            if A is not None and self.corners is not None:
                pts = np.hstack([self.corners, np.ones((4, 1), dtype=np.float32)])
                self.corners = (A @ pts.T).T.astype(np.float32)
                self.H = build_homography(self.corners, dst=self._dst)

        # 2. Periodic automatic (re)detection.
        if self.auto and (self._frame_idx % self.refresh_every == 1
                          or self.corners is None):
            quad = detect_court_quad(frame)
            if quad is not None and self.corners is not None:
                # M11 (AUDIT_FINDINGS.md): detect_court_quad's own checks
                # (area, convexity, landscape shape) all pass for a quad
                # that is simply the WRONG region - a shadow, a different
                # part of the court, an ad board with similar colour. Bound
                # how far the candidate's centroid can have moved from the
                # current one: CMC already tracks smooth camera motion
                # between refreshes, so a genuine court cannot have jumped
                # further than a generous fraction of the frame diagonal in
                # `refresh_every` frames. A bigger jump is a bad detection,
                # not real motion - reject it like any other bad quad.
                diag = float(np.hypot(self.frame_w, self.frame_h))
                jump = float(np.linalg.norm(
                    quad.mean(axis=0) - self.corners.mean(axis=0)
                ))
                if jump > 0.25 * diag:
                    quad = None
            if quad is not None:
                self.set_auto_quad(quad)
                self.n_auto_success += 1
            else:
                self.n_auto_fail += 1   # keep previous calibration

    # ------------------------------------------------------------------
    #: EMA weight for a fresh auto-detected quad (0 = never move, 1 = snap).
    #: A low weight stops the drawn court outline (and the on/off-court mask)
    #: wobbling frame-to-frame between two near-equivalent detections while
    #: still tracking genuine slow camera drift over a few refreshes.
    AUTO_QUAD_EMA = 0.35

    def set_auto_quad(self, quad: np.ndarray) -> None:
        """Adopt an auto-detected NEAR-HALF-court quad (net = top edge).

        The quad is corner-wise EMA-blended with the current one (when the two
        are already in the same corner order) so the outline is temporally
        stable; the first detection is taken as-is.
        """
        quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        if self.corners is not None and self.corners.shape == (4, 2):
            prev = self.corners.astype(np.float32)
            # only blend when corners still correspond (small per-corner move);
            # a large move is a genuine cut / re-lock, take it whole.
            per_corner = np.linalg.norm(quad - prev, axis=1)
            diag = float(np.hypot(self.frame_w, self.frame_h))
            if float(per_corner.max()) < 0.12 * diag:
                quad = (1.0 - self.AUTO_QUAD_EMA) * prev + self.AUTO_QUAD_EMA * quad
                quad = quad.astype(np.float32)
        self.corners = quad
        self._dst = _HALF_DST
        self.H = build_homography(quad, dst=_HALF_DST)

    # ------------------------------------------------------------------
    def pixel_to_court(self, px: float, py: float) -> tuple[float, float]:
        """Map one pixel to court cm (homography, or linear fallback)."""
        if self.H is not None and not self.force_linear:
            pt = np.array([[[px, py]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self.H)
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        return (
            px / max(self.frame_w, 1) * COURT_W_CM,
            py / max(self.frame_h, 1) * COURT_H_CM,
        )

    def net_y_cm_current(self) -> float:
        """Where the net actually is, in whatever court-Y this calibrator's
        `pixel_to_court` currently produces (H6, AUDIT_FINDINGS.md).

        Under real homography, this is exactly NET_Y_CM by construction -
        set_auto_quad built self.H so the quad's own top edge maps to
        _HALF_DST's top edge, which IS NET_Y_CM - so transforming the
        quad's own top edge back through pixel_to_court reproduces it, no
        special-casing needed. Under --court-coords linear specifically
        (or before any quad is ever detected), pixel_to_court is a naive
        py/frame_h*COURT_H_CM proportional scale with no perspective
        correction, and NET_Y_CM (frame vertical midpoint) is routinely
        wrong for a real broadcast angle - measured on video (4): net
        pixel-row 289 -> court_y ~361, not 450. Deriving it from the
        detected quad's own top edge (still passed through the SAME
        pixel_to_court call, so it's consistent with every other
        court-space distance this calibrator produces) fixes that without
        a mode-specific branch. Falls back to the NET_Y_CM constant only
        when no quad has ever been detected - nothing else to derive it
        from.
        """
        if self.corners is None:
            return NET_Y_CM
        mid_top_x = float((self.corners[0][0] + self.corners[1][0]) / 2.0)
        mid_top_y = float((self.corners[0][1] + self.corners[1][1]) / 2.0)
        return self.pixel_to_court(mid_top_x, mid_top_y)[1]

    # ------------------------------------------------------------------
    def in_court(self, px: float, py: float) -> bool:
        """True when a pixel lies inside the court polygon (+margin).
        Without a calibration polygon everything passes (no-op mask)."""
        if self.corners is None:
            return True
        poly = self.corners.astype(np.float32)
        d = cv2.pointPolygonTest(poly, (float(px), float(py)), True)
        return d >= -self.mask_margin_px

    def filter_detections(
        self, xyxy: np.ndarray, conf: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop detections whose FOOT POINT is outside the court polygon –
        referees, bench players, coaches, crowd (spec §3C)."""
        if self.corners is None or len(xyxy) == 0:
            return xyxy, conf
        keep = [
            i for i, (x1, y1, x2, y2) in enumerate(xyxy)
            if self.in_court((x1 + x2) / 2.0, y2)
        ]
        if not keep:
            return (np.empty((0, 4), dtype=xyxy.dtype),
                    np.empty((0,), dtype=conf.dtype))
        idx = np.asarray(keep, dtype=int)
        return xyxy[idx], conf[idx]

    # ------------------------------------------------------------------
    def court_mask(self) -> np.ndarray | None:
        """Binary uint8 mask (255 inside court) or None when uncalibrated."""
        if self.corners is None:
            return None
        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.corners.astype(np.int32)], 255)
        return mask

    def draw(self, frame: np.ndarray) -> None:
        """Overlay the current court polygon on a frame (in place)."""
        if self.corners is None:
            return
        cv2.polylines(
            frame, [self.corners.astype(np.int32)], isClosed=True,
            color=(0, 220, 220), thickness=2, lineType=cv2.LINE_AA,
        )


__all__ = [
    "COURT_W_CM", "COURT_H_CM", "NET_Y_CM",
    "order_corners", "build_homography", "detect_court_quad",
    "CameraMotionEstimator", "CourtCalibrator",
]
