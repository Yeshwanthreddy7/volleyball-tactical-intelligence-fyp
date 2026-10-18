"""
strict_render.py - display-only hardening of the annotated video.

WHAT THIS IS / IS NOT
---------------------
This layer post-processes ONLY what gets DRAWN in ``pipeline.py --output-video``
(player boxes, jersey numbers, ball marker). It does NOT touch the feature
tensor that feeds the tactical classifier, so train/serve parity is preserved
(the classifier was trained on the un-hardened extraction). Enable it with
``pipeline.py --strict-render``.

CONDITIONS IT ENFORCES  (as far as automatic CV on arbitrary broadcast footage
allows - this is not, and cannot be, perfect)
-----------------------------------------------------------------------------
  * Stable numbers 1..N .. identity comes from identity.SlotManager (persistent
                           slot for the life of a track, 15-frame occlusion
                           bridge, hard cap of N). The displayed number is
                           ``slot + 1``; a slot freed by a player who leaves is
                           re-used by the next joiner (lowest free number).
                           Raw tracker ids - which churn after occlusions - are
                           never shown.
  * Court gate ........... a box whose foot point is outside the auto-detected
                           court polygon is not drawn; a player who steps off
                           court loses the box that same frame; referees / bench
                           / crowd never get a box. No-op until --auto-court has
                           locked a quad (prints one warning).
  * Confirmation ........ a slot must be occupied for ``confirm_frames``
                           consecutive frames before it is drawn - removes the
                           one-or-two-frame opponent flicker at the net before
                           the colour vote settles, and transient false tracks.
  * Ball anti-stuck ..... the ball marker is hidden after ``ball_max_coast``
                           consecutive predicted (undetected) frames instead of
                           being frozen at its last predicted position.

WHAT IT CANNOT DO
-----------------
Invent detections the detector missed; separate two teams in near-identical
kit; prevent every same-team id swap inside a block; work before the court
quad locks. "Perfect" tracking on arbitrary footage is out of reach here.
"""
from __future__ import annotations

import sys

import numpy as np


class StrictRenderer:
    """Per-frame hardening of the drawn player set + ball. Stateful."""

    def __init__(
        self,
        n_slots: int = 6,
        confirm_frames: int = 5,
        ball_max_coast: int = 8,
    ) -> None:
        self.n_slots = int(n_slots)
        self.confirm_frames = max(1, int(confirm_frames))
        self.ball_max_coast = max(0, int(ball_max_coast))

        self._slot_seen: dict[int, int] = {}   # slot -> consecutive frames occupied
        self._ball_coast = 0
        self._warned_no_court = False
        # diagnostics
        self.n_court_dropped = 0
        self.n_unconfirmed_dropped = 0
        self.n_ball_hidden = 0

    # ------------------------------------------------------------------
    def update(
        self,
        tracked_xyxy: np.ndarray,
        track_ids: np.ndarray,
        id_to_slot: "dict[int, int]",
        calibrator,
        ball_center: "np.ndarray | None",
        ball_is_predicted: bool,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray | None]":
        """Return ``(display_xyxy, display_numbers, display_ball)`` for one frame.

        ``display_numbers`` are jersey numbers 1..n_slots (int array), sorted
        ascending and aligned with ``display_xyxy``.
        """
        xyxy = np.asarray(tracked_xyxy, dtype=float).reshape(-1, 4)
        tids = [int(t) for t in np.asarray(track_ids).reshape(-1)]

        court_ok = (
            calibrator is not None
            and getattr(calibrator, "corners", None) is not None
        )
        if not court_ok and not self._warned_no_court:
            print(
                "[strict-render] court polygon not locked yet - court gate "
                "inactive until --auto-court finds the lines.",
                file=sys.stderr,
            )
            self._warned_no_court = True

        # ── slot -> box, after the court gate ────────────────────────────
        slot_box: dict[int, np.ndarray] = {}
        for tid, box in zip(tids, xyxy):
            slot = id_to_slot.get(tid)
            if slot is None or not (0 <= int(slot) < self.n_slots):
                continue
            foot_x = float((box[0] + box[2]) / 2.0)
            foot_y = float(box[3])
            if court_ok and not calibrator.in_court(foot_x, foot_y):
                self.n_court_dropped += 1
                continue
            slot_box.setdefault(int(slot), box)   # first tid wins a shared slot

        # ── confirmation counters ───────────────────────────────────────
        for s in range(self.n_slots):
            self._slot_seen[s] = self._slot_seen.get(s, 0) + 1 if s in slot_box else 0

        disp_boxes: list[np.ndarray] = []
        disp_nums: list[int] = []
        for s, box in sorted(slot_box.items()):
            if self._slot_seen.get(s, 0) >= self.confirm_frames:
                disp_boxes.append(box)
                disp_nums.append(s + 1)
            else:
                self.n_unconfirmed_dropped += 1

        d_xyxy = (np.asarray(disp_boxes, dtype=float)
                  if disp_boxes else np.empty((0, 4), dtype=float))
        d_nums = (np.asarray(disp_nums, dtype=int)
                  if disp_nums else np.empty((0,), dtype=int))

        # ── ball anti-stuck ─────────────────────────────────────────────
        if ball_center is None:
            d_ball = None
        elif ball_is_predicted:
            self._ball_coast += 1
            if self._ball_coast > self.ball_max_coast:
                d_ball = None
                self.n_ball_hidden += 1
            else:
                d_ball = ball_center
        else:
            self._ball_coast = 0
            d_ball = ball_center

        return d_xyxy, d_nums, d_ball

    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"strict-render: {self.n_court_dropped} box-frames dropped off-court, "
            f"{self.n_unconfirmed_dropped} unconfirmed, "
            f"{self.n_ball_hidden} frames with ball hidden (coast > "
            f"{self.ball_max_coast})."
        )


__all__ = ["StrictRenderer"]
