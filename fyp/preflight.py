"""
preflight.py - fail loudly BEFORE producing a confident-looking wrong answer.

WHY THIS EXISTS
---------------
Every serious failure in this project's history was silent. The pipeline ran to
completion, printed a formatted report, rendered a video, and was wrong:

  * §12.8  a half-court quad mapped onto the full-court plane deleted the whole
           front row; output: 100% "TRANSITION", no error.
  * §12.9  off-court coaches claimed all six identity slots; output: 100%
           "TRANSITION", no error.
  * §13    the custom detector scored 0.06 max confidence on players in the
           demo video; output: zero boxes, 100% "TRANSITION", no error.

In all three the console said "no players" while the report said "analysis
complete". A tactical report computed from zero or twelve players is not a
degraded result - it is a meaningless one, and it must not be presented as an
answer. This module samples the actual segment that is about to be processed,
measures what survives each stage, and refuses to continue when the population
is degenerate.

The verdict logic (`assess`) is deliberately separated from frame sampling and
is pure arithmetic on a stats dict, so it is unit-testable with no video, no
weights and no cv2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Six players per side in indoor volleyball. Detections legitimately fall below
# six (occlusion behind the net band, a player leaving frame during a dig) and
# rise slightly above it during substitutions, so the ACCEPTABLE band is wide;
# only clearly impossible populations are rejected.
EXPECTED_TEAM_SIZE = 6
MIN_TEAM_MEDIAN = 3.0        # below this the six slots cannot be filled
MAX_TEAM_MEDIAN = 9.0        # above this the team split is leaking opponents
MIN_RAW_MEDIAN = 4.5         # M10: below this, the detector is not seeing the rally at all
LOW_RAW_MEDIAN = 8.0         # M10: a full rally frame shows ~12 (both teams + officials);
                              # 4.5-8 passes today with zero comment, but is still suspiciously low
MIN_BALL_RECALL = 0.20       # below this, ball-derived features are noise

FATAL, WARN, OK = "FATAL", "WARN", "OK"


@dataclass
class Finding:
    level: str
    stage: str
    message: str
    remedy: str


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)

    @property
    def fatal(self) -> bool:
        return any(f.level == FATAL for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARN]

    def render(self) -> str:
        if not self.findings:
            return "  All preflight checks passed."
        lines = []
        for f in self.findings:
            mark = "FATAL" if f.level == FATAL else "WARN "
            lines.append(f"  [{mark}] {f.stage}: {f.message}")
            lines.append(f"          -> {f.remedy}")
        return "\n".join(lines)


def assess(stats: dict) -> Verdict:
    """
    Turn measured per-stage populations into a go / no-go verdict.

    `stats` keys (all optional except the medians):
      raw_person_median      persons per frame straight from the detector
      masked_person_median   after the court mask
      team_person_median     after the team split (what the tracker feeds slots)
      ball_recall            fraction of sampled frames with a ball detection
      n_frames               how many frames were sampled
    """
    v = Verdict()
    raw = float(stats.get("raw_person_median", 0.0))
    masked = float(stats.get("masked_person_median", 0.0))
    team = float(stats.get("team_person_median", 0.0))
    ball = stats.get("ball_recall")
    n_frames = int(stats.get("n_frames", 0))

    # --- Stage 0: could any frames be read at all? --------------------------
    # L20 (AUDIT_FINDINGS.md): a short or unreadable segment leaves the raw/
    # masked/team medians at their 0.0 default (median of an empty sample),
    # which Stage 1 below would misdiagnose as "the detector is not working"
    # and recommend a detector-weights swap - the wrong remedy for a segment
    # that was never actually read. Catch the real cause first.
    if n_frames == 0:
        v.findings.append(Finding(
            FATAL, "frame read",
            "zero frames could be read from the sampled segment - this is "
            "not a detector problem, no detector ran",
            "Check --start-frame is within the video's actual frame count "
            "and the file is not truncated/corrupted. Try "
            "fyp/diagnose_video.py on the same file and --start-frame.",
        ))
        return v

    # --- Stage 1: does the detector see people at all? ---------------------
    if raw < MIN_RAW_MEDIAN:
        v.findings.append(Finding(
            FATAL, "detection",
            f"median {raw:.1f} people/frame - a rally frame shows 12 players "
            f"plus officials, so the detector is not working on this footage",
            "The volleyball fine-tune collapses on unseen courts (measured "
            "0.1 players/frame on videoplayback (4)). Use stock COCO weights "
            "for players and the fine-tune only for the ball: "
            "--yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt",
        ))
        return v      # nothing downstream can be meaningful; stop here.
    elif raw < LOW_RAW_MEDIAN:
        # M10 (AUDIT_FINDINGS.md): this used to pass with zero comment. A
        # real rally frame shows ~12 people; 4.5-8 is still suspiciously
        # low (half-blind detector, half the roster off-court, heavy
        # occlusion) - not certainly broken, so WARN rather than FATAL, but
        # not silent either.
        v.findings.append(Finding(
            WARN, "detection",
            f"median {raw:.1f} people/frame - low for a rally frame (both "
            f"teams + officials is normally ~12); not fatal, but downstream "
            f"population checks may be starved by this rather than by a "
            f"real team-split or court-mask problem",
            "Check --conf-threshold isn't too high and --imgsz matches the "
            "footage resolution; compare against fyp/diagnose_video.py.",
        ))

    # --- Stage 2: does the court mask keep them? ---------------------------
    if masked <= 0.0:
        v.findings.append(Finding(
            FATAL, "court mask",
            f"the detector found {raw:.1f} people/frame but the court mask "
            f"kept none - the calibration disagrees with the image",
            "Check the auto-detected court quad, or supply --court-corners "
            "explicitly. Run fyp/diagnose_video.py to see the quad drawn.",
        ))
        return v

    # --- Stage 3: is the analysed population one team? ---------------------
    if team <= 0.0:
        v.findings.append(Finding(
            FATAL, "team split",
            f"{masked:.1f} people/frame survive the court mask but none "
            f"survive the team split - every window will be Unclassified",
            "The geometric split fails when the court plane is mis-mapped. "
            "Use --team-split colour, and verify --team-side matches the half "
            "the analysed team actually occupies.",
        ))
    elif team < MIN_TEAM_MEDIAN:
        v.findings.append(Finding(
            FATAL, "team split",
            f"median {team:.1f} players/frame reach the six identity slots; "
            f"below {MIN_TEAM_MEDIAN:.0f} the formation features (spacing, "
            f"centroid, synchronisation) describe missing players",
            "Lower --conf-threshold, raise --imgsz, or widen the court mask; "
            "confirm the analysed team is on the --team-side you asked for.",
        ))
    elif team > MAX_TEAM_MEDIAN:
        # M9 (AUDIT_FINDINGS.md): with --team-side bottom/top, a median this
        # high is only explainable by opponents leaking into "our" formation
        # - every downstream feature (spacing, centroid, sync) is computed
        # over the wrong set of people, so this was a WARN sitting in front
        # of a fully-computed, confidently-wrong report. --team-side all
        # means BOTH teams are the intended population, so the same number
        # there is expected, not a leak - stays a WARN.
        #
        # H7 (AUDIT_FINDINGS.md): `team` here is ALWAYS the crude geometric
        # estimate (see _preflight_sample), which is a known, EXPECTED
        # over-counter (both front rows near the net) - not what will
        # actually run when --team-split colour is requested. Trust the
        # colour fit's own verdict (below, from a real batch fit on these
        # same sampled frames) when one is available and healthy, rather
        # than aborting on a proxy metric that was never going to be used.
        team_side = stats.get("team_side", "bottom")
        colour_report = stats.get("colour_report")
        colour_will_correct = (
            stats.get("team_split") == "colour"
            and colour_report is not None
            and not colour_report.get("degenerate", True)
        )
        level = WARN if (team_side == "all" or colour_will_correct) else FATAL
        note = (
            " - the colour fit below is healthy and will correct this at "
            "serve time" if colour_will_correct else ""
        )
        v.findings.append(Finding(
            level, "team split",
            f"median {team:.1f} players/frame (geometric estimate) - more "
            f"than one team's worth, so opponents are leaking into the "
            f"analysed formation" + note
            + ("" if team_side == "all" else f" (--team-side {team_side})"),
            "Use --team-split colour (the geometric split cannot separate the "
            "two front rows at the net) and check --team-clusters covers the "
            "libero's contrasting jersey.",
        ))

    # H7: the colour fit's OWN verdict, independent of the geometric count
    # above - catches the case the geometric count can't: a colour fit
    # that found a plausible NUMBER of players but the WRONG six (tracking
    # the opponent's team), which a person-count check is blind to.
    colour_report = stats.get("colour_report")
    if stats.get("team_split") == "colour" and stats.get("team_side") != "all":
        if colour_report is None:
            v.findings.append(Finding(
                WARN, "colour team split",
                "not enough evidence in the preflight sample to test-fit the "
                "colour model - its quality is unverified before the real run",
                "Usually resolves once the real run's longer warm-up window "
                "accumulates more evidence; increase --preflight-frames if "
                "this persists.",
            ))
        elif colour_report.get("degenerate"):
            gap = colour_report.get("near_far_court_y_gap")
            gap_txt = f", near/far court-Y gap {gap:.0f} cm" if gap is not None else ""
            v.findings.append(Finding(
                FATAL, "colour team split",
                "a test fit of the colour model on this same sample is "
                f"degenerate (agreement with position "
                f"{colour_report.get('geometric_agreement')}{gap_txt}) - "
                "colour will not reliably separate the two teams at serve "
                "time, independent of how many players the geometric count saw",
                "Similar kits, or the analysed side needs --team-clusters "
                "raised to cover a contrasting libero jersey; inspect "
                "fyp/diagnose_video.py on this segment.",
            ))
    elif abs(team - EXPECTED_TEAM_SIZE) > 2.0:
        v.findings.append(Finding(
            WARN, "team split",
            f"median {team:.1f} players/frame vs the expected "
            f"{EXPECTED_TEAM_SIZE} - formation features will be noisier",
            "Usually harmless (occlusion at the net band); inspect a rendered "
            "frame if the tactical labels look implausible.",
        ))

    # --- Stage 4: is the ball usable? --------------------------------------
    if ball is not None and float(ball) < MIN_BALL_RECALL:
        v.findings.append(Finding(
            WARN, "ball",
            f"ball found in {100 * float(ball):.0f}% of sampled frames - the "
            f"ball-derived features and the Coordinated-Attack / "
            f"Delayed-Support rules are unreliable at this recall",
            "Raise --imgsz (measured recall 15% at 640 vs 77% at 1280) and "
            "pass --ball-model fyp/volleyball_best.pt; report ball-dependent "
            "results as low-confidence.",
        ))

    return v


def summarise(stats: dict) -> str:
    """One-line-per-stage table of what the preflight actually measured."""
    rows = [
        ("frames sampled", stats.get("n_frames")),
        ("people/frame (detector)", stats.get("raw_person_median")),
        ("people/frame (after court mask)", stats.get("masked_person_median")),
        ("players/frame (after team split)", stats.get("team_person_median")),
    ]
    out = []
    for label, value in rows:
        if value is None:
            continue
        out.append(f"  {label:<34}: {value:g}")
    if stats.get("ball_recall") is not None:
        out.append(f"  {'ball recall':<34}: "
                   f"{100 * float(stats['ball_recall']):.0f}%")
    return "\n".join(out)


__all__ = [
    "assess", "summarise", "Verdict", "Finding",
    "FATAL", "WARN", "OK",
    "EXPECTED_TEAM_SIZE", "MIN_TEAM_MEDIAN", "MAX_TEAM_MEDIAN",
    "MIN_RAW_MEDIAN", "MIN_BALL_RECALL",
]
