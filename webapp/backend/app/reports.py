"""Plain-text (Notepad-friendly), CSV and JSON reports for one analysis.

One report per video, matching the annotated video exactly:
  * tactical_report  - your team (near side): when it was in Coordinated
                       Attack, Coordinated Defense, Delayed Support or Spacing
                       Breakdown, with timestamps and confidence. These are the
                       moments captioned in the video.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

TACTICS = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]  # = new_render.SHOWN_TACTICS
OPP_ACTIONS = ["SERVE", "SPIKE", "BLOCK"]
TACTIC_MEANING = {
    "Coordinated Attack": "fast approach by front-court attackers, base players held near the net",
    "Coordinated Defense": "the team shifts as a rigid unit",
    "Delayed Support": "the nearest player reacts late after ball impact",
    "Spacing Breakdown": "structural failure - formation too spread or too clustered",
}
RULE = "=" * 78
THIN = "-" * 78


def ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def summarise(result: dict) -> dict:
    opp = Counter(e["action"] for e in result["events"])
    segs = result["tactics"]["segments"]
    called = sum(s["end"] - s["start"] for s in segs)
    tac = {}
    for t in TACTICS:
        ss = [s for s in segs if s["tactic"] == t]
        dur = sum(s["end"] - s["start"] for s in ss)
        tac[t] = {"segments": len(ss), "seconds": round(dur, 2),
                  "share": round(dur / called, 4) if called else 0.0,
                  "avg_confidence": round(sum(s["confidence"] for s in ss) / len(ss), 4) if ss else 0.0}
    v = result["video"]
    return {
        "opponent_actions": {a: opp[a] for a in OPP_ACTIONS},
        "opponent_total": sum(opp.values()),
        "tactic_segments": len(segs),
        "live_seconds": round(called, 2),
        "analysed_seconds": round(v["analysed_end"] - v["analysed_start"], 2),
        "tactics": tac,
        "dominant_tactic": max(TACTICS, key=lambda t: tac[t]["seconds"]) if segs else None,
    }


def _header(title: str, result: dict, video_name: str) -> list[str]:
    v = result["video"]
    court = result["court"]
    if court["method"] == "auto":
        cal = f"automatic ({', '.join(court['lines'])})"
    elif court["method"] == "known venue":
        cal = f"known venue, verified court lines ({court.get('venue')})"
    else:
        cal = "generic end-court model (court lines not found)"
    return [
        RULE, title.center(78), "Volleyball Tactical Analysis  -  volleyballtacticalanalyze.com".center(78), RULE,
        f"Video            : {video_name}",
        f"Analysed part    : {ts(v['analysed_start'])} - {ts(v['analysed_end'])}"
        f"   ({v['analysed_end'] - v['analysed_start']:.1f} s, {v['fps']:.2f} fps, {v['width']}x{v['height']})",
        "Your team        : near side of the net (bottom of the picture, blue)",
        "Opponent         : far side of the net (top of the picture, red)",
        f"Court calibration: {cal}",
        f"Generated        : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]


def tactical_report_txt(result: dict, video_name: str) -> str:
    s = result["summary"]
    segs = result["tactics"]["segments"]
    L = _header("TEAM TACTICAL REPORT  -  YOUR TEAM (NEAR SIDE)", result, video_name)
    L += ["SUMMARY", THIN,
          f"  {'TACTIC':<22}{'SEGMENTS':>9}{'TIME (s)':>11}{'SHARE':>9}{'AVG CONF.':>12}", THIN]
    for name in TACTICS:
        x = s["tactics"][name]
        L.append(f"  {name:<22}{x['segments']:>9}{x['seconds']:>11.1f}{x['share'] * 100:>8.1f}%"
                 f"{x['avg_confidence'] * 100:>11.1f}%")
    L += [THIN,
          f"  Dominant tactic      : {s['dominant_tactic'] or '-'}",
          f"  Tactic time          : {s['live_seconds']:.1f} s of {s['analysed_seconds']:.1f} s analysed",
          "  Same moments as the captions in the annotated video. Nothing is reported between",
          "  rallies, in replays, with fewer than 3 of your players visible, or when the model",
          "  is not sure enough (50% for Attack / Defense, 20% for the two rarer tactics).",
          "",
          "WHAT THE TACTICS MEAN", THIN]
    for name in TACTICS:
        L.append(f"  {name:<20}: {TACTIC_MEANING[name]}")
    L += ["", "TIMELINE  (times are positions in the original video)", THIN,
          f"  {'#':>4}  {'FROM':<10} {'TO':<10} {'LENGTH':>7}  {'TACTIC':<21} {'CONF.':>6}", THIN]
    if not segs:
        L.append("  No clear team tactic was found in this part of the video.")
    for i, g in enumerate(segs, 1):
        star = " *" if g["tactic"] in ("Delayed Support", "Spacing Breakdown") else ""
        L.append(f"  {i:>4}  {ts(g['start']):<10} {ts(g['end']):<10} {g['end'] - g['start']:>6.1f}s"
                 f"  {g['tactic']:<21} {g['confidence'] * 100:>5.0f}%{star}")
    L += ["", THIN, "Notes",
          "  * All four tactics are reported, exactly as captioned in the annotated video.",
          "    Click a time to play the video from that moment.",
          "  * On videos the model never saw, Coordinated Attack / Defense calls were right about",
          "    85% of the time, Spacing Breakdown about 29% and Delayed Support about 26% (they",
          "    had few training examples). Rows marked (*) are less reliable: check them by eye.",
          "  * Model: the project's deployed HistGradientBoosting classifier (tactical_ml_v6.joblib).",
          "    29-sample (about 1 s) windows of your team's six court positions, sliding every 1/3 s,",
          "    balanced for class frequency, smoothed, with a minimum length of 0.8 s per call.",
          "  * CONF. is the model's own probability for the called tactic. Attack / Defense",
          "    need at least 50%, Delayed Support / Spacing Breakdown at least 20% (they are",
          "    rarer, so their probabilities stay lower).",
          "  * Leave-one-video-out accuracy of this model is 67.0% (macro F1 0.51) on 948 clips;",
          "    treat short or low-confidence segments as indicative, not certain.",
          RULE]
    return "\r\n".join(L) + "\r\n"


def opponent_report_txt(result: dict, video_name: str) -> str:
    s = result["summary"]
    ev = result["events"]
    L = _header("OPPONENT ACTION REPORT  -  SERVE / SPIKE / BLOCK", result, video_name)
    L += ["SUMMARY", THIN] + [f"  {a:<8}: {s['opponent_actions'][a]}" for a in OPP_ACTIONS] + [
        f"  {'TOTAL':<8}: {s['opponent_total']}",
        "  Only the opponent's serves, spikes and blocks are reported; their other touches are ignored.",
        "", "TIMELINE  (times are positions in the original video)", THIN,
        f"  {'#':>4}  {'TIME':<10} {'ACTION':<8} {'PLAYER':<8}", THIN]
    if not ev:
        L.append("  No opponent serve, spike or block was detected in the analysed part of the video.")
    for i, e in enumerate(ev, 1):
        pid = f"#{e['player_track_id']}" if e["player_track_id"] >= 0 else "-"
        L.append(f"  {i:>4}  {ts(e['time']):<10} {e['action']:<8} {pid:<8}")
    L += ["", THIN, "Notes",
          "  * Actions come from ball-touch events (sharp changes in the ball path next to a player),",
          "    the volleyball touch order and pose (contact above the head).",
          "  * PLAYER is the tracker id of the opponent player (opponents are not marked in the video).",
          RULE]
    return "\r\n".join(L) + "\r\n"


def write_all(result: dict, out_dir: Path, video_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tactical_report.txt").write_text(tactical_report_txt(result, video_name), encoding="utf-8")
    with open(out_dir / "tactical_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "start_seconds", "end_seconds", "tactic", "confidence"])
        for g in result["tactics"]["segments"]:
            w.writerow([ts(g["start"]), ts(g["end"]), f"{g['start']:.2f}", f"{g['end']:.2f}",
                        g["tactic"], f"{g['confidence']:.3f}"])
    (out_dir / "tactical_report.json").write_text(json.dumps(
        {"video": video_name, "team": "near side", "model": result["tactics"]["model"],
         "summary": result["summary"]["tactics"], "segments": result["tactics"]["segments"]},
        indent=1), encoding="utf-8")
