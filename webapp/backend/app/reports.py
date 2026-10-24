"""Plain-text (Notepad-friendly), CSV and JSON reports for one analysis.

Two separate reports per video:
  * tactical_report  - your team (near side): when it was in Coordinated
                       Attack, Coordinated Defense, Delayed Support or
                       Spacing Breakdown, with timestamps and confidence.
  * opponent_report  - the opponent (far side): only SERVE, SPIKE and BLOCK,
                       each with its timestamp.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

TACTICS = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
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
    cal = (f"automatic ({', '.join(court['lines'])})" if court["method"] == "auto"
           else "generic end-court model (court lines not found)")
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
          f"  Live play classified : {s['live_seconds']:.1f} s of {s['analysed_seconds']:.1f} s analysed"
          " (no call between rallies, in replays or with fewer than 3 of your players visible)",
          "",
          "WHAT THE TACTICS MEAN", THIN]
    for name in TACTICS:
        L.append(f"  {name:<20}: {TACTIC_MEANING[name]}")
    L += ["", "TIMELINE  (times are positions in the original video)", THIN,
          f"  {'#':>4}  {'FROM':<10} {'TO':<10} {'LENGTH':>7}  {'TACTIC':<21} {'CONF.':>6}", THIN]
    if not segs:
        L.append("  No live play with enough visible players was found, so no tactic was called.")
    for i, g in enumerate(segs, 1):
        L.append(f"  {i:>4}  {ts(g['start']):<10} {ts(g['end']):<10} {g['end'] - g['start']:>6.1f}s"
                 f"  {g['tactic']:<21} {g['confidence'] * 100:>5.0f}%")
    L += ["", THIN, "Notes",
          "  * Model: the project's deployed HistGradientBoosting classifier (tactical_ml_v6.joblib).",
          "    29-sample (about 1 s) windows of your team's six court positions, sliding every 1/3 s,",
          "    balanced for class frequency, smoothed, with a minimum length of 0.8 s per call.",
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
          "  * PLAYER is the tracker id of the opponent player in the annotated video.",
          RULE]
    return "\r\n".join(L) + "\r\n"


def write_all(result: dict, out_dir: Path, video_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tactical_report.txt").write_text(tactical_report_txt(result, video_name), encoding="utf-8")
    (out_dir / "opponent_report.txt").write_text(opponent_report_txt(result, video_name), encoding="utf-8")
    with open(out_dir / "tactical_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "start_seconds", "end_seconds", "tactic", "confidence"])
        for g in result["tactics"]["segments"]:
            w.writerow([ts(g["start"]), ts(g["end"]), f"{g['start']:.2f}", f"{g['end']:.2f}",
                        g["tactic"], f"{g['confidence']:.3f}"])
    with open(out_dir / "opponent_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "seconds", "frame", "action", "player_track_id"])
        for e in result["events"]:
            w.writerow([ts(e["time"]), f"{e['time']:.2f}", e["frame"], e["action"], e["player_track_id"]])
    (out_dir / "tactical_report.json").write_text(json.dumps(
        {"video": video_name, "team": "near side", "model": result["tactics"]["model"],
         "summary": result["summary"]["tactics"], "segments": result["tactics"]["segments"]},
        indent=1), encoding="utf-8")
    (out_dir / "opponent_report.json").write_text(json.dumps(
        {"video": video_name, "team": "far side (opponent)",
         "summary": result["summary"]["opponent_actions"], "events": result["events"]},
        indent=1), encoding="utf-8")
