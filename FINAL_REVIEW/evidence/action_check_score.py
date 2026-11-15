"""Opponent-action check, step 2: score the human marks (added 2026-11-15).

A person looked at each 3-frame strip in action_check/sheet_*.jpg and judged
whether the named player really did the named action at that moment.
CORRECT lists the event ids judged right; all others were judged wrong
(typical errors: a reception called SERVE, a standing player called SPIKE,
the wrong player picked in a crowd at the net).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORRECT = {1, 10, 14, 18, 20, 25, 27,                 # SERVE
           0, 5, 9, 15, 16, 17, 22, 24, 26, 30}       # SPIKE
ev = json.loads((HERE / "action_check" / "events.json").read_text())
res = {}
for act in ("SERVE", "SPIKE", "BLOCK"):
    e = [x for x in ev if x["action"] == act]
    ok = sum(x["id"] in CORRECT for x in e)
    res[act] = {"checked": len(e), "correct": ok, "precision": ok / len(e) if e else None}
res["all"] = {"checked": len(ev), "correct": sum(x["id"] in CORRECT for x in ev)}
res["all"]["precision"] = res["all"]["correct"] / len(ev)
res["recall_note"] = ("Not measured with labels. Indication of misses: in the 5-minute 'videoplayback.mp4' "
                      "segment the rules found 1 serve (both teams), while a 5-minute stretch has roughly "
                      "10-20 rallies; serves are often shown in a close-up camera shot, which the live-play "
                      "filter removes.")
(HERE / "action_check_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
