"""Ball-detector check, step 2: score the human marks (added 2026-11-15).

A person looked at every crop in ball_check/sheet_*.jpg and marked it.
NOT_BALL lists the crop ids that are not the match ball (crowd, shirts,
spare balls on the scorer's table, shoes). Everything else is the match ball.
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOT_BALL = {0, 2, 3, 5, 6, 7, 9, 10, 29, 35, 38, 39, 50, 52, 65, 69}
SPARE_BALLS_ON_TABLE = {29, 35, 38, 39}       # real balls, but beside the court

s = json.loads((HERE / "ball_check" / "detections.json").read_text())
det = [x for x in s if "id" in x]
ok = [x for x in det if x["id"] not in NOT_BALL]
per = Counter(x["video"] for x in s)
per_ok = Counter(x["video"] for x in ok)
res = {
    "play_frames": len(s), "frames_with_a_ball_box": len(det), "boxes_that_are_the_match_ball": len(ok),
    "precision": len(ok) / len(det),
    "precision_after_court_filter": len(ok) / (len(det) - len(SPARE_BALLS_ON_TABLE)),
    "ball_found_rate_single_frame": len(ok) / len(s),
    "per_video_found": {v: f"{per_ok[v]}/{per[v]}" for v in sorted(per)},
    "note": "single frames, no tracking. In the app, gaps up to 2 s are bridged on a flight arc and balls beside the court are dropped.",
}
(HERE / "ball_check_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
