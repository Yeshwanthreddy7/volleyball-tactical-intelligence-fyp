import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "NEW_OUTPUTS" / "code"))
sys.path.insert(0, str(ROOT / "webapp" / "backend"))
pa = pytest.importorskip("player_analytics")

FPS = 30.0
BASE = {1: (1.5, 11.0), 2: (4.5, 11.0), 3: (7.5, 11.0), 4: (1.5, 16.0), 5: (4.5, 16.0), 6: (7.5, 16.0)}


def frames(n, move=lambda pid, f: (0.0, 0.0)):
    out = []
    for f in range(n):
        plist = []
        for pid, (x, y) in BASE.items():
            dx, dy = move(pid, f)
            plist.append({"id": pid, "team": 1, "court": np.array([x + dx, y + dy]),
                          "box": [100 * pid, 400, 100 * pid + 40, 520]})
        out.append(plist)
    return out


def test_attack_credits_the_two_fastest_towards_the_net():
    n = 60
    pl = frames(n, lambda pid, f: (0.0, -0.03 * f) if pid in (1, 2) else (0.0, 0.0))
    lab = np.zeros(n, int)                                   # all Coordinated Attack
    segs, stats, best = pa.analyse(pl, lab, FPS, 0, 1, np.ones(n, bool))
    assert segs[0]["tactic"] == "Coordinated Attack"
    assert sorted(segs[0]["credit"]) == [1, 2]
    assert "drove the attack" in segs[0]["why"]
    by = {s["id"]: s for s in stats}
    assert by[1]["attacks_driven"] == 1 and by[4]["attacks_driven"] == 0
    assert set(best) == set(BASE)


def test_delayed_support_names_the_player_who_did_not_move():
    n = 60
    pl = frames(n, lambda pid, f: (0.0, 0.0) if pid == 5 else (0.04 * f, 0.0))
    lab = np.full(n, 2)                                      # Delayed Support
    segs, stats, _ = pa.analyse(pl, lab, FPS, 0, 1, np.ones(n, bool))
    assert segs[0]["fault"] == [5]
    assert {s["id"]: s for s in stats}[5]["late_reactions"] == 1


def test_stitching_joins_a_track_that_continues_after_a_short_break():
    trk = {1: {f: np.array([4.0, 12.0 + 0.01 * f]) for f in range(30)},
           9: {f: np.array([4.1, 12.4]) for f in range(45, 90)}}       # same player, new ID after 0.5 s
    merged, alias = pa.stitch(trk, FPS)
    assert alias[9] == 1 and len(merged) == 1


def test_coach_rule_answers_and_retrieval():
    coach = pytest.importorskip("app.coach")
    hits = coach.search_kb("drills for late reactions")
    assert hits and "Delayed Support" in hits[0]["source"]
    result = {"video": {"analysed_start": 0, "analysed_end": 10},
              "summary": {"tactics": {t: {"segments": 1, "seconds": 2.0} for t in coach.TACTICS},
                          "dominant_tactic": "Coordinated Attack", "opponent_actions": {"SERVE": 1, "SPIKE": 0, "BLOCK": 0}},
              "tactics": {"segments": [{"tactic": "Delayed Support", "start": 2.0, "end": 3.0, "confidence": 0.3,
                                        "why": "Player #5 reacted late", "credit": [], "fault": [5]}]},
              "players": [{"id": 5, "track_ids": [5], "seconds_on_court": 9.0, "distance_m": 10.0, "avg_speed": 1.0,
                           "top_speed": 3.0, "attacks_driven": 0, "defences_in": 0, "late_reactions": 1,
                           "spacing_errors": 0, "defence_out_of_step": 0, "impact_score": -6.7, "impact": "negative"}],
              "events": []}
    assert "Player #5" in coach.rule_summary(result)
    out = coach.rule_answer(result, "How did player #5 play?")
    assert "late" in out["answer"] and "0:02.0" in out["answer"]
