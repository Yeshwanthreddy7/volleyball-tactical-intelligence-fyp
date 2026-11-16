"""Player-wise performance and responsibility for each team-tactic moment.

The tactic model (team_tactics.py) names WHAT the team did each second. This
module explains WHO made it happen, using the same tracked court positions
(metres, X 0..9 across, Y 0..18 far->near, net at Y = 9, our half Y > 9):

  Coordinated Attack  : the two players moving fastest towards the net drove it
                        (the HGB feature speed_top2 vs speed_bot4, made concrete).
  Coordinated Defense : the team shifted as one unit; the player least in step
                        with the team's movement is named as the weak link.
  Delayed Support     : the player who moved least while the team moved is named
                        as the late reactor.
  Spacing Breakdown   : the player who left the biggest gap to the nearest
                        team-mate, or the two players standing too close.

Per player over the whole analysis: time on court, distance covered, top and
average speed, how often they drove an attack / were part of a defence, how
often they were named for a late reaction or a spacing error, and an impact
score (+1 per credit, -1 per fault, per minute on court).

Player names are tracker IDs ("Player #7"). The tracker can occasionally swap
IDs between two players who cross, so each player also gets a picture (see
best_frames) for the coach to recognise them.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

CLASSES = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
MIN_TRACK_S = 1.0          # ignore IDs seen for less than this (tracker noise)
MAX_SPEED_MS = 9.0         # faster than an elite sprint = a tracking glitch, not running
GAP_TOO_BIG_M = 4.0        # nearest team-mate further than this = a hole in the formation
TOO_CLOSE_M = 1.0          # two players closer than this = crowding
STITCH_GAP_S = 3.0         # a new track starting this soon after one ended may be the same player
STITCH_BASE_M = 1.5        # ...if it starts within this distance (+ RUN_MS per second of gap)
RUN_MS = 3.0
STATIC_MS = 0.3            # people who barely move (line judges inside the margin) are not players


def _in_our_half(q, margin=0.5):
    X, Y = float(q["court"][0]), float(q["court"][1])
    return -margin <= X <= 9 + margin and 9 - margin <= Y <= 18 + margin


def tracks(players, team, cam_ok):
    """{id: {frame: court_xy}} for our in-court players on normal camera frames."""
    out = defaultdict(dict)
    for f, plist in enumerate(players):
        if not cam_ok[f]:
            continue
        for q in plist:
            if q["team"] == team and q["id"] >= 0 and _in_our_half(q):
                out[q["id"]][f] = np.asarray(q["court"], np.float64)
    return out


def stitch(trk, fps):
    """Join track fragments of the same player: the tracker gives a new ID after an
    occlusion or camera cut. A fragment that starts shortly after another ended, near
    where it ended, and does not overlap it in time, continues that player.
    Returns (merged {pid: {frame: xy}}, alias {track_id: pid})."""
    order = sorted(trk, key=lambda i: min(trk[i]))
    chains, alias = {}, {}
    for i in order:
        f0 = min(trk[i]); p0 = trk[i][f0]
        best, best_d = None, None
        for pid, ch in chains.items():
            last = max(ch)
            gap = (f0 - last) / fps
            if gap <= 0 or gap > STITCH_GAP_S:
                continue
            d = float(np.linalg.norm(ch[last] - p0))
            if d <= STITCH_BASE_M + RUN_MS * gap and (best_d is None or d < best_d):
                best, best_d = pid, d
        if best is None:
            chains[i] = dict(trk[i]); alias[i] = i
        else:
            chains[best].update(trk[i]); alias[i] = best
    return chains, alias


def _speeds(tr, frames, fps):
    """Per-frame speed (m/s) of one track over the given frames (gaps skipped)."""
    v = []
    for f in frames:
        if f in tr and f - 1 in tr:
            step = float(np.linalg.norm(tr[f] - tr[f - 1]))
            if step * fps <= MAX_SPEED_MS:
                v.append(step * fps)
    return v


def _pos(tr, a, b):
    """Mean position of a track inside [a, b], or None."""
    pts = [tr[f] for f in range(a, b + 1) if f in tr]
    return np.mean(pts, 0) if pts else None


def explain_segment(seg, trk, fps):
    """Who made this tactic moment happen. Returns dict(why, credit, fault, players)."""
    a, b, tactic = seg["a"], seg["b"], seg["tactic"]
    ids = [i for i, tr in trk.items() if sum(1 for f in range(a, b + 1) if f in tr) >= 0.5 * (b - a + 1)]
    if len(ids) < 3:
        return dict(why="Too few of our players were tracked here to say who was responsible.",
                    credit=[], fault=[], players=ids)
    frames = range(a + 1, b + 1)
    speed = {i: (float(np.mean(s)) if (s := _speeds(trk[i], frames, fps)) else 0.0) for i in ids}
    start = {i: _pos(trk[i], a, min(b, a + int(0.2 * fps))) for i in ids}
    end = {i: _pos(trk[i], max(a, b - int(0.2 * fps)), b) for i in ids}
    move = {i: (end[i] - start[i]) if start[i] is not None and end[i] is not None else np.zeros(2) for i in ids}
    team_speed = float(np.mean(list(speed.values())))

    if tactic == "Coordinated Attack":
        # towards the net = Y decreasing in our half
        net_ward = {i: speed[i] * (1.0 if move[i][1] < 0 else 0.5) for i in ids}
        top = sorted(ids, key=lambda i: -net_ward[i])[:2]
        rest = [speed[i] for i in ids if i not in top]
        why = (f"Players #{top[0]} and #{top[1]} drove the attack, moving {speed[top[0]]:.1f} and "
               f"{speed[top[1]]:.1f} m/s towards the net while the other {len(rest)} held their base "
               f"positions ({np.mean(rest):.1f} m/s on average).")
        return dict(why=why, credit=top, fault=[], players=ids)

    if tactic == "Coordinated Defense":
        team_move = np.mean([move[i] for i in ids], 0)
        off = {i: float(np.linalg.norm(move[i] - team_move)) for i in ids}
        worst = max(ids, key=lambda i: off[i])
        why = (f"The team shifted together as one unit ({np.linalg.norm(team_move):.1f} m as a block). "
               f"Player #{worst} was the least in step ({off[worst]:.1f} m away from the team's movement).")
        fault = [worst] if off[worst] > 1.5 else []
        return dict(why=why, credit=[i for i in ids if i not in fault], fault=fault, players=ids)

    if tactic == "Delayed Support":
        late = min(ids, key=lambda i: speed[i])
        why = (f"Player #{late} reacted late: moving {speed[late]:.1f} m/s while the team averaged "
               f"{team_speed:.1f} m/s, so support arrived late.")
        return dict(why=why, credit=[], fault=[late], players=ids)

    # Spacing Breakdown
    mid = (a + b) // 2
    pos = {i: _pos(trk[i], max(a, mid - 3), min(b, mid + 3)) for i in ids}
    pos = {i: p for i, p in pos.items() if p is not None}
    if len(pos) < 3:
        return dict(why="Too few positions to measure the spacing.", credit=[], fault=[], players=ids)
    keys = list(pos)
    D = np.array([[np.linalg.norm(pos[i] - pos[j]) for j in keys] for i in keys])
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    far = keys[int(nn.argmax())]
    i0, j0 = np.unravel_index(int(D.argmin()), D.shape)
    gap_excess = nn.max() - GAP_TOO_BIG_M
    crowd_excess = TOO_CLOSE_M - D.min()
    if gap_excess >= crowd_excess:
        why = (f"Player #{far} was isolated: {nn.max():.1f} m from the nearest team-mate, "
               f"leaving a hole in the formation.")
        fault = [far]
    else:
        a_id, b_id = keys[i0], keys[j0]
        why = (f"Players #{a_id} and #{b_id} crowded each other ({D.min():.1f} m apart), "
               f"leaving space uncovered elsewhere.")
        fault = [a_id, b_id]
    return dict(why=why, credit=[], fault=fault, players=ids)


def analyse(players, lab, fps, start, team, cam_ok):
    """Returns (segments_with_explanations, player_stats, best_frames).

    lab: per-frame tactic index (-1 = no call), aligned with `players`.
    best_frames: {id: (frame, box)} - the frame with each player's largest box,
    used for the player's picture."""
    raw = tracks(players, team, cam_ok)
    trk, alias = stitch(raw, fps)
    min_frames = int(MIN_TRACK_S * fps)
    trk = {i: tr for i, tr in trk.items() if len(tr) >= min_frames}
    # drop people who barely move over a long time (officials standing inside the margin)
    for i in list(trk):
        sp = _speeds(trk[i], sorted(trk[i]), fps)
        if len(trk[i]) > 3 * fps and sp and float(np.mean(sp)) < STATIC_MS:
            del trk[i]

    # run-length segments of the tactic calls
    segs, s0 = [], 0
    for f in range(1, len(lab) + 1):
        if f == len(lab) or lab[f] != lab[s0]:
            if lab[s0] >= 0:
                segs.append(dict(a=s0, b=f - 1, tactic=CLASSES[int(lab[s0])]))
            s0 = f
    for s in segs:
        s.update(explain_segment(s, trk, fps))
        s["start_s"] = round((start + s["a"]) / fps, 2)
        s["end_s"] = round((start + s["b"] + 1) / fps, 2)

    stats = {}
    for i, tr in trk.items():
        fr = sorted(tr)
        sp = _speeds(tr, fr, fps)
        dist = sum(float(np.linalg.norm(tr[f] - tr[f - 1]))
                   for f in fr if f - 1 in tr and np.linalg.norm(tr[f] - tr[f - 1]) * fps <= MAX_SPEED_MS)
        st = dict(id=int(i), track_ids=sorted(int(k) for k, v in alias.items() if v == i), seconds_on_court=round(len(fr) / fps, 1), distance_m=round(dist, 1),
                  avg_speed=round(float(np.mean(sp)), 2) if sp else 0.0,
                  top_speed=round(float(np.percentile(sp, 95)), 2) if sp else 0.0,
                  attacks_driven=0, defences_in=0, late_reactions=0, spacing_errors=0, defence_out_of_step=0)
        stats[int(i)] = st
    for s in segs:
        for i in s["credit"]:
            if i in stats:
                stats[i]["attacks_driven" if s["tactic"] == "Coordinated Attack" else "defences_in"] += 1
        for i in s["fault"]:
            if i in stats:
                key = {"Delayed Support": "late_reactions", "Spacing Breakdown": "spacing_errors",
                       "Coordinated Defense": "defence_out_of_step"}.get(s["tactic"])
                if key:
                    stats[i][key] += 1
    for st in stats.values():
        plus = st["attacks_driven"] + 0.5 * st["defences_in"]
        minus = st["late_reactions"] + st["spacing_errors"] + 0.5 * st["defence_out_of_step"]
        minutes = max(st["seconds_on_court"] / 60.0, 0.25)
        st["impact_score"] = round((plus - minus) / minutes, 2)
        st["impact"] = "positive" if plus > minus else ("negative" if minus > plus else "neutral")

    best = {}
    for f, plist in enumerate(players):
        if not cam_ok[f]:
            continue
        for q in plist:
            if q["team"] != team or int(q["id"]) not in alias:
                continue
            i = int(alias[int(q["id"])])
            if i in stats:
                x0, y0, x1, y1 = [float(v) for v in q["box"]]
                area = (x1 - x0) * (y1 - y0)
                if i not in best or area > best[i][2]:
                    best[i] = (f, (x0, y0, x1, y1), area)
    best = {i: (f, box) for i, (f, box, _) in best.items()}
    for s in segs:
        s.pop("a"); s.pop("b")
    ranked = sorted(stats.values(), key=lambda s: -s["seconds_on_court"])
    return segs, ranked, best
