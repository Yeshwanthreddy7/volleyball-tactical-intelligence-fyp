"""Coach assistant: turns one analysis into answers a coach can act on.

Two layers:

* Rule-based (always available, no internet / API key): a plain-English match
  summary built from the measured facts - tactic time, who drove the attacks,
  who reacted late, who opened gaps - plus coaching tips retrieved from the
  knowledge base in coach_kb/.

* AI agent (when ANTHROPIC_API_KEY is set): Claude answers the coach's free
  questions ("why did we lose shape at 2:10?", "how did player #13 play?").
  It is grounded - it can only see this analysis through tools that read the
  result (tactic moments with their explanations, player stats, opponent
  actions) and a retrieval tool over the coaching knowledge base (RAG). It is
  told to cite times and players for every claim.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

KB_DIR = Path(__file__).with_name("coach_kb")
MODEL = os.environ.get("VTA_COACH_MODEL", "claude-opus-5-5")
TACTICS = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
LESS_RELIABLE = ("Delayed Support", "Spacing Breakdown")


def ts(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s // 60):d}:{s % 60:04.1f}"


# ---------------------------------------------------------------- knowledge base (RAG)
@lru_cache(maxsize=1)
def _kb():
    """Chunks = one '##' section of one document, with its source name."""
    chunks = []
    for f in sorted(KB_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        for part in re.split(r"\n(?=## )", text):
            body = part.strip()
            if body and not body.startswith("# "):
                head = body.splitlines()[0].lstrip("# ").strip()
                chunks.append({"source": f"{title} - {head}", "text": body})
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform([c["source"] + " " + c["text"] for c in chunks])
    return chunks, vec, mat


def search_kb(query: str, k: int = 3) -> list[dict]:
    chunks, vec, mat = _kb()
    sims = (mat @ vec.transform([query]).T).toarray().ravel()
    order = sims.argsort()[::-1][:k]
    return [{**chunks[i], "score": round(float(sims[i]), 3)} for i in order if sims[i] > 0]


# ---------------------------------------------------------------- facts from one analysis
def main_players(result: dict, n: int = 6) -> list[dict]:
    return sorted(result.get("players", []), key=lambda p: -p["seconds_on_court"])[:n]


def moments(result: dict, tactic: str = "", start: float = -1, end: float = -1) -> list[dict]:
    out = []
    for g in result["tactics"]["segments"]:
        if tactic and g["tactic"].lower() != tactic.lower():
            continue
        if start >= 0 and g["end"] < start or end >= 0 and g["start"] > end:
            continue
        out.append({"from": ts(g["start"]), "to": ts(g["end"]), "tactic": g["tactic"],
                    "confidence": round(g["confidence"], 2), "why": g.get("why", ""),
                    "credited_players": g.get("credit", []), "players_at_fault": g.get("fault", []),
                    "less_reliable": g["tactic"] in LESS_RELIABLE})
    return out


def player_view(result: dict, pid: int) -> dict | None:
    p = next((p for p in result.get("players", []) if p["id"] == pid or pid in p.get("track_ids", [])), None)
    if not p:
        return None
    mine = [m for m in moments(result) if p["id"] in m["players_at_fault"]
            or (p["id"] in m["credited_players"] and m["tactic"] == "Coordinated Attack")]
    return {**p, "moments": mine}


def overview(result: dict) -> dict:
    s = result["summary"]
    return {"analysed": f"{ts(result['video']['analysed_start'])} - {ts(result['video']['analysed_end'])}",
            "tactics": {t: {"moments": s["tactics"][t]["segments"], "seconds": s["tactics"][t]["seconds"]}
                        for t in TACTICS if t in s["tactics"]},
            "dominant_tactic": s.get("dominant_tactic"),
            "opponent_actions": s.get("opponent_actions"),
            "main_players": [{"id": p["id"], "seconds": p["seconds_on_court"], "impact": p["impact"],
                              "impact_score": p["impact_score"]} for p in main_players(result)]}


# ---------------------------------------------------------------- rule-based summary
def rule_summary(result: dict) -> str:
    s = result["summary"]
    tac = s["tactics"]
    segs = result["tactics"]["segments"]
    L = []
    if not segs:
        return "No team tactic was found in this part of the video (no live play with enough of your players visible)."
    total = sum(tac[t]["seconds"] for t in TACTICS if t in tac) or 1
    share = {t: tac[t]["seconds"] / total for t in TACTICS if t in tac}
    att, dfn = share.get("Coordinated Attack", 0), share.get("Coordinated Defense", 0)
    bad = share.get("Delayed Support", 0) + share.get("Spacing Breakdown", 0)
    L.append(f"Overview: your team spent {att:.0%} of the classified play in Coordinated Attack, {dfn:.0%} in "
             f"Coordinated Defense and {bad:.0%} in problem states (Delayed Support or Spacing Breakdown).")
    if s.get("dominant_tactic"):
        L.append(f"Most common pattern: {s['dominant_tactic']}.")

    players = result.get("players", [])
    if players:
        drivers = sorted([p for p in players if p["attacks_driven"]], key=lambda p: -p["attacks_driven"])[:2]
        if drivers:
            L.append("Attack leaders: " + ", ".join(f"Player #{p['id']} ({p['attacks_driven']} attacks driven)" for p in drivers) + ".")
        late = sorted([p for p in players if p["late_reactions"]], key=lambda p: -p["late_reactions"])[:2]
        if late:
            L.append("Late reactions: " + ", ".join(f"Player #{p['id']} ({p['late_reactions']}x)" for p in late)
                     + " - check ready position and first step.")
        gaps = sorted([p for p in players if p["spacing_errors"]], key=lambda p: -p["spacing_errors"])[:2]
        if gaps:
            L.append("Spacing problems: " + ", ".join(f"Player #{p['id']} ({p['spacing_errors']}x)" for p in gaps)
                     + " - check return to base after each touch.")
        busiest = max(players, key=lambda p: p["distance_m"])
        L.append(f"Most work rate: Player #{busiest['id']} covered {busiest['distance_m']:.0f} m "
                 f"(top speed {busiest['top_speed']:.1f} m/s).")

    worst = [g for g in segs if g["tactic"] in LESS_RELIABLE and g.get("why")][:3]
    if worst:
        L.append("Moments to review: " + "; ".join(f"{ts(g['start'])} {g['tactic']} - {g['why']}" for g in worst))

    focus = max(LESS_RELIABLE, key=lambda t: tac.get(t, {}).get("seconds", 0))
    if tac.get(focus, {}).get("seconds", 0) > 0:
        tips = search_kb(f"{focus} coaching drills", k=1)
        if tips:
            drill = next((ln for ln in tips[0]["text"].splitlines() if ln.startswith("- ")), "")
            L.append(f"Training focus: {focus}. Suggested drill ({tips[0]['source']}): {drill[2:]}")
    opp = s.get("opponent_actions") or {}
    if sum(opp.values()):
        L.append(f"Opponent: {opp.get('SERVE', 0)} serves, {opp.get('SPIKE', 0)} spikes, {opp.get('BLOCK', 0)} blocks detected.")
    L.append("Note: Delayed Support and Spacing Breakdown calls are less reliable; check them in the video. "
             "Player numbers are tracker numbers, not shirt numbers.")
    return "\n".join(L)


def rule_answer(result: dict, question: str) -> dict:
    """Keyword-routed answer when no AI is available."""
    q = question.lower()
    m = re.search(r"(?:player|#)\s*#?\s*(\d+)", q)
    if m:
        p = player_view(result, int(m.group(1)))
        if not p:
            return {"answer": f"Player #{m.group(1)} was not one of your tracked players in this analysis.", "sources": []}
        lines = [f"Player #{p['id']}: {p['seconds_on_court']:.0f} s on court, {p['distance_m']:.0f} m covered, "
                 f"top speed {p['top_speed']:.1f} m/s. Drove {p['attacks_driven']} attacks, part of "
                 f"{p['defences_in']} defences, {p['late_reactions']} late reactions, {p['spacing_errors']} spacing errors "
                 f"(impact: {p['impact']})."]
        lines += [f"{mm['from']} {mm['tactic']}: {mm['why']}" for mm in p["moments"][:6]]
        return {"answer": "\n".join(lines), "sources": []}
    if re.search(r"summar|overall|how did (we|the team)|match report", q):
        return {"answer": rule_summary(result), "sources": []}
    if re.search(r"\b(train|drill|practi[cs]e|improve|work on|next)\b", q):
        tac = result["summary"]["tactics"]
        focus = max(TACTICS[2:], key=lambda t: tac.get(t, {}).get("seconds", 0))
        tips = search_kb(f"{focus} coaching drills", k=1)
        lines = [f"Biggest problem in this clip: {focus} ({tac.get(focus, {}).get('segments', 0)} moments)."]
        lines += [f"{m['from']}: {m['why']}" for m in moments(result, focus)[:4]]
        if tips:
            lines += [f"Drills ({tips[0]['source']}):"] + [ln[2:] for ln in tips[0]["text"].splitlines() if ln.startswith("- ")]
        return {"answer": "\n".join(lines), "sources": [t["source"] for t in tips]}
    keys = {"Delayed Support": r"late|react|slow|support", "Spacing Breakdown": r"gap|spac|shape|crowd|isolat",
            "Coordinated Attack": r"attack|drove|spike|hitter", "Coordinated Defense": r"defen|block"}
    for t in TACTICS:
        if t.lower() in q or re.search(keys[t], q):
            ms = moments(result, t)
            tips = search_kb(f"{t} {question}", k=2)
            lines = [f"{len(ms)} {t} moments."] + [f"{x['from']}: {x['why']}" for x in ms[:6]]
            lines += [f"Tip ({k['source']}): " + " ".join(k["text"].splitlines()[1:4]) for k in tips[:1]]
            return {"answer": "\n".join(lines), "sources": [k["source"] for k in tips]}
    tips = search_kb(question, k=2)
    if tips and tips[0]["score"] > 0.15:
        return {"answer": "\n".join(k["text"] for k in tips[:1]), "sources": [k["source"] for k in tips[:1]]}
    return {"answer": rule_summary(result), "sources": []}


# ---------------------------------------------------------------- AI agent (Claude)
SYSTEM = """You are the coaching assistant of a volleyball tactical-analysis app. A coach is asking about ONE analysed match clip of their own team (the near-side team).

Ground every statement in the tools: call them to read tactic moments (with the measured explanation of who made each happen), player statistics, opponent actions, and the coaching knowledge base. Never invent players, times or numbers.

Style:
- Short, practical answers a coach can act on. Plain English.
- Cite the video time (m:ss.s) and the player (Player #N) for every claim, so the coach can click and check.
- When you recommend training, take it from search_coaching_knowledge and name the source.
- Delayed Support and Spacing Breakdown calls are less reliable (about 1 in 4 correct on unseen videos); say so when they matter to your answer.
- Player numbers are tracker numbers, not shirt numbers; the tracker can occasionally swap two players who cross.
- If the data cannot answer the question, say so."""


def ai_available() -> bool:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _tools(result: dict):
    from anthropic import beta_tool

    @beta_tool
    def get_match_overview() -> str:
        """Overview of the analysed clip: time per tactic, dominant tactic, opponent action counts, main players with impact."""
        return json.dumps(overview(result))

    @beta_tool
    def list_tactic_moments(tactic: str = "", from_seconds: float = -1, to_seconds: float = -1) -> str:
        """List the team's tactic moments with time, confidence and the measured explanation (who drove it, who was at fault).

        Args:
            tactic: Optional filter: Coordinated Attack, Coordinated Defense, Delayed Support or Spacing Breakdown. Empty = all.
            from_seconds: Optional start of a time window in seconds of the original video (-1 = no limit).
            to_seconds: Optional end of a time window in seconds of the original video (-1 = no limit).
        """
        return json.dumps(moments(result, tactic, from_seconds, to_seconds)[:60])

    @beta_tool
    def list_players() -> str:
        """Player-wise table: time on court, distance, speed, attacks driven, defences, late reactions, spacing errors, impact."""
        return json.dumps(result.get("players", []))

    @beta_tool
    def get_player(player_id: int) -> str:
        """One player's statistics and every tactic moment they were credited or blamed for.

        Args:
            player_id: Tracker number of the player (as in "Player #7").
        """
        p = player_view(result, player_id)
        return json.dumps(p) if p else f"Player #{player_id} is not one of the tracked players in this analysis."

    @beta_tool
    def get_opponent_actions() -> str:
        """Opponent serves, spikes and blocks with their times (seconds of the original video)."""
        return json.dumps([{"time": ts(e["time"]), "action": e["action"]} for e in result.get("events", [])])

    @beta_tool
    def search_coaching_knowledge(query: str) -> str:
        """Search the coaching knowledge base (tactic definitions, causes of faults, training drills, how to read the report).

        Args:
            query: What to look for, e.g. "drills for late reactions".
        """
        return json.dumps(search_kb(query, k=3))

    return [get_match_overview, list_tactic_moments, list_players, get_player,
            get_opponent_actions, search_coaching_knowledge]


def ai_answer(result: dict, question: str, history: list[dict] | None = None) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-8:]
                if m.get("role") in ("user", "assistant") and m.get("content")]
    messages.append({"role": "user", "content": question})
    params = dict(model=MODEL, max_tokens=16000, system=SYSTEM, tools=_tools(result), messages=messages,
                  output_config={"effort": "medium"}, max_iterations=8)
    try:
        runner = client.beta.messages.tool_runner(**params, betas=["server-side-fallback-2026-07-01"],
                                                  fallbacks="default")
        final = runner.until_done()
    except anthropic.BadRequestError:
        final = client.beta.messages.tool_runner(**params).until_done()
    if final.stop_reason == "refusal":
        return {"answer": "The AI assistant declined this question. Try rephrasing it.", "sources": [], "mode": "ai"}
    text = "\n".join(b.text for b in final.content if b.type == "text").strip()
    return {"answer": text or "No answer was produced.", "sources": [], "mode": "ai"}


def answer(result: dict, question: str, history: list[dict] | None = None) -> dict:
    question = (question or "").strip()[:2000]
    if not question:
        return {"answer": "Ask a question about this analysis.", "sources": [], "mode": "rules"}
    if ai_available():
        try:
            return ai_answer(result, question, history)
        except Exception as exc:  # noqa: BLE001 - fall back to the rules, but say why
            out = rule_answer(result, question)
            out["note"] = f"AI assistant unavailable ({type(exc).__name__}); showing the rule-based answer."
            out["mode"] = "rules"
            return out
    out = rule_answer(result, question)
    out["mode"] = "rules"
    return out


def summary(result: dict, use_ai: bool = False) -> dict:
    if use_ai and ai_available():
        try:
            r = ai_answer(result, "Write a match summary for the coach: what worked, what went wrong and who was "
                                  "involved (with times), each player's impact, and the two most useful training "
                                  "drills for this team. Under 250 words, short sections.")
            return {"summary": r["answer"], "mode": "ai"}
        except Exception as exc:  # noqa: BLE001
            return {"summary": rule_summary(result), "mode": "rules",
                    "note": f"AI summary unavailable ({type(exc).__name__})."}
    return {"summary": rule_summary(result), "mode": "rules"}
