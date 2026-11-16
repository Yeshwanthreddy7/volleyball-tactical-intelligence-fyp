import { useEffect, useRef, useState } from "react";
import { api, apiUrl } from "../api.js";

// Times like 2:10.4 or 02:10.40 in any answer become links that play the video there.
const TIME = /\b(\d{1,2}):(\d{2}(?:\.\d{1,2})?)\b/g;

function Linked({ text, onSeek }) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(TIME)) {
    out.push(text.slice(last, m.index));
    const sec = +m[1] * 60 + parseFloat(m[2]);
    out.push(<a key={m.index} href="#player" onClick={(e) => { e.preventDefault(); onSeek(sec); }}>{m[0]}</a>);
    last = m.index + m[0].length;
  }
  out.push(text.slice(last));
  return <>{out}</>;
}

function Paragraphs({ text, onSeek }) {
  return (text || "").split("\n").filter((l) => l.trim()).map((l, i) => (
    <p key={i} style={{ margin: "0 0 8px" }}><Linked text={l.replace(/\*\*/g, "")} onSeek={onSeek} /></p>
  ));
}

const QUICK = [
  "Summarise our match for me",
  "Who reacted late and when?",
  "Who drove our attacks?",
  "Where did our spacing break down?",
  "What should we train next?",
];

export default function CoachPanel({ jobId, onSeek }) {
  const [sum, setSum] = useState(null);
  const [busy, setBusy] = useState(false);
  const [chat, setChat] = useState([]);
  const [q, setQ] = useState("");
  const [err, setErr] = useState(null);
  const end = useRef(null);

  useEffect(() => {
    api.coachSummary(jobId, false).then(setSum).catch((e) => setErr(e.message));
  }, [jobId]);
  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }); }, [chat]);

  const aiSummary = () => {
    setBusy(true);
    api.coachSummary(jobId, true).then(setSum).catch((e) => setErr(e.message)).finally(() => setBusy(false));
  };
  const ask = (question) => {
    if (!question.trim() || busy) return;
    const history = chat.map((m) => ({ role: m.role, content: m.text }));
    setChat((c) => [...c, { role: "user", text: question }]);
    setQ("");
    setBusy(true);
    api.coachAsk(jobId, { question, history })
      .then((r) => setChat((c) => [...c, { role: "assistant", text: r.answer, mode: r.mode, note: r.note, sources: r.sources }]))
      .catch((e) => setChat((c) => [...c, { role: "assistant", text: `Error: ${e.message}`, mode: "error" }]))
      .finally(() => setBusy(false));
  };

  return (
    <div className="grid cols-2" style={{ alignItems: "start" }}>
      <div className="card">
        <div className="row">
          <h3 style={{ margin: 0 }}>Coach summary</h3>
          <span className="spacer" />
          {sum && <span className={`pill ${sum.mode === "ai" ? "ok" : ""}`}>{sum.mode === "ai" ? "AI (Claude)" : "Rule-based"}</span>}
        </div>
        {err && <p className="muted">{err}</p>}
        {!sum && !err && <p className="muted">Loading…</p>}
        {sum && <div style={{ marginTop: 12 }}><Paragraphs text={sum.summary} onSeek={onSeek} /></div>}
        {sum?.note && <p className="small faint">{sum.note}</p>}
        {sum && sum.mode !== "ai" && (
          sum.ai_available
            ? <button className="btn sm" disabled={busy} onClick={aiSummary}>{busy ? "Writing…" : "✨ Write AI summary"}</button>
            : <p className="small faint">AI summary: set ANTHROPIC_API_KEY on the server to enable it. The summary above is computed directly from the measurements.</p>
        )}
      </div>

      <div className="card coach-chat">
        <h3 style={{ marginTop: 0 }}>Ask the coach assistant</h3>
        <p className="small muted" style={{ marginTop: 0 }}>
          Answers come from this analysis only (tactic moments, player stats, opponent actions) and a coaching
          knowledge base. Click any time to play the video there.
        </p>
        <div className="chips">
          {QUICK.map((t) => <button key={t} className="chip" disabled={busy} onClick={() => ask(t)}>{t}</button>)}
        </div>
        <div className="chat-log">
          {chat.length === 0 && <p className="faint small">Try “How did player #13 play?” or “Why did we lose shape at 0:27?”</p>}
          {chat.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === "assistant" ? <Paragraphs text={m.text} onSeek={onSeek} /> : <p style={{ margin: 0 }}>{m.text}</p>}
              {m.role === "assistant" && (
                <div className="small faint">
                  {m.mode === "ai" ? "Claude, grounded in this analysis" : m.mode === "rules" ? "Rule-based answer" : ""}
                  {m.sources?.length ? ` · sources: ${m.sources.join(", ")}` : ""}
                  {m.note ? ` · ${m.note}` : ""}
                </div>
              )}
            </div>
          ))}
          {busy && <div className="msg assistant"><p className="faint" style={{ margin: 0 }}>Thinking…</p></div>}
          <div ref={end} />
        </div>
        <form className="row" onSubmit={(e) => { e.preventDefault(); ask(q); }} style={{ marginTop: 10 }}>
          <input className="input" style={{ flex: 1 }} value={q} maxLength={2000} placeholder="Ask about a player, a moment or what to train…"
            onChange={(e) => setQ(e.target.value)} />
          <button className="btn sm" disabled={busy || !q.trim()}>Ask</button>
        </form>
      </div>
    </div>
  );
}

export function PlayersPanel({ jobId, players, segments, onSeek }) {
  if (!players?.length) {
    return <p className="muted">Player analysis is available for analyses made after this feature was added. Upload the video again to see it.</p>;
  }
  const main = [...players].sort((a, b) => b.seconds_on_court - a.seconds_on_court);
  const ts = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  return (
    <div>
      <p className="small muted" style={{ marginTop: 0 }}>
        Player numbers are tracker numbers (not shirt numbers) - use the picture to recognise each player.
        Attacks = drove a Coordinated Attack; Late / Gaps = named in Delayed Support / Spacing Breakdown (less reliable, check in the video).
      </p>
      <div className="grid cols-3">
        {main.map((p) => {
          const mine = segments.filter((g) => (g.fault || []).includes(p.id)
            || ((g.credit || []).includes(p.id) && g.tactic === "Coordinated Attack"));
          return (
            <div key={p.id} className="card player-card">
              <div className="row" style={{ alignItems: "flex-start", gap: 14 }}>
                {p.picture
                  ? <img src={apiUrl(`/api/analyses/${jobId}/players/${p.picture}`)} alt={`Player #${p.id}`} className="player-pic" />
                  : <div className="player-pic empty">#{p.id}</div>}
                <div style={{ flex: 1 }}>
                  <div className="row"><h3 style={{ margin: 0 }}>Player #{p.id}</h3><span className="spacer" />
                    <span className={`pill ${p.impact === "positive" ? "ok" : p.impact === "negative" ? "warn" : ""}`}>{p.impact}</span></div>
                  <div className="small muted">{p.seconds_on_court.toFixed(0)} s on court · {p.distance_m.toFixed(0)} m · top {p.top_speed.toFixed(1)} m/s</div>
                  <div className="kpis">
                    <span><b>{p.attacks_driven}</b> attacks</span><span><b>{p.defences_in}</b> defences</span>
                    <span><b>{p.late_reactions}</b> late</span><span><b>{p.spacing_errors}</b> gaps</span>
                  </div>
                </div>
              </div>
              {mine.length > 0 && (
                <ul className="moments">
                  {mine.slice(0, 5).map((g, i) => (
                    <li key={i}><a href="#player" onClick={(e) => { e.preventDefault(); onSeek(g.start); }}>{ts(g.start)}</a> {g.tactic}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
