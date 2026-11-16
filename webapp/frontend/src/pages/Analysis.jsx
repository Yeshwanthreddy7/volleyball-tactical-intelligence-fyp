import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, apiUrl, fileUrl, trackJob } from "../api.js";
import { TACTICS, TACTIC_HEX } from "../content.js";
import VideoPlayer from "../components/VideoPlayer.jsx";
import Notepad from "../components/Notepad.jsx";
import CoachPanel, { PlayersPanel } from "../components/CoachPanel.jsx";

const STAGES = [
  ["queued", "Waiting in queue"],
  ["calibrating", "Court calibration"],
  ["detecting", "Player and ball detection"],
  ["analyzing", "Opponent actions and team tactics"],
  ["rendering", "Rendering annotated video"],
  ["encoding", "Encoding MP4 qualities"],
  ["reporting", "Writing reports"],
];
const STAGE_INDEX = { queued: 0, starting: 1, probing: 1, calibrating: 1, detecting: 2, analyzing: 3, rendering: 4, encoding: 5, reporting: 6, done: 7 };

export function ts(sec) {
  sec = Math.max(0, sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = (sec % 60).toFixed(2).padStart(5, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${String(m).padStart(2, "0")}:${s}`;
}

export function StatusPill({ status }) {
  const cls = { done: "ok", failed: "err", cancelled: "warn", processing: "info", queued: "" }[status] ?? "";
  return <span className={`pill ${cls}`}>{status}</span>;
}

function Progress({ job, onCancel }) {
  const idx = STAGE_INDEX[job.stage] ?? 0;
  const [eta, setEta] = useState(null);
  const hist = useRef([]);
  useEffect(() => {
    // ETA from the progress rate over the last few minutes
    const now = Date.now();
    hist.current = [...hist.current.filter((p) => now - p.t < 240000), { t: now, p: job.progress || 0 }];
    const a = hist.current[0], b = hist.current[hist.current.length - 1];
    if (job.status === "processing" && b.p - a.p > 0.5 && b.t - a.t > 20000) {
      setEta(((100 - b.p) * (b.t - a.t)) / (b.p - a.p) / 1000);
    }
  }, [job.progress, job.status]);

  return (
    <div className="card">
      <div className="row">
        <h3 style={{ margin: 0 }}>{job.status === "queued" ? "Queued" : "Analysing your video"}</h3>
        <span className="spacer" />
        <StatusPill status={job.status} />
      </div>
      <p className="muted" style={{ marginTop: 8 }}>
        {job.message}
        {job.status === "queued" && job.queue_position ? ` - position ${job.queue_position} in the queue` : ""}
      </p>
      <div className="row small" style={{ marginBottom: 6 }}>
        <b>{(job.progress || 0).toFixed(0)}%</b>
        <span className="spacer" />
        {eta !== null && <span className="muted">about {Math.max(1, Math.round(eta / 60))} min left</span>}
      </div>
      <div className="bar"><span style={{ width: `${job.progress || 0}%` }} /></div>
      <div className="stages">
        {STAGES.map(([key, label], i) => (
          <div key={key} className={`stage ${i < idx ? "done" : i === idx ? "active" : ""}`}>
            <span className="ic">{i < idx ? "✓" : ""}</span>{label}
          </div>
        ))}
      </div>
      <p className="small faint" style={{ marginTop: 18 }}>
        Keep this website open (any page of it) until the analysis finishes. If you close the website, the
        analysis is cancelled automatically after a few minutes so it does not hold up other users.
      </p>
      <button className="btn danger sm" onClick={onCancel}>Cancel analysis</button>
    </div>
  );
}

function Timeline({ result, duration, current, onSeek }) {
  const off = result.video.analysed_start;
  const len = duration || result.video.analysed_end - off || 1;
  const pos = (t) => `${(100 * (t - off)) / len}%`;
  const click = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(off + ((e.clientX - r.left) / r.width) * len);
  };
  return (
    <div>
      <div className="timeline" onClick={click} title="Click to jump">
        {result.tactics.segments.map((s, i) => (
          <div key={i} className="tseg" title={`${s.tactic} ${ts(s.start)}-${ts(s.end)}`}
            style={{ left: pos(s.start), width: `${(100 * (s.end - s.start)) / len}%`, background: TACTIC_HEX[s.tactic] }} />
        ))}
        <div className="head" style={{ left: pos(off + current) }} />
      </div>
      <div className="legend">
        {SHOWN.map((t) => <span key={t.name}><i className="dot" style={{ background: t.hex }} />{t.name}</span>)}
      </div>
    </div>
  );
}

// The four team tactics - the same moments captioned in the annotated video.
const SHOWN_NAMES = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"];
const SHOWN = TACTICS.filter((t) => SHOWN_NAMES.includes(t.name));
// right ~26% / ~29% of the time on unseen videos (Attack / Defense ~85%)
const LESS_RELIABLE = ["Delayed Support", "Spacing Breakdown"];

function AllActions({ segments, events, onSeek }) {
  const rows = useMemo(() => [
    ...segments.map((g) => ({ time: g.start, end: g.end, name: g.tactic, team: "Your team", hex: TACTIC_HEX[g.tactic], why: g.why })),
    ...(events || []).map((e) => ({ time: e.time, end: null, name: e.action, team: "Opponent (report only)", hex: "#c0392b" })),
  ].sort((a, b) => a.time - b.time), [segments, events]);
  return (
    <div className="table-wrap" style={{ maxHeight: 520, overflowY: "auto" }}>
      <table>
        <thead><tr><th>#</th><th>Time</th><th>Team</th><th>Action</th><th /></tr></thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={5} className="muted">No clear Coordinated Attack or Coordinated Defense was found.</td></tr>}
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="num">{i + 1}</td>
              <td className="num"><a href="#player" onClick={(ev) => { ev.preventDefault(); onSeek(r.time); }}>{ts(r.time)}{r.end != null && ` - ${ts(r.end)}`}</a></td>
              <td>{r.team}</td>
              <td><span className="chip-tactic"><i className="dot" style={{ background: r.hex }} />{r.name}</span>
                {LESS_RELIABLE.includes(r.name) && <span className="small faint"> · less reliable, check by eye</span>}
                {r.why && <div className="small muted">{r.why}</div>}</td>
              <td><button className="btn ghost sm" onClick={() => onSeek(r.time)}>▶ Play</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TacticTable({ segments, onSeek }) {
  return (
    <div className="table-wrap" style={{ maxHeight: 520, overflowY: "auto" }}>
      <table>
        <thead><tr><th>#</th><th>From</th><th>To</th><th>Length</th><th>Tactic</th><th>Confidence</th><th /></tr></thead>
        <tbody>
          {segments.length === 0 && <tr><td colSpan={7} className="muted">No clear Coordinated Attack or Coordinated Defense was found.</td></tr>}
          {segments.map((g, i) => (
            <tr key={i}>
              <td className="num">{i + 1}</td>
              <td className="num"><a href="#player" onClick={(ev) => { ev.preventDefault(); onSeek(g.start); }}>{ts(g.start)}</a></td>
              <td className="num">{ts(g.end)}</td>
              <td className="num">{(g.end - g.start).toFixed(1)} s</td>
              <td><span className="chip-tactic"><i className="dot" style={{ background: TACTIC_HEX[g.tactic] }} />{g.tactic}</span></td>
              <td className="num">{(g.confidence * 100).toFixed(0)}%</td>
              <td><button className="btn ghost sm" onClick={() => onSeek(g.start)}>▶ Play</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Results({ job, onDelete }) {
  // older analyses also hold the other tactics and opponent actions: hide them
  const r = useMemo(() => ({ ...job.result, tactics: { ...job.result.tactics,
    segments: job.result.tactics.segments.filter((g) => SHOWN_NAMES.includes(g.tactic)) } }), [job.result]);
  const player = useRef(null);
  const [tab, setTab] = useState("coach");
  const [cur, setCur] = useState(0);
  const off = r.video.analysed_start;
  const dur = r.video.analysed_end - off;
  const seek = (t) => player.current?.seek(t - off);
  const s = r.summary;
  const renditions = r.files;
  const tacticRows = useMemo(() => SHOWN.map((t) => ({ ...t, ...s.tactics[t.name] })), [s]);

  return (
    <>
      <div className="grid cols-4" style={{ marginBottom: 22 }}>
        {SHOWN.map((t) => (
          <div key={t.name} className="stat">
            <div className="v" style={{ color: t.hex }}>{s.tactics[t.name]?.segments ?? 0}</div>
            <div className="l">{t.name.toLowerCase()} moments ({(s.tactics[t.name]?.seconds ?? 0).toFixed(0)} s)</div>
          </div>
        ))}
      </div>

      <div id="player">
        <VideoPlayer ref={player} renditions={renditions} duration={dur}
          srcFor={(k) => fileUrl(job.id, renditions[k].file)}
          poster={fileUrl(job.id, "poster.jpg")}
          onTime={setCur}
          onDownload={(k) => fileUrl(job.id, renditions[k].file, true)} />
      </div>
      <Timeline result={r} duration={dur} current={cur} onSeek={seek} />

      <div className="card" style={{ marginTop: 22 }}>
        <h3>Your team's tactics <span className="faint small">(near side, blue · dominant: {s.dominant_tactic || "-"})</span></h3>
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead><tr><th>Tactic</th><th>Segments</th><th>Time</th><th>Share of rally time</th><th>Avg. confidence</th></tr></thead>
            <tbody>
              {tacticRows.map((t) => (
                <tr key={t.name}>
                  <td><span className="chip-tactic"><i className="dot" style={{ background: t.hex }} />{t.name}</span></td>
                  <td className="num">{t.segments}</td>
                  <td className="num">{t.seconds.toFixed(1)} s</td>
                  <td>
                    <div className="meter">
                      <div className="track"><div className="fill" style={{ width: `${t.share * 100}%`, background: t.hex }} /></div>
                      <span className="num small">{(t.share * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="num">{t.segments ? `${(t.avg_confidence * 100).toFixed(0)}%` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginTop: 28 }}>
        <div className="tabs">
          {[["coach", "🧠 Coach assistant"], ["players", "👥 Players"], ["all", "All tactic moments"], ["tactical", "📄 Notepad: team tactics"], ["playersnote", "📄 Notepad: players"], ["actions", "📄 Notepad: opponent actions"], ["segments", "Segments with confidence"], ["court", "Court calibration"]].map(([k, l]) => (
            <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
        {tab === "all" && <AllActions segments={r.tactics.segments} events={r.events} onSeek={seek} />}
        {tab === "tactical" && (
          <Notepad title="tactical_report.txt" url={fileUrl(job.id, "tactical_report.txt")} onSeek={seek}
            downloads={[
              { label: "⬇ tactical_report.txt", href: fileUrl(job.id, "tactical_report.txt", true), primary: true },
              { label: "CSV", href: fileUrl(job.id, "tactical_report.csv", true) },
              { label: "JSON", href: fileUrl(job.id, "tactical_report.json", true) },
            ]} />
        )}
        {tab === "coach" && <CoachPanel jobId={job.id} onSeek={seek} />}
        {tab === "players" && <PlayersPanel jobId={job.id} players={r.players} segments={r.tactics.segments} onSeek={seek} />}
        {tab === "playersnote" && (
          <Notepad title="player_report.txt" url={fileUrl(job.id, "player_report.txt")} onSeek={seek}
            downloads={[{ label: "⬇ player_report.txt", href: fileUrl(job.id, "player_report.txt", true), primary: true }]} />
        )}
        {tab === "actions" && (
          <Notepad title="opponent_report.txt" url={fileUrl(job.id, "opponent_report.txt")} onSeek={seek}
            downloads={[
              { label: "⬇ opponent_report.txt", href: fileUrl(job.id, "opponent_report.txt", true), primary: true },
              { label: "CSV", href: fileUrl(job.id, "opponent_report.csv", true) },
            ]} />
        )}
        {tab === "segments" && <TacticTable segments={r.tactics.segments} onSeek={seek} />}
        {tab === "court" && (
          <div className="card">
            <p>
              {r.court.method === "auto"
                ? <>Court found automatically from the lines: <b>{r.court.lines.join(", ")}</b>{r.court.fit_rmse_px != null && ` (fit error ${r.court.fit_rmse_px.toFixed(2)} px)`}.</>
                : r.court.method === "known venue"
                  ? <>This venue matches a court already verified in the project dataset ({r.court.venue}), so its checked court lines were used: <b>{r.court.lines.join(", ")}</b>.</>
                  : "The court lines could not be found in this video, so a generic end-court model was used. Positions and tactics may be less accurate."}
            </p>
            {r.court.annotated && <img src={apiUrl(`/api/analyses/${job.id}/court.png`)} alt="Detected court lines on the empty-court base frame" style={{ borderRadius: 10 }} />}
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 22 }}>
        <h3>Details</h3>
        <dl className="kv">
          <dt>Video</dt><dd>{job.original_name} · {r.video.width}×{r.video.height} · {r.video.fps.toFixed(2)} fps</dd>
          <dt>Analysed part</dt><dd>{ts(r.video.analysed_start)} - {ts(r.video.analysed_end)} ({r.video.analysed_frames.toLocaleString()} frames)</dd>
          <dt>Your team</dt><dd>near side (bottom of the picture) · opponent: far side</dd>
          <dt>Mode</dt><dd>{r.mode === "accurate" ? "Accurate (1280 px)" : "Fast (960 px)"}</dd>
          <dt>Tactical model</dt><dd>{r.tactics.model}</dd>
        </dl>
        <div className="row" style={{ marginTop: 16 }}>
          <Link className="btn secondary sm" to="/analyze">Analyse another video</Link>
          <button className="btn danger sm" onClick={onDelete}>Delete this analysis</button>
        </div>
      </div>
    </>
  );
}

export default function Analysis() {
  const { id } = useParams();
  const nav = useNavigate();
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true, timer;
    const tick = async () => {
      try {
        const j = await api.analysis(id);
        if (!alive) return;
        setJob(j);
        if (j.status === "queued" || j.status === "processing") {
          trackJob(j.id); // this page keeps the analysis alive while it is open
          timer = setTimeout(tick, 2000);
        }
      } catch (e) {
        if (alive) setErr(e.message);
      }
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [id]);

  const cancel = async () => {
    if (!confirm("Cancel this analysis?")) return;
    try { setJob(await api.cancel(id)); } catch (e) { alert(e.message); }
  };
  const del = async () => {
    if (!confirm("Delete this analysis and all its files? This cannot be undone.")) return;
    try { await api.remove(id); nav("/my-analyses"); } catch (e) { alert(e.message); }
  };

  if (err) return <div className="container section empty"><h2>Analysis not found</h2><p>{err}</p><Link className="btn" to="/analyze">Upload a video</Link></div>;
  if (!job) return <div className="container section empty"><span className="spinner" /></div>;

  return (
    <div className="container section">
      <div className="eyebrow">Analysis</div>
      <div className="row" style={{ marginBottom: 22 }}>
        <h2 style={{ margin: 0, overflowWrap: "anywhere" }}>{job.original_name}</h2>
        <StatusPill status={job.status} />
      </div>
      {(job.status === "queued" || job.status === "processing") && <Progress job={job} onCancel={cancel} />}
      {job.status === "failed" && (
        <div className="card">
          <div className="alert err">{job.message || "The analysis failed."}</div>
          <p className="muted" style={{ marginTop: 14 }}>Check that the file is a volleyball match filmed from behind a baseline, then try again.</p>
          <div className="row"><Link className="btn" to="/analyze">Try another video</Link><button className="btn danger sm" onClick={del}>Delete</button></div>
        </div>
      )}
      {job.status === "cancelled" && (
        <div className="card"><p>This analysis was cancelled.</p><div className="row"><Link className="btn" to="/analyze">New analysis</Link><button className="btn danger sm" onClick={del}>Delete</button></div></div>
      )}
      {job.status === "done" && job.result && <Results job={job} onDelete={del} />}
    </div>
  );
}
