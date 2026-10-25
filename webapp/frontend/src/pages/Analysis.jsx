import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, apiUrl, fileUrl } from "../api.js";
import { OPP_HEX, TACTICS, TACTIC_HEX } from "../content.js";
import VideoPlayer from "../components/VideoPlayer.jsx";
import Notepad from "../components/Notepad.jsx";

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
        You can close this page - the analysis keeps running on the server and appears under "My analyses" in this browser.
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
        {result.events.map((e, i) => (
          <div key={i} className="tick" title={`${ts(e.time)} opponent ${e.action}`}
            style={{ left: pos(e.time), background: OPP_HEX[e.action] || "#fff" }} />
        ))}
        <div className="head" style={{ left: pos(off + current) }} />
      </div>
      <div className="legend">
        {TACTICS.map((t) => <span key={t.name}><i className="dot" style={{ background: t.hex }} />{t.name}</span>)}
        {Object.entries(OPP_HEX).map(([a, c]) => <span key={a}><i className="dot" style={{ background: c, borderRadius: 2 }} />opponent {a.toLowerCase()}</span>)}
      </div>
    </div>
  );
}

function EventsTable({ events, onSeek }) {
  const [act, setAct] = useState("all");
  const rows = events.filter((e) => act === "all" || e.action === act);
  return (
    <div>
      <div className="row" style={{ marginBottom: 12 }}>
        <div className="seg">
          {["all", "SERVE", "SPIKE", "BLOCK"].map((k) => (
            <button key={k} className={act === k ? "on" : ""} onClick={() => setAct(k)}>{k === "all" ? "All" : k}</button>
          ))}
        </div>
        <span className="small muted">{rows.length} opponent actions</span>
      </div>
      <div className="table-wrap" style={{ maxHeight: 520, overflowY: "auto" }}>
        <table>
          <thead><tr><th>#</th><th>Time</th><th>Action</th><th>Opponent player</th><th /></tr></thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan={5} className="muted">No opponent serve, spike or block detected.</td></tr>}
            {rows.map((e, i) => (
              <tr key={i}>
                <td className="num">{i + 1}</td>
                <td className="num"><a href="#player" onClick={(ev) => { ev.preventDefault(); onSeek(e.time); }}>{ts(e.time)}</a></td>
                <td><span className="chip-tactic"><i className="dot" style={{ background: OPP_HEX[e.action] }} />{e.action}</span></td>
                <td>{e.player_track_id >= 0 ? `#${e.player_track_id}` : "-"}</td>
                <td><button className="btn ghost sm" onClick={() => onSeek(e.time)}>▶ Play</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TacticTable({ segments, onSeek }) {
  return (
    <div className="table-wrap" style={{ maxHeight: 520, overflowY: "auto" }}>
      <table>
        <thead><tr><th>#</th><th>From</th><th>To</th><th>Length</th><th>Tactic</th><th>Confidence</th><th /></tr></thead>
        <tbody>
          {segments.length === 0 && <tr><td colSpan={7} className="muted">No tactic was called (no live play with 3+ of your players visible).</td></tr>}
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
  const r = job.result;
  const player = useRef(null);
  const [tab, setTab] = useState("tactical");
  const [cur, setCur] = useState(0);
  const off = r.video.analysed_start;
  const dur = r.video.analysed_end - off;
  const seek = (t) => player.current?.seek(t - off);
  const s = r.summary;
  const renditions = r.files;
  const tacticRows = useMemo(() => TACTICS.map((t) => ({ ...t, ...s.tactics[t.name] })), [s]);

  return (
    <>
      <div className="grid cols-4" style={{ marginBottom: 22 }}>
        <div className="stat"><div className="v">{s.tactic_segments}</div><div className="l">your team tactic segments ({s.live_seconds.toFixed(0)} s of live play)</div></div>
        <div className="stat"><div className="v">{s.opponent_total}</div><div className="l">opponent: {s.opponent_actions.SERVE} serves · {s.opponent_actions.SPIKE} spikes · {s.opponent_actions.BLOCK} blocks</div></div>
        <div className="stat"><div className="v">{(100 * s.live_seconds / Math.max(1, s.analysed_seconds)).toFixed(0)}%</div><div className="l">of analysed time in live play</div></div>
        <div className="stat">
          <div className="v" style={{ fontSize: "1.15rem", paddingTop: 8, color: s.dominant_tactic ? TACTIC_HEX[s.dominant_tactic] : undefined }}>
            {s.dominant_tactic || "-"}
          </div>
          <div className="l">dominant team tactic</div>
        </div>
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
        <h3>Your team's tactics <span className="faint small">(near side, blue)</span></h3>
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
          {[["tactical", "📄 Team tactical report"], ["actions", "📄 Opponent action report"], ["segments", "Tactic segments"], ["table", "Opponent actions"], ["court", "Court calibration"]].map(([k, l]) => (
            <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
        {tab === "tactical" && (
          <Notepad title="tactical_report.txt" url={fileUrl(job.id, "tactical_report.txt")} onSeek={seek}
            downloads={[
              { label: "⬇ tactical_report.txt", href: fileUrl(job.id, "tactical_report.txt", true), primary: true },
              { label: "CSV", href: fileUrl(job.id, "tactical_report.csv", true) },
              { label: "JSON", href: fileUrl(job.id, "tactical_report.json", true) },
            ]} />
        )}
        {tab === "actions" && (
          <Notepad title="opponent_report.txt" url={fileUrl(job.id, "opponent_report.txt")} onSeek={seek}
            downloads={[
              { label: "⬇ opponent_report.txt", href: fileUrl(job.id, "opponent_report.txt", true), primary: true },
              { label: "CSV", href: fileUrl(job.id, "opponent_report.csv", true) },
              { label: "JSON", href: fileUrl(job.id, "opponent_report.json", true) },
            ]} />
        )}
        {tab === "segments" && <TacticTable segments={r.tactics.segments} onSeek={seek} />}
        {tab === "table" && <EventsTable events={r.events} onSeek={seek} />}
        {tab === "court" && (
          <div className="card">
            <p>
              {r.court.method === "auto"
                ? <>Court found automatically from the lines: <b>{r.court.lines.join(", ")}</b>{r.court.fit_rmse_px != null && ` (fit error ${r.court.fit_rmse_px.toFixed(2)} px)`}.</>
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
        if (j.status === "queued" || j.status === "processing") timer = setTimeout(tick, 2000);
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
