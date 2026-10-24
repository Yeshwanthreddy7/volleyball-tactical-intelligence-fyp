import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, uploadFile } from "../api.js";
import { TACTICS } from "../content.js";

const MB = 1024 * 1024;
const EXTS = [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"];

function fmtDur(s) {
  if (!isFinite(s)) return "-";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}
function parseTime(v) {
  if (!v.trim()) return null;
  const parts = v.trim().split(":").map(Number);
  if (parts.some((p) => !isFinite(p) || p < 0)) return NaN;
  return parts.reduce((acc, p) => acc * 60 + p, 0);
}

export default function Analyze() {
  const nav = useNavigate();
  const input = useRef(null);
  const abort = useRef(null);
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [meta, setMeta] = useState(null);
  const [preview, setPreview] = useState(null);
  const [drag, setDrag] = useState(false);
  const [mode, setMode] = useState("accurate");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [err, setErr] = useState(null);
  const [up, setUp] = useState(null); // {sent,total,rate}

  const maxMb = health?.max_upload_mb || 500;
  useEffect(() => { api.health().then(setHealth).catch(() => {}); }, []);
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const choose = (f) => {
    setErr(null);
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!EXTS.includes(ext)) return setErr(`"${f.name}" is not a supported video. Use ${EXTS.join(", ")}.`);
    if (f.size > maxMb * MB) return setErr(`The file is ${(f.size / MB).toFixed(0)} MB. The limit is ${maxMb} MB.`);
    setFile(f);
    setMeta(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => setMeta({ duration: v.duration, w: v.videoWidth, h: v.videoHeight });
    v.onerror = () => setMeta({ duration: NaN });
    v.src = url;
  };

  const s = parseTime(start), e = parseTime(end);
  const timeErr = Number.isNaN(s) || Number.isNaN(e) ? "Use mm:ss or h:mm:ss."
    : s !== null && e !== null && e <= s ? "End must be after start."
    : meta?.duration && s !== null && s >= meta.duration ? "Start is after the end of the video." : null;
  const analysedSec = meta?.duration ? Math.max(0, Math.min(e ?? meta.duration, meta.duration) - (s ?? 0)) : null;
  // measured on the project's RTX 3050 laptop GPU: detection about 9 frames/s
  // at 1280 px (13 at 960 px), rendering about 9 frames/s, plus ~1.5 min setup
  const estMin = analysedSec
    ? Math.ceil((analysedSec * 30 * (1 / (mode === "accurate" ? 9 : 13) + 1 / 9)) / 60 + 1.5) : null;

  const submit = async () => {
    if (!file || timeErr) return;
    setErr(null);
    abort.current = new AbortController();
    const t0 = performance.now();
    try {
      setUp({ sent: 0, total: file.size, rate: 0 });
      const uid = await uploadFile(file, (sent, total) => {
        const secs = (performance.now() - t0) / 1000;
        setUp({ sent, total, rate: secs > 0.5 ? sent / secs : 0 });
      }, abort.current.signal);
      const job = await api.create({ upload_id: uid, mode, start: s ?? 0, end: e ?? null });
      nav(`/analysis/${job.id}`);
    } catch (ex) {
      setUp(null);
      if (!ex.aborted) setErr(ex.message);
    }
  };

  const pct = up ? (100 * up.sent) / up.total : 0;
  const eta = up?.rate ? (up.total - up.sent) / up.rate : null;

  // static deployment (e.g. Vercel): no GPU server behind this site
  if (health?.status === "static") {
    return (
      <div className="container section">
        <div className="eyebrow">Analyze a match</div>
        <h1>Video analysis runs on a GPU</h1>
        <p className="lead">
          This online version is the project portal. Analysing a match needs an NVIDIA GPU and up to 500 MB
          uploads, so it runs in the desktop app or on the project's analysis server.
        </p>
        <div className="grid cols-2" style={{ marginTop: 26 }}>
          <div className="card">
            <div className="icon-badge">💻</div>
            <h3>Desktop app (Windows)</h3>
            <p className="muted small">
              Installs everything on your PC and analyses videos locally: annotated MP4, team tactical report
              and opponent serve / spike / block report. Windows 10/11, 8 GB RAM, NVIDIA GPU recommended.
            </p>
            {health.desktop_url
              ? <a className="btn" href={health.desktop_url} target="_blank" rel="noreferrer">⬇ Download the installer</a>
              : <span className="pill">Download link coming soon</span>}
          </div>
          <div className="card">
            <div className="icon-badge">🌐</div>
            <h3>Live analysis server</h3>
            <p className="muted small">
              When the project's GPU laptop is online, the full website with uploads is available at its live link.
            </p>
            {health.live_url
              ? <a className="btn secondary" href={health.live_url} target="_blank" rel="noreferrer">Open the live server ↗</a>
              : <span className="pill">Ask the team for the current live link</span>}
          </div>
        </div>
        <div className="section tight">
          <h3>What you will get</h3>
          <div className="grid cols-4" style={{ marginTop: 12 }}>
            {TACTICS.map((t) => (
              <div key={t.name} className="card" style={{ borderTop: `4px solid ${t.hex}` }}>
                <b>{t.name}</b>
                <p className="small muted" style={{ marginTop: 6 }}>{t.text}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container section">
      <div className="eyebrow">Analyze a match</div>
      <h1>Upload a volleyball video</h1>
      <p className="lead">
        Broadcast end-court footage works best (camera behind one baseline). You get an annotated MP4
        in several qualities, a timestamped team tactical report and an opponent action report.
      </p>

      {health && !health.gpu?.available && (
        <div className="alert warn" style={{ margin: "16px 0" }}>
          The analysis server has no GPU right now, so processing will be slow.
        </div>
      )}

      <div className="grid cols-2" style={{ marginTop: 26, alignItems: "start" }}>
        <div className="card">
          {!file ? (
            <div
              className={`dropzone ${drag ? "drag" : ""}`}
              onClick={() => input.current?.click()}
              onDragOver={(ev) => { ev.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(ev) => { ev.preventDefault(); setDrag(false); choose(ev.dataTransfer.files[0]); }}
              role="button" tabIndex={0}
              onKeyDown={(ev) => ev.key === "Enter" && input.current?.click()}
            >
              <div className="big">🎬</div>
              <h3>Drop your video here or click to browse</h3>
              <p className="muted small">MP4, MOV, M4V, AVI, MKV or WEBM · up to {maxMb} MB</p>
            </div>
          ) : (
            <div>
              <video src={preview} controls muted style={{ width: "100%", borderRadius: 10, background: "#000", maxHeight: 300 }} />
              <div className="row" style={{ marginTop: 12 }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 700, overflowWrap: "anywhere" }}>{file.name}</div>
                  <div className="small muted">
                    {(file.size / MB).toFixed(1)} MB
                    {meta?.duration > 0 && ` · ${fmtDur(meta.duration)}`}
                    {meta?.w > 0 && ` · ${meta.w}×${meta.h}`}
                  </div>
                </div>
                <span className="spacer" />
                {!up && <button className="btn ghost sm" onClick={() => { setFile(null); setPreview(null); }}>Change</button>}
              </div>
            </div>
          )}
          <input ref={input} type="file" accept={EXTS.join(",") + ",video/*"} hidden
            onChange={(ev) => { choose(ev.target.files[0]); ev.target.value = ""; }} />

          {up && (
            <div style={{ marginTop: 18 }}>
              <div className="row small" style={{ marginBottom: 6 }}>
                <b>Uploading… {pct.toFixed(0)}%</b>
                <span className="spacer" />
                <span className="muted">
                  {(up.sent / MB).toFixed(0)} / {(up.total / MB).toFixed(0)} MB
                  {up.rate > 0 && ` · ${(up.rate / MB).toFixed(1)} MB/s`}
                  {eta !== null && isFinite(eta) && ` · ${fmtDur(eta)} left`}
                </span>
              </div>
              <div className="bar"><span style={{ width: `${pct}%` }} /></div>
              <button className="btn danger sm" style={{ marginTop: 12 }} onClick={() => abort.current?.abort()}>Cancel upload</button>
            </div>
          )}
          {err && <div className="alert err" style={{ marginTop: 16 }}>{err}</div>}
        </div>

        <div className="card">
          <h3>Analysis options</h3>
          <div className="field" style={{ marginTop: 14 }}>
            <span className="label">Your team = the team on the near side (bottom of the picture)</span>
            <span className="small muted">
              Your team gets the four tactics; the far-side opponent gets only SERVE, SPIKE and BLOCK.
              The tactical model was trained on the near-side team. Teams change ends between sets, so use
              the time range below to analyse the sets in which your team plays at the bottom.
            </span>
          </div>

          <div className="field" style={{ marginTop: 18 }}>
            <span className="label">Processing mode</span>
            <div className="seg">
              <button className={mode === "accurate" ? "on" : ""} onClick={() => setMode("accurate")}>Accurate · 1280 px</button>
              <button className={mode === "fast" ? "on" : ""} onClick={() => setMode("fast")}>Fast · 960 px</button>
            </div>
            <span className="small muted">Accurate uses the resolution the models were trained at (best ball recall).</span>
          </div>

          <div className="field" style={{ marginTop: 18 }}>
            <span className="label">Part of the video to analyse (optional)</span>
            <div className="row" style={{ flexWrap: "nowrap" }}>
              <input className="input" placeholder="Start  0:00" value={start} onChange={(ev) => setStart(ev.target.value)} />
              <span className="muted">to</span>
              <input className="input" placeholder={`End  ${meta?.duration ? fmtDur(meta.duration) : "end"}`} value={end} onChange={(ev) => setEnd(ev.target.value)} />
            </div>
            {timeErr ? <span className="small" style={{ color: "#fca5a5" }}>{timeErr}</span>
              : analysedSec !== null && (
                <span className="small muted">
                  {fmtDur(analysedSec)} of video · estimated processing about {estMin} min on the project GPU
                  {health?.queue > 0 && ` (+ ${health.queue} analysis in queue)`}
                </span>
              )}
          </div>

          <button className="btn" style={{ width: "100%", marginTop: 22 }} disabled={!file || !!up || !!timeErr} onClick={submit}>
            {up ? "Uploading…" : "Upload and analyse"}
          </button>
          <p className="small faint" style={{ marginTop: 12 }}>
            Videos and results are kept for 14 days and are only listed in this browser.
          </p>
        </div>
      </div>

      <div className="section tight">
        <h3>What you will get</h3>
        <div className="grid cols-4" style={{ marginTop: 12 }}>
          {TACTICS.map((t) => (
            <div key={t.name} className="card" style={{ borderTop: `4px solid ${t.hex}` }}>
              <b>{t.name}</b>
              <p className="small muted" style={{ marginTop: 6 }}>{t.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
