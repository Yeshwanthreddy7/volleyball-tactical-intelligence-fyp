import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, clientId, fileUrl, setClientId, VALID_CODE } from "../api.js";
import { TACTIC_HEX } from "../content.js";
import { StatusPill } from "./Analysis.jsx";

export default function History() {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);
  const [code, setCode] = useState(clientId());
  const [entry, setEntry] = useState("");
  const [note, setNote] = useState(null);

  const switchTo = (next, msg) => {
    setCode(setClientId(next));
    setItems(null);
    setErr(null);
    setEntry("");
    setNote(msg);
  };
  const restore = () => {
    const c = entry.trim();
    if (!VALID_CODE.test(c)) return setNote("That does not look like a history code.");
    switchTo(c, "History restored.");
  };
  const fresh = () => {
    if (window.confirm("Start a fresh, empty history in this browser? Keep your current code if you want to come back to the old uploads."))
      switchTo(null, "Fresh history started.");
  };
  const copy = () => navigator.clipboard?.writeText(code).then(() => setNote("Code copied."), () => {});

  useEffect(() => {
    let alive = true, timer;
    const load = () => api.analyses()
      .then((x) => {
        if (!alive) return;
        setItems(x);
        if (x.some((j) => j.status === "queued" || j.status === "processing")) timer = setTimeout(load, 4000);
      })
      .catch((e) => alive && setErr(e.message));
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [code]);

  return (
    <div className="container section">
      <div className="eyebrow">History</div>
      <div className="row" style={{ marginBottom: 20 }}>
        <h1 style={{ margin: 0 }}>My analyses</h1>
        <span className="spacer" />
        <Link className="btn" to="/analyze">+ New analysis</Link>
      </div>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="small muted">
          Uploads are saved to this browser. Use your history code to open them on another device, or start fresh.
        </div>
        <div className="row" style={{ marginTop: 10, gap: 8 }}>
          <code style={{ userSelect: "all", wordBreak: "break-all" }}>{code}</code>
          <button className="btn secondary" onClick={copy}>Copy code</button>
          <button className="btn secondary" onClick={fresh}>Start fresh</button>
        </div>
        <div className="row" style={{ marginTop: 10, gap: 8 }}>
          <input value={entry} onChange={(e) => setEntry(e.target.value)} placeholder="Paste a history code"
                 onKeyDown={(e) => e.key === "Enter" && restore()} style={{ flex: 1, minWidth: 200 }} />
          <button className="btn secondary" onClick={restore}>Restore</button>
        </div>
        {note && <div className="small" style={{ marginTop: 8 }}>{note}</div>}
      </div>
      {err && <div className="alert err">{err}</div>}
      {!items && !err && <div className="empty"><span className="spinner" /></div>}
      {items && items.length === 0 && (
        <div className="card empty">
          <h3>No analyses yet</h3>
          <p>Upload a match video to get an annotated video and reports.</p>
          <Link className="btn" to="/analyze">Analyse a video</Link>
        </div>
      )}
      <div className="grid cols-3">
        {items?.map((j) => (
          <Link key={j.id} to={`/analysis/${j.id}`} className="card hover" style={{ color: "inherit", textDecoration: "none", padding: 0, overflow: "hidden" }}>
            <div style={{ aspectRatio: "16/7", background: "#000", display: "grid", placeItems: "center" }}>
              {j.status === "done"
                ? <img src={fileUrl(j.id, "poster.jpg")} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                : <span className="muted small">{j.status === "processing" ? `${(j.progress || 0).toFixed(0)}% · ${j.message}` : j.message}</span>}
            </div>
            <div style={{ padding: 16 }}>
              <div className="row" style={{ flexWrap: "nowrap" }}>
                <b style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{j.original_name}</b>
                <span className="spacer" />
                <StatusPill status={j.status} />
              </div>
              <div className="small muted" style={{ marginTop: 6 }}>
                {new Date(j.created_at).toLocaleString()} · {(j.size / 1048576).toFixed(0)} MB
              </div>
              {j.result?.summary && (
                <div className="small" style={{ marginTop: 8 }}>
                  {j.result.summary.tactic_segments} tactic segments · {j.result.summary.opponent_total} opponent actions
                  {j.result.summary.dominant_tactic && (
                    <> · <span style={{ color: TACTIC_HEX[j.result.summary.dominant_tactic] }}>{j.result.summary.dominant_tactic}</span></>
                  )}
                </div>
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
