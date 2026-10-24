import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fileUrl } from "../api.js";
import { TACTIC_HEX } from "../content.js";
import { StatusPill } from "./Analysis.jsx";

export default function History() {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);

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
  }, []);

  return (
    <div className="container section">
      <div className="eyebrow">History</div>
      <div className="row" style={{ marginBottom: 20 }}>
        <h1 style={{ margin: 0 }}>My analyses</h1>
        <span className="spacer" />
        <Link className="btn" to="/analyze">+ New analysis</Link>
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
