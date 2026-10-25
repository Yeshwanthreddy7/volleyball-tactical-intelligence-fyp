import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, apiUrl } from "../api.js";
import { TACTICS } from "../content.js";
import { Gallery } from "./Methodology.jsx";

const slug = (t) => t.toLowerCase().replace(/ /g, "_");
const ts = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;

// One folder per tactic, filled with clips cut from the visitor's own analyses.
function TacticFolders() {
  const [jobs, setJobs] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState(slug(TACTICS[0].name));

  useEffect(() => {
    api.analyses().then((j) => setJobs(j.filter((x) => x.status === "done"))).catch((e) => setErr(e.message));
  }, []);

  const folders = useMemo(() => {
    const f = Object.fromEntries(TACTICS.map((t) => [slug(t.name), []]));
    for (const j of jobs || []) {
      for (const [k, list] of Object.entries(j.result?.clips || {})) {
        for (const c of list) f[k]?.push({ ...c, job: j, url: apiUrl(`/api/analyses/${j.id}/clips/${k}/${c.file}`) });
      }
    }
    return f;
  }, [jobs]);

  const total = Object.values(folders).reduce((a, l) => a + l.length, 0);
  const active = TACTICS.find((t) => slug(t.name) === tab);

  return (
    <div className="section tight">
      <h2>Your video, by tactic</h2>
      <p className="muted">
        After you upload a match, the best moments of each tactic are cut into short clips and grouped here,
        one folder per tactic.
      </p>
      {err && <div className="alert warn">{err}</div>}
      {!err && jobs === null && <div className="empty"><span className="spinner" /></div>}
      {jobs && total === 0 && (
        <div className="card empty">
          <h3>No clips yet</h3>
          <p>{jobs.length ? "Your finished analyses have no tactic segments long enough for a clip yet." : "Upload a match video and its tactic clips will appear here."}</p>
          <Link className="btn" to="/analyze">Analyse a video</Link>
        </div>
      )}
      {jobs && total > 0 && (
        <>
          <div className="tabs">
            {TACTICS.map((t) => (
              <button key={t.name} className={tab === slug(t.name) ? "on" : ""} onClick={() => setTab(slug(t.name))}>
                📁 {t.name} <span className="faint">({folders[slug(t.name)].length})</span>
              </button>
            ))}
          </div>
          <p className="small muted" style={{ marginTop: 0 }}>
            <span className="dot" style={{ background: active.hex, marginRight: 6 }} />{active.text}
          </p>
          {folders[tab].length === 0
            ? <div className="card empty">No {active.name} segment was found in your videos yet.</div>
            : (
              <div className="grid cols-3">
                {folders[tab].map((c) => (
                  <div key={c.url} className="card" style={{ padding: 0, overflow: "hidden", borderTop: `4px solid ${active.hex}` }}>
                    <video src={c.url} controls preload="metadata" playsInline style={{ width: "100%", display: "block", background: "#000" }} />
                    <div style={{ padding: 14 }}>
                      <b style={{ overflowWrap: "anywhere" }}>{c.job.original_name}</b>
                      <div className="small muted">{ts(c.start)} – {ts(c.end)} · confidence {(c.confidence * 100).toFixed(0)}%</div>
                      <div className="row" style={{ marginTop: 8 }}>
                        <Link className="btn ghost sm" to={`/analysis/${c.job.id}`}>Open analysis</Link>
                        <a className="btn secondary sm" href={c.url} download>⬇ Clip</a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
        </>
      )}
    </div>
  );
}

export default function Demos() {
  const [demos, setDemos] = useState(null);
  const [figs, setFigs] = useState([]);
  useEffect(() => {
    api.demos().then(setDemos).catch(() => setDemos([]));
    api.figures().then((f) => setFigs(f.filter((x) => x.group === "actions"))).catch(() => {});
  }, []);

  return (
    <div className="container section">
      <div className="eyebrow">Demonstration videos</div>
      <h1>See the output</h1>
      <p className="lead">
        Annotated clips produced by the project pipeline. Blue = your team (near side) with its relation graph and
        tactic-coloured outline, red = opponent, star = opponent player serving, spiking or blocking.
      </p>

      <TacticFolders />

      <div className="section tight">
        <h2>Sample videos</h2>
        {demos === null && <div className="empty"><span className="spinner" /></div>}
        {demos?.length === 0 && <div className="card empty">No sample videos available.</div>}
        <div className="grid cols-2" style={{ marginTop: 16 }}>
          {demos?.map((d) => (
            <div key={d.file} className="card" style={{ padding: 0, overflow: "hidden" }}>
              <video src={`/api/project/demos/${d.file}`} poster={d.poster ? `/api/project/demos/${d.poster}` : undefined}
                controls preload="none" playsInline style={{ width: "100%", display: "block", background: "#000" }} />
              <div style={{ padding: 18 }}>
                <h3>{d.title}</h3>
                <p className="muted small" style={{ marginBottom: 10 }}>{d.description}</p>
                <a className="btn secondary sm" href={`/api/project/demos/${d.file}`} download>⬇ Download MP4</a>
              </div>
            </div>
          ))}
        </div>
      </div>
      {figs.length > 0 && (
        <div className="section tight">
          <h2>One frame per action</h2>
          <Gallery items={figs} />
        </div>
      )}
      <div className="card row" style={{ marginTop: 20 }}>
        <div style={{ flex: "1 1 300px" }}><h3>Try it on your own match</h3><p className="muted small" style={{ margin: 0 }}>Upload up to 500 MB and get the same output for your video.</p></div>
        <Link className="btn" to="/analyze">Analyse a video</Link>
      </div>
    </div>
  );
}
