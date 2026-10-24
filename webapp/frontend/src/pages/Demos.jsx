import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Gallery } from "./Methodology.jsx";

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
        Annotated clips produced by the project pipeline. Red = far team, blue = near team, star = player
        on the ball, white arrows = relation graph, the banner shows the action and the team tactic.
      </p>
      {demos === null && <div className="empty"><span className="spinner" /></div>}
      {demos?.length === 0 && <div className="card empty">Demo videos have not been generated yet (run <code>webapp/scripts/make_demos.py</code>).</div>}
      <div className="grid cols-2" style={{ marginTop: 24 }}>
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
