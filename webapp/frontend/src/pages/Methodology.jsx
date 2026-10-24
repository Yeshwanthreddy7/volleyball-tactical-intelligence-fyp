import { useEffect, useState } from "react";
import { api, repoFile } from "../api.js";
import { PIPELINE } from "../content.js";

export function Gallery({ items }) {
  const [big, setBig] = useState(null);
  return (
    <>
      <div className="grid cols-3">
        {items.map((f) => (
          <figure key={f.path} className="figure" style={{ margin: 0 }}>
            <div className="img" onClick={() => setBig(f)}><img src={repoFile(f.path)} alt={f.title} loading="lazy" /></div>
            <figcaption className="cap">{f.title}</figcaption>
          </figure>
        ))}
      </div>
      {big && (
        <div className="lightbox" onClick={() => setBig(null)}>
          <img src={repoFile(big.path)} alt={big.title} />
        </div>
      )}
    </>
  );
}

export default function Methodology() {
  const [figs, setFigs] = useState([]);
  useEffect(() => { api.figures().then(setFigs).catch(() => {}); }, []);
  const main = figs.filter((f) => f.group !== "actions");
  const actions = figs.filter((f) => f.group === "actions");

  return (
    <div className="container section">
      <div className="eyebrow">Methodology</div>
      <h1>From broadcast video to a tactical report</h1>
      <p className="lead">
        Every step below is the project's own research code. The website runs the same scripts
        and the same trained models as the experiments; nothing is re-implemented for the web.
      </p>

      <div className="pipeline" style={{ marginTop: 30 }}>
        {PIPELINE.map((s, i) => (
          <div key={s.title} className="step">
            <div className="n">{i + 1}</div>
            <div>
              <h3 style={{ marginBottom: 4 }}>{s.title}</h3>
              <p className="muted" style={{ marginBottom: 6 }}>{s.text}</p>
              <code>{s.code}</code>
            </div>
          </div>
        ))}
      </div>

      <div className="section tight">
        <h2>Key design decisions</h2>
        <div className="grid cols-2" style={{ marginTop: 16 }}>
          <div className="card"><h3>Dual detector</h3><p className="muted small">Players come from stock YOLO weights (a canonical COCO class); the ball from the volleyball fine-tune, which reaches 67% recall vs 16% for stock weights. Inference at 1280 px: ball recall rises from 15% at 640 px to 77%.</p></div>
          <div className="card"><h3>Scale-invariant features</h3><p className="muted small">With one uncalibrated camera, absolute distances encode camera identity. The perm_invariant_v2 descriptor uses dimensionless ratios and the z-scored temporal shape of every channel, so the model cannot memorise which video it is watching.</p></div>
          <div className="card"><h3>Identity-stable slots</h3><p className="muted small">Tracker ids switch during occlusions. Slots survive gaps of up to 15 frames, identities are recovered before features are built, and missing players are NaN, never a fake (0,0) court corner.</p></div>
          <div className="card"><h3>Honest evaluation</h3><p className="muted small">Leave-one-video-out validation (the model never sees the test match), a majority-class baseline, bootstrap confidence intervals and McNemar tests; one command (reproduce.py) regenerates every number.</p></div>
        </div>
      </div>

      {main.length > 0 && (
        <div className="section tight">
          <h2>Diagrams and figures</h2>
          <p className="muted">Loaded live from the project repository. Click to enlarge.</p>
          <Gallery items={main} />
        </div>
      )}
      {actions.length > 0 && (
        <div className="section tight">
          <h2>Detected action examples</h2>
          <Gallery items={actions} />
        </div>
      )}
    </div>
  );
}
