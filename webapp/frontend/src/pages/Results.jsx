import { useEffect, useState } from "react";
import { api } from "../api.js";
import { TACTIC_HEX } from "../content.js";
import { Gallery } from "./Methodology.jsx";

const pct = (v) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);

function Meter({ v, max = 1, hi }) {
  if (v == null) return <span className="faint">-</span>;
  return (
    <div className="meter">
      <div className="track"><div className={`fill ${hi ? "hi" : ""}`} style={{ width: `${(100 * v) / max}%` }} /></div>
      <span className="num small">{v.toFixed(3)}</span>
    </div>
  );
}

export default function Results() {
  const [m, setM] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api.metrics().then(setM).catch((e) => setErr(e.message)); }, []);
  if (err) return <div className="container section"><div className="alert err">{err}</div></div>;
  if (!m) return <div className="container section empty"><span className="spinner" /></div>;

  const counts = m.class_counts || {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const maxF1 = Math.max(...m.models.map((x) => x.lovo_macro_f1 || 0), 0.01);

  return (
    <div className="container section">
      <div className="eyebrow">Results</div>
      <h1>Model comparison</h1>
      <p className="lead">
        {m.models.length} models trained on {m.n_clips} labelled clips and evaluated with
        leave-one-video-out (LOVO) validation: each match is held out in turn, so the score shows
        how the model does on a match it has never seen. Read live from <code>ml_results_v6</code> and <code>deep_results_v6</code>.
      </p>

      <div className="grid cols-4" style={{ margin: "26px 0" }}>
        {m.models.filter((x) => x.deployed).map((x) => (
          <div key="d" className="stat" style={{ borderColor: "var(--accent)" }}>
            <div className="v">{pct(x.lovo_accuracy)}</div><div className="l">LOVO accuracy · {x.name}</div>
          </div>
        ))}
        {m.models.filter((x) => x.deployed).map((x) => (
          <div key="f" className="stat"><div className="v">{x.lovo_macro_f1.toFixed(3)}</div><div className="l">LOVO macro F1 (4 classes)</div></div>
        ))}
        <div className="stat"><div className="v">{pct(m.majority_baseline?.accuracy)}</div><div className="l">majority-class baseline accuracy</div></div>
        <div className="stat"><div className="v">{m.majority_baseline?.macro_f1?.toFixed(3) ?? "-"}</div><div className="l">majority-class baseline macro F1</div></div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Model</th><th>Family</th><th>LOVO macro F1</th><th>LOVO accuracy</th><th>LOVO κ</th><th>5-fold macro F1</th><th>5-fold accuracy</th></tr>
          </thead>
          <tbody>
            {m.models.map((x) => (
              <tr key={x.key} className={x.deployed ? "deployed" : ""}>
                <td><b>{x.name}</b>{x.deployed && <span className="pill warn" style={{ marginLeft: 8 }}>deployed</span>}</td>
                <td className="muted">{x.family}</td>
                <td><Meter v={x.lovo_macro_f1} max={maxF1} hi={x.deployed} /></td>
                <td className="num">{pct(x.lovo_accuracy)}</td>
                <td className="num">{x.lovo_kappa?.toFixed(3) ?? "-"}</td>
                <td className="num">{x.kfold_macro_f1?.toFixed(3) ?? "-"}</td>
                <td className="num">{pct(x.kfold_accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 10 }}>
        The top models are within about 0.01 LOVO macro F1 of each other. HistGradientBoosting is deployed
        because it has the best LOVO accuracy among them, handles missing players natively and needs no
        feature scaling. Deep models are scored with pooled LOVO predictions.
      </p>

      <div className="grid cols-2" style={{ marginTop: 34, alignItems: "start" }}>
        <div className="card">
          <h3>Training data by class</h3>
          {Object.entries(counts).map(([k, v]) => (
            <div key={k} style={{ margin: "12px 0" }}>
              <div className="row small"><span>{k}</span><span className="spacer" /><span className="num">{v} clips · {(100 * v / total).toFixed(1)}%</span></div>
              <div className="bar" style={{ height: 8 }}><span style={{ width: `${(100 * v) / total}%`, background: TACTIC_HEX[k] }} /></div>
            </div>
          ))}
          <p className="small muted" style={{ marginBottom: 0 }}>Delayed Support and Spacing Breakdown are minority classes, which is why macro F1 (equal weight per class) is the headline metric.</p>
        </div>
        <div className="card">
          <h3>Per-class F1 of the deep models</h3>
          <div className="table-wrap" style={{ marginTop: 10 }}>
            <table>
              <thead><tr><th>Model</th>{["Attack", "Defense", "Delayed sup.", "Spacing br."].map((t) => <th key={t}>{t}</th>)}</tr></thead>
              <tbody>
                {m.models.filter((x) => x.per_class_f1).map((x) => (
                  <tr key={x.key}>
                    <td>{x.name}</td>
                    {Object.keys(TACTIC_HEX).map((t) => <td key={t} className="num">{x.per_class_f1[t]?.toFixed(2) ?? "-"}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="section tight">
        <h2>Figures</h2>
        <Gallery items={[
          { title: "Train vs test macro F1, all models", path: "model_comparison/trainvstest_macrof1.png" },
          { title: "Train vs test accuracy, all models", path: "model_comparison/trainvstest_accuracy.png" },
          { title: "Per-class F1, four tactics", path: "figures_perclass_f1/perclass_f1_all4.png" },
        ]} />
      </div>
    </div>
  );
}
