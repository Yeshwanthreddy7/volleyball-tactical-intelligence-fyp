import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { OBJECTIVES, PROBLEM, TACTICS } from "../content.js";

export default function Home() {
  const [demo, setDemo] = useState(null);
  const [m, setM] = useState(null);
  useEffect(() => {
    api.demos().then((d) => setDemo(d[0] || null)).catch(() => {});
    api.metrics().then(setM).catch(() => {});
  }, []);
  const deployed = m?.models?.find((x) => x.deployed);

  return (
    <>
      <section className="hero">
        <div className="container hero-grid">
          <div>
            <div className="eyebrow">Final Year Project · Computer Vision + Machine Learning</div>
            <h1>Automated <span className="hl">volleyball tactical analysis</span> from broadcast video</h1>
            <p className="lead">
              Upload a match. The system tracks all twelve players and the ball, tells you second by second
              when your team was in Coordinated Attack, Coordinated Defense, Delayed Support or Spacing
              Breakdown, and time-stamps every opponent serve, spike and block.
            </p>
            <div className="row cta">
              <Link className="btn" to="/analyze">Analyse a video →</Link>
              <Link className="btn secondary" to="/methodology">How it works</Link>
              <Link className="btn ghost" to="/demos">Watch demos</Link>
            </div>
          </div>
          <div className="hero-media">
            {demo ? (
              <video src={`/api/project/demos/${demo.file}`} poster={demo.poster ? `/api/project/demos/${demo.poster}` : undefined}
                autoPlay muted loop playsInline />
            ) : (
              <img src="/api/project/file?path=PAPER_OUTPUT%2Faction_examples%2FSPIKE.png" alt="Annotated spike example" />
            )}
          </div>
        </div>
      </section>

      <section className="section tight">
        <div className="container grid cols-4">
          <div className="stat"><div className="v">{m?.n_clips ?? "948"}</div><div className="l">labelled tactical clips</div></div>
          <div className="stat"><div className="v">{m?.models?.length ?? 16}</div><div className="l">models compared (ML + deep)</div></div>
          <div className="stat"><div className="v">{deployed ? `${(deployed.lovo_accuracy * 100).toFixed(1)}%` : "67.0%"}</div><div className="l">leave-one-video-out accuracy (deployed HGB)</div></div>
          <div className="stat"><div className="v">4 + 3</div><div className="l">team tactics + opponent actions, time-stamped</div></div>
        </div>
      </section>

      <section className="section" id="problem">
        <div className="container grid cols-2" style={{ alignItems: "start" }}>
          <div>
            <div className="eyebrow">Problem statement</div>
            <h2>Team coordination is hard to measure</h2>
            <p className="lead">{PROBLEM.short}</p>
            <p className="muted">{PROBLEM.gap}</p>
          </div>
          <div className="card">
            <h3>Why it is difficult</h3>
            <ul className="muted" style={{ paddingLeft: 18, margin: 0 }}>
              {PROBLEM.challenges.map((c) => <li key={c} style={{ marginBottom: 6 }}>{c}</li>)}
            </ul>
          </div>
        </div>
      </section>

      <section className="section" id="objectives" style={{ background: "var(--bg-2)", borderBlock: "1px solid var(--border)" }}>
        <div className="container">
          <div className="eyebrow">Project objectives</div>
          <h2>What the system does</h2>
          <div className="grid cols-3" style={{ marginTop: 22 }}>
            {OBJECTIVES.map((o) => (
              <div key={o.title} className="card hover">
                <div className="icon-badge">{o.icon}</div>
                <h3>{o.title}</h3>
                <p className="muted small">{o.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="eyebrow">Four tactical classes</div>
          <h2>What the tactical report tells you</h2>
          <div className="grid cols-4" style={{ marginTop: 22 }}>
            {TACTICS.map((t) => (
              <div key={t.name} className="card" style={{ borderTop: `4px solid ${t.hex}` }}>
                <h3 style={{ color: t.hex }}>{t.name}</h3>
                <p className="muted small">{t.text}</p>
              </div>
            ))}
          </div>
          <div className="card" style={{ marginTop: 22 }}>
            <div className="row">
              <div style={{ flex: "1 1 360px" }}>
                <h3>Two reports for every video</h3>
                <p className="muted small" style={{ marginBottom: 0 }}>
                  <b>Team tactical report</b>: when your team was in each of the four tactics, for how long, and
                  how confident the model is. <b>Opponent action report</b>: every opponent serve, spike and block
                  with its exact time. Both open in Notepad (TXT) and also come as CSV and JSON.
                </p>
              </div>
              <Link className="btn" to="/analyze">Try it now</Link>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
