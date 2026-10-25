import { useState } from "react";
import { TEAM } from "../content.js";

const initials = (n) => n.replace(/^Dr\.\s*/, "").split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();

// member photo (webapp/images -> frontend/public/team); initials if missing or not loaded
function Photo({ m }) {
  const [failed, setFailed] = useState(false);
  if (!m.photo || failed) return <div className="avatar" style={{ margin: "0 auto 12px" }}>{initials(m.name)}</div>;
  return (
    <img src={m.photo} alt={m.name} loading="lazy" onError={() => setFailed(true)}
      style={{ width: 132, height: 132, borderRadius: "50%", objectFit: "cover", display: "block",
               margin: "4px auto 16px", border: "3px solid var(--accent)", boxShadow: "var(--shadow)" }} />
  );
}

export default function Team({ github }) {
  return (
    <div className="container section">
      <div className="eyebrow">Team details</div>
      <h1>{TEAM.group}</h1>
      <p className="lead">{TEAM.department}</p>
      <div className="grid" style={{ marginTop: 26, gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
        {TEAM.members.map((m) => (
          <div key={m.name} className="card center">
            <Photo m={m} />
            <h3>{m.name}</h3>
            {m.role && <div className="muted small">{m.role}</div>}
            <div className="faint small" style={{ fontFamily: "var(--mono)" }}>{m.id}</div>
          </div>
        ))}
      </div>
      <div className="card" style={{ marginTop: 22 }}>
        <div className="row">
          <div className="avatar" style={{ margin: 0, background: "linear-gradient(135deg, var(--blue), #1e3a8a)", color: "#fff" }}>{initials(TEAM.guide.name)}</div>
          <div><div className="faint small">Project guide</div><h3 style={{ margin: 0 }}>{TEAM.guide.name}</h3><div className="muted small">{TEAM.guide.role}</div></div>
        </div>
      </div>
      <div className="card row" style={{ marginTop: 22 }}>
        <div style={{ flex: "1 1 300px" }}>
          <h3>Source code</h3>
          <p className="muted small" style={{ margin: 0 }}>The full research pipeline, models, experiments and this website.</p>
        </div>
        <a className="btn" href={github} target="_blank" rel="noreferrer">Open GitHub repository ↗</a>
      </div>
    </div>
  );
}
