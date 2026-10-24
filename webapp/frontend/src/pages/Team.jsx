import { TEAM } from "../content.js";

const initials = (n) => n.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();

export default function Team({ github }) {
  return (
    <div className="container section">
      <div className="eyebrow">Team details</div>
      <h1>{TEAM.group}</h1>
      <p className="lead">{TEAM.department}</p>
      <div className="grid cols-4" style={{ marginTop: 26 }}>
        {TEAM.members.map((m) => (
          <div key={m.name} className="card center">
            <div className="avatar" style={{ margin: "0 auto 12px" }}>{initials(m.name)}</div>
            <h3>{m.name}</h3>
            <div className="muted small">{m.role}</div>
            <div className="faint small">{m.id}</div>
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
