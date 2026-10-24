import { MILESTONES } from "../content.js";

export default function Progress() {
  const done = MILESTONES.filter((m) => m.state === "done").length;
  return (
    <div className="container section">
      <div className="eyebrow">Project progress</div>
      <h1>Timeline</h1>
      <p className="lead">{done} of {MILESTONES.length} milestones complete.</p>
      <div className="bar" style={{ maxWidth: 520, margin: "10px 0 34px" }}>
        <span style={{ width: `${(100 * done) / MILESTONES.length}%` }} />
      </div>
      <div className="milestones">
        {MILESTONES.map((m) => (
          <div key={m.title} className={`ms ${m.state}`}>
            <div className="card">
              <div className="row">
                <span className="pill">{m.date}</span>
                <span className={`pill ${m.state === "done" ? "ok" : m.state === "now" ? "warn" : ""}`}>
                  {m.state === "done" ? "Completed" : m.state === "now" ? "In progress" : "Planned"}
                </span>
              </div>
              <h3 style={{ marginTop: 10 }}>{m.title}</h3>
              <p className="muted" style={{ margin: 0 }}>{m.text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
