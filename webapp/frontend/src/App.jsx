import { useEffect, useState } from "react";
import { NavLink, Link, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api.js";
import Home from "./pages/Home.jsx";
import Analyze from "./pages/Analyze.jsx";
import Analysis from "./pages/Analysis.jsx";
import History from "./pages/History.jsx";
import Methodology from "./pages/Methodology.jsx";
import Results from "./pages/Results.jsx";
import Progress from "./pages/Progress.jsx";
import Docs from "./pages/Docs.jsx";
import Demos from "./pages/Demos.jsx";
import Team from "./pages/Team.jsx";

export const GITHUB_FALLBACK = "https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp";

const LINKS = [
  ["/", "Home"],
  ["/analyze", "Analyze video"],
  ["/my-analyses", "My analyses"],
  ["/methodology", "Methodology"],
  ["/results", "Results"],
  ["/progress", "Progress"],
  ["/docs", "Documentation"],
  ["/demos", "Demos"],
  ["/team", "Team"],
];

export function BallIcon({ className = "ball" }) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true">
      <circle cx="32" cy="32" r="29" fill="#f59e0b" stroke="#0b1220" strokeWidth="3" />
      <path d="M32 3c-6 12-6 26 0 29M32 32c13 2 24-5 28-14M32 32c-7 11-19 16-28 13"
        fill="none" stroke="#0b1220" strokeWidth="3.5" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const [open, setOpen] = useState(false);
  const [github, setGithub] = useState(GITHUB_FALLBACK);
  const loc = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo(0, 0); }, [loc.pathname]);
  useEffect(() => { api.info().then((i) => i.github && setGithub(i.github)).catch(() => {}); }, []);

  return (
    <>
      <header className="nav">
        <div className="container nav-inner">
          <Link to="/" className="brand">
            <BallIcon />
            <span>Volleyball Tactical Analysis<small>Final Year Project · Group 56</small></span>
          </Link>
          <button className="nav-toggle" aria-label="Menu" onClick={() => setOpen((o) => !o)}>☰</button>
          <nav className={`nav-links ${open ? "open" : ""}`}>
            {LINKS.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>
            ))}
            <a href={github} target="_blank" rel="noreferrer">GitHub ↗</a>
          </nav>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/analysis/:id" element={<Analysis />} />
          <Route path="/my-analyses" element={<History />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/results" element={<Results />} />
          <Route path="/progress" element={<Progress />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/docs/:slug" element={<Docs />} />
          <Route path="/demos" element={<Demos />} />
          <Route path="/team" element={<Team github={github} />} />
          <Route path="*" element={
            <div className="container section empty">
              <h2>Page not found</h2>
              <Link className="btn" to="/">Go home</Link>
            </div>} />
        </Routes>
      </main>
      <footer className="footer">
        <div className="container row">
          <span>© 2026 Volleyball Tactical Analysis · Group 56 Final Year Project</span>
          <span className="spacer" />
          <Link to="/docs">Documentation</Link>
          <a href={github} target="_blank" rel="noreferrer">GitHub repository</a>
        </div>
      </footer>
    </>
  );
}
