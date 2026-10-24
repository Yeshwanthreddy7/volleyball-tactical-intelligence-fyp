import { useEffect, useState } from "react";

// Shows a .txt report the way Windows Notepad would, with every timestamp
// clickable: clicking a timeline row jumps the video to that moment.
const TS = /(?:(\d+):)?(\d{2}):(\d{2}\.\d{2})/;

function toSeconds(m) {
  return (m[1] ? +m[1] * 3600 : 0) + +m[2] * 60 + parseFloat(m[3]);
}

export default function Notepad({ url, title, onSeek, downloads = [] }) {
  const [text, setText] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setText(null);
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setText)
      .catch((e) => setErr(e.message));
  }, [url]);

  const lines = (text || "").replace(/\r/g, "").split("\n");
  // only timeline rows (they start with a row number) seek the video
  const rowRe = /^\s+\d+\s+/;

  return (
    <div>
      <div className="notepad">
        <div className="np-title">
          <span className="np-icon" />
          <span>{title} - Notepad</span>
          <span className="win">─ ☐ ✕</span>
        </div>
        <div className="np-menu"><span>File</span><span>Edit</span><span>Format</span><span>View</span><span>Help</span></div>
        <div className="np-body">
          {err && <div className="np-line">Could not load report: {err}</div>}
          {text === null && !err && <div className="np-line">Loading…</div>}
          {text !== null && lines.map((ln, i) => {
            const m = rowRe.test(ln) ? ln.match(TS) : null;
            if (!m || !onSeek) return <div key={i} className="np-line">{ln}</div>;
            const at = m.index;
            return (
              <div key={i} className="np-line click" title="Play from here" onClick={() => onSeek(toSeconds(m))}>
                {ln.slice(0, at)}<span className="ts">{m[0]}</span>{ln.slice(at + m[0].length)}
              </div>
            );
          })}
        </div>
        <div className="np-status">
          <span>{lines.length} lines</span><span>UTF-8</span><span>Windows (CRLF)</span><span>100%</span>
        </div>
      </div>
      {downloads.length > 0 && (
        <div className="row" style={{ marginTop: 12 }}>
          {downloads.map((d) => (
            <a key={d.href} className={`btn sm ${d.primary ? "" : "secondary"}`} href={d.href}>{d.label}</a>
          ))}
        </div>
      )}
    </div>
  );
}
