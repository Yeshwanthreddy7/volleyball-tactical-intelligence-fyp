import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";

// YouTube-style quality menu over the MP4 renditions the engine produced.
// "Auto" picks the largest rendition that fits the player's on-screen size
// and the measured connection speed, and re-checks when the window resizes.

const LABELS = {
  ultra: "Ultra HD",
  high: "High",
  medium: "Medium",
  low: "Low",
};
const ORDER = ["ultra", "high", "medium", "low"];

function pickAuto(renditions, el, duration) {
  const usable = ORDER.filter((k) => renditions[k]);
  if (!usable.length) return null;
  // on-screen height follows from the player width (the video's own height is
  // unknown until its metadata loads)
  const cssW = el?.parentElement?.clientWidth || window.innerWidth;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const downlink = navigator.connection?.downlink; // Mbit/s, Chromium only
  for (const k of usable) {
    const r = renditions[k];
    const needH = ((cssW * r.height) / r.width) * dpr;
    const mbps = duration > 0 ? (r.size * 8) / duration / 1e6 : 0;
    const fitsScreen = r.height <= needH * 1.25 || k === usable[usable.length - 1];
    const fitsNet = !downlink || mbps <= downlink * 0.8;
    if (fitsScreen && fitsNet) return k;
  }
  return usable[usable.length - 1];
}

const VideoPlayer = forwardRef(function VideoPlayer({ renditions, srcFor, poster, duration, onTime, onDownload }, ref) {
  const video = useRef(null);
  const [choice, setChoice] = useState("auto");
  const [menu, setMenu] = useState(false);
  const [autoPick, setAutoPick] = useState(null);
  const pending = useRef(null);

  const active = choice === "auto" ? autoPick : choice;
  const src = active ? srcFor(active) : null;

  useImperativeHandle(ref, () => ({
    seek(t) {
      const v = video.current;
      if (!v) return;
      v.currentTime = Math.max(0, t);
      v.play().catch(() => {});
      v.scrollIntoView({ behavior: "smooth", block: "center" });
    },
  }));

  useEffect(() => {
    const update = () => setAutoPick(pickAuto(renditions, video.current, duration));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [renditions, duration]);

  // switching quality keeps the position and play state, like YouTube
  const switchTo = (k) => {
    const v = video.current;
    if (v && !v.paused) pending.current = { t: v.currentTime, play: true };
    else if (v) pending.current = { t: v.currentTime, play: false };
    setChoice(k);
    setMenu(false);
  };
  const onLoaded = () => {
    const v = video.current;
    if (v && pending.current) {
      v.currentTime = pending.current.t;
      if (pending.current.play) v.play().catch(() => {});
      pending.current = null;
    }
  };

  const options = useMemo(() => ORDER.filter((k) => renditions[k]), [renditions]);
  const label = (k) => `${LABELS[k]} · ${renditions[k].height}p`;

  return (
    <div className="player">
      {src && (
        <video ref={video} key={src} src={src} poster={poster} controls playsInline preload="metadata"
          onLoadedMetadata={onLoaded} onTimeUpdate={(e) => onTime?.(e.currentTarget.currentTime)} />
      )}
      <div className="player-bar">
        <span className="pill info">MP4 · H.264</span>
        <span className="small muted">
          Playing: <b style={{ color: "var(--text)" }}>{active ? (choice === "auto" ? `Auto (${renditions[active].height}p)` : label(active)) : "-"}</b>
        </span>
        <span className="spacer" />
        <div className="quality">
          <button className="btn secondary sm" onClick={() => setMenu((m) => !m)} aria-haspopup="menu">
            ⚙ Quality
          </button>
          {menu && (
            <div className="quality-menu" role="menu">
              <button className={choice === "auto" ? "on" : ""} onClick={() => switchTo("auto")}>
                <span>Auto</span><span className="faint small">{autoPick ? `${renditions[autoPick].height}p` : ""}</span>
              </button>
              {options.map((k) => (
                <button key={k} className={choice === k ? "on" : ""} onClick={() => switchTo(k)}>
                  <span>{LABELS[k]}{renditions[k].height >= 720 && <span className="hd">HD</span>}</span>
                  <span className="faint small">{renditions[k].height}p · {(renditions[k].size / 1e6).toFixed(0)} MB</span>
                </button>
              ))}
            </div>
          )}
        </div>
        {onDownload && active && (
          <a className="btn sm" href={onDownload(active)}>⬇ Download MP4 ({renditions[active].height}p)</a>
        )}
      </div>
    </div>
  );
});

export default VideoPlayer;
