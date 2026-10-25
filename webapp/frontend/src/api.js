// Thin API client. Every browser gets a random client id (kept in
// localStorage) so "My analyses" only lists the visitor's own uploads.

function makeId() {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

let memoryId = null;
export function clientId() {
  try {
    let id = localStorage.getItem("vta-client-id") || memoryId;
    if (!id) {
      id = makeId();
      localStorage.setItem("vta-client-id", id);
    }
    return id;
  } catch {
    memoryId = memoryId || makeId();
    return memoryId;
  }
}

// The client id doubles as a "history code": entering it on another device
// shows the same uploads; resetting it starts a fresh, empty history.
export const VALID_CODE = /^[A-Za-z0-9-]{16,64}$/;
export function setClientId(id) {
  const next = id || makeId();
  memoryId = next;
  try { localStorage.setItem("vta-client-id", next); } catch { /* memory only */ }
  return next;
}

// On the static (Vercel) site, uploads and analyses go to the project's GPU
// laptop when it is online: its tunnel URL is published as live_url in
// /api/health. Portal pages always come from the site itself.
let LIVE = "";
export const OFFLINE =
  "The analysis server is not running. Start it with webapp\\run_website.bat (or open the Desktop app), then refresh this page.";
const ANALYSIS = /^\/api\/(analyses|uploads)(\/|$|\?)/;
const base = (path) => (LIVE && ANALYSIS.test(path) ? LIVE : "") + path;
export const apiUrl = base;

async function probeLive(url) {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 25000); // a sleeping cloud server needs time to wake
    const r = await fetch(url.replace(/\/$/, "") + "/api/health", { signal: ctl.signal });
    clearTimeout(t);
    const h = await r.json();
    return h.status === "ok" ? h : null;
  } catch {
    return null;
  }
}

// resolves once we know whether a live analysis server is reachable
export const ready = (async () => {
  try {
    const h = await (await fetch("/api/health")).json();
    if (h.status === "static" && h.live_url) {
      const live = await probeLive(h.live_url);
      if (live) {
        LIVE = h.live_url.replace(/\/$/, "");
        return { ...live, via: LIVE };
      }
    }
    return h;
  } catch {
    return null;
  }
})();

async function request(path, { method = "GET", body, headers = {} } = {}) {
  if (ANALYSIS.test(path)) await ready;
  let res;
  try {
    res = await fetch(base(path), {
    method,
    headers: {
      "X-Client-Id": clientId(),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error(OFFLINE);
  }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

export const api = {
  health: () => ready.then((h) => h || request("/api/health")),
  info: () => request("/api/project/info"),
  metrics: () => request("/api/project/metrics"),
  docs: () => request("/api/project/docs"),
  doc: (slug) => request(`/api/project/docs/${slug}`),
  figures: () => request("/api/project/figures"),
  demos: () => request("/api/project/demos"),
  analyses: () => request("/api/analyses"),
  analysis: (id) => request(`/api/analyses/${id}`),
  create: (body) => request("/api/analyses", { method: "POST", body }),
  cancel: (id) => request(`/api/analyses/${id}/cancel`, { method: "POST" }),
  remove: (id) => request(`/api/analyses/${id}`, { method: "DELETE" }),
};

export const fileUrl = (id, name, download = false) =>
  base(`/api/analyses/${id}/files/${name}${download ? "?download=1" : ""}`);
export const repoFile = (path) => `/api/project/file?path=${encodeURIComponent(path)}`;

function putChunk(url, blob, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", base(url));
    xhr.setRequestHeader("X-Client-Id", clientId());
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else {
        let msg = `Upload failed (${xhr.status})`;
        try { msg = JSON.parse(xhr.responseText).detail || msg; } catch { /* keep default */ }
        reject(Object.assign(new Error(msg), { status: xhr.status }));
      }
    };
    xhr.onerror = () => reject(Object.assign(new Error("Network error during upload"), { status: 0 }));
    xhr.onabort = () => reject(Object.assign(new Error("Upload cancelled"), { aborted: true }));
    if (signal) signal.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(blob);
  });
}

// Uploads `file` in chunks (Cloudflare caps a single request at 100 MB),
// retrying each chunk up to 4 times. onProgress(bytesSent, total).
export async function uploadFile(file, onProgress, signal) {
  await ready;
  const init = await request("/api/uploads", { method: "POST", body: { filename: file.name, size: file.size } });
  const { upload_id: uid, chunk_bytes: size, chunks } = init;
  let done = 0;
  for (let i = 0; i < chunks; i++) {
    const blob = file.slice(i * size, Math.min(file.size, (i + 1) * size));
    for (let attempt = 0; ; attempt++) {
      try {
        await putChunk(`/api/uploads/${uid}/chunks/${i}`, blob, (n) => onProgress(done + n, file.size), signal);
        break;
      } catch (err) {
        if (err.aborted || attempt >= 3 || (err.status >= 400 && err.status < 500)) throw err;
        await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
      }
    }
    done += blob.size;
    onProgress(done, file.size);
  }
  return uid;
}

// ---- keep-alive: an analysis runs only while the website is open somewhere.
// Every open page sends a heartbeat for this browser's unfinished analyses;
// the server cancels an analysis when the heartbeats stop (website closed).
const ACTIVE_KEY = "vta-active-jobs";

function readActive() {
  try { return JSON.parse(localStorage.getItem(ACTIVE_KEY) || "[]"); } catch { return []; }
}
function writeActive(ids) {
  try { localStorage.setItem(ACTIVE_KEY, JSON.stringify([...new Set(ids)].slice(-20))); } catch { /* private mode */ }
}
export function trackJob(id) {
  writeActive([...readActive(), id]);
}

export async function sendHeartbeat() {
  const ids = readActive();
  if (!ids.length) return;
  try {
    const r = await request("/api/analyses/heartbeat", { method: "POST", body: { ids } });
    const still = ids.filter((id) => ["queued", "processing"].includes(r.status?.[id]));
    writeActive(still);
  } catch { /* server offline: try again on the next beat */ }
}

export function startHeartbeat() {
  sendHeartbeat();
  const t = setInterval(sendHeartbeat, 20000);
  const onVisible = () => document.visibilityState === "visible" && sendHeartbeat();
  document.addEventListener("visibilitychange", onVisible);
  return () => { clearInterval(t); document.removeEventListener("visibilitychange", onVisible); };
}
