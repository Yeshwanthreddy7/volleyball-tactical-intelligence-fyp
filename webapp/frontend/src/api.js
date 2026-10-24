// Thin API client. Every browser gets a random client id (kept in
// localStorage) so "My analyses" only lists the visitor's own uploads.

function makeId() {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

let memoryId = null;
export function clientId() {
  try {
    let id = localStorage.getItem("vta-client-id");
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

async function request(path, { method = "GET", body, headers = {} } = {}) {
  const res = await fetch(path, {
    method,
    headers: {
      "X-Client-Id": clientId(),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

export const api = {
  health: () => request("/api/health"),
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
  `/api/analyses/${id}/files/${name}${download ? "?download=1" : ""}`;
export const repoFile = (path) => `/api/project/file?path=${encodeURIComponent(path)}`;

function putChunk(url, blob, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
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
