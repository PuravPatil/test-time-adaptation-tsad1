// Thin client for the FastAPI backend (demo/api). Every call goes to /api,
// which Vite (dev) or nginx (Docker) forwards to the API server.
const BASE = import.meta.env.VITE_API_URL ?? "/api";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function detailOf(body, status) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
  if (status === 0) return "The API is not reachable. Is it running on port 8000?";
  return `Request failed (HTTP ${status})`;
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, options);
  } catch {
    throw new ApiError(detailOf(null, 0), 0);
  }
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) throw new ApiError(detailOf(body, res.status), res.status);
  return body;
}

const enc = encodeURIComponent;

export const api = {
  health: () => request("/health"),
  experiments: () => request("/experiments"),
  run: (exp, ds, arm, thr, seed) => request(`/runs/${enc(exp)}/${enc(ds)}/${enc(arm)}/${enc(thr)}/${seed}`),
  compare: ({ experiment, a, b, threshold = "fixed", metrics = ["auroc"] }) => {
    const q = new URLSearchParams({ experiment, a, b, threshold });
    metrics.forEach((m) => q.append("metrics", m));
    return request(`/compare?${q}`);
  },
  samples: () => request("/samples"),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/uploads", { method: "POST", body: form });
  },
  createJob: (body) =>
    request("/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  job: (id) => request(`/jobs/${enc(id)}`),
  jobSeries: (id, points = 600) => request(`/jobs/${enc(id)}/series?points=${points}`),
  downloadUrl: (id) => `${BASE}/jobs/${enc(id)}/download`,

  /** Open the job's WebSocket. handlers: {status, metrics, series, end, error}. Returns a close function. */
  streamJob(id, handlers, { points = 600, chunk = 30, delay = 0.03 } = {}) {
    const path = `/jobs/${enc(id)}/stream?points=${points}&chunk=${chunk}&delay=${delay}`;
    const url = BASE.startsWith("http")
      ? BASE.replace(/^http/, "ws") + path
      : `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${BASE}${path}`;
    const ws = new WebSocket(url);
    let ended = false;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "end") ended = true;
      handlers[msg.type]?.(msg);
    };
    ws.onclose = () => {
      if (!ended) handlers.error?.({ detail: "stream closed early" });
    };
    return () => {
      ended = true;
      ws.close();
    };
  },
};
