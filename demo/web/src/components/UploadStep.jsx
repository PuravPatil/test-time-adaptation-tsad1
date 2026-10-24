import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

const FORMATS = [".csv", ".tsv", ".txt", ".xlsx", ".parquet", ".json", ".npy"];

export default function UploadStep({ maxMb, onUploaded, onJob }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState(null);
  const [sample, setSample] = useState("");
  const [starting, setStarting] = useState(false);
  const input = useRef(null);

  useEffect(() => {
    api.samples().then(
      (list) => {
        setSamples(list);
        if (list.length) setSample((list.find((s) => s.name === "smd_1-7") || list[0]).name);
      },
      () => setSamples([])
    );
  }, []);

  async function send(file) {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    try {
      onUploaded(await api.upload(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function runSample() {
    setStarting(true);
    setError(null);
    try {
      const job = await api.createJob({ sample, method: "m2n2" });
      onJob(job.job_id);
    } catch (e) {
      setError(e.message);
      setStarting(false);
    }
  }

  return (
    <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="row">
        <label
          className={`dropzone ${over ? "over" : ""} ${busy ? "busy" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            send(e.dataTransfer.files?.[0]);
          }}
        >
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#C6F432" strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 16V4M6 10l6-6 6 6M4 20h16" />
          </svg>
          <span className="big-text">{busy ? "Reading your file…" : "Drop a file, or browse"}</span>
          <span className="muted" style={{ maxWidth: 460, lineHeight: 1.5 }}>
            A numeric time series: one row per timestep, one column per sensor
            {maxMb ? `. Up to ${maxMb} MB.` : "."}
          </span>
          <input
            ref={input}
            className="sr-only"
            type="file"
            accept={FORMATS.join(",")}
            disabled={busy}
            onChange={(e) => {
              send(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <span className="btn" aria-hidden="true">Browse files</span>
          <span className="chips">
            {FORMATS.map((f) => (
              <span key={f} className="chip">{f}</span>
            ))}
          </span>
        </label>

        <div className="side">
          {samples && samples.length > 0 && (
            <div className="sample-card">
              <span className="label-caps">No file handy?</span>
              <span className="title">Run a sample from the Server Machine Dataset</span>
              <label className="field" style={{ color: "#0B0D12" }}>
                <span className="sr-only">Sample</span>
                <select value={sample} onChange={(e) => setSample(e.target.value)}>
                  {samples.map((s) => (
                    <option key={s.name} value={s.name}>{s.name.replace("smd_", "SMD machine ")}</option>
                  ))}
                </select>
              </label>
              <button type="button" className="btn dark" onClick={runSample} disabled={starting || !sample}>
                {starting ? "Starting…" : "Run M2N2 on it"}
              </button>
            </div>
          )}
          <div className="card facts">
            <strong>What works</strong>
            <span>
              Any numeric series, univariate or multivariate. The model trains on the normal part of your data, then
              adapts as it streams the rest.
            </span>
            <strong style={{ marginTop: 6 }}>What doesn't</strong>
            <span>Free text, images, or tables without a time order.</span>
          </div>
        </div>
      </div>
      {error && <div className="error-box" role="alert">{error}</div>}
    </div>
  );
}
