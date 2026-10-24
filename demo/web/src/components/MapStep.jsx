import { useMemo, useState } from "react";
import { api } from "../api.js";

const NONE = "__none__";
export const METHODS = {
  m2n2: "M2N2 (detrend + adapt)",
  detrend_only: "Detrend only (frozen)",
  static: "Plain MLP (no detrend, no adapt)",
};

export default function MapStep({ upload, onBack, onJob }) {
  const numeric = upload.columns.filter((c) => c.numeric).map((c) => c.name);
  const textCols = upload.columns.filter((c) => !c.numeric).map((c) => c.name);
  const missing = upload.columns.reduce((n, c) => n + c.missing, 0);

  const [timeCol, setTimeCol] = useState(upload.time_col ? "auto" : NONE);
  const [labelCol, setLabelCol] = useState(upload.label_col ?? NONE);
  const [mode, setMode] = useState("all");
  const [picked, setPicked] = useState(() => new Set(numeric));
  const [trainFrac, setTrainFrac] = useState(0.5);
  const [method, setMethod] = useState("m2n2");
  const [windowSize, setWindowSize] = useState(5);
  const [threshold, setThreshold] = useState("fixed");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const resolvedTime = timeCol === "auto" ? upload.time_col : timeCol === NONE ? null : timeCol;
  const resolvedLabel = labelCol === NONE ? null : labelCol;
  const candidates = useMemo(
    () => numeric.filter((c) => c !== resolvedTime && c !== resolvedLabel),
    [numeric, resolvedTime, resolvedLabel]
  );
  const chosen = mode === "all" ? candidates : candidates.filter((c) => picked.has(c));
  const nTrain = Math.round(upload.n_rows * trainFrac);

  async function submit(e) {
    e.preventDefault();
    if (!chosen.length) return setError("Pick at least one feature column.");
    setBusy(true);
    setError(null);
    try {
      const job = await api.createJob({
        upload_id: upload.upload_id,
        method,
        time_col: timeCol === "auto" ? "auto" : resolvedTime,
        label_col: resolvedLabel,
        feature_cols: mode === "all" ? null : chosen,
        train_frac: trainFrac,
        window: Number(windowSize) || null,
        threshold,
      });
      onJob(job.job_id);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  const toggle = (c) =>
    setPicked((prev) => {
      const next = new Set(prev);
      next.has(c) ? next.delete(c) : next.add(c);
      return next;
    });

  return (
    <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div className="row">
        <form className="card form-grid" onSubmit={submit}>
          <div className="full file-head">
            <span className="name">{upload.filename}</span>
            <span className="mono muted" style={{ fontSize: 13 }}>
              {upload.n_rows.toLocaleString()} rows · {upload.n_columns} columns · {upload.n_numeric} numeric
            </span>
          </div>

          <label className="field">
            Time column
            <select value={timeCol} onChange={(e) => setTimeCol(e.target.value)}>
              <option value="auto">Auto-detect{upload.time_col ? ` (${upload.time_col})` : " (none found)"}</option>
              <option value={NONE}>Row order (no timestamp)</option>
              {upload.columns.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </label>

          <label className="field">
            Label column (optional)
            <select value={labelCol} onChange={(e) => setLabelCol(e.target.value)}>
              <option value={NONE}>None — scores and alarms only</option>
              {numeric.map((c) => (
                <option key={c} value={c}>{c}{c === upload.label_col ? " (guessed)" : ""}</option>
              ))}
            </select>
            <span className="hint">0 = normal, anything above 0 = anomaly. Needed for F1, AUROC and AUPRC.</span>
          </label>

          <label className="field">
            Features
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="all">All numeric columns ({candidates.length})</option>
              <option value="choose">Choose columns…</option>
            </select>
          </label>

          <label className="field">
            Model
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {Object.entries(METHODS).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
          </label>

          {mode === "choose" && (
            <fieldset className="full field" style={{ border: 0, padding: 0, margin: 0 }}>
              <legend style={{ marginBottom: 8 }}>Feature columns ({chosen.length} selected)</legend>
              <div className="checks">
                {candidates.map((c) => (
                  <label key={c}>
                    <input type="checkbox" checked={picked.has(c)} onChange={() => toggle(c)} />
                    {c}
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          <label className="field">
            Normal training part: first {Math.round(trainFrac * 100)}% ({nTrain.toLocaleString()} rows)
            <span className="range-row">
              <input type="range" min="0.1" max="0.9" step="0.05" value={trainFrac}
                onChange={(e) => setTrainFrac(Number(e.target.value))} />
            </span>
            <span className="hint">These rows are assumed normal and used for training.</span>
          </label>

          <div className="row" style={{ gap: 16 }}>
            <label className="field" style={{ flex: "1 1 120px" }}>
              Window
              <input type="number" min="2" max="200" value={windowSize} onChange={(e) => setWindowSize(e.target.value)} />
            </label>
            <label className="field" style={{ flex: "1 1 160px" }}>
              Threshold
              <select value={threshold} onChange={(e) => setThreshold(e.target.value)}>
                <option value="fixed">Fixed τ (paper setting)</option>
                <option value="streaming">Streaming τ</option>
              </select>
            </label>
          </div>

          {error && <div className="full error-box" role="alert">{error}</div>}
          <div className="full row" style={{ gap: 12 }}>
            <button type="submit" className="btn primary big" style={{ flex: "1 1 260px" }} disabled={busy}>
              {busy ? "Starting…" : "Train and stream"}
            </button>
            <button type="button" className="btn big" onClick={onBack}>Choose another file</button>
          </div>
        </form>

        <div className="card side" style={{ flex: "1 1 300px" }}>
          <span className="label-caps">Checks before training</span>
          <div className="check-list">
            <div><span className="ok">✓</span>{upload.n_numeric} numeric column{upload.n_numeric === 1 ? "" : "s"} found</div>
            {missing > 0
              ? <div><span className="warn">!</span>{missing.toLocaleString()} missing values will be filled from the previous timestep</div>
              : <div><span className="ok">✓</span>No missing values</div>}
            {textCols.filter((c) => c !== resolvedTime).length > 0 && (
              <div><span className="warn">!</span>Ignored (not numeric): {textCols.filter((c) => c !== resolvedTime).join(", ")}</div>
            )}
            {resolvedLabel
              ? <div><span className="ok">✓</span>Labels from “{resolvedLabel}”: metrics will be computed</div>
              : <div><span className="warn">!</span>No label column: you will get scores and alarms, not accuracy</div>}
            <div><span className="warn">!</span>Settings were tuned on SMD; on new data treat results as exploratory</div>
          </div>
        </div>
      </div>

      <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <span className="label-caps">First rows</span>
        <div className="table-box">
          <table>
            <thead>
              <tr>
                {upload.columns.map((c) => (
                  <th key={c.name} className={c.numeric ? "num" : undefined} scope="col">{c.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {upload.preview.map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => <td key={j}>{v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <span className="muted" style={{ fontSize: 13 }}>Green headers are numeric columns.</span>
      </div>
    </div>
  );
}
