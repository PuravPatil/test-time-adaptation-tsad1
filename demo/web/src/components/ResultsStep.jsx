import { useEffect, useState } from "react";
import { api } from "../api.js";
import { METHODS } from "./MapStep.jsx";
import ScoreChart from "./ScoreChart.jsx";

const POINTS = 600;
const STAGES = [
  ["loading", "Loading"],
  ["training", "Training"],
  ["scoring", "Streaming"],
  ["evaluating", "Evaluating"],
];
const ORDER = ["queued", "loading", "training", "scoring", "evaluating", "done"];
const EMPTY = { scores: [], preds: [], labels: [], n: 0, total: 0 };
const pick = (m) => ({ state: m.state, stage: m.stage, progress: m.progress, error: m.error });

export default function ResultsStep({ jobId, onRestart }) {
  const [status, setStatus] = useState({ state: "queued", stage: "queued", progress: 0, error: null });
  const [request, setRequest] = useState(null);
  const [info, setInfo] = useState(null); // metrics, tau, detector_stats, dataset, ms_per_step
  const [series, setSeries] = useState(EMPTY);
  const [live, setLive] = useState(true);

  useEffect(() => {
    let closed = false;
    let timer = null;
    let ended = false;
    api.job(jobId).then((j) => !closed && setRequest(j.request), () => {});

    // REST fallback if the WebSocket cannot connect (e.g. a proxy without WS support)
    const poll = async () => {
      try {
        const j = await api.job(jobId);
        if (closed) return;
        setStatus(pick(j));
        if (j.state === "done") {
          setInfo(j);
          const s = await api.jobSeries(jobId, POINTS);
          if (closed) return;
          setSeries({ scores: s.scores, preds: s.preds, labels: s.labels || [], n: s.n, total: s.scores.length });
          setLive(false);
        } else if (j.state === "failed") {
          setLive(false);
        } else {
          timer = setTimeout(poll, 800);
        }
      } catch (e) {
        if (!closed) {
          setStatus((s) => ({ ...s, state: "failed", error: e.message }));
          setLive(false);
        }
      }
    };

    const close = api.streamJob(jobId, {
      status: (m) => setStatus(pick(m)),
      metrics: (m) => {
        setInfo(m);
        setSeries(EMPTY);
      },
      series: (m) =>
        setSeries((prev) => ({
          scores: prev.scores.concat(m.scores),
          preds: prev.preds.concat(m.preds),
          labels: m.labels ? prev.labels.concat(m.labels) : prev.labels,
          n: m.n,
          total: Math.min(m.n, POINTS),
        })),
      end: () => {
        ended = true;
        setLive(false);
      },
      error: () => {
        if (!ended && !closed) poll();
      },
    }, { points: POINTS, chunk: 20, delay: 0.03 });

    return () => {
      closed = true;
      close();
      clearTimeout(timer);
    };
  }, [jobId]);

  const failed = status.state === "failed";
  const done = status.state === "done" && !live;
  const stageIdx = ORDER.indexOf(status.stage);
  const stats = info?.detector_stats || {};
  const metrics = info?.metrics;
  const ds = info?.dataset;
  const adapted = request?.method === "m2n2";

  return (
    <div id="results" className="fade-up" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="chart-head">
          <h3>
            {request ? (request.sample ? request.sample.replace("smd_", "SMD machine ") : "Your upload") : "Run"}
            {request && <span className="muted" style={{ fontWeight: 500 }}> · {METHODS[request.method] || request.method}</span>}
          </h3>
          <div className="stages" aria-hidden="true">
            {STAGES.map(([key, label]) => {
              const i = ORDER.indexOf(key);
              const cls = stageIdx > i || status.stage === "done" ? "done" : stageIdx === i && !failed ? "active" : "";
              return <span key={key} className={`stage ${cls}`}>{label}</span>;
            })}
          </div>
        </div>
        <div className="bar" aria-hidden="true"><div style={{ width: `${Math.round((status.progress || 0) * 100)}%` }} /></div>
        <p className="sr-only" aria-live="polite">
          {failed ? `Job failed: ${status.error}` : status.state === "done" ? "Job finished" : `Job ${status.stage}`}
        </p>
        {failed && <div className="error-box" role="alert">This run failed: {status.error}</div>}
      </div>

      {info && (
        <div className="metric-grid">
          {metrics ? (
            <>
              <Metric k="AUROC" v={fmt(metrics.auroc)} />
              <Metric k="AUPRC" v={fmt(metrics.auprc)} />
              <Metric k="F1" v={fmt(metrics.f1)} sub={`F1-PA ${fmt(metrics.f1_pa)} (reference)`} />
            </>
          ) : (
            <div className="metric" style={{ gridColumn: "span 2" }}>
              <span className="k">No label column</span>
              <span className="sub" style={{ fontSize: 15, lineHeight: 1.5 }}>
                Scores and alarms only. Add a 0/1 label column to get AUROC, AUPRC and F1.
              </span>
            </div>
          )}
          {!metrics && <Metric k="τ (threshold)" v={info.tau > 0 ? sci(info.tau) : "moving"} sub="score at or above τ = alarm" />}
          <Metric
            hot={adapted}
            k={adapted ? "Steps used to adapt" : "Adaptation"}
            v={adapted && stats.normal_step_rate != null ? pct(stats.normal_step_rate) : "off"}
            sub={ds?.test_anomaly_rate != null ? `labelled anomalies: ${pct(ds.test_anomaly_rate, 1)}` : undefined}
          />
        </div>
      )}

      <div className="card chart" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="chart-head">
          <span className="label-caps">
            {live && series.scores.length > 0 ? "Streaming the test part…" : done ? "Test part" : "Waiting for scores…"}
          </span>
          <span className="legend">
            <span><span style={{ width: 18, height: 3, background: "#C6F432" }} />score</span>
            {info?.tau > 0 && <span><span style={{ width: 18, borderTop: "2px dashed #F3F4F6" }} />τ</span>}
            {series.labels.length > 0 && <span><span style={{ width: 14, height: 10, background: "#FF9466", opacity: 0.45 }} />true anomaly</span>}
            <span><span style={{ width: 14, height: 6, background: "#FF9466" }} />alarm</span>
          </span>
        </div>
        <ScoreChart scores={series.scores} preds={series.preds} labels={series.labels} total={series.total}
          tau={info?.tau} live={live} n={series.n || ds?.n_test} />
      </div>

      {info && <AllMetrics metrics={metrics} info={info} stats={stats} adapted={adapted} ds={ds} />}

      {ds && (
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <span className="label-caps">What was run</span>
          <span className="muted" style={{ lineHeight: 1.6 }}>
            Trained on {ds.n_train?.toLocaleString()} timesteps (assumed normal), streamed {ds.n_test?.toLocaleString()}
            {ds.n_features ? ` · ${ds.n_features} features` : ""}
            {info?.ms_per_step != null ? ` · ${msFmt(info.ms_per_step)} ms per timestep` : ""}
            {stats.n_updates != null && adapted ? ` · ${stats.n_updates.toLocaleString()} model updates` : ""}
          </span>
          {ds.warnings?.length > 0 && (
            <ul className="warn-list">{ds.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          )}
        </div>
      )}

      <div className="row" style={{ gap: 12 }}>
        <a className={`btn ${done ? "primary" : ""}`} href={done ? api.downloadUrl(jobId) : undefined}
          aria-disabled={!done} style={done ? undefined : { opacity: 0.5, pointerEvents: "none" }}>
          Download scores (.csv)
        </a>
        <button type="button" className="btn" onClick={onRestart}>Upload another</button>
      </div>
    </div>
  );
}

function Metric({ k, v, sub, hot }) {
  return (
    <div className={`metric ${hot ? "hot" : ""}`}>
      <span className="k">{k}</span>
      <span className="v">{v}</span>
      {sub && <span className="sub">{sub}</span>}
    </div>
  );
}

// Every metric the API returns (audit.evaluation.metrics.evaluate), with what it means.
// PA%K at K=0 equals F1-PA and at K=100 equals F1, so those two are not repeated.
const METRIC_ROWS = [
  ["auroc", "AUROC", "How well scores rank anomalies above normal points (threshold-free)"],
  ["auprc", "AUPRC", "Average precision over all thresholds (threshold-free)"],
  ["pr_auc", "PR-AUC", "Area under the precision-recall curve, the way the paper reports AUPRC"],
  ["f1", "F1", "Point-wise F1 at τ"],
  ["f1_pa", "F1-PA", "Point-adjusted: a whole anomaly counts if any point of it is caught"],
  ["pa_k_20", "PA%K · K=20", "An anomaly counts if more than 20% of it is caught"],
  ["pa_k_50", "PA%K · K=50", "An anomaly counts if more than 50% of it is caught"],
  ["pa_k_80", "PA%K · K=80", "An anomaly counts if more than 80% of it is caught"],
  ["f1_best", "Best F1", "Best F1 with a threshold chosen on the labels (reference only)"],
];

function AllMetrics({ metrics, info, stats, adapted, ds }) {
  const run = [
    ["τ (threshold)", info.tau > 0 ? sci(info.tau) : "moving (streaming)", "A score at or above τ raises an alarm"],
    ["Timesteps used to adapt", adapted && stats.normal_step_rate != null ? pct(stats.normal_step_rate, 1) : "off",
      "Timesteps below τ that updated the model"],
    ["Model updates", adapted && stats.n_updates != null ? stats.n_updates.toLocaleString() : "0",
      stats.n_windows != null ? `over ${stats.n_windows.toLocaleString()} windows` : "windows that updated the model"],
    ["Labelled anomalies", ds?.test_anomaly_rate != null ? pct(ds.test_anomaly_rate, 1) : "—", "Share of the test part"],
    ["Speed", info.ms_per_step != null ? `${msFmt(info.ms_per_step)} ms` : "—", "Per timestep, adaptation included"],
  ];
  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <span className="label-caps">All metrics</span>
      {metrics ? (
        <div className="all-metrics">
          {METRIC_ROWS.map(([key, label, hint]) => <Row key={key} label={label} value={fmt(metrics[key])} hint={hint} />)}
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }}>No label column, so accuracy metrics can't be computed for this run.</p>
      )}
      <span className="label-caps" style={{ marginTop: 8 }}>This run</span>
      <div className="all-metrics">
        {run.map(([label, value, hint]) => <Row key={label} label={label} value={value} hint={hint} />)}
      </div>
    </div>
  );
}

function Row({ label, value, hint }) {
  return (
    <div className="metric-row">
      <span className="k">{label}</span>
      <span className="v">{value}</span>
      <span className="hint">{hint}</span>
    </div>
  );
}

const fmt = (x) => (x == null ? "—" : x.toFixed(3));
const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;
const sci = (x) => (x >= 0.01 ? x.toFixed(4) : x.toExponential(2));
const msFmt = (x) => (x >= 0.01 ? x.toFixed(3) : x.toPrecision(2));
