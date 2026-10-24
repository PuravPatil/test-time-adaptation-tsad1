import { pct, signed } from "../findings.js";

export default function AuditSection({ findings: f }) {
  return (
    <section id="audit" className="audit" aria-labelledby="audit-title">
      <h2 id="audit-title" className="section-title">What the audit found.</h2>
      {!f ? (
        <p className="muted" style={{ gridColumn: "1 / -1", margin: 0, fontSize: 15 }}>
          The findings appear here once the API is online and the <span className="mono">smd_ablation</span> experiment
          has been run.
        </p>
      ) : (
        <>
          <div className="finding">
            <span className="num">{signed(f.dtF1.diff, 2)}</span>
            <p>
              F1 from detrending alone (EMA vs none, no adaptation), higher on {f.dtF1.up} of {f.dtF1.n} SMD machines.
            </p>
          </div>
          <div className="finding grey">
            <span className="num">{signed(f.ttaAuroc.diff, 3)}</span>
            <p>
              AUROC from adding test-time updates on top of detrending: up on {f.ttaAuroc.up}, down on {f.ttaAuroc.down}
              {" "}machines (Wilcoxon p = {f.ttaAuroc.p?.toFixed(2)}).
            </p>
          </div>
          {f.update?.rate != null && (
            <div className="finding hot">
              <span className="num">{pct(f.update.rate)}</span>
              <p>
                of timesteps used for model updates on machine {f.update.machine}
                {f.update.anomalyRate != null ? `, where ${pct(f.update.anomalyRate, 1)} are labelled anomalies` : ""}.
              </p>
            </div>
          )}
          <p className="muted" style={{ gridColumn: "1 / -1", margin: 0, fontSize: 13 }}>
            {f.size.runs} runs: {f.size.machines} machines × {f.size.arms} arms × {f.size.seeds} seeds, read live from
            results/runs/smd_ablation through the API.
          </p>
        </>
      )}
    </section>
  );
}
