import { pct, signed } from "../findings.js";

const GENERIC = [
  { text: "Upload any numeric time series" },
  { text: ".csv .tsv .txt .xlsx .parquet .json .npy" },
  { text: "Trains on the normal part, adapts as it streams", hot: true },
  { text: "Scores, alarms and metrics, live" },
];

function items(f) {
  if (!f) return GENERIC;
  const out = [
    { text: `Detrending: ${signed(f.dtF1.diff, 2)} F1 on ${f.dtF1.up}/${f.dtF1.n} machines` },
    { text: `Adaptation on top: ${signed(f.ttaAuroc.diff, 3)} AUROC` },
  ];
  if (f.update?.rate != null) out.push({ text: `${pct(f.update.rate)} of steps used for updates on ${f.update.machine}`, hot: true });
  out.push({ text: `${f.size.machines} machines · ${f.size.arms} arms · ${f.size.seeds} seeds` });
  return out;
}

export default function Ticker({ findings }) {
  const list = items(findings);
  return (
    <div className="ticker">
      <p className="sr-only">{list.map((i) => i.text).join(". ")}</p>
      <div className="marq" aria-hidden="true">
        {[...list, ...list].map((item, i) => (
          <span key={i} style={{ display: "contents" }}>
            <span className={item.hot ? "hot" : undefined}>{item.text}</span>
            <span className="star">✦</span>
          </span>
        ))}
      </div>
    </div>
  );
}
