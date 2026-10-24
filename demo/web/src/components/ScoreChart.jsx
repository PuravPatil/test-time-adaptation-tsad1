import { useMemo } from "react";

// Anomaly score on a log scale, the threshold τ, labelled anomalies (shaded) and
// alarms (ticks under the axis). `total` fixes the x scale so a live replay fills
// the chart left to right.
const X0 = 64, X1 = 944, Y0 = 14, Y1 = 214, ALARM_Y = 222;

function bands(bits, x, step, y, h) {
  let d = "";
  for (let i = 0; i < bits.length; ) {
    if (bits[i] > 0) {
      let j = i;
      while (j < bits.length && bits[j] > 0) j++;
      d += `M${(x(i) - step / 2).toFixed(1)} ${y}h${((j - i) * step).toFixed(1)}v${h}h${(-(j - i) * step).toFixed(1)}Z`;
      i = j;
    } else i++;
  }
  return d || "M0 0";
}

const tick = (k) => (k >= 4 || k < -3 ? `1e${k}` : k >= 0 ? String(10 ** k) : (10 ** k).toFixed(-k));

export default function ScoreChart({ scores, preds, labels, total, tau, live, n }) {
  const count = Math.max(total || scores.length, 1);
  const step = (X1 - X0) / count;
  const x = (i) => X0 + (i + 0.5) * step;
  const logs = useMemo(() => scores.map((s) => Math.log10(Math.max(s ?? 1e-12, 1e-12))), [scores]);
  const tauLog = tau > 0 ? Math.log10(tau) : null;

  const [lo, hi] = useMemo(() => {
    if (!logs.length) return [-4, 0];
    let mn = Infinity, mx = -Infinity;
    for (const v of logs) { if (v < mn) mn = v; if (v > mx) mx = v; }
    if (tauLog != null) { mn = Math.min(mn, tauLog); mx = Math.max(mx, tauLog); }
    const l = Math.floor(mn);
    return [l, Math.max(Math.ceil(mx), l + 1)];
  }, [logs, tauLog]);
  const y = (v) => Y1 - ((v - lo) / (hi - lo)) * (Y1 - Y0);

  const line = logs.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join("") || "M0 0";
  const ticks = [];
  const every = Math.ceil((hi - lo) / 6);
  for (let k = lo; k <= hi; k += every) ticks.push(k);

  return (
    <svg viewBox="0 0 960 250" role="img"
      aria-label={`Anomaly score over ${n ? n.toLocaleString() : "the"} test timesteps, with threshold, labelled anomalies and alarms`}>
      {labels && labels.length > 0 && <path d={bands(labels, x, step, Y0, Y1 - Y0)} fill="#FF9466" fillOpacity="0.22" />}
      <g stroke="#1F2430" strokeWidth="1">
        {ticks.map((k) => <line key={k} x1={X0} x2={X1} y1={y(k)} y2={y(k)} />)}
      </g>
      <g fontFamily="JetBrains Mono, monospace" fontSize="11" fill="#8E95A3" textAnchor="end">
        {ticks.map((k) => <text key={k} x={X0 - 8} y={y(k) + 4}>{tick(k)}</text>)}
      </g>
      <path key={live ? "live" : "done"} className={live ? undefined : "draw"} pathLength={live ? undefined : 1}
        d={line} fill="none" stroke="#C6F432" strokeWidth="1.6" strokeLinejoin="round" />
      {tauLog != null && (
        <line x1={X0} x2={X1} y1={y(tauLog)} y2={y(tauLog)} stroke="#F3F4F6" strokeWidth="1.2" strokeDasharray="6 5" />
      )}
      {preds && preds.length > 0 && <path d={bands(preds, x, step, ALARM_Y, 10)} fill="#FF9466" />}
      {live && scores.length > 0 && (
        <line x1={x(scores.length - 1)} x2={x(scores.length - 1)} y1={Y0} y2={ALARM_Y + 12} stroke="#F3F4F6" strokeOpacity="0.4" />
      )}
      <g fontFamily="JetBrains Mono, monospace" fontSize="11" fill="#8E95A3">
        <text x={X0} y="248">t = 0</text>
        <text x={X1} y="248" textAnchor="end">t = {n ? n.toLocaleString() : "…"}</text>
      </g>
    </svg>
  );
}
