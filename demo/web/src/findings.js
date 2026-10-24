import { useEffect, useState } from "react";
import { api } from "./api.js";

// The audit's headline numbers, computed live from the saved smd_ablation runs
// through the API (never hard-coded). Null until loaded, or if that experiment is missing.
const EXP = "smd_ablation";
const arm = (norm, adapt) => `m2n2__normalization-${norm}__adapt-${adapt ? "True" : "False"}`;

export function useFindings(enabled) {
  const [findings, setFindings] = useState(null);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    (async () => {
      try {
        const abl = (await api.experiments()).find((e) => e.name === EXP);
        if (!abl) return;
        const [dt, tta, run] = await Promise.all([
          api.compare({ experiment: EXP, a: arm("ema", false), b: arm("none", false), metrics: ["f1"] }),
          api.compare({ experiment: EXP, a: arm("ema", true), b: arm("ema", false), metrics: ["auroc"] }),
          abl.datasets.includes("smd_1-7")
            ? api.run(EXP, "smd_1-7", arm("ema", true), "fixed", abl.seeds[0] ?? 0).catch(() => null)
            : null,
        ]);
        const d = dt.summary[0];
        const t = tta.summary[0];
        const stats = run?.detector_stats || {};
        if (!alive) return;
        setFindings({
          dtF1: { diff: d.mean_diff, up: d.datasets_up, n: d.n_datasets, p: d.wilcoxon_p },
          ttaAuroc: { diff: t.mean_diff, up: t.datasets_up, down: t.datasets_down, n: t.n_datasets, p: t.wilcoxon_p },
          update: run
            ? { rate: stats.normal_step_rate ?? stats.update_rate, anomalyRate: run.test_anomaly_rate, machine: "1-7" }
            : null,
          size: { runs: abl.n_runs, machines: abl.datasets.length, arms: abl.arms.length, seeds: abl.seeds.length },
        });
      } catch {
        /* API offline or experiment incomplete: show no findings rather than wrong ones */
      }
    })();
    return () => {
      alive = false;
    };
  }, [enabled]);
  return findings;
}

export const signed = (x, digits = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(digits);
export const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;
