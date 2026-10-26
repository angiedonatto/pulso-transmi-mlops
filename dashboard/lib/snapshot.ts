import type { MonitoringSnapshot } from "@/lib/types";

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function records(value: unknown): value is Record<string, unknown>[] {
  return Array.isArray(value) && value.every(isRecord);
}

// Accept the canonical snapshot contract, including empty metric dictionaries
// for insufficient data. Reject malformed JSON instead of crashing or inventing data.
export function parseSnapshot(value: unknown): MonitoringSnapshot | null {
  if (!isRecord(value) || typeof value.generated_at !== "string" || !Number.isFinite(Date.parse(value.generated_at))) return null;
  if (typeof value.overall_status !== "string" || typeof value.drift_status !== "string" || typeof value.retrain_recommended !== "boolean") return null;
  if (value.model_id !== null && typeof value.model_id !== "string") return null;
  const report = value.report;
  if (!isRecord(report) || !isRecord(report.drift)) return null;
  const drift = report.drift;
  if (typeof drift.status !== "string" || typeof drift.retrain_recommended !== "boolean") return null;
  if (!Array.isArray(drift.reasons) || !drift.reasons.every((r) => typeof r === "string")) return null;
  if (!["performance", "data_drift", "residual_drift"].every((key) => isRecord(drift[key]))) return null;
  if (!["station_alerts", "horizon_alerts"].every((key) => records(drift[key]))) return null;
  if (!["latest_observation", "current_cycle", "data_quality", "operations", "prediction_traces"].every((key) => isRecord(report[key]))) return null;
  if (!["active_model", "latest_submission", "latest_pipeline_run"].every((key) => report[key] === null || isRecord(report[key]))) return null;
  if (!["alerts", "stream_cursors", "runtime_ingestion_events"].every((key) => records(report[key]))) return null;
  if (!(report.alerts as Record<string, unknown>[]).every((a) => ["severity", "timestamp", "category", "title", "explanation"].every((key) => typeof a[key] === "string"))) return null;
  for (const key of ["cycle_accuracy", "station_accuracy", "horizon_accuracy"]) {
    const rows = (drift.performance as Record<string, unknown>)[key];
    if (rows !== undefined && !records(rows)) return null;
  }
  for (const [source, keys] of [[drift.data_drift, ["station_metrics", "alerts"]], [drift.residual_drift, ["by_station", "by_horizon"]]] as const) {
    for (const key of keys) {
      const rows = (source as Record<string, unknown>)[key];
      if (rows !== undefined && !records(rows)) return null;
    }
  }
  return value as MonitoringSnapshot;
}

export function snapshotIsStale(generatedAt: string, now = Date.now()): boolean {
  const age = now - Date.parse(generatedAt);
  return !Number.isFinite(age) || age < -5 * 60_000 || age > 15 * 60_000;
}
