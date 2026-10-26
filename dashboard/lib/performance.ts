import "server-only";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { isRecord } from "@/lib/snapshot";

export type PerformancePoint = {
  cycle_id: string; accepted_at: string | null; accuracy: number | null;
  rolling6_accuracy: number | null; rolling12_accuracy: number | null; mae: number | null; bias_pct: number | null;
  prediction_count: number | null; resolved_targets: number | null; expected_predictions: number | null;
};
export type GroupMetric = {
  key: string; label: string; accuracy: number | null; mae: number | null;
  bias_pct: number | null; count: number | null; actual_sum: number | null;
};

const number = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const rows = (value: unknown): Record<string, unknown>[] => Array.isArray(value) ? value.filter(isRecord) : [];

// Presentation only. Never join predictions to observations or calculate scoring,
// rolling windows, drift thresholds or a new health status in the frontend.
export async function getPerformanceOverview() {
  const result = await getMonitoringSnapshot();
  if (!result.snapshot) return { available: false as const, message: result.message };
  const snapshot = result.snapshot;
  const performance = snapshot.report.drift.performance;
  const history: PerformancePoint[] = rows(performance.cycle_accuracy).map((row) => ({
    cycle_id: String(row.cycle_id), accepted_at: typeof row.accepted_at === "string" ? row.accepted_at : null,
    accuracy: number(row.mean_station_accuracy ?? row.cycle_accuracy),
    rolling6_accuracy: number(row.rolling6_accuracy), rolling12_accuracy: number(row.rolling12_accuracy),
    mae: number(row.mae), bias_pct: null,
    prediction_count: number(row.count), resolved_targets: number(row.resolved_targets),
    expected_predictions: number(row.expected_predictions),
  }));
  const grouped = (value: unknown, key: string): GroupMetric[] => rows(value).map((row) => ({
    key: String(row[key]), label: String(row[key]) + (key === "horizon_minutes" ? " min" : ""),
    accuracy: number(row.recent_accuracy), mae: number(row.mae), bias_pct: null,
    count: number(row.count), actual_sum: number(row.actual_sum),
  }));
  return {
    available: true as const, snapshot,
    champion: snapshot.report.active_model,
    status: performance.status ?? "UNKNOWN",
    history, stations: grouped(performance.station_accuracy, "station_id"),
    horizons: grouped(performance.horizon_accuracy, "horizon_minutes"),
    recent_accuracy: number(performance.recent_accuracy),
    reference_accuracy: number(performance.reference_accuracy),
    accuracy_drop_pp: number(performance.accuracy_drop_pp),
    latest_cycle: history.at(-1) ?? null,
    resolved_predictions: number(performance.overall?.count),
  };
}
