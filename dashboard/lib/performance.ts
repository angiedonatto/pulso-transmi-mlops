import { getSupabaseReader, readRows, type ReadIssue } from "@/lib/supabase";

const PREDICTION_SCAN_ROWS = 2400;
const RECENT_CYCLES = 6;

type ModelRow = {
  model_id: string;
  model_name: string;
  model_type: string;
  trained_until: string;
  artifact_uri: string | null;
  is_active: boolean;
};

type PredictionRow = {
  cycle_id: string | null;
  station_id: string;
  target_observed_at: string;
  horizon_minutes: number;
  prediction: number;
  created_at: string;
};

type CycleRow = {
  cycle_id: string;
  data_cutoff: string;
  closes_at: string;
  expected_predictions: number;
};

type ObservationRow = {
  station_id: string;
  observed_at: string;
  demand: number;
};

type ScoredPrediction = PredictionRow & {
  cycle_id: string;
  actual: number;
  abs_error: number;
  signed_error: number;
};

export type PerformancePoint = {
  cycle_id: string;
  data_cutoff: string | null;
  accuracy: number | null;
  rolling6_accuracy: number | null;
  mae: number | null;
  bias_pct: number | null;
  prediction_count: number;
  resolved_targets: number;
  expected_predictions: number | null;
};

export type GroupMetric = {
  key: string;
  label: string;
  accuracy: number | null;
  mae: number | null;
  bias_pct: number | null;
  count: number;
  actual_sum: number;
};

export type PerformanceOverview = {
  source: "supabase" | "not_configured";
  fetched_at: string;
  status: "HEALTHY" | "WATCH" | "INSUFFICIENT_DATA" | "UNKNOWN";
  issues: ReadIssue[];
  champion: ModelRow | null;
  history: PerformancePoint[];
  stations: GroupMetric[];
  horizons: GroupMetric[];
  recent_accuracy: number | null;
  reference_accuracy: number | null;
  accuracy_drop_pp: number | null;
  latest_cycle: PerformancePoint | null;
  resolved_predictions: number;
  scanned_predictions: number;
};

export async function getPerformanceOverview(): Promise<PerformanceOverview> {
  const fetchedAt = new Date().toISOString();
  const db = getSupabaseReader();
  if (!db) return emptyOverview("not_configured", fetchedAt, []);

  const activeModels = await readRows<ModelRow>(
    "model_versions",
    db.select("model_versions", "model_id,model_name,model_type,trained_until,artifact_uri,is_active").eq("is_active", true).order("trained_until", { ascending: false }).limit(2),
  );
  if (!activeModels.ok) return emptyOverview("supabase", fetchedAt, [activeModels.issue]);

  const issues: ReadIssue[] = [];
  const champion = activeModels.rows[0] ?? null;
  if (!champion) return emptyOverview("supabase", fetchedAt, issues);

  const predictions = await readRows<PredictionRow>(
    "predictions",
    db
      .select("predictions", "cycle_id,station_id,target_observed_at,horizon_minutes,prediction,created_at")
      .eq("model_id", champion.model_id)
      .not("cycle_id", "is", null)
      .order("created_at", { ascending: false })
      .limit(PREDICTION_SCAN_ROWS),
  );
  if (!predictions.ok) return emptyOverview("supabase", fetchedAt, [predictions.issue], champion);

  const predictionRows = predictions.rows.filter((row): row is PredictionRow & { cycle_id: string } => typeof row.cycle_id === "string" && row.cycle_id.length > 0);
  if (!predictionRows.length) return buildOverview({ fetchedAt, champion, issues, scored: [], cycles: [], scannedPredictions: 0 });

  const cycleIds = unique(predictionRows.map((row) => row.cycle_id));
  const stationIds = unique(predictionRows.map((row) => row.station_id));
  const targetTimes = unique(predictionRows.map((row) => normalizeTimestamp(row.target_observed_at)));

  const [cycles, observations] = await Promise.all([
    readRows<CycleRow>("forecast_cycles", db.select("forecast_cycles", "cycle_id,data_cutoff,closes_at,expected_predictions").in("cycle_id", cycleIds)),
    readRows<ObservationRow>("observations", db.select("observations", "station_id,observed_at,demand").in("station_id", stationIds).in("observed_at", targetTimes)),
  ]);
  if (!cycles.ok) issues.push(cycles.issue);
  if (!observations.ok) issues.push(observations.issue);

  const actualByTarget = new Map<string, number>();
  if (observations.ok) {
    for (const row of observations.rows) {
      actualByTarget.set(targetKey(row.station_id, row.observed_at), row.demand);
    }
  }

  const scored: ScoredPrediction[] = [];
  for (const row of predictionRows) {
    const actual = actualByTarget.get(targetKey(row.station_id, row.target_observed_at));
    if (actual === undefined || !Number.isFinite(actual)) continue;
    const prediction = Number(row.prediction);
    if (!Number.isFinite(prediction)) continue;
    scored.push({
      ...row,
      prediction,
      actual,
      abs_error: Math.abs(prediction - actual),
      signed_error: prediction - actual,
    });
  }

  return buildOverview({
    fetchedAt,
    champion,
    issues,
    scored,
    cycles: cycles.ok ? cycles.rows : [],
    scannedPredictions: predictionRows.length,
  });
}

function buildOverview({
  fetchedAt,
  champion,
  issues,
  scored,
  cycles,
  scannedPredictions,
}: {
  fetchedAt: string;
  champion: ModelRow;
  issues: ReadIssue[];
  scored: ScoredPrediction[];
  cycles: CycleRow[];
  scannedPredictions: number;
}): PerformanceOverview {
  const cycleById = new Map(cycles.map((cycle) => [cycle.cycle_id, cycle]));
  const byCycle = groupBy(scored, (row) => row.cycle_id);
  const history: PerformancePoint[] = [...byCycle.entries()]
    .map(([cycleId, rows]) => {
      const cycle = cycleById.get(cycleId);
      const summary = summarize(rows);
      return {
        cycle_id: cycleId,
        data_cutoff: cycle?.data_cutoff ?? latest(rows.map((row) => row.target_observed_at)),
        accuracy: summary.accuracy,
        rolling6_accuracy: null,
        mae: summary.mae,
        bias_pct: summary.bias_pct,
        prediction_count: rows.length,
        resolved_targets: rows.length,
        expected_predictions: cycle?.expected_predictions ?? null,
      };
    })
    .sort((a, b) => String(a.data_cutoff ?? a.cycle_id).localeCompare(String(b.data_cutoff ?? b.cycle_id)));

  for (let index = 0; index < history.length; index += 1) {
    const window = history.slice(Math.max(0, index - RECENT_CYCLES + 1), index + 1).map((point) => point.accuracy).filter(isNumber);
    history[index].rolling6_accuracy = window.length ? average(window) : null;
  }

  const latestCycles = history.slice(-RECENT_CYCLES).map((point) => point.cycle_id);
  const referenceCycles = history.slice(0, Math.max(0, history.length - RECENT_CYCLES)).map((point) => point.cycle_id);
  const recentRows = scored.filter((row) => latestCycles.includes(row.cycle_id));
  const referenceRows = scored.filter((row) => referenceCycles.includes(row.cycle_id));
  const recent = summarize(recentRows);
  const reference = summarize(referenceRows);
  const drop = isNumber(recent.accuracy) && isNumber(reference.accuracy) ? reference.accuracy - recent.accuracy : null;
  const status = !scored.length ? "INSUFFICIENT_DATA" : drop !== null && drop > 3 ? "WATCH" : "HEALTHY";

  return {
    source: "supabase",
    fetched_at: fetchedAt,
    status,
    issues,
    champion,
    history,
    stations: metricsBy(scored, (row) => row.station_id, (key) => key).sort(sortByAccuracy),
    horizons: metricsBy(scored, (row) => String(row.horizon_minutes), (key) => `${key} min`).sort((a, b) => Number(a.key) - Number(b.key)),
    recent_accuracy: recent.accuracy,
    reference_accuracy: reference.accuracy,
    accuracy_drop_pp: drop,
    latest_cycle: history.at(-1) ?? null,
    resolved_predictions: scored.length,
    scanned_predictions: scannedPredictions,
  };
}

function emptyOverview(source: PerformanceOverview["source"], fetchedAt: string, issues: ReadIssue[], champion: ModelRow | null = null): PerformanceOverview {
  return {
    source,
    fetched_at: fetchedAt,
    status: source === "not_configured" ? "UNKNOWN" : "INSUFFICIENT_DATA",
    issues,
    champion,
    history: [],
    stations: [],
    horizons: [],
    recent_accuracy: null,
    reference_accuracy: null,
    accuracy_drop_pp: null,
    latest_cycle: null,
    resolved_predictions: 0,
    scanned_predictions: 0,
  };
}

function metricsBy(rows: ScoredPrediction[], keyFor: (row: ScoredPrediction) => string, labelFor: (key: string) => string): GroupMetric[] {
  return [...groupBy(rows, keyFor).entries()].map(([key, group]) => ({ key, label: labelFor(key), ...summarize(group), count: group.length }));
}

function summarize(rows: ScoredPrediction[]): Omit<GroupMetric, "key" | "label" | "count"> {
  const actualSum = rows.reduce((sum, row) => sum + Math.max(0, row.actual), 0);
  const absError = rows.reduce((sum, row) => sum + row.abs_error, 0);
  const signedError = rows.reduce((sum, row) => sum + row.signed_error, 0);
  return {
    accuracy: actualSum > 0 ? Math.max(0, 100 * (1 - absError / actualSum)) : null,
    mae: rows.length ? absError / rows.length : null,
    bias_pct: actualSum > 0 ? 100 * (signedError / actualSum) : null,
    actual_sum: actualSum,
  };
}

function groupBy<T>(rows: T[], keyFor: (row: T) => string): Map<string, T[]> {
  const grouped = new Map<string, T[]>();
  for (const row of rows) {
    const key = keyFor(row);
    grouped.set(key, [...(grouped.get(key) ?? []), row]);
  }
  return grouped;
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function targetKey(stationId: string, timestamp: string): string {
  return `${stationId}|${normalizeTimestamp(timestamp)}`;
}

function normalizeTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? timestamp : date.toISOString();
}

function latest(values: string[]): string | null {
  return values.map((value) => new Date(value).getTime()).filter(Number.isFinite).sort((a, b) => b - a).map((value) => new Date(value).toISOString())[0] ?? null;
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sortByAccuracy(a: GroupMetric, b: GroupMetric): number {
  if (a.accuracy === null) return 1;
  if (b.accuracy === null) return -1;
  return a.accuracy - b.accuracy;
}
