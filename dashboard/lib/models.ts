import "server-only";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { getSupabaseReader, readRows, type ReadIssue, type SupabaseReader } from "@/lib/supabase";
import { utcTimestamp } from "@/lib/format";

// Candidate I is resolved by model_name (never by model_id); the champion by is_active.
export const SHADOW_MODEL_NAME = "candidate_i_f4_oos_g2_snr_gate";

// ~10 cycles x 48 targets: enough to find the latest cycle both models predicted.
const CYCLE_SCAN_ROWS = 480;
// Shadow Candidate I runs hourly after the champion's official run, so trailing the
// champion by one cycle is expected; two or more means the shadow workflow is stalling.
const SHADOW_LAG_WATCH = 2;
const DELIVERED_STATUSES = new Set(["accepted", "idempotent_replay"]);
const MODEL_COLUMNS = "model_id,run_id,model_name,model_type,artifact_uri,trained_until,is_active,params,feature_config";

export type HealthStatus = "HEALTHY" | "WATCH" | "FAILED" | "INSUFFICIENT_DATA" | "UNKNOWN";
export type CheckState = "pass" | "watch" | "fail" | "unknown";
export type Check = { label: string; state: CheckState; detail: string };

export type ModelVersion = {
  model_id: string;
  run_id: string;
  model_name: string;
  model_type: string;
  artifact_uri: string | null;
  trained_until: string;
  is_active: boolean;
  params: Record<string, unknown> | null;
  feature_config: Record<string, unknown> | null;
};

type CycleRow = { cycle_id: string; data_cutoff: string; closes_at: string; expected_predictions: number };
type TargetRow = { station_id: string; target_at: string };
type PredictionRow = { model_id: string; station_id: string; target_observed_at: string; created_at: string };
type ReceiptRow = { model_id: string; status: string; accepted_at: string | null };
type TraceRow = Record<string, unknown> & { model_id: string; station_id: string };
type RecentCycle = { cycle_id: string; predictions: number; last_created_at: string };

export type ModelCoverage = {
  role: "champion" | "shadow";
  model_id: string;
  model_name: string;
  predictions: number;
  aligned: boolean;
  last_created_at: string | null;
};

export type CycleComparison = {
  cycle: CycleRow;
  cycle_targets: number;
  models: ModelCoverage[];
  shared_targets: number;
  resolved_targets: number | null;
};

export type ChampionHealth = {
  status: HealthStatus;
  checks: Check[];
  latest_cycle_id: string | null;
};

export type ShadowHealth = {
  status: HealthStatus;
  checks: Check[];
  predictions: number | null;
  expected_predictions: number | null;
  receipts_total: number | null;
  receipts_official: number | null;
  lag_cycles: number | null;
  lag_is_lower_bound: boolean;
};

export type AdaptationRow = {
  station_id: string;
  shadow_factor: number | null;
  gate: number | null;
  z: number | null;
  history_slots: number | null;
  champion_factor: number | null;
};

export type Adaptation =
  | { state: "available"; cycle_id: string; rows: AdaptationRow[]; has_gate: boolean; has_z: boolean; has_champion: boolean }
  | { state: "unavailable"; title: string; detail: string };

export type ModelsOverview = {
  source: "supabase" | "not_configured";
  fetched_at: string;
  issues: ReadIssue[];
  champion: ModelVersion | null;
  shadow: ModelVersion | null;
  champion_health: ChampionHealth;
  shadow_health: ShadowHealth;
  comparison: CycleComparison | null;
  adaptation: Adaptation;
};

export async function getModelsOverview(): Promise<ModelsOverview> {
  const fetchedAt = new Date().toISOString();
  const db = getSupabaseReader();
  if (!db) return emptyOverview("not_configured", fetchedAt, []);

  const models = await readRows<ModelVersion>("model_versions", db.select("model_versions", MODEL_COLUMNS).order("trained_until", { ascending: false }));
  if (!models.ok) return emptyOverview("supabase", fetchedAt, [models.issue]);

  const issues: ReadIssue[] = [];
  const active = models.rows.filter((model) => model.is_active);
  const champion = active.length === 1 ? active[0] : null;
  const shadow = models.rows.find((model) => model.model_name === SHADOW_MODEL_NAME) ?? null;

  const [championCycles, shadowCycles, receiptsTotal, receiptsOfficial] = await Promise.all([
    champion ? recentCycles(db, champion.model_id) : null,
    shadow ? recentCycles(db, shadow.model_id) : null,
    shadow ? readRows("submission_receipts", db.select("submission_receipts", "receipt_id", { count: "exact", head: true }).eq("model_id", shadow.model_id)) : null,
    shadow
      ? readRows("submission_receipts", db.select("submission_receipts", "receipt_id", { count: "exact", head: true }).eq("model_id", shadow.model_id).eq("is_official", true))
      : null,
  ]);
  for (const result of [championCycles, shadowCycles, receiptsTotal, receiptsOfficial]) {
    if (result && !result.ok) issues.push(result.issue);
  }

  const championList = championCycles?.ok ? championCycles.rows : [];
  const shadowList = shadowCycles?.ok ? shadowCycles.rows : [];
  const commonCycleId = champion && shadow ? await latestSharedCycle(db, champion.model_id, championList, shadowList) : null;
  const latestChampionCycle = championList[0] ?? null;

  const [latestCycleRow, latestReceipt, comparisonData] = await Promise.all([
    latestChampionCycle ? readRows<CycleRow>("forecast_cycles", db.select("forecast_cycles", "cycle_id,data_cutoff,closes_at,expected_predictions").eq("cycle_id", latestChampionCycle.cycle_id).limit(1)) : null,
    latestChampionCycle
      ? readRows<ReceiptRow>("submission_receipts", db.select("submission_receipts", "model_id,status,accepted_at").eq("cycle_id", latestChampionCycle.cycle_id).eq("is_official", true).limit(1))
      : null,
    commonCycleId && champion && shadow ? loadComparison(db, commonCycleId, champion, shadow) : null,
  ]);
  for (const result of [latestCycleRow, latestReceipt]) {
    if (result && !result.ok) issues.push(result.issue);
  }
  if (comparisonData) issues.push(...comparisonData.issues);

  const comparison = comparisonData?.comparison ?? null;
  const shadowCoverage = comparison?.models.find((model) => model.role === "shadow") ?? null;
  const lagIndex = commonCycleId ? championList.findIndex((cycle) => cycle.cycle_id === commonCycleId) : -1;
  const lag = commonCycleId ? (lagIndex >= 0 ? lagIndex : championList.length) : null;

  return {
    source: "supabase",
    fetched_at: fetchedAt,
    issues,
    champion,
    shadow,
    champion_health: championHealth(active.length, champion, latestChampionCycle, latestCycleRow?.ok ? latestCycleRow.rows[0] ?? null : null, latestReceipt),
    shadow_health: shadowHealth({
      champion,
      shadow,
      comparison,
      shadowCoverage,
      hasShadowPredictions: shadowList.length > 0,
      receiptsTotal: receiptsTotal?.ok ? receiptsTotal.count : null,
      receiptsOfficial: receiptsOfficial?.ok ? receiptsOfficial.count : null,
      lag,
      lagIsLowerBound: commonCycleId !== null && lagIndex < 0,
    }),
    comparison,
    adaptation: comparisonData?.adaptation ?? {
      state: "unavailable",
      title: "Trace data aún no está persistido",
      detail: shadow ? "Aún no hay un ciclo con predictions del champion y del shadow." : `${SHADOW_MODEL_NAME} no está registrado en model_versions.`,
    },
  };
}

// Distinct cycles (newest first) from the model's latest prediction rows. Only the newest
// cycle's count is used; the oldest cycle in the window may be cut off by the row limit.
async function recentCycles(db: SupabaseReader, modelId: string) {
  const result = await readRows<{ cycle_id: string; created_at: string }>(
    "predictions",
    db.select("predictions", "cycle_id,created_at").eq("model_id", modelId).not("cycle_id", "is", null).order("created_at", { ascending: false }).limit(CYCLE_SCAN_ROWS),
  );
  if (!result.ok) return result;
  const cycles = new Map<string, RecentCycle>();
  for (const row of result.rows) {
    const cycle = cycles.get(row.cycle_id);
    if (cycle) cycle.predictions += 1;
    else cycles.set(row.cycle_id, { cycle_id: row.cycle_id, predictions: 1, last_created_at: row.created_at });
  }
  return { ok: true as const, rows: [...cycles.values()], count: null };
}

async function latestSharedCycle(db: SupabaseReader, championId: string, championList: RecentCycle[], shadowList: RecentCycle[]): Promise<string | null> {
  const championIds = new Set(championList.map((cycle) => cycle.cycle_id));
  const inWindow = shadowList.find((cycle) => championIds.has(cycle.cycle_id));
  if (inWindow) return inWindow.cycle_id;
  // The shadow trails further than the champion scan window: check its newest cycles directly.
  for (const cycle of shadowList.slice(0, 3)) {
    const result = await readRows("predictions", db.select("predictions", "prediction_id", { count: "exact", head: true }).eq("cycle_id", cycle.cycle_id).eq("model_id", championId));
    if (result.ok && (result.count ?? 0) > 0) return cycle.cycle_id;
  }
  return null;
}

async function loadComparison(db: SupabaseReader, cycleId: string, champion: ModelVersion, shadow: ModelVersion) {
  const issues: ReadIssue[] = [];
  const modelIds = [champion.model_id, shadow.model_id];
  const [cycles, targets, predictions, traces] = await Promise.all([
    readRows<CycleRow>("forecast_cycles", db.select("forecast_cycles", "cycle_id,data_cutoff,closes_at,expected_predictions").eq("cycle_id", cycleId).limit(1)),
    readRows<TargetRow>("forecast_cycle_targets", db.select("forecast_cycle_targets", "station_id,target_at").eq("cycle_id", cycleId)),
    readRows<PredictionRow>("predictions", db.select("predictions", "model_id,station_id,target_observed_at,created_at").eq("cycle_id", cycleId).in("model_id", modelIds)),
    readRows<TraceRow>("prediction_traces", db.select("prediction_traces", "*").eq("cycle_id", cycleId).in("model_id", modelIds)),
  ]);
  for (const result of [cycles, targets, predictions]) {
    if (!result.ok) issues.push(result.issue);
  }
  const adaptation = buildAdaptation(cycleId, champion, shadow, traces);
  if (!cycles.ok || !cycles.rows[0] || !targets.ok || !predictions.ok) return { issues, comparison: null, adaptation };

  const cycleKeys = new Set(targets.rows.map((row) => targetKey(row.station_id, row.target_at)));
  const coverage = (role: ModelCoverage["role"], model: ModelVersion): ModelCoverage => {
    const rows = predictions.rows.filter((row) => row.model_id === model.model_id);
    const keys = new Set(rows.map((row) => targetKey(row.station_id, row.target_observed_at)));
    return {
      role,
      model_id: model.model_id,
      model_name: model.model_name,
      predictions: rows.length,
      aligned: rows.length > 0 && sameSet(keys, cycleKeys),
      last_created_at: rows.map((row) => row.created_at).sort().at(-1) ?? null,
    };
  };
  const models = [coverage("champion", champion), coverage("shadow", shadow)];
  const championKeys = new Set(predictions.rows.filter((row) => row.model_id === champion.model_id).map((row) => targetKey(row.station_id, row.target_observed_at)));
  const shared = predictions.rows.filter((row) => row.model_id === shadow.model_id && championKeys.has(targetKey(row.station_id, row.target_observed_at))).length;

  const resolved = await resolvedTargets(cycleId);

  return {
    issues,
    adaptation,
    comparison: {
      cycle: cycles.rows[0],
      cycle_targets: cycleKeys.size,
      models,
      shared_targets: shared,
      resolved_targets: resolved,
    } satisfies CycleComparison,
  };
}

// Coverage is taken only from the canonical snapshot for the exact same cycle.
// The frontend never downloads observations or computes scoring/coverage from demand.
async function resolvedTargets(cycleId: string) {
  const { snapshot } = await getMonitoringSnapshot();
  const cycle = snapshot?.report.current_cycle;
  const counts = cycle?.resolved_targets as { actual?: unknown } | undefined;
  return cycle?.cycle_id === cycleId && typeof counts?.actual === "number" ? counts.actual : null;
}

function buildAdaptation(cycleId: string, champion: ModelVersion, shadow: ModelVersion, traces: Awaited<ReturnType<typeof readRows<TraceRow>>>): Adaptation {
  if (!traces.ok) {
    return {
      state: "unavailable",
      title: traces.issue.kind === "missing_table" ? "Trace data aún no está persistido" : "Trace data no disponible",
      detail: traces.issue.kind === "missing_table" ? `${traces.issue.message} Los factores de adaptación por estación no se pueden leer hasta que exista.` : traces.issue.message,
    };
  }
  const shadowRows = traces.rows.filter((row) => row.model_id === shadow.model_id);
  if (!shadowRows.length) {
    return { state: "unavailable", title: "Trace data aún no está persistido", detail: `No hay filas en prediction_traces para ${shadow.model_name} en el ciclo ${cycleId}.` };
  }
  const championFactors = new Map<string, number | null>();
  for (const row of traces.rows) {
    if (row.model_id === champion.model_id && !championFactors.has(row.station_id)) championFactors.set(row.station_id, num(row.calibration_factor));
  }
  const byStation = new Map<string, AdaptationRow>();
  for (const row of shadowRows) {
    if (byStation.has(row.station_id)) continue;
    byStation.set(row.station_id, {
      station_id: row.station_id,
      shadow_factor: num(row.calibration_factor),
      gate: num(row.gate),
      z: num(row.z),
      history_slots: num(row.feedback_samples),
      champion_factor: championFactors.get(row.station_id) ?? null,
    });
  }
  return {
    state: "available",
    cycle_id: cycleId,
    rows: [...byStation.values()].sort((a, b) => a.station_id.localeCompare(b.station_id)),
    has_gate: shadowRows.some((row) => "gate" in row),
    has_z: shadowRows.some((row) => "z" in row),
    has_champion: championFactors.size > 0,
  };
}

function championHealth(
  activeCount: number,
  champion: ModelVersion | null,
  latest: RecentCycle | null,
  cycle: CycleRow | null,
  receipt: Awaited<ReturnType<typeof readRows<ReceiptRow>>> | null,
): ChampionHealth {
  const checks: Check[] = [
    activeCount === 1
      ? { label: "Registry", state: "pass", detail: "Exactamente una fila en model_versions tiene is_active=true." }
      : { label: "Registry", state: "fail", detail: activeCount === 0 ? "Ningún modelo tiene is_active=true." : `${activeCount} modelos tienen is_active=true.` },
  ];
  if (champion && !latest) {
    checks.push({ label: "Latest cycle", state: "unknown", detail: "No encontré predictions del champion." });
  }
  if (champion && latest) {
    const expected = cycle?.expected_predictions ?? null;
    checks.push(
      expected === null
        ? { label: "Latest cycle", state: "unknown", detail: `${latest.predictions} predictions; conteo esperado no disponible.` }
        : { label: "Latest cycle", state: latest.predictions === expected ? "pass" : "watch", detail: `${latest.predictions}/${expected} predictions persistidas.` },
    );
    checks.push(officialReceiptCheck(champion, cycle, receipt));
  }
  if (latest) {
    const age = Date.now() - Date.parse(latest.last_created_at);
    checks.push({ label: "Freshness", state: !Number.isFinite(age) || age < -300000 ? "unknown" : age > 900000 ? "watch" : "pass",
      detail: `Última predicción registrada: ${utcTimestamp(latest.last_created_at)}. Este registro no certifica actividad en tiempo real.` });
  }
  return { status: rollup(checks, !champion || !latest), checks, latest_cycle_id: latest?.cycle_id ?? null };
}

function officialReceiptCheck(champion: ModelVersion, cycle: CycleRow | null, receipt: Awaited<ReturnType<typeof readRows<ReceiptRow>>> | null): Check {
  if (!receipt || !receipt.ok) return { label: "Official submission", state: "unknown", detail: "No se pudo leer submission_receipts." };
  const row = receipt.rows[0];
  if (row && DELIVERED_STATUSES.has(row.status) && row.model_id === champion.model_id) {
    return { label: "Official submission", state: "pass", detail: `${row.status} en ${utcTimestamp(row.accepted_at)}.` };
  }
  if (row) return { label: "Official submission", state: "watch", detail: row.model_id !== champion.model_id ? `El receipt oficial pertenece a otro model_id (${row.status}).` : `El receipt oficial está en ${row.status}.` };
  const closed = cycle ? Date.parse(cycle.closes_at) < Date.now() : false;
  return closed
    ? { label: "Official submission", state: "fail", detail: "El ciclo cerró sin receipt oficial." }
    : { label: "Official submission", state: "watch", detail: "Esperando el receipt oficial del último ciclo." };
}

function shadowHealth(input: {
  champion: ModelVersion | null;
  shadow: ModelVersion | null;
  comparison: CycleComparison | null;
  shadowCoverage: ModelCoverage | null;
  hasShadowPredictions: boolean;
  receiptsTotal: number | null;
  receiptsOfficial: number | null;
  lag: number | null;
  lagIsLowerBound: boolean;
}): ShadowHealth {
  const { champion, shadow, comparison, shadowCoverage, receiptsTotal, receiptsOfficial, lag, lagIsLowerBound } = input;
  const expected = comparison?.cycle.expected_predictions ?? null;
  const checks: Check[] = [
    shadow
      ? { label: "Registered", state: "pass", detail: `${SHADOW_MODEL_NAME} encontrado en model_versions.` }
      : { label: "Registered", state: "fail", detail: `${SHADOW_MODEL_NAME} no encontrado en model_versions.` },
  ];
  if (shadow) {
    checks.push(
      shadow.is_active
        ? { label: "Shadow mode", state: "fail", detail: "is_active=true: un modelo shadow nunca debe estar activo." }
        : { label: "Shadow mode", state: "pass", detail: "is_active=false." },
      champion && champion.model_id !== shadow.model_id
        ? { label: "Champion unaffected", state: "pass", detail: `${champion.model_name} es el único modelo activo.` }
        : { label: "Champion unaffected", state: "fail", detail: "No hay un champion activo único distinto del shadow." },
      receiptsTotal === null
        ? { label: "Official submissions", state: "unknown", detail: "No se pudo leer submission_receipts." }
        : receiptsTotal === 0
          ? { label: "Official submissions", state: "pass", detail: "0 filas en submission_receipts para el modelo shadow." }
          : { label: "Official submissions", state: "fail", detail: `${receiptsTotal} filas en submission_receipts (${receiptsOfficial ?? "?"} oficiales) para el modelo shadow.` },
    );
    if (!comparison || !shadowCoverage) {
      checks.push({ label: "Shadow coverage", state: "unknown", detail: input.hasShadowPredictions ? "No hay ciclo compartido con el champion." : "Aún no hay shadow predictions persistidas." });
    } else {
      const aligned = comparison.models.every((model) => model.aligned);
      checks.push(
        {
          label: "Shadow coverage",
          state: shadowCoverage.predictions === expected ? "pass" : "watch",
          detail: `${shadowCoverage.predictions}/${expected} predictions en el último ciclo compartido.`,
        },
        {
          label: "Target alignment",
          state: aligned ? "pass" : "watch",
          detail: aligned ? "Champion y shadow predijeron exactamente los targets del ciclo." : "Los targets del champion y shadow difieren de forecast_cycle_targets.",
        },
        lagCheck(lag, lagIsLowerBound),
      );
    }
  }
  return {
    status: rollup(checks, !shadow || !input.hasShadowPredictions),
    checks,
    predictions: shadowCoverage?.predictions ?? null,
    expected_predictions: expected,
    receipts_total: receiptsTotal,
    receipts_official: receiptsOfficial,
    lag_cycles: lag,
    lag_is_lower_bound: lagIsLowerBound,
  };
}

function lagCheck(lag: number | null, lowerBound: boolean): Check {
  if (lag === null) return { label: "Freshness", state: "unknown", detail: "No se pudo determinar el lag del shadow." };
  if (lag === 0) return { label: "Freshness", state: "pass", detail: "Shadow al día con el último ciclo del champion." };
  if (lag < SHADOW_LAG_WATCH) return { label: "Freshness", state: "pass", detail: "1 ciclo detrás del champion; la siguiente corrida horaria del shadow lo cubre." };
  return { label: "Freshness", state: "watch", detail: `${lowerBound ? "Al menos " : ""}${lag} ciclos detrás del champion.` };
}

function rollup(checks: Check[], insufficient: boolean): HealthStatus {
  if (checks.some((check) => check.state === "fail")) return "FAILED";
  if (insufficient) return "INSUFFICIENT_DATA";
  if (checks.some((check) => check.state === "watch")) return "WATCH";
  if (checks.some((check) => check.state === "unknown")) return "UNKNOWN";
  return "HEALTHY";
}

function emptyOverview(source: ModelsOverview["source"], fetchedAt: string, issues: ReadIssue[]): ModelsOverview {
  const detail = source === "not_configured" ? "Supabase no está configurado para el dashboard." : "No se pudo leer model_versions.";
  return {
    source,
    fetched_at: fetchedAt,
    issues,
    champion: null,
    shadow: null,
    champion_health: { status: "UNKNOWN", checks: [{ label: "Registry", state: "unknown", detail }], latest_cycle_id: null },
    shadow_health: {
      status: "UNKNOWN",
      checks: [{ label: "Registered", state: "unknown", detail }],
      predictions: null,
      expected_predictions: null,
      receipts_total: null,
      receipts_official: null,
      lag_cycles: null,
      lag_is_lower_bound: false,
    },
    comparison: null,
    adaptation: { state: "unavailable", title: "Trace data aún no está persistido", detail },
  };
}

function targetKey(stationId: string, timestamp: string): string {
  const date = new Date(timestamp);
  return `${stationId}|${Number.isNaN(date.getTime()) ? timestamp : date.toISOString()}`;
}

function sameSet(a: Set<string>, b: Set<string>): boolean {
  return a.size === b.size && [...a].every((key) => b.has(key));
}

function num(value: unknown): number | null {
  const parsed = typeof value === "string" ? Number(value) : value;
  return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : null;
}
