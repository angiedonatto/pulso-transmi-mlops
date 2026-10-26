import { DataTable } from "@/components/data-table";
import { DataUnavailable } from "@/components/data-unavailable";
import { MetricCard } from "@/components/metric-card";
import { PerformanceCharts } from "@/components/performance-charts";
import { getPerformanceOverview, type GroupMetric, type PerformancePoint } from "@/lib/performance";
import { numberValue, pct, shortDate, text } from "@/lib/format";
import { AlertTriangle, BarChart3, Gauge, History, TrendingDown } from "lucide-react";

export default async function AccuracyPage() {
  const performance = await getPerformanceOverview();
  if (!performance.available) return <DataUnavailable message={performance.message} />;
  const latest = performance.latest_cycle;

  return (
    <div className="space-y-6">
      <section className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">Historial de desempeño</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">Accuracy del snapshot canónico</h2>
          <p className="mt-1 text-sm text-slate-400">
            Métricas persistidas por drift_monitor.py. Modelo registrado en el snapshot: {text(performance.champion?.model_name)}.
          </p>
        </div>
        <div className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-right text-xs text-slate-400">
          <p>Snapshot: {shortDate(performance.snapshot.generated_at)}</p>
          <p className="mt-1">Fuente: monitoring_snapshots · información histórica</p>
        </div>
      </section>

      <section className="grid gap-3 md:grid-cols-3">
        <MetricCard icon={Gauge} label="Accuracy reciente del snapshot" value={pct(performance.recent_accuracy)} status={performance.status} detail={`Referencia histórica: ${pct(performance.reference_accuracy)}`} />
        <MetricCard icon={BarChart3} label="Predicciones evaluadas" value={numberValue(performance.resolved_predictions)} detail="Conteo del reporte canónico; n/a si no está registrado." />
        <MetricCard icon={TrendingDown} label="Cambio vs referencia" value={performance.accuracy_drop_pp === null ? "n/a" : `${performance.accuracy_drop_pp.toFixed(1)} pp`} status={performance.status} detail="Positivo significa caída frente al historial." />
      </section>

      <section className="grid gap-3 md:grid-cols-3">
        <MetricCard icon={History} label="Último ciclo evaluado" value={text(latest?.cycle_id, "sin ciclo")} detail={`Aceptado: ${shortDate(latest?.accepted_at ?? null)}`} />
        <MetricCard icon={Gauge} label="Accuracy último ciclo" value={pct(latest?.accuracy)} detail={`${numberValue(latest?.resolved_targets)} targets resueltos`} />
        <MetricCard icon={BarChart3} label="MAE último ciclo" value={latest?.mae === null || latest?.mae === undefined ? "n/a" : latest.mae.toFixed(1)} detail={`Bias: ${pct(latest?.bias_pct)}`} />
      </section>

      {performance.history.length ? (
        <PerformanceCharts history={performance.history} stations={performance.stations} horizons={performance.horizons} />
      ) : (
        <Notice title="Aún no hay historial evaluable">
          El snapshot no contiene una serie evaluable de ciclos oficiales resueltos. Las métricas ausentes se muestran como n/a.
        </Notice>
      )}

      <section className="grid gap-5 xl:grid-cols-2">
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Desempeño por estación</h2>
          <DataTable columns={["estacion", "accuracy", "mae", "bias_pct", "registros"]} rows={stationRows(performance.stations)} />
        </div>
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Desempeño por horizonte</h2>
          <DataTable columns={["horizonte", "accuracy", "mae", "bias_pct", "registros"]} rows={horizonRows(performance.horizons)} />
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold text-white">Historial por ciclo</h2>
        <DataTable columns={["ciclo", "aceptado_en", "accuracy", "rolling_6", "mae", "targets_resueltos", "targets_esperados"]} rows={cycleRows(performance.history)} />
      </section>
    </div>
  );
}

function stationRows(rows: GroupMetric[]) {
  return rows.map((row) => ({
    estacion: row.label,
    accuracy: pct(row.accuracy),
    mae: row.mae === null ? "n/a" : row.mae.toFixed(1),
    bias_pct: pct(row.bias_pct),
    registros: row.count,
  }));
}

function horizonRows(rows: GroupMetric[]) {
  return rows.map((row) => ({
    horizonte: row.label,
    accuracy: pct(row.accuracy),
    mae: row.mae === null ? "n/a" : row.mae.toFixed(1),
    bias_pct: pct(row.bias_pct),
    registros: row.count,
  }));
}

function cycleRows(rows: PerformancePoint[]) {
  return rows.slice().reverse().map((row) => ({
    ciclo: row.cycle_id,
    aceptado_en: shortDate(row.accepted_at),
    accuracy: pct(row.accuracy),
    rolling_6: pct(row.rolling6_accuracy),
    mae: row.mae === null ? "n/a" : row.mae.toFixed(1),
    targets_resueltos: row.resolved_targets,
    targets_esperados: row.expected_predictions ?? "n/a",
  }));
}

function Notice({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-amber-300/25 bg-amber-300/[0.06] p-4 text-sm text-amber-50">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div>
        <p className="font-semibold">{title}</p>
        <p className="mt-1 text-slate-300">{children}</p>
      </div>
    </div>
  );
}
