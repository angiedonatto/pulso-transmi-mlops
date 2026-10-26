import { DataUnavailable } from "@/components/data-unavailable";
import { AlertChart } from "@/components/analytics";
import { RadioTower, RefreshCw, SignalHigh } from "lucide-react";
import { DataTable } from "@/components/data-table";
import { MetricCard } from "@/components/metric-card";
import { StatusPill } from "@/components/status-pill";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { text } from "@/lib/format";

export default async function DriftPage() {
  const result = await getMonitoringSnapshot();
  if (!result.snapshot) return <DataUnavailable message={result.message} />;
  const snapshot = result.snapshot;
  const drift = snapshot.report.drift;
  const dataDrift = drift.data_drift as { status?: string; alerts?: Array<Record<string, unknown>>; station_metrics?: Array<Record<string, unknown>> };
  const residual = drift.residual_drift as { status?: string; by_station?: Array<Record<string, unknown>>; by_horizon?: Array<Record<string, unknown>> };

  return (
    <div className="space-y-5">
      <AlertChart alerts={snapshot.report.alerts} />
      <section className="grid gap-3 md:grid-cols-3">
        <MetricCard icon={RadioTower} label="Drift general" value={drift.status} status={drift.status} detail={drift.reasons.join(" · ") || "Sin razones de drift"} />
        <MetricCard icon={SignalHigh} label="Data drift" value={text(dataDrift.status, "UNKNOWN")} status={text(dataDrift.status, "UNKNOWN")} detail="PSI, KS y checks de distribución desde drift_monitor.py" />
        <MetricCard icon={RefreshCw} label="Retrain" value={drift.retrain_recommended ? "Recomendado" : "No recomendado en el snapshot"} status={drift.retrain_recommended ? "DRIFT" : "HEALTHY"} detail="Calculado por la lógica canónica de drift" />
      </section>
      <div className="rounded-lg border border-white/10 bg-[#10151d]/88 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-white">Razones canónicas</h2>
          <StatusPill status={drift.status} />
        </div>
        <ul className="mt-4 grid gap-2 md:grid-cols-2">
          {(drift.reasons.length ? drift.reasons : ["No se reportaron razones en el snapshot."]).map((reason) => (
            <li key={reason} className="rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-sm text-slate-300">{reason}</li>
          ))}
        </ul>
      </div>
      <section className="grid gap-5 xl:grid-cols-2">
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Alertas por estación</h2>
          <DataTable columns={["station_id", "type", "severity", "recent_accuracy", "delta_pp", "signed_relative_bias"]} rows={drift.station_alerts as Array<Record<string, unknown>>} />
        </div>
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Alertas por horizonte</h2>
          <DataTable columns={["horizon_minutes", "type", "severity", "recent_accuracy", "delta_pp", "signed_relative_bias"]} rows={drift.horizon_alerts as Array<Record<string, unknown>>} />
        </div>
      </section>
      <section className="grid gap-5 xl:grid-cols-2">
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Métricas de data drift</h2>
          <DataTable columns={["station_id", "median_relative_change", "dispersion_relative_change", "psi", "ks_statistic", "severity"]} rows={(dataDrift.station_metrics || dataDrift.alerts || []) as Array<Record<string, unknown>>} />
        </div>
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Bias residual por estación</h2>
          <DataTable columns={["station_id", "bias", "signed_relative_bias", "mae", "count"]} rows={(residual.by_station || []) as Array<Record<string, unknown>>} />
        </div>
      </section>
    </div>
  );
}
