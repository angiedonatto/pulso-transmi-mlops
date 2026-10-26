import { DataUnavailable } from "@/components/data-unavailable";
import { AlertTriangle, ServerCog, ShieldAlert } from "lucide-react";
import { DataTable } from "@/components/data-table";
import { MetricCard } from "@/components/metric-card";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { numberValue, shortDate, text } from "@/lib/format";

export default async function OperationsPage() {
  const result = await getMonitoringSnapshot();
  if (!result.snapshot) return <DataUnavailable message={result.message} />;
  const snapshot = result.snapshot;
  const latest = snapshot.report.latest_pipeline_run;
  const ops = snapshot.report.operations;

  return (
    <div className="space-y-5">
      <section className="grid gap-3 md:grid-cols-3">
        <MetricCard icon={ServerCog} label="Último pipeline" value={text(latest?.status, "Sin corrida")} status={text(latest?.status, "UNKNOWN")} detail={`Inició ${shortDate(latest?.started_at)}`} />
        <MetricCard icon={AlertTriangle} label="Corridas fallidas" value={numberValue(ops.failed_pipeline_runs)} status={typeof ops.failed_pipeline_runs !== "number" ? "UNKNOWN" : ops.failed_pipeline_runs > 0 ? "FAILED" : "HEALTHY"} detail="Desde pipeline_runs" />
        <MetricCard icon={ShieldAlert} label="Eventos críticos de ingestión" value={numberValue(ops.critical_ingestion_events)} status={typeof ops.critical_ingestion_events !== "number" ? "UNKNOWN" : ops.critical_ingestion_events > 0 ? "FAILED" : "HEALTHY"} detail="Eventos de validación runtime" />
      </section>
      <section className="grid gap-5 xl:grid-cols-2">
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Alertas del snapshot</h2>
          <DataTable columns={["severity", "timestamp", "category", "title", "entity"]} rows={snapshot.report.alerts as unknown as Array<Record<string, unknown>>} />
        </div>
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Últimos eventos de ingestión</h2>
          <DataTable columns={["event_type", "severity", "created_at", "reason_codes"]} rows={snapshot.report.runtime_ingestion_events} />
        </div>
      </section>
    </div>
  );
}
