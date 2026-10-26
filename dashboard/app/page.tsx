import { DataUnavailable } from "@/components/data-unavailable";
import { AlertTriangle, BrainCircuit, CheckCircle2, Gauge, RadioTower, Send, ServerCog } from "lucide-react";
import { MetricCard } from "@/components/metric-card";
import { StatusPill } from "@/components/status-pill";
import { DataTable } from "@/components/data-table";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { numberValue, pct, shortDate, text } from "@/lib/format";

export default async function OverviewPage() {
  const result = await getMonitoringSnapshot();
  if (!result.snapshot) return <DataUnavailable message={result.message} />;
  const snapshot = result.snapshot;
  const { report } = snapshot;
  const perf = report.drift.performance;
  const cycle = report.current_cycle;
  const model = report.active_model;
  const latestSubmission = report.latest_submission;

  return (
    <div className="space-y-6">
      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={Gauge} label="Estado del sistema" value={snapshot.overall_status} status={snapshot.overall_status} detail={`${report.alerts.length} alertas del snapshot`} />
        <MetricCard icon={BrainCircuit} label="Modelo del snapshot" value={text(model?.model_name, "Sin champion")} detail={text(model?.model_type)} status={model ? "HEALTHY" : "FAILED"} />
        <MetricCard icon={CheckCircle2} label="Última accuracy resuelta" value={pct(perf.recent_accuracy)} detail={`Rolling 6: ${pct(perf.recent6_accuracy)} · Rolling 12: ${pct(perf.recent12_accuracy)}`} status={perf.status || "UNKNOWN"} />
        <MetricCard icon={RadioTower} label="Estado de drift" value={snapshot.drift_status} status={snapshot.drift_status} detail={snapshot.retrain_recommended ? "Retrain recomendado" : "Retrain no recomendado"} />
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-lg border border-white/10 bg-[#10151d]/88 p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">Último ciclo del snapshot</p>
              <h2 className="mt-2 text-xl font-semibold text-white">{text(cycle.cycle_id, "Sin ciclo activo")}</h2>
              <p className="mt-1 text-sm text-slate-400">Corte de datos {shortDate(cycle.data_cutoff)} · cierra {shortDate(cycle.closes_at)}</p>
            </div>
            <StatusPill status={text(cycle.status, "WAITING")} />
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            <HealthRow label="Predictions" value={`${numberValue(cycle.actual_predictions)} / ${numberValue(cycle.expected_predictions)}`} status={text(cycle.prediction_status, "WAITING")} />
            <HealthRow label="Submission" value={text(cycle.submission_status, "WAITING")} status={text(cycle.submission_status, "WAITING")} />
            <HealthRow label="Oficial" value={cycle.is_official ? "sí" : "no"} status={cycle.is_official ? "OK" : "WAITING"} />
            <HealthRow label="Targets resueltos" value={`${numberValue((cycle.resolved_targets as { actual?: number })?.actual)} / ${numberValue((cycle.resolved_targets as { expected?: number })?.expected)}`} status="WAITING" />
          </div>
        </div>

        <div className="rounded-lg border border-white/10 bg-[#10151d]/88 p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">Última submission</p>
              <h2 className="mt-2 text-xl font-semibold text-white">{text(latestSubmission?.status, "Sin receipt")}</h2>
            </div>
            <Send className="size-5 text-cyan-100" />
          </div>
          <dl className="mt-5 space-y-3 text-sm">
            <Info label="Aceptada en" value={shortDate(latestSubmission?.accepted_at)} />
            <Info label="Submission ID" value={text(latestSubmission?.submission_id)} />
            <Info label="Oficial" value={latestSubmission?.is_official ? "sí" : "no"} />
          </dl>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
        <div className="rounded-lg border border-white/10 bg-[#10151d]/88 p-5">
          <div className="flex items-center gap-2">
            <AlertTriangle className="size-4 text-amber-200" />
            <h2 className="text-lg font-semibold text-white">Centro de alertas</h2>
          </div>
          <div className="mt-4 space-y-3">
            {report.alerts.length ? report.alerts.map((alert) => (
              <div key={`${alert.title}-${alert.timestamp}`} className="rounded-md border border-white/10 bg-white/[0.035] p-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-medium text-white">{alert.title}</p>
                  <StatusPill status={alert.severity === "critical" ? "FAILED" : "WATCH"} compact />
                </div>
                <p className="mt-1 text-sm text-slate-400">{alert.explanation}</p>
                <p className="mt-2 text-xs uppercase tracking-[0.12em] text-slate-500">{alert.category} · {text(alert.entity)}</p>
              </div>
            )) : <p className="text-sm text-slate-400">No hay alertas del snapshot.</p>}
          </div>
        </div>

        <div>
          <div className="mb-3 flex items-center gap-2">
            <ServerCog className="size-4 text-cyan-100" />
            <h2 className="text-lg font-semibold text-white">Últimos eventos operativos</h2>
          </div>
          <DataTable columns={["event_type", "severity", "created_at", "reason_codes"]} rows={report.runtime_ingestion_events.slice(-5)} />
        </div>
      </section>
    </div>
  );
}

function HealthRow({ label, value, status }: { label: string; value: string; status: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-white/10 bg-white/[0.035] px-3 py-3">
      <div>
        <p className="text-sm font-medium text-white">{label}</p>
        <p className="mt-1 text-xs text-slate-400">{value}</p>
      </div>
      <StatusPill status={status} compact />
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/8 pb-3 last:border-b-0">
      <dt className="text-slate-400">{label}</dt>
      <dd className="max-w-[220px] truncate font-medium text-slate-100">{value}</dd>
    </div>
  );
}
