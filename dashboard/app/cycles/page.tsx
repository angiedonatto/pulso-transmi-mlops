import { DataUnavailable } from "@/components/data-unavailable";
import { DataTable } from "@/components/data-table";
import { StatusPill } from "@/components/status-pill";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { numberValue, shortDate, text } from "@/lib/format";

export default async function CyclesPage() {
  const result = await getMonitoringSnapshot();
  if (!result.snapshot) return <DataUnavailable message={result.message} />;
  const snapshot = result.snapshot;
  const cycle = snapshot.report.current_cycle;
  const rows = [
    {
      cycle: text(cycle.cycle_id),
      data_cutoff: shortDate(cycle.data_cutoff),
      model: text(snapshot.report.active_model?.model_name),
      prediction_count: `${numberValue(cycle.actual_predictions)} / ${numberValue(cycle.expected_predictions)}`,
      submission: text(cycle.submission_status),
      official: cycle.is_official ? "sí" : "no",
      receipt: text((cycle.receipt as Record<string, unknown> | undefined)?.submission_id),
      resolved_targets: `${numberValue((cycle.resolved_targets as { actual?: number })?.actual)} / ${numberValue((cycle.resolved_targets as { expected?: number })?.expected)}`,
    },
  ];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">Forecast cycles</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">Salud del ciclo y estado de submission</h2>
        </div>
        <StatusPill status={text(cycle.prediction_status, "WAITING")} />
      </div>
      <DataTable columns={["cycle", "data_cutoff", "model", "prediction_count", "submission", "official", "receipt", "resolved_targets"]} rows={rows} />
    </div>
  );
}
