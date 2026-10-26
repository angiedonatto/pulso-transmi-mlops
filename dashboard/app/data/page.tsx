import { DataUnavailable } from "@/components/data-unavailable";
import { Database, Rows3, ShieldCheck } from "lucide-react";
import { DataTable } from "@/components/data-table";
import { MetricCard } from "@/components/metric-card";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { numberValue, shortDate, text } from "@/lib/format";

export default async function DataPage() {
  const result = await getMonitoringSnapshot();
  if (!result.snapshot) return <DataUnavailable message={result.message} />;
  const snapshot = result.snapshot;
  const latest = snapshot.report.latest_observation;
  const quality = snapshot.report.data_quality;

  return (
    <div className="space-y-5">
      <section className="grid gap-3 md:grid-cols-3">
        <MetricCard icon={Database} label="Último observed_at" value={shortDate(latest.observed_at)} status={text(latest.freshness_status, "UNKNOWN")} detail={`${numberValue(latest.minutes_old)} minutos de antigüedad`} />
        <MetricCard icon={ShieldCheck} label="Calidad de datos" value={text(quality.status, "UNKNOWN")} status={text(quality.status, "UNKNOWN")} detail={`${numberValue(quality.failed_checks)} fallidos · ${numberValue(quality.warning_checks)} warnings`} />
        <MetricCard icon={Rows3} label="Stream cursor" value={text(snapshot.report.stream_cursors[0]?.stream_name, "No cursor")} detail={`Rows written: ${numberValue(snapshot.report.stream_cursors[0]?.rows_written)}`} />
      </section>
      <section className="grid gap-5 xl:grid-cols-2">
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Stream cursors</h2>
          <DataTable columns={["stream_name", "confirmed_at", "last_observed_at", "rows_written", "confirmed_cursor"]} rows={snapshot.report.stream_cursors} />
        </div>
        <div>
          <h2 className="mb-3 text-lg font-semibold text-white">Eventos de ingestión runtime</h2>
          <DataTable columns={["event_type", "severity", "station_id", "observed_at", "created_at", "reason_codes"]} rows={snapshot.report.runtime_ingestion_events} />
        </div>
      </section>
    </div>
  );
}
