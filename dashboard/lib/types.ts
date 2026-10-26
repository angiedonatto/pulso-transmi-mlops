export type Status = "HEALTHY" | "WATCH" | "DRIFT" | "FAILED" | "ACCEPTED" | "WAITING" | "NOT RESOLVED" | "OK" | "PASSED" | "UNKNOWN" | "INSUFFICIENT_DATA";

export type AlertItem = {
  severity: "critical" | "warning" | "info";
  timestamp: string;
  category: string;
  title: string;
  explanation: string;
  entity?: string | number | null;
};

export type SnapshotResult =
  | { source: "supabase"; snapshot: MonitoringSnapshot }
  | { source: "unavailable"; snapshot: null; reason: "not_configured" | "query_error" | "empty" | "invalid_snapshot"; message: string };

export type MonitoringSnapshot = {
  generated_at: string;
  model_id: string | null;
  overall_status: Status;
  drift_status: Status;
  retrain_recommended: boolean;
  report: {
    drift: {
      status: Status;
      retrain_recommended: boolean;
      reasons: string[];
      performance: {
        status?: Status;
        recent6_accuracy?: number | null;
        recent12_accuracy?: number | null;
        recent_accuracy?: number | null;
        reference_accuracy?: number | null;
        accuracy_drop_pp?: number | null;
        cycle_accuracy?: Array<Record<string, number | string | null>>;
        station_accuracy?: Array<Record<string, number | string | null>>;
        horizon_accuracy?: Array<Record<string, number | string | null>>;
        overall?: Record<string, number | string | null>;
      };
      data_drift: Record<string, unknown>;
      residual_drift: Record<string, unknown>;
      station_alerts: Array<Record<string, unknown>>;
      horizon_alerts: Array<Record<string, unknown>>;
    };
    active_model: Record<string, unknown> | null;
    latest_observation: Record<string, unknown>;
    current_cycle: Record<string, unknown>;
    latest_submission: Record<string, unknown> | null;
    latest_pipeline_run: Record<string, unknown> | null;
    data_quality: Record<string, unknown>;
    stream_cursors: Array<Record<string, unknown>>;
    runtime_ingestion_events: Array<Record<string, unknown>>;
    prediction_traces: { available: boolean; recent_factors: Array<Record<string, unknown>> };
    operations: Record<string, unknown>;
    alerts: AlertItem[];
  };
};
