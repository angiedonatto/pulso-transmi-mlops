from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from drift_monitor import DriftConfig, build_drift_report


SUCCESS_RECEIPT_STATUSES = {"accepted", "idempotent_replay"}
TERMINAL_BAD_STATUSES = {"failed", "rejected"}


@dataclass(frozen=True)
class MonitoringSnapshot:
    generated_at: str
    model_id: str | None
    overall_status: str
    drift_status: str
    retrain_recommended: bool
    report: dict[str, Any]

    def row(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "model_id": self.model_id,
            "overall_status": self.overall_status,
            "drift_status": self.drift_status,
            "retrain_recommended": self.retrain_recommended,
            "report": self.report,
        }


def build_monitoring_snapshot(
    *,
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    submission_receipts: pd.DataFrame,
    forecast_cycles: pd.DataFrame,
    forecast_cycle_targets: pd.DataFrame,
    model_versions: pd.DataFrame,
    pipeline_runs: pd.DataFrame | None = None,
    data_quality_checks: pd.DataFrame | None = None,
    stream_cursors: pd.DataFrame | None = None,
    runtime_ingestion_events: pd.DataFrame | None = None,
    prediction_traces: pd.DataFrame | None = None,
    config: DriftConfig | None = None,
    generated_at: str | None = None,
) -> MonitoringSnapshot:
    drift = build_drift_report(
        observations=observations,
        predictions=predictions,
        submission_receipts=submission_receipts,
        forecast_cycles=forecast_cycles,
        forecast_cycle_targets=forecast_cycle_targets,
        model_versions=model_versions,
        config=config,
    )
    generated_at = generated_at or _now_iso()
    active_model = _active_model(model_versions)
    official_model_id = active_model.get("model_id") if active_model and bool(active_model.get("is_active")) else None
    latest_cycle = _latest_cycle(forecast_cycles)
    latest_receipt = _latest_receipt(submission_receipts)
    latest_pipeline = _latest_pipeline(pipeline_runs)
    latest_observation = _latest_observation(observations)
    current_cycle = _cycle_health(latest_cycle, _official_rows(predictions, official_model_id), submission_receipts, forecast_cycle_targets)
    quality = _quality_summary(data_quality_checks)
    operations = _operations_summary(pipeline_runs, runtime_ingestion_events)
    alerts = _alerts(
        drift_status=drift.status,
        current_cycle=current_cycle,
        latest_observation=latest_observation,
        latest_receipt=latest_receipt,
        latest_pipeline=latest_pipeline,
        quality=quality,
        station_alerts=drift.station_alerts,
    )
    overall_status = _overall_status(drift.status, alerts)
    report = {
        "drift": drift.to_dict(),
        "active_model": active_model,
        "latest_observation": latest_observation,
        "current_cycle": current_cycle,
        "latest_submission": latest_receipt,
        "latest_pipeline_run": latest_pipeline,
        "data_quality": quality,
        "stream_cursors": _records(stream_cursors, limit=10),
        "runtime_ingestion_events": _records(runtime_ingestion_events, limit=20),
        "prediction_traces": _trace_summary(_official_rows(prediction_traces, official_model_id)),
        "operations": operations,
        "alerts": alerts,
    }
    return MonitoringSnapshot(
        generated_at=generated_at,
        model_id=active_model.get("model_id") if active_model else None,
        overall_status=overall_status,
        drift_status=drift.status,
        retrain_recommended=drift.retrain_recommended,
        report=_json_value(report),
    )


def prediction_count_health(*, expected_predictions: int, actual_predictions: int) -> str:
    if expected_predictions <= 0:
        return "NOT RESOLVED"
    if actual_predictions == expected_predictions:
        return "OK"
    if actual_predictions == 0:
        return "WAITING"
    return "FAILED"


def _official_rows(frame: pd.DataFrame | None, model_id: str | None) -> pd.DataFrame | None:
    """Rows of the active (official) model only: shadow challengers predict the same cycles
    under their own model_id and must not change the official cycle health or traces."""
    if frame is None or frame.empty or model_id is None or "model_id" not in frame:
        return frame
    return frame[frame["model_id"].astype(str) == str(model_id)]


def _cycle_health(
    cycle: dict[str, Any] | None,
    predictions: pd.DataFrame,
    receipts: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, Any]:
    if not cycle:
        return {"status": "WAITING", "prediction_status": "WAITING", "expected_predictions": 0, "actual_predictions": 0}
    cycle_id = str(cycle["cycle_id"])
    expected = int(cycle.get("expected_predictions") or 0)
    actual = int(predictions[predictions.get("cycle_id") == cycle_id].shape[0]) if not predictions.empty and "cycle_id" in predictions else 0
    resolved = _resolved_targets(cycle_id, targets)
    receipt = _latest_receipt(receipts[receipts.get("cycle_id") == cycle_id].copy()) if not receipts.empty and "cycle_id" in receipts else None
    prediction_status = prediction_count_health(expected_predictions=expected, actual_predictions=actual)
    submission_status = str(receipt.get("status", "WAITING")).upper() if receipt else "WAITING"
    status = "FAILED" if prediction_status == "FAILED" or submission_status in {"FAILED", "REJECTED"} else "HEALTHY"
    return {
        "cycle_id": cycle_id,
        "data_cutoff": cycle.get("data_cutoff"),
        "closes_at": cycle.get("closes_at"),
        "expected_predictions": expected,
        "actual_predictions": actual,
        "prediction_status": prediction_status,
        "submission_status": submission_status,
        "is_official": bool(receipt.get("is_official")) if receipt else False,
        "receipt": receipt,
        "resolved_targets": resolved,
        "status": status,
    }


def _alerts(
    *,
    drift_status: str,
    current_cycle: dict[str, Any],
    latest_observation: dict[str, Any],
    latest_receipt: dict[str, Any] | None,
    latest_pipeline: dict[str, Any] | None,
    quality: dict[str, Any],
    station_alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    now = _now_iso()
    if current_cycle.get("prediction_status") == "FAILED":
        alerts.append(_alert("critical", now, "cycle", "Prediction count mismatch", "actual predictions differ from expected_predictions", current_cycle.get("cycle_id")))
    if current_cycle.get("submission_status") in {"WAITING", "FAILED", "REJECTED"}:
        alerts.append(_alert("warning", now, "submission", "Submission needs attention", "latest cycle has no accepted official receipt", current_cycle.get("cycle_id")))
    if latest_observation.get("freshness_status") == "STALE":
        alerts.append(_alert("warning", now, "data", "Data stale", "latest observation is outside the expected freshness window", latest_observation.get("observed_at")))
    if latest_pipeline and str(latest_pipeline.get("status")).lower() == "failed":
        alerts.append(_alert("critical", now, "pipeline", "Failed pipeline run", str(latest_pipeline.get("error_message") or "pipeline failed"), latest_pipeline.get("run_id")))
    if quality.get("failed_checks", 0) > 0:
        alerts.append(_alert("critical", now, "data_quality", "Data quality failure", "one or more data quality checks failed", quality.get("latest_check_at")))
    if drift_status in {"WATCH", "DRIFT"}:
        severity = "critical" if drift_status == "DRIFT" else "warning"
        alerts.append(_alert(severity, now, "drift", f"Drift status {drift_status}", "canonical drift monitor reported a non-healthy state", drift_status))
    for station in station_alerts[:8]:
        severity = "critical" if station.get("severity") == "drift" else "warning"
        alerts.append(_alert(severity, now, "station", "Station performance alert", str(station.get("type", "station alert")), station.get("station_id")))
    return alerts


def _overall_status(drift_status: str, alerts: list[dict[str, Any]]) -> str:
    if any(alert["severity"] == "critical" for alert in alerts):
        return "FAILED" if any(alert["category"] in {"cycle", "pipeline", "data_quality"} for alert in alerts) else "DRIFT"
    if drift_status == "DRIFT":
        return "DRIFT"
    if drift_status == "WATCH" or alerts:
        return "WATCH"
    if drift_status == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    return "HEALTHY"


def _active_model(models: pd.DataFrame) -> dict[str, Any] | None:
    if models.empty:
        return None
    active = models[models.get("is_active", False) == True].copy()
    row = active.iloc[0] if not active.empty else models.sort_values("trained_until").iloc[-1]
    return _json_value(row.to_dict())


def _latest_cycle(cycles: pd.DataFrame) -> dict[str, Any] | None:
    if cycles.empty:
        return None
    frame = cycles.copy()
    frame["data_cutoff"] = pd.to_datetime(frame["data_cutoff"], utc=True)
    return _json_value(frame.sort_values("data_cutoff").iloc[-1].to_dict())


def _latest_receipt(receipts: pd.DataFrame) -> dict[str, Any] | None:
    if receipts.empty:
        return None
    frame = receipts.copy()
    frame["created_at"] = pd.to_datetime(frame.get("created_at"), utc=True)
    return _json_value(frame.sort_values("created_at").iloc[-1].to_dict())


def _latest_pipeline(pipeline_runs: pd.DataFrame | None) -> dict[str, Any] | None:
    if pipeline_runs is None or pipeline_runs.empty:
        return None
    frame = pipeline_runs.copy()
    frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True)
    return _json_value(frame.sort_values("started_at").iloc[-1].to_dict())


def _latest_observation(observations: pd.DataFrame) -> dict[str, Any]:
    if observations.empty:
        return {"freshness_status": "STALE", "observed_at": None, "minutes_old": None}
    latest = pd.to_datetime(observations["observed_at"], utc=True).max()
    minutes_old = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 60.0
    return {
        "observed_at": latest.isoformat().replace("+00:00", "Z"),
        "minutes_old": round(minutes_old, 1),
        "freshness_status": "STALE" if minutes_old > 45 else "FRESH",
    }


def _resolved_targets(cycle_id: str, targets: pd.DataFrame) -> dict[str, int]:
    if targets.empty or "cycle_id" not in targets:
        return {"actual": 0, "expected": 0}
    expected = int(targets[targets["cycle_id"] == cycle_id].shape[0])
    return {"actual": 0, "expected": expected}


def _quality_summary(checks: pd.DataFrame | None) -> dict[str, Any]:
    if checks is None or checks.empty:
        return {"status": "UNKNOWN", "failed_checks": 0, "warning_checks": 0, "latest_check_at": None}
    status_col = checks.get("status", pd.Series(dtype=str)).astype(str).str.lower()
    severity_col = checks.get("severity", pd.Series(dtype=str)).astype(str).str.lower()
    failed = int(((status_col == "failed") | severity_col.isin(["critical", "quarantine"])).sum())
    warning = int(((status_col == "warning") | (severity_col == "warning")).sum())
    latest = checks.get("checked_at")
    latest_value = pd.to_datetime(latest, utc=True).max().isoformat().replace("+00:00", "Z") if latest is not None else None
    return {"status": "FAILED" if failed else "WATCH" if warning else "PASSED", "failed_checks": failed, "warning_checks": warning, "latest_check_at": latest_value}


def _operations_summary(pipeline_runs: pd.DataFrame | None, events: pd.DataFrame | None) -> dict[str, Any]:
    failed_runs = 0 if pipeline_runs is None or pipeline_runs.empty else int((pipeline_runs.get("status").astype(str).str.lower() == "failed").sum())
    critical_events = 0 if events is None or events.empty else int(events.get("severity").astype(str).str.lower().isin(["critical", "quarantine"]).sum())
    return {"failed_pipeline_runs": failed_runs, "critical_ingestion_events": critical_events}


def _trace_summary(traces: pd.DataFrame | None) -> dict[str, Any]:
    if traces is None or traces.empty:
        return {"available": False, "recent_factors": []}
    frame = traces.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    recent = frame.sort_values("created_at").tail(48)
    return {"available": True, "recent_factors": _records(recent[["station_id", "calibration_factor", "feedback_samples", "target_at"]], limit=48)}


def _records(frame: pd.DataFrame | None, *, limit: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [_json_value(row) for row in frame.tail(limit).to_dict("records")]


def _alert(severity: str, timestamp: str, category: str, title: str, explanation: str, entity: Any) -> dict[str, Any]:
    return {"severity": severity, "timestamp": timestamp, "category": category, "title": title, "explanation": explanation, "entity": entity}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
