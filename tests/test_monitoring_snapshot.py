from __future__ import annotations

import pandas as pd

from cycle_predictions import CyclePrediction, prediction_trace_payloads
from monitoring_snapshot import build_monitoring_snapshot, prediction_count_health
from test_drift_monitor import CONFIG, build_tables


def test_prediction_count_health_flags_mismatch() -> None:
    assert prediction_count_health(expected_predictions=48, actual_predictions=48) == "OK"
    assert prediction_count_health(expected_predictions=48, actual_predictions=47) == "FAILED"
    assert prediction_count_health(expected_predictions=48, actual_predictions=0) == "WAITING"


def test_monitoring_snapshot_reuses_official_drift_report() -> None:
    tables = build_tables(reference_error=8.0, recent_error=35.0)
    snapshot = build_monitoring_snapshot(
        **tables,
        pipeline_runs=pd.DataFrame(),
        data_quality_checks=pd.DataFrame(),
        stream_cursors=pd.DataFrame(),
        runtime_ingestion_events=pd.DataFrame(),
        prediction_traces=pd.DataFrame(),
        config=CONFIG,
        generated_at="2026-01-20T00:00:00Z",
    )

    assert snapshot.drift_status == "DRIFT"
    assert snapshot.retrain_recommended is True
    assert snapshot.report["drift"]["performance"]["cycle_count"] == 12


def test_missing_latest_receipt_generates_submission_alert() -> None:
    tables = build_tables()
    tables["submission_receipts"] = tables["submission_receipts"].iloc[:-1].copy()
    snapshot = build_monitoring_snapshot(
        **tables,
        pipeline_runs=pd.DataFrame(),
        data_quality_checks=pd.DataFrame(),
        stream_cursors=pd.DataFrame(),
        runtime_ingestion_events=pd.DataFrame(),
        prediction_traces=pd.DataFrame(),
        config=CONFIG,
        generated_at="2026-01-20T00:00:00Z",
    )

    assert any(alert["category"] == "submission" for alert in snapshot.report["alerts"])


def test_candidate_g_trace_payloads_are_separate_observability_rows() -> None:
    rows = prediction_trace_payloads(
        [
            CyclePrediction(
                cycle_id="cycle_1",
                model_id="11111111-1111-4111-8111-111111111111",
                station_id="03000",
                target_at="2026-01-01T00:15:00Z",
                horizon_minutes=15,
                prediction=123.0,
                trace={
                    "candidate": "candidate_g_v1",
                    "base_prediction": 100.0,
                    "calibration_factor": 1.23,
                    "final_prediction": 123.0,
                    "feedback_samples": 8,
                    "data_cutoff": "2026-01-01T00:00:00Z",
                },
            ),
            CyclePrediction(
                cycle_id="cycle_1",
                model_id="11111111-1111-4111-8111-111111111111",
                station_id="02300",
                target_at="2026-01-01T00:15:00Z",
                horizon_minutes=15,
                prediction=90.0,
                trace=None,
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["calibration_factor"] == 1.23
    assert rows[0]["final_prediction"] == 123.0
