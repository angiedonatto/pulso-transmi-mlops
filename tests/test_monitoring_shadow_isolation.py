from __future__ import annotations

import pandas as pd

import candidate_i_snr_gate as ci
from monitoring_snapshot import build_monitoring_snapshot
from test_drift_monitor import CONFIG, build_tables

# Monitoring side of the Candidate I shadow (kept with the monitoring work, not with the
# shadow integration): shadow predictions share cycles with the champion under their own
# model_id and must not change the official cycle health, traces or drift report.
I_ID = "33333333-3333-4333-8333-333333333333"


def test_monitoring_ignores_shadow_rows() -> None:
    tables = build_tables()
    base = build_monitoring_snapshot(**tables, prediction_traces=pd.DataFrame(), config=CONFIG, generated_at="2026-01-20T00:00:00Z")
    active_id = tables["model_versions"]["model_id"].iloc[0]
    latest = tables["forecast_cycles"].sort_values("data_cutoff")["cycle_id"].iloc[-1]
    shadow_rows = tables["predictions"][tables["predictions"]["cycle_id"] == latest].assign(model_id=I_ID)
    assert len(shadow_rows) > 0
    tables["predictions"] = pd.concat([tables["predictions"], shadow_rows], ignore_index=True)
    tables["model_versions"] = pd.concat([tables["model_versions"], pd.DataFrame([{"model_id": I_ID, "is_active": False, "trained_until": ci.TRAINED_UNTIL}])], ignore_index=True)
    traces = pd.DataFrame(
        [
            {"model_id": active_id, "station_id": "03000", "calibration_factor": 1.1, "feedback_samples": 8, "target_at": "2026-01-19T00:15:00Z", "created_at": "2026-01-19T00:01:00Z"},
            {"model_id": I_ID, "station_id": "03000", "calibration_factor": 0.5, "feedback_samples": 32, "target_at": "2026-01-19T00:15:00Z", "created_at": "2026-01-19T00:05:00Z"},
        ]
    )
    with_shadow = build_monitoring_snapshot(**tables, prediction_traces=traces, config=CONFIG, generated_at="2026-01-20T00:00:00Z")
    assert with_shadow.report["current_cycle"] == base.report["current_cycle"]
    assert not any(alert["title"] == "Prediction count mismatch" for alert in with_shadow.report["alerts"])
    assert [row["calibration_factor"] for row in with_shadow.report["prediction_traces"]["recent_factors"]] == [1.1]
    assert with_shadow.report["drift"] == base.report["drift"]
