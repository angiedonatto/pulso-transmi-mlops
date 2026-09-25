from __future__ import annotations

from datetime import timedelta

import pandas as pd

from drift_monitor import DriftConfig, build_drift_report


CONFIG = DriftConfig(
    min_cycles=6,
    recent_cycles=6,
    reference_cycles=6,
    min_rows_per_window=20,
    min_station_rows=4,
    min_station_horizon_rows=4,
    min_psi_rows=20,
    min_ks_rows=10,
    min_new_observations_since_training=20,
)


def build_tables(
    *,
    cycles: int = 12,
    recent_error: float = 0.0,
    reference_error: float = 0.0,
    actual_value: float = 100.0,
    isolated_bad_cycle: bool = False,
    recent_cycle_errors: list[float] | None = None,
    station_recent_errors: dict[str, float] | None = None,
    horizon_recent_errors: dict[int, float] | None = None,
    distribution_shift: bool = False,
    include_new_station: bool = False,
    trained_until: str = "2026-01-06T00:00:00Z",
):
    stations = ["03000", "02300"]
    horizons = [15, 30]
    station_recent_errors = station_recent_errors or {}
    horizon_recent_errors = horizon_recent_errors or {}
    cycles_rows = []
    targets_rows = []
    predictions_rows = []
    observations_rows = []
    receipts_rows = []
    base = pd.Timestamp("2026-01-01T08:00:00Z")
    model_id = "11111111-1111-4111-8111-111111111111"

    for cycle_idx in range(cycles):
        cycle_id = f"cycle_{cycle_idx:02d}"
        data_cutoff = base + timedelta(days=cycle_idx)
        cycles_rows.append(
            {
                "cycle_id": cycle_id,
                "data_cutoff": data_cutoff,
                "closes_at": data_cutoff + timedelta(hours=1),
                "expected_predictions": len(stations) * len(horizons),
            }
        )
        receipts_rows.append(
            {
                "cycle_id": cycle_id,
                "model_id": model_id,
                "status": "accepted",
                "is_official": True,
                "accepted_at": data_cutoff + timedelta(hours=1),
                "created_at": data_cutoff + timedelta(hours=1),
            }
        )
        for station_id in stations:
            for horizon in horizons:
                target_at = data_cutoff + timedelta(minutes=horizon)
                error = reference_error if cycle_idx < cycles - 6 else recent_error
                if recent_cycle_errors is not None and cycle_idx >= cycles - 6:
                    recent_idx = cycle_idx - (cycles - 6)
                    error = recent_cycle_errors[recent_idx]
                if isolated_bad_cycle and cycle_idx == cycles - 1:
                    error = 60.0
                if cycle_idx >= cycles - 6 and station_id in station_recent_errors:
                    error = station_recent_errors[station_id]
                if cycle_idx >= cycles - 6 and horizon in horizon_recent_errors:
                    error = horizon_recent_errors[horizon]
                actual = actual_value
                prediction = actual - error
                targets_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "station_id": station_id,
                        "target_at": target_at,
                        "horizon_minutes": horizon,
                        "target_order": len(targets_rows),
                    }
                )
                predictions_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "model_id": model_id,
                        "station_id": station_id,
                        "target_observed_at": target_at,
                        "horizon_minutes": horizon,
                        "prediction": prediction,
                    }
                )
                observations_rows.append({"station_id": station_id, "observed_at": target_at, "demand": actual})

    obs_base = pd.Timestamp("2025-12-01T00:00:00Z")
    for idx in range(120):
        observed_at = obs_base + timedelta(minutes=15 * idx)
        is_recent = idx >= 60
        for station_id in stations:
            demand = 100
            if distribution_shift and is_recent:
                demand = 180 + (idx % 7)
            observations_rows.append({"station_id": station_id, "observed_at": observed_at, "demand": demand})
        if include_new_station and idx >= 80:
            observations_rows.append({"station_id": "99999", "observed_at": observed_at, "demand": 120})

    return {
        "observations": pd.DataFrame(observations_rows),
        "predictions": pd.DataFrame(predictions_rows),
        "submission_receipts": pd.DataFrame(receipts_rows),
        "forecast_cycles": pd.DataFrame(cycles_rows),
        "forecast_cycle_targets": pd.DataFrame(targets_rows),
        "model_versions": pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "is_active": True,
                    "trained_until": trained_until,
                }
            ]
        ),
    }


def report(**kwargs):
    tables = build_tables(**kwargs)
    return build_drift_report(config=CONFIG, **tables)


def test_stable_distribution_is_healthy() -> None:
    result = report()

    assert result.status == "HEALTHY"
    assert result.retrain_recommended is False


def test_single_bad_cycle_does_not_mark_drift() -> None:
    result = report(isolated_bad_cycle=True)

    assert result.status in ("HEALTHY", "WATCH")
    assert result.status != "DRIFT"
    assert result.retrain_recommended is False


def test_sustained_performance_degradation_marks_drift() -> None:
    result = report(reference_error=8.0, recent_error=35.0)

    assert result.status == "DRIFT"
    assert result.retrain_recommended is True


def test_strong_distribution_change_marks_drift() -> None:
    result = report(distribution_shift=True)

    assert result.data_drift["status"] == "DRIFT"
    assert result.status in ("WATCH", "DRIFT")


def test_sustained_negative_bias_is_watch_or_drift() -> None:
    result = report(reference_error=0.0, recent_error=16.0)

    assert result.residual_drift["status"] in ("WATCH", "DRIFT")
    assert result.status in ("WATCH", "DRIFT")


def test_insufficient_sample_is_insufficient_data() -> None:
    result = report(cycles=3)

    assert result.status == "INSUFFICIENT_DATA"
    assert result.retrain_recommended is False


def test_single_station_deterioration_generates_station_alert() -> None:
    result = report(station_recent_errors={"03000": 35.0})

    assert any(alert.get("station_id") == "03000" for alert in result.station_alerts)


def test_single_horizon_deterioration_generates_horizon_alert() -> None:
    result = report(horizon_recent_errors={30: 35.0})

    assert any(alert.get("horizon_minutes") == 30 for alert in result.horizon_alerts)


def test_new_station_is_data_drift_signal() -> None:
    result = report(include_new_station=True)

    assert "99999" in result.data_drift["new_stations"]
    assert result.data_drift["status"] == "DRIFT"


def test_recovery_after_drift_returns_to_healthy_or_watch() -> None:
    result = report(reference_error=35.0, recent_error=10.0)

    assert result.status in ("HEALTHY", "WATCH")
    assert result.status != "DRIFT"


def test_large_absolute_bias_with_small_relative_bias_is_not_strong_alert() -> None:
    result = report(actual_value=10000.0, recent_error=100.0)

    assert result.residual_drift["overall_recent_bias"] == -100.0
    assert abs(result.residual_drift["overall_recent_signed_relative_bias"]) < CONFIG.relative_bias_watch_abs
    assert result.residual_drift["status"] == "HEALTHY"
    assert result.status != "DRIFT"


def test_sustained_negative_relative_bias_marks_residual_drift() -> None:
    result = report(actual_value=100.0, recent_error=16.0)

    assert result.residual_drift["overall_recent_signed_relative_bias"] <= -0.16
    assert result.residual_drift["status"] == "DRIFT"


def test_single_biased_period_followed_by_recovery_does_not_mark_residual_drift() -> None:
    result = report(recent_cycle_errors=[30.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert result.residual_drift["status"] != "DRIFT"


def test_relative_bias_by_horizon_is_reported_and_alerted() -> None:
    result = report(horizon_recent_errors={30: 18.0})

    horizon_30 = next(row for row in result.residual_drift["by_horizon"] if row["horizon_minutes"] == 30)
    assert horizon_30["signed_relative_bias"] <= -0.18
    assert any(alert.get("horizon_minutes") == 30 and alert.get("type") == "horizon_bias" for alert in result.horizon_alerts)


def test_relative_bias_by_station_is_reported_and_alerted() -> None:
    result = report(station_recent_errors={"03000": 18.0})

    station = next(row for row in result.residual_drift["by_station"] if row["station_id"] == "03000")
    assert station["signed_relative_bias"] <= -0.18
    assert any(alert.get("station_id") == "03000" and alert.get("type") == "station_bias" for alert in result.station_alerts)
