from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from runtime_data_validation import RuntimeDataValidator, RuntimeValidationContext
from stream_sync import ObservationRow


def validate(rows, **kwargs):
    context = RuntimeValidationContext(**kwargs)
    return RuntimeDataValidator().validate(rows, context)


def event_reasons(result):
    return {reason for event in result.events for reason in event.reason_codes}


def checks_by_name(result):
    return {check.check_name: check for check in result.checks}


def test_demand_int_passes() -> None:
    result = validate([{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 120}])

    assert result.data_health == "HEALTHY"
    assert result.prediction_readiness == "READY"
    assert result.valid_rows == [ObservationRow("03000", "2026-01-01T05:00:00Z", 120)]


def test_numeric_string_and_integer_float_are_recovered() -> None:
    result = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": "120"},
            {"station_id": "02300", "observed_at": "2026-01-01T05:15:00Z", "demand": "120.0"},
            {"station_id": "03000", "observed_at": "2026-01-01T05:30:00Z", "demand": 121.0},
        ]
    )

    assert [row.demand for row in result.valid_rows] == [120, 120, 121]
    assert "demand_integer_recovered" in event_reasons(result)
    assert result.data_health == "DEGRADED"


def test_decimal_float_is_quarantined_without_rounding() -> None:
    result = validate([{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 120.7}])

    assert result.valid_rows == []
    assert "demand_not_integer" in event_reasons(result)
    assert result.quarantined_rows == 1


def test_null_negative_nan_infinity_and_bool_demand_are_quarantined() -> None:
    result = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": None},
            {"station_id": "03000", "observed_at": "2026-01-01T05:15:00Z", "demand": -1},
            {"station_id": "03000", "observed_at": "2026-01-01T05:30:00Z", "demand": math.nan},
            {"station_id": "03000", "observed_at": "2026-01-01T05:45:00Z", "demand": math.inf},
            {"station_id": "03000", "observed_at": "2026-01-01T06:00:00Z", "demand": True},
        ]
    )

    assert result.valid_rows == []
    assert {"demand_not_integer", "demand_negative", "demand_not_finite", "demand_bool"} <= event_reasons(result)
    assert result.quarantined_rows == 5


def test_extreme_outlier_warns_but_remains_valid() -> None:
    history = [
        ObservationRow("03000", f"2025-12-{day:02d}T05:00:00Z", value)
        for day, value in enumerate([100, 101, 99, 100, 100, 102, 98, 100, 101, 99, 100, 100], start=1)
    ]
    result = validate(
        [{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 999}],
        history_rows=history,
    )

    assert result.valid_rows[0].demand == 999
    assert "demand_outlier" in event_reasons(result)
    assert result.data_health == "DEGRADED"


def test_station_id_keeps_leading_zero_and_numeric_is_recovered_only_if_known() -> None:
    result = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10},
            {"station_id": 3000, "observed_at": "2026-01-01T05:15:00Z", "demand": 11},
            {"station_id": "999", "observed_at": "2026-01-01T05:30:00Z", "demand": 12},
        ],
        known_station_ids={"03000"},
    )

    assert result.valid_rows[0].station_id == "03000"
    assert result.valid_rows[1].station_id == "03000"
    assert "station_id_zero_padded" in event_reasons(result)
    assert "station_id_not_recoverable" in event_reasons(result)


def test_new_station_is_quarantined_because_catalog_fk_would_fail() -> None:
    result = validate([{"station_id": "99999", "observed_at": "2026-01-01T05:00:00Z", "demand": 10}], known_station_ids={"03000"})

    assert result.valid_rows == []
    assert "new_station" in event_reasons(result)


def test_duplicate_identical_and_conflicting_within_batch() -> None:
    result = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10},
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10},
            {"station_id": "02300", "observed_at": "2026-01-01T05:00:00Z", "demand": 20},
            {"station_id": "02300", "observed_at": "2026-01-01T05:00:00Z", "demand": 21},
        ]
    )

    assert [row.station_id for row in result.valid_rows] == ["03000"]
    assert "duplicate_identical_batch" in event_reasons(result)
    assert "duplicate_conflicting_batch" in event_reasons(result)


def test_duplicate_conflicting_against_existing_observation_is_quarantined_when_revisions_disabled() -> None:
    result = validate(
        [{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10}],
        existing_observations={("03000", "2026-01-01T05:00:00Z"): 11},
        accept_revisions=False,
    )

    assert result.valid_rows == []
    assert "duplicate_conflicting_existing" in event_reasons(result)


def test_republished_value_for_existing_observation_is_written_as_revision() -> None:
    result = validate(
        [{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10, "released_at": "2026-01-02T00:00:00Z"}],
        existing_observations={("03000", "2026-01-01T05:00:00Z"): 11},
    )

    assert result.rows_to_write == [ObservationRow("03000", "2026-01-01T05:00:00Z", 10)]
    revision = next(event for event in result.events if "value_revised" in event.reason_codes)
    assert revision.severity == "warning"
    assert revision.details["previous_demand"] == 11 and revision.details["revised_demand"] == 10
    assert checks_by_name(result)["runtime.revisions"].actual_value == 1
    assert result.data_health == "DEGRADED" and result.prediction_readiness == "READY"


def test_conflicting_versions_in_one_batch_keep_the_latest_publication() -> None:
    result = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10, "released_at": "2026-01-02T00:00:00Z"},
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 14, "released_at": "2026-01-02T00:30:00Z"},
        ]
    )

    assert result.rows_to_write == [ObservationRow("03000", "2026-01-01T05:00:00Z", 14)]
    assert "duplicate_conflicting_batch_resolved" in event_reasons(result)
    tied = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10, "released_at": "2026-01-02T00:00:00Z"},
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 14, "released_at": "2026-01-02T00:00:00Z"},
        ]
    )
    assert tied.valid_rows == []  # same publication time: ambiguous, quarantined
    assert "duplicate_conflicting_batch" in event_reasons(tied)


def test_abrupt_change_is_written_and_only_warned() -> None:
    start = datetime(2026, 1, 5, 5, 0, tzinfo=timezone.utc)
    history = [ObservationRow("03000", (start - timedelta(days=d)).isoformat().replace("+00:00", "Z"), 100 + d % 3) for d in range(1, 15)]
    result = validate([{"station_id": "03000", "observed_at": start.isoformat().replace("+00:00", "Z"), "demand": 1500}], history_rows=history)

    assert result.rows_to_write == [ObservationRow("03000", "2026-01-05T05:00:00Z", 1500)]
    assert "demand_outlier" in event_reasons(result)
    assert all(event.severity != "quarantine" for event in result.events)


def test_identical_existing_duplicate_is_idempotent_and_not_written_or_warned() -> None:
    row = {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10}
    result = validate([row], existing_observations={("03000", "2026-01-01T05:00:00Z"): 10})

    checks = checks_by_name(result)
    assert result.valid_rows == [ObservationRow("03000", "2026-01-01T05:00:00Z", 10)]
    assert result.rows_to_write == []
    assert result.events == []
    assert result.data_health == "HEALTHY"
    assert checks["runtime.duplicates.identical_existing"].severity == "pass"
    assert checks["runtime.duplicates.identical_existing"].actual_value == 1


def test_identical_existing_late_rows_do_not_count_as_late_arrivals() -> None:
    result = validate(
        [{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10}],
        existing_observations={("03000", "2026-01-01T05:00:00Z"): 10},
        last_observed_at="2026-01-02T00:00:00Z",
    )

    ordering = checks_by_name(result)["runtime.timestamp.ordering"]
    assert result.rows_to_write == []
    assert ordering.status == "passed"
    assert ordering.details["late_arrivals"] == 0
    assert result.data_health == "HEALTHY"


def test_new_late_arrival_warns_and_is_kept_for_write() -> None:
    result = validate(
        [{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10}],
        last_observed_at="2026-01-02T00:00:00Z",
    )

    ordering = checks_by_name(result)["runtime.timestamp.ordering"]
    assert result.rows_to_write == [ObservationRow("03000", "2026-01-01T05:00:00Z", 10)]
    assert ordering.status == "warning"
    assert ordering.details["late_arrivals"] == 1
    assert result.data_health == "DEGRADED"


def test_timestamp_without_timezone_is_quarantined() -> None:
    result = validate([{"station_id": "03000", "observed_at": "2026-01-01T05:00:00", "demand": 10}])

    assert result.valid_rows == []
    assert "timestamp_missing_timezone" in event_reasons(result)


def test_out_of_order_late_arrival_and_gap_warn_without_dropping_rows() -> None:
    result = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:30:00Z", "demand": 10},
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 11},
        ],
        last_observed_at="2026-01-01T05:15:00Z",
    )

    assert len(result.valid_rows) == 2
    checks = {check.check_name: check for check in result.checks}
    assert checks["runtime.timestamp.ordering"].status == "warning"
    assert checks["runtime.timestamp.gaps"].status == "warning"


def test_extra_column_required_missing_partial_corruption_and_empty_response() -> None:
    result = validate(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10, "extra": "x"},
            {"station_id": "03000", "observed_at": "2026-01-01T05:15:00Z"},
        ]
    )
    empty = validate([])

    assert len(result.valid_rows) == 1
    assert "missing_required_field" in event_reasons(result)
    assert {check.check_name: check.status for check in result.checks}["runtime.schema.extra_fields"] == "warning"
    assert empty.data_health == "DEGRADED"
    assert empty.prediction_readiness == "NOT_READY"


def test_released_at_is_known_metadata_but_unknown_extra_field_warns() -> None:
    released = validate(
        [
            {
                "station_id": "03000",
                "observed_at": "2026-01-01T05:00:00Z",
                "demand": 10,
                "released_at": "2026-01-01T05:01:00Z",
            }
        ]
    )
    unknown = validate(
        [
            {
                "station_id": "03000",
                "observed_at": "2026-01-01T05:00:00Z",
                "demand": 10,
                "mystery_field": "x",
            }
        ]
    )

    assert checks_by_name(released)["runtime.schema.extra_fields"].status == "passed"
    assert released.raw_payload["data"][0]["released_at"] == "2026-01-01T05:01:00Z"
    assert checks_by_name(unknown)["runtime.schema.extra_fields"].status == "warning"
    assert checks_by_name(unknown)["runtime.schema.extra_fields"].details["extra_fields"] == ["mystery_field"]


def test_many_existing_identical_duplicates_and_new_rows_are_aggregated_without_noise() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    existing_rows = [
        {
            "station_id": "03000",
            "observed_at": (start + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z"),
            "demand": 100 + index % 7,
        }
        for index in range(3708)
    ]
    new_start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    new_rows = [
        {
            "station_id": "03000",
            "observed_at": (new_start + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z"),
            "demand": 150 + index % 5,
        }
        for index in range(24)
    ]
    existing = {(row["station_id"], row["observed_at"]): row["demand"] for row in existing_rows}

    result = validate(
        existing_rows + new_rows,
        existing_observations=existing,
        last_observed_at="2026-02-15T00:00:00Z",
    )

    checks = checks_by_name(result)
    assert len(result.valid_rows) == 3732
    assert len(result.rows_to_write) == 24
    assert len(result.events) == 0
    assert checks["runtime.duplicates.identical_existing"].severity == "pass"
    assert checks["runtime.duplicates.identical_existing"].actual_value == 3708
    assert checks["runtime.timestamp.ordering"].details["late_arrivals"] == 0
    assert result.data_health == "HEALTHY"


def test_all_rows_missing_required_column_is_critical() -> None:
    result = validate([{"station_id": "03000", "demand": 10}])

    assert result.data_health == "CRITICAL"
    assert result.prediction_readiness == "NOT_READY"


def test_model_compatibility_is_separate_from_data_health() -> None:
    result = RuntimeDataValidator().validate(
        [{"station_id": "03000", "observed_at": "2026-01-01T05:00:00Z", "demand": 10}],
        RuntimeValidationContext(known_station_ids={"03000"}, champion_station_ids={"02300", "03000"}),
    )

    assert result.data_health == "HEALTHY"
    assert result.prediction_readiness == "DEGRADED"
    assert result.model_compatibility.missing_expected == ("02300",)
