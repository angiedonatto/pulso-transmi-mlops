from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np

from stream_sync import ObservationRow


Severity = Literal["pass", "warning", "quarantine", "critical"]
QualityStatus = Literal["passed", "warning", "failed"]
DataHealth = Literal["HEALTHY", "DEGRADED", "CRITICAL"]
PredictionReadiness = Literal["READY", "DEGRADED", "NOT_READY"]

STATION_ID_RE = re.compile(r"^[0-9]{5}$")
REQUIRED_FIELDS = ("station_id", "observed_at", "demand")
EXPECTED_FIELDS = set(REQUIRED_FIELDS)
BOGOTA_TZ = ZoneInfo("America/Bogota")


@dataclass(frozen=True)
class RuntimeQualityCheck:
    check_name: str
    status: QualityStatus
    severity: Severity
    actual_value: int | None = None
    expected_value: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None


@dataclass(frozen=True)
class RuntimeIngestionEvent:
    event_type: str
    severity: Severity
    reason_codes: tuple[str, ...]
    raw_payload: Any
    normalized_payload: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)
    station_id: str | None = None
    observed_at: str | None = None
    row_hash: str | None = None
    idempotency_key: str | None = None
    resolution_status: str | None = None


@dataclass(frozen=True)
class ModelCompatibilityReport:
    known_to_model: tuple[str, ...]
    unknown_to_model: tuple[str, ...]
    missing_expected: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeValidationContext:
    known_station_ids: set[str] | None = None
    champion_station_ids: set[str] | None = None
    last_observed_at: str | None = None
    existing_observations: dict[tuple[str, str], int] = field(default_factory=dict)
    history_rows: list[ObservationRow] = field(default_factory=list)
    stream_name: str = "observations"
    cursor: str | None = None
    empty_response_count: int = 0
    empty_response_critical_threshold: int = 3


@dataclass(frozen=True)
class RuntimeValidationResult:
    raw_payload: Any
    valid_rows: list[ObservationRow]
    events: list[RuntimeIngestionEvent]
    checks: list[RuntimeQualityCheck]
    data_health: DataHealth
    prediction_readiness: PredictionReadiness
    model_compatibility: ModelCompatibilityReport
    batch_hash: str

    @property
    def has_critical(self) -> bool:
        return self.data_health == "CRITICAL"

    @property
    def quarantined_rows(self) -> int:
        return sum(1 for event in self.events if event.severity == "quarantine")


class RuntimeDataValidator:
    def validate(self, raw_rows: list[Any], context: RuntimeValidationContext | None = None, *, raw_payload: Any | None = None) -> RuntimeValidationResult:
        context = context or RuntimeValidationContext()
        payload = raw_payload if raw_payload is not None else {"data": [_jsonable(row) for row in raw_rows]}
        batch_hash = _hash_payload(payload)
        events: list[RuntimeIngestionEvent] = []
        checks: list[RuntimeQualityCheck] = []

        if not raw_rows:
            next_count = context.empty_response_count + 1
            is_critical = next_count >= context.empty_response_critical_threshold
            severity: Severity = "critical" if is_critical else "warning"
            status: QualityStatus = "failed" if is_critical else "warning"
            events.append(
                _event(
                    "batch",
                    severity,
                    ("empty_response",),
                    payload,
                    None,
                    {
                        "message": "stream page contained no rows",
                        "consecutive_empty_responses": next_count,
                        "critical_threshold": context.empty_response_critical_threshold,
                    },
                    batch_hash,
                    context.cursor,
                    None,
                )
            )
            checks.append(
                _check(
                    "runtime.empty_response",
                    status,
                    severity,
                    actual=0,
                    expected=None,
                    details={
                        "message": "stream page contained no rows",
                        "consecutive_empty_responses": next_count,
                        "critical_threshold": context.empty_response_critical_threshold,
                    },
                    batch_hash=batch_hash,
                    cursor=context.cursor,
                )
            )
            compatibility = ModelCompatibilityReport((), (), tuple(sorted(context.champion_station_ids or ())))
            health: DataHealth = "CRITICAL" if is_critical else "DEGRADED"
            return RuntimeValidationResult(payload, [], events, checks, health, "NOT_READY", compatibility, batch_hash)

        normalized_candidates: list[tuple[int, ObservationRow, dict[str, Any], Any]] = []
        quarantined_indexes: set[int] = set()
        missing_required_counts = {field_name: 0 for field_name in REQUIRED_FIELDS}
        extra_fields: set[str] = set()

        for index, raw in enumerate(raw_rows):
            raw_dict = _row_to_dict(raw)
            row_hash = _hash_payload(raw_dict)
            if raw_dict is None:
                quarantined_indexes.add(index)
                events.append(
                    _event("row", "quarantine", ("row_not_object",), raw, None, {"row_index": index}, batch_hash, context.cursor, row_hash)
                )
                continue

            missing = [field_name for field_name in REQUIRED_FIELDS if field_name not in raw_dict]
            for field_name in missing:
                missing_required_counts[field_name] += 1
            extra_fields.update(set(raw_dict) - EXPECTED_FIELDS)
            if missing:
                quarantined_indexes.add(index)
                events.append(
                    _event("row", "quarantine", ("missing_required_field",), raw_dict, None, {"row_index": index, "missing": missing}, batch_hash, context.cursor, row_hash)
                )
                continue

            row_events: list[RuntimeIngestionEvent] = []
            station_id = _normalize_station_id(raw_dict["station_id"], context.known_station_ids, row_events, raw_dict, batch_hash, context.cursor, row_hash, index)
            observed_at = _normalize_timestamp(raw_dict["observed_at"], row_events, raw_dict, batch_hash, context.cursor, row_hash, index)
            demand = _normalize_demand(raw_dict["demand"], row_events, raw_dict, batch_hash, context.cursor, row_hash, index)
            events.extend(row_events)
            if station_id is None or observed_at is None or demand is None:
                quarantined_indexes.add(index)
                continue

            normalized = {"station_id": station_id, "observed_at": observed_at, "demand": demand}
            normalized_candidates.append((index, ObservationRow(station_id, observed_at, demand), normalized, raw_dict))

        critical_missing = [field_name for field_name, count in missing_required_counts.items() if count == len(raw_rows)]
        if critical_missing:
            checks.append(
                _check(
                    "runtime.schema.required_fields",
                    "failed",
                    "critical",
                    actual=len(critical_missing),
                    expected=0,
                    details={"missing_from_all_rows": critical_missing},
                    batch_hash=batch_hash,
                    cursor=context.cursor,
                )
            )
        elif any(missing_required_counts.values()):
            checks.append(
                _check(
                    "runtime.schema.required_fields",
                    "warning",
                    "quarantine",
                    actual=sum(missing_required_counts.values()),
                    expected=0,
                    details={"missing_counts": missing_required_counts},
                    batch_hash=batch_hash,
                    cursor=context.cursor,
                )
            )
        else:
            checks.append(_check("runtime.schema.required_fields", "passed", "pass", 0, 0, {}, batch_hash, context.cursor))

        if extra_fields:
            checks.append(
                _check(
                    "runtime.schema.extra_fields",
                    "warning",
                    "warning",
                    actual=len(extra_fields),
                    expected=0,
                    details={"extra_fields": sorted(extra_fields)},
                    batch_hash=batch_hash,
                    cursor=context.cursor,
                )
            )
        else:
            checks.append(_check("runtime.schema.extra_fields", "passed", "pass", 0, 0, {}, batch_hash, context.cursor))

        valid_rows = _deduplicate_and_check_existing(normalized_candidates, context, events, batch_hash)
        _add_order_gap_coverage_checks(valid_rows, context, checks, events, batch_hash)
        _add_outlier_warnings(valid_rows, context, events, checks, batch_hash)
        compatibility = _model_compatibility(valid_rows, context)
        if compatibility.unknown_to_model or compatibility.missing_expected:
            checks.append(
                _check(
                    "runtime.model_compatibility",
                    "warning",
                    "warning",
                    actual=len(compatibility.unknown_to_model),
                    expected=0,
                    details={
                        "unknown_to_model": list(compatibility.unknown_to_model),
                        "missing_expected": list(compatibility.missing_expected),
                    },
                    batch_hash=batch_hash,
                    cursor=context.cursor,
                )
            )
        else:
            checks.append(_check("runtime.model_compatibility", "passed", "pass", 0, 0, {}, batch_hash, context.cursor))

        data_health = _data_health(checks, events)
        prediction_readiness = _prediction_readiness(data_health, valid_rows, compatibility)
        return RuntimeValidationResult(payload, valid_rows, events, checks, data_health, prediction_readiness, compatibility, batch_hash)


def _normalize_station_id(
    value: Any,
    known_station_ids: set[str] | None,
    events: list[RuntimeIngestionEvent],
    raw: dict[str, Any],
    batch_hash: str,
    cursor: str | None,
    row_hash: str,
    index: int,
) -> str | None:
    if isinstance(value, bool):
        events.append(_event("row", "quarantine", ("invalid_station_id",), raw, None, {"row_index": index}, batch_hash, cursor, row_hash))
        return None
    raw_text = str(value).strip() if isinstance(value, int) else value.strip() if isinstance(value, str) else None
    if raw_text is None or not raw_text.isdigit():
        events.append(_event("row", "quarantine", ("invalid_station_id",), raw, None, {"row_index": index, "value": repr(value)}, batch_hash, cursor, row_hash))
        return None
    if STATION_ID_RE.fullmatch(raw_text):
        if known_station_ids is not None and raw_text not in known_station_ids:
            events.append(_event("row", "quarantine", ("new_station",), raw, None, {"row_index": index, "station_id": raw_text}, batch_hash, cursor, row_hash, station_id=raw_text))
            return None
        return raw_text
    padded = raw_text.zfill(5)
    if known_station_ids is not None and padded in known_station_ids:
        events.append(_event("row", "warning", ("station_id_zero_padded",), raw, {"station_id": padded}, {"row_index": index, "original": raw_text}, batch_hash, cursor, row_hash, station_id=padded))
        return padded
    events.append(_event("row", "quarantine", ("station_id_not_recoverable",), raw, None, {"row_index": index, "candidate": padded}, batch_hash, cursor, row_hash))
    return None


def _normalize_timestamp(
    value: Any,
    events: list[RuntimeIngestionEvent],
    raw: dict[str, Any],
    batch_hash: str,
    cursor: str | None,
    row_hash: str,
    index: int,
) -> str | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            events.append(_event("row", "quarantine", ("invalid_timestamp",), raw, None, {"row_index": index, "value": value}, batch_hash, cursor, row_hash))
            return None
    else:
        events.append(_event("row", "quarantine", ("invalid_timestamp",), raw, None, {"row_index": index, "value": repr(value)}, batch_hash, cursor, row_hash))
        return None
    if parsed.tzinfo is None:
        events.append(_event("row", "quarantine", ("timestamp_missing_timezone",), raw, None, {"row_index": index}, batch_hash, cursor, row_hash))
        return None
    utc_time = parsed.astimezone(timezone.utc)
    if utc_time.second != 0 or utc_time.microsecond != 0 or utc_time.minute not in (0, 15, 30, 45):
        events.append(_event("row", "quarantine", ("timestamp_not_quarter_hour",), raw, None, {"row_index": index, "observed_at": _ts(utc_time)}, batch_hash, cursor, row_hash))
        return None
    return _ts(utc_time)


def _normalize_demand(
    value: Any,
    events: list[RuntimeIngestionEvent],
    raw: dict[str, Any],
    batch_hash: str,
    cursor: str | None,
    row_hash: str,
    index: int,
) -> int | None:
    if isinstance(value, bool):
        events.append(_event("row", "quarantine", ("demand_bool",), raw, None, {"row_index": index}, batch_hash, cursor, row_hash))
        return None
    normalized: int | None = None
    recovered = False
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            events.append(_event("row", "quarantine", ("demand_not_finite",), raw, None, {"row_index": index}, batch_hash, cursor, row_hash))
            return None
        if value.is_integer():
            normalized = int(value)
            recovered = True
    elif isinstance(value, str):
        try:
            decimal_value = Decimal(value.strip())
        except (InvalidOperation, ValueError):
            events.append(_event("row", "quarantine", ("demand_not_numeric",), raw, None, {"row_index": index, "value": value}, batch_hash, cursor, row_hash))
            return None
        if not decimal_value.is_finite():
            events.append(_event("row", "quarantine", ("demand_not_finite",), raw, None, {"row_index": index, "value": value}, batch_hash, cursor, row_hash))
            return None
        if decimal_value == decimal_value.to_integral_value():
            normalized = int(decimal_value)
            recovered = True
    if normalized is None:
        events.append(_event("row", "quarantine", ("demand_not_integer",), raw, None, {"row_index": index, "value": repr(value)}, batch_hash, cursor, row_hash))
        return None
    if normalized < 0:
        events.append(_event("row", "quarantine", ("demand_negative",), raw, None, {"row_index": index, "value": normalized}, batch_hash, cursor, row_hash))
        return None
    if recovered:
        events.append(_event("row", "warning", ("demand_integer_recovered",), raw, {"demand": normalized}, {"row_index": index, "original": value}, batch_hash, cursor, row_hash))
    return normalized


def _deduplicate_and_check_existing(
    candidates: list[tuple[int, ObservationRow, dict[str, Any], dict[str, Any]]],
    context: RuntimeValidationContext,
    events: list[RuntimeIngestionEvent],
    batch_hash: str,
) -> list[ObservationRow]:
    by_key: dict[tuple[str, str], list[tuple[int, ObservationRow, dict[str, Any], dict[str, Any]]]] = {}
    for item in candidates:
        _, row, _, _ = item
        by_key.setdefault((row.station_id, row.observed_at), []).append(item)
    valid: list[ObservationRow] = []
    for key, items in by_key.items():
        demands = {item[1].demand for item in items}
        if len(demands) > 1:
            for index, row, normalized, raw in items:
                events.append(
                    _event("row", "quarantine", ("duplicate_conflicting_batch",), raw, normalized, {"row_index": index, "demands": sorted(demands)}, batch_hash, context.cursor, _hash_payload(raw), row.station_id, row.observed_at)
                )
            continue
        first_index, first_row, first_normalized, first_raw = items[0]
        if len(items) > 1:
            for index, row, normalized, raw in items[1:]:
                events.append(
                    _event("row", "warning", ("duplicate_identical_batch",), raw, normalized, {"row_index": index, "kept_row_index": first_index}, batch_hash, context.cursor, _hash_payload(raw), row.station_id, row.observed_at)
                )
        existing = context.existing_observations.get(key)
        if existing is not None and existing != first_row.demand:
            events.append(
                _event("row", "quarantine", ("duplicate_conflicting_existing",), first_raw, first_normalized, {"existing_demand": existing, "incoming_demand": first_row.demand}, batch_hash, context.cursor, _hash_payload(first_raw), first_row.station_id, first_row.observed_at)
            )
            continue
        if existing == first_row.demand:
            events.append(
                _event("row", "warning", ("duplicate_identical_existing",), first_raw, first_normalized, {"existing_demand": existing}, batch_hash, context.cursor, _hash_payload(first_raw), first_row.station_id, first_row.observed_at)
            )
        valid.append(first_row)
    return valid


def _add_order_gap_coverage_checks(
    rows: list[ObservationRow],
    context: RuntimeValidationContext,
    checks: list[RuntimeQualityCheck],
    events: list[RuntimeIngestionEvent],
    batch_hash: str,
) -> None:
    parsed = [(row, datetime.fromisoformat(row.observed_at.replace("Z", "+00:00"))) for row in rows]
    out_of_order = sum(1 for idx in range(1, len(parsed)) if parsed[idx][1] < parsed[idx - 1][1])
    late = 0
    if context.last_observed_at:
        cutoff = datetime.fromisoformat(context.last_observed_at.replace("Z", "+00:00"))
        late = sum(1 for _, timestamp in parsed if timestamp <= cutoff)
    checks.append(_check("runtime.timestamp.ordering", "warning" if out_of_order or late else "passed", "warning" if out_of_order or late else "pass", out_of_order + late, 0, {"out_of_order": out_of_order, "late_arrivals": late}, batch_hash, context.cursor))

    gaps = 0
    by_station: dict[str, list[datetime]] = {}
    for row, timestamp in parsed:
        by_station.setdefault(row.station_id, []).append(timestamp)
    for timestamps in by_station.values():
        ordered = sorted(set(timestamps))
        gaps += sum(1 for idx in range(1, len(ordered)) if (ordered[idx] - ordered[idx - 1]).total_seconds() > 15 * 60)
    checks.append(_check("runtime.timestamp.gaps", "warning" if gaps else "passed", "warning" if gaps else "pass", gaps, 0, {}, batch_hash, context.cursor))

    if context.known_station_ids:
        observed = {row.station_id for row in rows}
        missing = sorted(context.known_station_ids - observed)
        coverage = len(observed & context.known_station_ids)
        checks.append(_check("runtime.station.coverage", "warning" if missing else "passed", "warning" if missing else "pass", coverage, len(context.known_station_ids), {"missing_station_ids": missing}, batch_hash, context.cursor))


def _add_outlier_warnings(
    rows: list[ObservationRow],
    context: RuntimeValidationContext,
    events: list[RuntimeIngestionEvent],
    checks: list[RuntimeQualityCheck],
    batch_hash: str,
) -> None:
    history: dict[tuple[str, int], list[int]] = {}
    for row in context.history_rows:
        timestamp = datetime.fromisoformat(row.observed_at.replace("Z", "+00:00")).astimezone(BOGOTA_TZ)
        history.setdefault((row.station_id, timestamp.hour * 4 + timestamp.minute // 15), []).append(row.demand)
    outliers = 0
    for row in rows:
        timestamp = datetime.fromisoformat(row.observed_at.replace("Z", "+00:00")).astimezone(BOGOTA_TZ)
        values = history.get((row.station_id, timestamp.hour * 4 + timestamp.minute // 15), [])
        if _is_outlier(row.demand, values):
            outliers += 1
            normalized = {"station_id": row.station_id, "observed_at": row.observed_at, "demand": row.demand}
            events.append(_event("row", "warning", ("demand_outlier",), normalized, normalized, {"history_count": len(values)}, batch_hash, context.cursor, _hash_payload(normalized), row.station_id, row.observed_at))
    checks.append(_check("runtime.demand.outliers", "warning" if outliers else "passed", "warning" if outliers else "pass", outliers, 0, {}, batch_hash, context.cursor))


def _is_outlier(value: int, history: list[int]) -> bool:
    if len(history) < 8:
        return False
    values = np.array(history, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return abs(value - median) / (1.4826 * mad) > 6.0
    if len(history) >= 12:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = float(q3 - q1)
        if iqr > 0:
            return value < q1 - 3.0 * iqr or value > q3 + 3.0 * iqr
    low, high = np.percentile(values, [1, 99])
    return value < low or value > high


def _model_compatibility(rows: list[ObservationRow], context: RuntimeValidationContext) -> ModelCompatibilityReport:
    observed = {row.station_id for row in rows}
    champion = context.champion_station_ids or set()
    if not champion:
        return ModelCompatibilityReport(tuple(sorted(observed)), (), ())
    return ModelCompatibilityReport(
        known_to_model=tuple(sorted(observed & champion)),
        unknown_to_model=tuple(sorted(observed - champion)),
        missing_expected=tuple(sorted(champion - observed)),
    )


def _data_health(checks: list[RuntimeQualityCheck], events: list[RuntimeIngestionEvent]) -> DataHealth:
    data_checks = [check for check in checks if check.check_name != "runtime.model_compatibility"]
    if any(check.severity == "critical" for check in data_checks) or any(event.severity == "critical" for event in events):
        return "CRITICAL"
    if any(check.severity in ("warning", "quarantine") for check in data_checks) or any(event.severity in ("warning", "quarantine") for event in events):
        return "DEGRADED"
    return "HEALTHY"


def _prediction_readiness(data_health: DataHealth, rows: list[ObservationRow], compatibility: ModelCompatibilityReport) -> PredictionReadiness:
    if data_health == "CRITICAL" or not rows:
        return "NOT_READY"
    if compatibility.unknown_to_model or compatibility.missing_expected:
        return "DEGRADED"
    return "READY"


def _event(
    event_type: str,
    severity: Severity,
    reason_codes: tuple[str, ...],
    raw_payload: Any,
    normalized_payload: dict[str, Any] | None,
    details: dict[str, Any],
    batch_hash: str,
    cursor: str | None,
    row_hash: str | None,
    station_id: str | None = None,
    observed_at: str | None = None,
) -> RuntimeIngestionEvent:
    key = _hash_payload({"batch_hash": batch_hash, "cursor": cursor, "row_hash": row_hash, "reason_codes": reason_codes, "event_type": event_type})
    resolution_status = "pending" if severity == "quarantine" else None
    return RuntimeIngestionEvent(event_type, severity, reason_codes, raw_payload, normalized_payload, details, station_id, observed_at, row_hash, key, resolution_status)


def _check(
    name: str,
    status: QualityStatus,
    severity: Severity,
    actual: int | None,
    expected: int | None,
    details: dict[str, Any],
    batch_hash: str,
    cursor: str | None,
) -> RuntimeQualityCheck:
    key = _hash_payload({"batch_hash": batch_hash, "cursor": cursor, "check_name": name})
    return RuntimeQualityCheck(name, status, severity, actual, expected, details, key)


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if isinstance(row, ObservationRow):
        return {"station_id": row.station_id, "observed_at": row.observed_at, "demand": row.demand}
    if isinstance(row, dict):
        return dict(row)
    return None


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, ObservationRow):
        return {"station_id": value.station_id, "observed_at": value.observed_at, "demand": value.demand}
    if isinstance(value, datetime):
        return _ts(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def api_error_validation_result(
    *,
    error_type: str,
    message: str,
    stream_name: str = "observations",
    cursor: str | None = None,
    raw_payload: Any | None = None,
) -> RuntimeValidationResult:
    payload = {
        "stream_name": stream_name,
        "cursor": cursor,
        "error_type": error_type,
        "message": message,
        "raw_payload": _jsonable(raw_payload),
    }
    batch_hash = _hash_payload(payload)
    event = _event(
        "api_error",
        "critical",
        (error_type,),
        payload,
        None,
        {"message": message, "cursor": cursor},
        batch_hash,
        cursor,
        None,
    )
    check = _check(
        "runtime.api_transport",
        "failed",
        "critical",
        actual=1,
        expected=0,
        details={"error_type": error_type, "message": message, "cursor": cursor},
        batch_hash=batch_hash,
        cursor=cursor,
    )
    return RuntimeValidationResult(
        payload,
        [],
        [event],
        [check],
        "CRITICAL",
        "NOT_READY",
        ModelCompatibilityReport((), (), ()),
        batch_hash,
    )
