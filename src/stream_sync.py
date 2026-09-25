from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx


STREAM_NAME = "observations"
STATION_ID_RE = re.compile(r"^[0-9]{5}$")
BOGOTA_TZ = ZoneInfo("America/Bogota")
DAY_NAMES_ES = {
    0: "lunes",
    1: "martes",
    2: "miercoles",
    3: "jueves",
    4: "viernes",
    5: "sabado",
    6: "domingo",
}


class StreamSyncError(RuntimeError):
    """Raised when incremental observation sync cannot continue safely."""


@dataclass(frozen=True)
class ObservationRow:
    station_id: str
    observed_at: str
    demand: int


@dataclass(frozen=True)
class CursorState:
    confirmed_cursor: str | None = None
    last_observed_at: str | None = None
    is_exhausted: bool = False


@dataclass(frozen=True)
class StreamPage:
    rows: list[Any]
    next_cursor: str | None
    raw_payload: Any | None = None


@dataclass(frozen=True)
class SyncResult:
    received: int
    written: int
    pages: int
    confirmed_cursor: str | None
    last_observed_at: str | None
    is_exhausted: bool = False


@dataclass(frozen=True)
class QuarantinedRuntimeEvent:
    event_id: str
    raw_payload: Any
    stream_name: str = STREAM_NAME
    cursor: str | None = None
    replay_count: int = 0
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuarantineReplayResult:
    scanned: int
    resolved: int
    still_invalid: int
    rows_written: int


class StreamApi(Protocol):
    def observations_page(self, *, cursor: str | None, limit: int) -> StreamPage:
        ...


class ObservationStore(Protocol):
    def get_cursor(self, stream_name: str = STREAM_NAME) -> CursorState:
        ...

    def upsert_observations(self, rows: list[ObservationRow]) -> int:
        ...

    def confirm_cursor(
        self,
        *,
        stream_name: str,
        cursor: str | None,
        rows_written: int,
        last_observed_at: str | None,
        is_exhausted: bool,
    ) -> None:
        ...


class PulsoStreamApi:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_url = base_url or os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
        resolved_key = api_key or os.getenv("PULSO_API_KEY")
        headers = {"User-Agent": "pulso-transmi-stream-sync/0.1.0"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        self._client = httpx.Client(
            base_url=resolved_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PulsoStreamApi":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def observations_page(self, *, cursor: str | None, limit: int) -> StreamPage:
        response = self._client.get(
            "/v1/stream/observations",
            params={key: value for key, value in {"cursor": cursor, "limit": limit}.items() if value is not None},
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise StreamSyncError("stream API returned invalid JSON") from exc
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []
        return StreamPage(rows=rows, next_cursor=payload.get("next_cursor") if isinstance(payload, dict) else None, raw_payload=payload)


class SupabaseRestObservationStore:
    def __init__(
        self,
        supabase_url: str | None = None,
        service_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        resolved_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        resolved_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not resolved_url or not resolved_key:
            raise StreamSyncError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{resolved_url}/rest/v1",
            headers={
                "apikey": resolved_key,
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupabaseRestObservationStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_cursor(self, stream_name: str = STREAM_NAME) -> CursorState:
        response = self._client.get(
            "/stream_cursors",
            params={"stream_name": f"eq.{stream_name}", "select": "confirmed_cursor,last_observed_at,is_exhausted"},
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return CursorState()
        return CursorState(
            confirmed_cursor=payload[0].get("confirmed_cursor"),
            last_observed_at=payload[0].get("last_observed_at"),
            is_exhausted=bool(payload[0].get("is_exhausted")),
        )

    def upsert_observations(self, rows: list[ObservationRow]) -> int:
        if not rows:
            return 0
        self._upsert_time_slots(rows)
        response = self._client.post(
            "/observations",
            params={"on_conflict": "station_id,observed_at"},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=[
                {"station_id": row.station_id, "observed_at": row.observed_at, "demand": row.demand}
                for row in rows
            ],
        )
        response.raise_for_status()
        return len(rows)

    def known_station_ids(self) -> set[str]:
        response = self._client.get("/stations", params={"select": "station_id"})
        response.raise_for_status()
        return {str(row["station_id"]) for row in response.json()}

    def existing_observations(self, rows: list[ObservationRow]) -> dict[tuple[str, str], int]:
        timestamps = sorted({row.observed_at for row in rows})
        if not timestamps:
            return {}
        result: dict[tuple[str, str], int] = {}
        wanted = {(row.station_id, row.observed_at) for row in rows}
        for chunk in _chunks(timestamps, 50):
            response = self._client.get(
                "/observations",
                params={
                    "observed_at": f"in.({','.join(chunk)})",
                    "select": "station_id,observed_at,demand",
                },
            )
            response.raise_for_status()
            for item in response.json():
                key = (str(item["station_id"]), _normalize_timestamp(item["observed_at"]))
                if key in wanted:
                    result[key] = int(item["demand"])
        return result

    def recent_observation_history(self, limit: int = 10000) -> list[ObservationRow]:
        response = self._client.get(
            "/observations",
            params={
                "select": "station_id,observed_at,demand",
                "order": "observed_at.desc",
                "limit": str(limit),
            },
        )
        response.raise_for_status()
        return [
            ObservationRow(str(row["station_id"]), _normalize_timestamp(row["observed_at"]), int(row["demand"]))
            for row in response.json()
        ]

    def active_model_station_ids(self) -> set[str]:
        response = self._client.get(
            "/model_versions",
            params={"is_active": "eq.true", "select": "feature_config", "limit": "1"},
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return set()
        mapping = rows[0].get("feature_config", {}).get("station_mapping", {})
        return set(mapping) if isinstance(mapping, dict) else set()

    def persist_runtime_validation(self, *, stream_name: str, cursor: str | None, result: Any) -> None:
        _validate_runtime_idempotency_keys(result)
        run_id = str(uuid4())
        self._post(
            "/pipeline_runs",
            {
                "run_id": run_id,
                "status": "running",
                "trigger_type": "runtime_data_validation",
            },
        )
        if result.events:
            self._post_upsert(
                "/runtime_ingestion_events",
                [
                    {
                        "run_id": run_id,
                        "stream_name": stream_name,
                        "cursor": cursor,
                        "batch_hash": result.batch_hash,
                        "row_hash": event.row_hash,
                        "idempotency_key": event.idempotency_key,
                        "event_type": event.event_type,
                        "severity": event.severity,
                        "station_id": event.station_id,
                        "observed_at": event.observed_at,
                        "raw_payload": event.raw_payload,
                        "normalized_payload": event.normalized_payload or {},
                        "details": event.details,
                        "reason_codes": list(event.reason_codes),
                        "resolution_status": event.resolution_status,
                    }
                    for event in result.events
                ],
                on_conflict="idempotency_key",
            )
        if result.checks:
            self._post_upsert(
                "/data_quality_checks",
                [
                    {
                        "run_id": run_id,
                        "check_name": check.check_name,
                        "status": check.status,
                        "severity": check.severity,
                        "actual_value": check.actual_value,
                        "expected_value": check.expected_value,
                        "details": check.details.get("message") if isinstance(check.details.get("message"), str) else None,
                        "details_json": check.details,
                        "idempotency_key": check.idempotency_key,
                    }
                    for check in result.checks
                ],
                on_conflict="idempotency_key",
            )
        self._patch("/pipeline_runs", {"status": "success", "finished_at": _now_iso()}, params={"run_id": f"eq.{run_id}"})

    def consecutive_empty_response_count(self, stream_name: str, cursor: str | None) -> int:
        params = {
            "stream_name": f"eq.{stream_name}",
            "event_type": "eq.batch",
            "reason_codes": "cs.{empty_response}",
            "select": "event_id",
            "limit": "1000",
        }
        params["cursor"] = "is.null" if cursor is None else f"eq.{cursor}"
        response = self._client.get("/runtime_ingestion_events", params=params)
        response.raise_for_status()
        return len(response.json())

    def quarantined_runtime_events(self, limit: int = 100) -> list[QuarantinedRuntimeEvent]:
        response = self._client.get(
            "/runtime_ingestion_events",
            params={
                "severity": "eq.quarantine",
                "resolution_status": "eq.pending",
                "select": "event_id,stream_name,cursor,raw_payload,replay_count,reason_codes",
                "order": "created_at.asc",
                "limit": str(limit),
            },
        )
        response.raise_for_status()
        return [
            QuarantinedRuntimeEvent(
                event_id=str(row["event_id"]),
                stream_name=str(row.get("stream_name") or STREAM_NAME),
                cursor=row.get("cursor"),
                raw_payload=row.get("raw_payload"),
                replay_count=int(row.get("replay_count") or 0),
                reason_codes=tuple(row.get("reason_codes") or ()),
            )
            for row in response.json()
        ]

    def mark_runtime_event_replay_result(self, event: QuarantinedRuntimeEvent, *, resolved: bool, details: dict[str, Any]) -> None:
        payload = {
            "resolution_status": "resolved" if resolved else "pending",
            "replay_count": event.replay_count + 1,
            "resolution_details": details,
        }
        if resolved:
            payload["resolved_at"] = _now_iso()
        self._patch("/runtime_ingestion_events", payload, params={"event_id": f"eq.{event.event_id}"})

    def _upsert_time_slots(self, rows: list[ObservationRow]) -> None:
        slots = {_time_slot_payload(row.observed_at)["observed_at"]: _time_slot_payload(row.observed_at) for row in rows}
        response = self._client.post(
            "/time_slots",
            params={"on_conflict": "observed_at"},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=list(slots.values()),
        )
        response.raise_for_status()

    def confirm_cursor(
        self,
        *,
        stream_name: str,
        cursor: str | None,
        rows_written: int,
        last_observed_at: str | None,
        is_exhausted: bool,
    ) -> None:
        response = self._client.post(
            "/stream_cursors",
            params={"on_conflict": "stream_name"},
            headers={"Prefer": "resolution=merge-duplicates"},
            json={
                "stream_name": stream_name,
                "confirmed_cursor": cursor,
                "rows_written": rows_written,
                "last_observed_at": last_observed_at,
                "is_exhausted": is_exhausted,
            },
        )
        response.raise_for_status()

    def _post(self, path: str, payload: Any) -> None:
        response = self._client.post(path, json=payload)
        response.raise_for_status()

    def _patch(self, path: str, payload: dict[str, Any], *, params: dict[str, str]) -> None:
        response = self._client.patch(path, params=params, json=payload)
        response.raise_for_status()

    def _post_upsert(self, path: str, payload: Any, *, on_conflict: str) -> None:
        response = self._client.post(
            path,
            params={"on_conflict": on_conflict},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=payload,
        )
        response.raise_for_status()


def sync_observation_stream(
    api: StreamApi,
    store: ObservationStore,
    *,
    stream_name: str = STREAM_NAME,
    limit: int = 5000,
    max_pages: int | None = None,
    empty_response_critical_threshold: int = 3,
) -> SyncResult:
    state = store.get_cursor(stream_name)
    cursor = state.confirmed_cursor
    received = 0
    written = 0
    pages = 0
    last_observed_at = state.last_observed_at
    is_exhausted = state.is_exhausted

    while True:
        if max_pages is not None and pages >= max_pages:
            break
        try:
            page = api.observations_page(cursor=cursor, limit=limit)
        except (httpx.TimeoutException, httpx.HTTPStatusError, StreamSyncError) as exc:
            _persist_api_error_if_supported(store, stream_name, cursor, exc)
            raise
        pages += 1
        validation = _validate_runtime_page(store, stream_name, cursor, last_observed_at, page, empty_response_critical_threshold)
        _persist_runtime_validation_if_supported(store, stream_name, cursor, validation)
        if validation.has_critical:
            raise StreamSyncError("runtime data validation reported CRITICAL health")
        if not page.rows:
            break

        rows_to_write = getattr(validation, "rows_to_write", validation.valid_rows)
        rows_written = store.upsert_observations(rows_to_write)
        page_last_observed_at = max((row.observed_at for row in rows_to_write), default=last_observed_at)
        is_exhausted = page.next_cursor is None
        next_cursor = page.next_cursor if page.next_cursor is not None else cursor
        store.confirm_cursor(
            stream_name=stream_name,
            cursor=next_cursor,
            rows_written=rows_written,
            last_observed_at=page_last_observed_at,
            is_exhausted=is_exhausted,
        )

        received += len(page.rows)
        written += rows_written
        cursor = next_cursor
        last_observed_at = page_last_observed_at
        if is_exhausted:
            break

    return SyncResult(
        received=received,
        written=written,
        pages=pages,
        confirmed_cursor=cursor,
        last_observed_at=last_observed_at,
        is_exhausted=is_exhausted,
    )


def _validate_runtime_page(
    store: ObservationStore,
    stream_name: str,
    cursor: str | None,
    last_observed_at: str | None,
    page: StreamPage,
    empty_response_critical_threshold: int,
) -> Any:
    from runtime_data_validation import RuntimeDataValidator, RuntimeValidationContext

    validator = RuntimeDataValidator()
    context = RuntimeValidationContext(
        known_station_ids=_store_call(store, "known_station_ids", default=None),
        champion_station_ids=_store_call(store, "active_model_station_ids", default=set()),
        last_observed_at=last_observed_at,
        history_rows=_store_call(store, "recent_observation_history", default=[]),
        stream_name=stream_name,
        cursor=cursor,
        empty_response_count=_store_call(store, "consecutive_empty_response_count", stream_name, cursor, default=0),
        empty_response_critical_threshold=empty_response_critical_threshold,
    )
    result = validator.validate(page.rows, context, raw_payload=page.raw_payload)
    if result.has_critical or not result.valid_rows:
        return result
    existing = _store_call(store, "existing_observations", result.valid_rows, default={})
    if not existing:
        return result
    context = RuntimeValidationContext(
        known_station_ids=context.known_station_ids,
        champion_station_ids=context.champion_station_ids,
        last_observed_at=last_observed_at,
        existing_observations=existing,
        history_rows=context.history_rows,
        stream_name=stream_name,
        cursor=cursor,
        empty_response_count=context.empty_response_count,
        empty_response_critical_threshold=empty_response_critical_threshold,
    )
    return validator.validate(page.rows, context, raw_payload=page.raw_payload)


def replay_quarantined_events(store: ObservationStore, *, limit: int = 100) -> QuarantineReplayResult:
    from runtime_data_validation import RuntimeDataValidator, RuntimeValidationContext

    method = getattr(store, "quarantined_runtime_events", None)
    if not callable(method):
        raise StreamSyncError("store does not support quarantine replay")

    validator = RuntimeDataValidator()
    scanned = 0
    resolved = 0
    still_invalid = 0
    rows_written = 0

    for event in method(limit=limit):
        scanned += 1
        context = RuntimeValidationContext(
            known_station_ids=_store_call(store, "known_station_ids", default=None),
            champion_station_ids=_store_call(store, "active_model_station_ids", default=set()),
            history_rows=_store_call(store, "recent_observation_history", default=[]),
            stream_name=event.stream_name,
            cursor=None,
        )
        raw_rows = [event.raw_payload]
        validation = validator.validate(raw_rows, context, raw_payload={"replay_event_id": event.event_id, "raw_payload": event.raw_payload})
        if not validation.has_critical and validation.valid_rows:
            existing = _store_call(store, "existing_observations", validation.valid_rows, default={})
            if existing:
                context = RuntimeValidationContext(
                    known_station_ids=context.known_station_ids,
                    champion_station_ids=context.champion_station_ids,
                    existing_observations=existing,
                    history_rows=context.history_rows,
                    stream_name=event.stream_name,
                    cursor=None,
                )
                validation = validator.validate(raw_rows, context, raw_payload={"replay_event_id": event.event_id, "raw_payload": event.raw_payload})

        event_resolved = not validation.has_critical and bool(validation.valid_rows)
        if event_resolved:
            rows_written += store.upsert_observations(getattr(validation, "rows_to_write", validation.valid_rows))
            resolved += 1
        else:
            still_invalid += 1

        marker = getattr(store, "mark_runtime_event_replay_result", None)
        if not callable(marker):
            raise StreamSyncError("store does not support quarantine replay marking")
        marker(
            event,
            resolved=event_resolved,
            details={
                "data_health": validation.data_health,
                "prediction_readiness": validation.prediction_readiness,
                "valid_rows": len(validation.valid_rows),
                "quarantined_rows": validation.quarantined_rows,
                "reason_codes": sorted({reason for replay_event in validation.events for reason in replay_event.reason_codes}),
            },
        )

    return QuarantineReplayResult(scanned, resolved, still_invalid, rows_written)


def _persist_runtime_validation_if_supported(
    store: ObservationStore,
    stream_name: str,
    cursor: str | None,
    result: Any,
) -> None:
    method = getattr(store, "persist_runtime_validation", None)
    if callable(method):
        method(stream_name=stream_name, cursor=cursor, result=result)


def _persist_api_error_if_supported(store: ObservationStore, stream_name: str, cursor: str | None, exc: BaseException) -> None:
    from runtime_data_validation import api_error_validation_result

    error_type = "timeout" if isinstance(exc, httpx.TimeoutException) else "http_5xx" if isinstance(exc, httpx.HTTPStatusError) else "invalid_json"
    result = api_error_validation_result(error_type=error_type, message=str(exc), stream_name=stream_name, cursor=cursor)
    _persist_runtime_validation_if_supported(store, stream_name, cursor, result)


def _validate_runtime_idempotency_keys(result: Any) -> None:
    for check in result.checks:
        if not isinstance(check.idempotency_key, str) or not check.idempotency_key:
            raise StreamSyncError(f"runtime quality check missing idempotency_key: {check.check_name}")
    for event in result.events:
        if not isinstance(event.idempotency_key, str) or not event.idempotency_key:
            raise StreamSyncError(f"runtime ingestion event missing idempotency_key: {event.reason_codes}")


def _store_call(store: ObservationStore, name: str, *args: Any, default: Any) -> Any:
    method = getattr(store, name, None)
    if not callable(method):
        return default
    return method(*args)


def _validate_observation(row: dict[str, Any]) -> ObservationRow:
    station_id = row.get("station_id")
    observed_at = row.get("observed_at")
    demand = row.get("demand")

    if not isinstance(station_id, str) or not STATION_ID_RE.fullmatch(station_id):
        raise StreamSyncError(f"invalid station_id: {station_id!r}")
    parsed = _parse_aware_utc(observed_at)
    if isinstance(demand, bool) or not isinstance(demand, int) or demand < 0:
        raise StreamSyncError(f"invalid demand: {demand!r}")
    return ObservationRow(
        station_id=station_id,
        observed_at=parsed.isoformat().replace("+00:00", "Z"),
        demand=demand,
    )


def _parse_aware_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise StreamSyncError(f"invalid observed_at: {value!r}")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise StreamSyncError(f"invalid observed_at: {value!r}") from exc
    if parsed.tzinfo is None:
        raise StreamSyncError(f"observed_at must include timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _time_slot_payload(observed_at: str) -> dict[str, Any]:
    utc_time = _parse_aware_utc(observed_at)
    local_time = utc_time.astimezone(BOGOTA_TZ)
    day_of_week = local_time.weekday()
    return {
        "observed_at": utc_time.isoformat().replace("+00:00", "Z"),
        "local_date": local_time.date().isoformat(),
        "hour": local_time.hour,
        "minute": local_time.minute,
        "day_of_week": day_of_week,
        "day_name": DAY_NAMES_ES[day_of_week],
        "is_weekend": day_of_week >= 5,
        "quarter_hour_block": local_time.hour * 4 + local_time.minute // 15,
    }


def _normalize_timestamp(value: Any) -> str:
    return _parse_aware_utc(value).isoformat().replace("+00:00", "Z")


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    with PulsoStreamApi() as api, SupabaseRestObservationStore() as store:
        result = sync_observation_stream(api, store)
    print(
        "stream sync complete: "
        f"received={result.received} written={result.written} "
        f"last_observed_at={result.last_observed_at} cursor_confirmed={result.confirmed_cursor is not None} "
        f"exhausted={result.is_exhausted}"
    )


if __name__ == "__main__":
    main()
