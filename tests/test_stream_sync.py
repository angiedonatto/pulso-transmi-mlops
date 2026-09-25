from __future__ import annotations

import httpx
import pytest
from datetime import datetime, timedelta, timezone

from stream_sync import (
    CursorState,
    ObservationRow,
    PulsoStreamApi,
    QuarantinedRuntimeEvent,
    StreamPage,
    StreamSyncError,
    replay_quarantined_events,
    sync_observation_stream,
)


class FakeApi:
    def __init__(self, pages: dict[str | None, StreamPage]) -> None:
        self.pages = pages
        self.calls: list[str | None] = []

    def observations_page(self, *, cursor: str | None, limit: int) -> StreamPage:
        self.calls.append(cursor)
        page = self.pages[cursor]
        if isinstance(page, BaseException):
            raise page
        return page


class MemoryStore:
    def __init__(
        self,
        cursor: str | None = None,
        fail_write: bool = False,
        fail_validation: bool = False,
        is_exhausted: bool = False,
        known_station_ids: set[str] | None = None,
        champion_station_ids: set[str] | None = None,
    ) -> None:
        self.cursor = cursor
        self.fail_write = fail_write
        self.fail_validation = fail_validation
        self.is_exhausted = is_exhausted
        self.rows: dict[tuple[str, str], ObservationRow] = {}
        self.confirmed: list[str | None] = []
        self.rows_written = 0
        self.last_observed_at: str | None = None
        self.validation_results = []
        self.validation_idempotency_keys: set[str] = set()
        self.known_stations = known_station_ids
        self.champion_stations = champion_station_ids or set()
        self.quarantine_events: list[QuarantinedRuntimeEvent] = []
        self.replay_marks: dict[str, dict] = {}

    def get_cursor(self, stream_name: str = "observations") -> CursorState:
        return CursorState(self.cursor, self.last_observed_at, self.is_exhausted)

    def upsert_observations(self, rows: list[ObservationRow]) -> int:
        if self.fail_write:
            raise StreamSyncError("write failed")
        for row in rows:
            self.rows[(row.station_id, row.observed_at)] = row
        self.rows_written += len(rows)
        return len(rows)

    def confirm_cursor(
        self,
        *,
        stream_name: str,
        cursor: str | None,
        rows_written: int,
        last_observed_at: str | None,
        is_exhausted: bool,
    ) -> None:
        self.cursor = cursor
        self.is_exhausted = is_exhausted
        self.confirmed.append(cursor)
        self.last_observed_at = last_observed_at

    def persist_runtime_validation(self, *, stream_name: str, cursor: str | None, result) -> None:
        if self.fail_validation:
            raise StreamSyncError("validation persistence failed")
        self.validation_results.append(result)
        for event in result.events:
            self.validation_idempotency_keys.add(event.idempotency_key)
        for check in result.checks:
            self.validation_idempotency_keys.add(check.idempotency_key)

    def consecutive_empty_response_count(self, stream_name: str, cursor: str | None) -> int:
        return sum(
            1
            for result in self.validation_results
            for event in result.events
            if event.event_type == "batch" and "empty_response" in event.reason_codes
        )

    def known_station_ids(self) -> set[str] | None:
        return self.known_stations

    def active_model_station_ids(self) -> set[str]:
        return self.champion_stations

    def existing_observations(self, rows: list[ObservationRow]) -> dict[tuple[str, str], int]:
        return {(row.station_id, row.observed_at): stored.demand for row in rows if (stored := self.rows.get((row.station_id, row.observed_at)))}

    def quarantined_runtime_events(self, limit: int = 100) -> list[QuarantinedRuntimeEvent]:
        return [
            event
            for event in self.quarantine_events[:limit]
            if self.replay_marks.get(event.event_id, {}).get("resolution_status") != "resolved"
        ]

    def mark_runtime_event_replay_result(self, event: QuarantinedRuntimeEvent, *, resolved: bool, details: dict) -> None:
        self.replay_marks[event.event_id] = {
            "resolution_status": "resolved" if resolved else "pending",
            "replay_count": event.replay_count + 1,
            "details": details,
        }


ROW_1 = ObservationRow("03000", "2026-09-09T05:00:00Z", 91)
ROW_2 = ObservationRow("02300", "2026-09-09T05:00:00Z", 141)
ROW_TERMINAL = ObservationRow("10009", "2026-09-10T16:30:00Z", 262)


def test_rerun_does_not_duplicate_observations() -> None:
    api = FakeApi({None: StreamPage([ROW_1, ROW_2], "cursor-1"), "cursor-1": StreamPage([], None)})
    store = MemoryStore()

    first = sync_observation_stream(api, store)
    second = sync_observation_stream(api, store)

    assert first.received == 2
    assert second.received == 0
    assert len(store.rows) == 2
    assert store.rows_written == 2


def test_cursor_does_not_advance_when_write_fails() -> None:
    api = FakeApi({None: StreamPage([ROW_1], "cursor-1")})
    store = MemoryStore(fail_write=True)

    with pytest.raises(StreamSyncError):
        sync_observation_stream(api, store)

    assert store.cursor is None
    assert store.confirmed == []
    assert store.rows == {}


def test_cursor_does_not_advance_when_runtime_validation_persistence_fails() -> None:
    api = FakeApi({None: StreamPage([ROW_1], "cursor-1")})
    store = MemoryStore(fail_validation=True)

    with pytest.raises(StreamSyncError):
        sync_observation_stream(api, store)

    assert store.cursor is None
    assert store.confirmed == []
    assert store.rows == {}


def test_cursor_advances_after_confirmed_write() -> None:
    api = FakeApi({None: StreamPage([ROW_1], "cursor-1")})
    store = MemoryStore()

    result = sync_observation_stream(api, store, max_pages=1)

    assert result.received == 1
    assert result.written == 1
    assert store.cursor == "cursor-1"
    assert store.confirmed == ["cursor-1"]
    assert store.last_observed_at == "2026-09-09T05:00:00Z"


def test_empty_stream_finishes_successfully() -> None:
    api = FakeApi({None: StreamPage([], None)})
    store = MemoryStore()

    result = sync_observation_stream(api, store)

    assert result.received == 0
    assert result.written == 0
    assert result.confirmed_cursor is None
    assert store.confirmed == []
    assert len(store.validation_results) == 1


def test_repeated_empty_responses_escalate_to_critical_without_confirming_cursor() -> None:
    api = FakeApi({None: StreamPage([], None)})
    store = MemoryStore()

    sync_observation_stream(api, store, empty_response_critical_threshold=3)
    sync_observation_stream(api, store, empty_response_critical_threshold=3)
    with pytest.raises(StreamSyncError):
        sync_observation_stream(api, store, empty_response_critical_threshold=3)

    assert store.confirmed == []
    assert store.validation_results[-1].data_health == "CRITICAL"


def test_terminal_page_marks_exhausted_without_deriving_cursor() -> None:
    api = FakeApi({"cursor-1": StreamPage([ROW_TERMINAL], None)})
    store = MemoryStore(cursor="cursor-1")

    result = sync_observation_stream(api, store)

    assert result.received == 1
    assert result.confirmed_cursor == "cursor-1"
    assert result.is_exhausted is True
    assert store.cursor == "cursor-1"
    assert store.is_exhausted is True
    assert store.last_observed_at == "2026-09-10T16:30:00Z"


def test_exhausted_stream_rechecks_cursor_without_dropping_late_rows_by_timestamp() -> None:
    api = FakeApi({"cursor-1": StreamPage([ROW_TERMINAL], None)})
    store = MemoryStore(cursor="cursor-1", is_exhausted=True)
    store.last_observed_at = "2026-09-10T16:30:00Z"

    result = sync_observation_stream(api, store)

    assert result.received == 1
    assert result.written == 1
    assert result.pages == 1
    assert result.confirmed_cursor == "cursor-1"
    assert result.last_observed_at == "2026-09-10T16:30:00Z"
    assert result.is_exhausted is True
    assert api.calls == ["cursor-1"]
    assert store.rows == {("10009", "2026-09-10T16:30:00Z"): ROW_TERMINAL}


def test_partially_corrupt_batch_writes_valid_rows_and_quarantines_bad_rows() -> None:
    api = FakeApi(
        {
            None: StreamPage(
                [
                    {"station_id": "03000", "observed_at": "2026-09-09T05:00:00Z", "demand": 91},
                    {"station_id": "02300", "observed_at": "2026-09-09T05:00:00Z", "demand": -1},
                ],
                "cursor-1",
            )
        }
    )
    store = MemoryStore()

    result = sync_observation_stream(api, store, max_pages=1)

    assert result.received == 2
    assert result.written == 1
    assert store.cursor == "cursor-1"
    assert store.rows == {("03000", "2026-09-09T05:00:00Z"): ROW_1}
    assert store.validation_results[0].quarantined_rows == 1


def test_degraded_runtime_result_with_valid_rows_is_processable_for_inference() -> None:
    api = FakeApi(
        {
            None: StreamPage(
                [{"station_id": "03000", "observed_at": "2026-09-09T05:00:00Z", "demand": "91"}],
                "cursor-1",
            )
        }
    )
    store = MemoryStore()

    result = sync_observation_stream(api, store, max_pages=1)

    assert result.written == 1
    assert store.cursor == "cursor-1"
    assert store.validation_results[0].data_health == "DEGRADED"
    assert store.validation_results[0].prediction_readiness == "READY"


def test_existing_identical_duplicates_are_not_rewritten_or_noisy() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    existing_rows = [
        ObservationRow(
            "03000",
            (start + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z"),
            100 + index % 7,
        )
        for index in range(3708)
    ]
    new_start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    new_rows = [
        ObservationRow(
            "03000",
            (new_start + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z"),
            150 + index % 5,
        )
        for index in range(24)
    ]
    api = FakeApi({None: StreamPage(existing_rows + new_rows, "cursor-1")})
    store = MemoryStore()
    store.last_observed_at = "2026-02-15T00:00:00Z"
    for row in existing_rows:
        store.rows[(row.station_id, row.observed_at)] = row

    result = sync_observation_stream(api, store, max_pages=1)
    validation = store.validation_results[0]
    checks = {check.check_name: check for check in validation.checks}

    assert result.received == 3732
    assert result.written == 24
    assert store.rows_written == 24
    assert len(validation.rows_to_write) == 24
    assert len([event for event in validation.events if "duplicate_identical_existing" in event.reason_codes]) == 0
    assert checks["runtime.duplicates.identical_existing"].actual_value == 3708
    assert checks["runtime.duplicates.identical_existing"].severity == "pass"
    assert checks["runtime.timestamp.ordering"].details["late_arrivals"] == 0
    assert validation.data_health == "HEALTHY"


def test_retry_after_write_failure_reuses_idempotent_validation_events() -> None:
    api = FakeApi({None: StreamPage([ROW_1], "cursor-1")})
    store = MemoryStore(fail_write=True)

    with pytest.raises(StreamSyncError):
        sync_observation_stream(api, store)
    first_keys = set(store.validation_idempotency_keys)
    store.fail_write = False
    result = sync_observation_stream(api, store, max_pages=1)

    assert result.written == 1
    assert store.cursor == "cursor-1"
    assert store.validation_idempotency_keys == first_keys


def test_pulso_stream_api_surfaces_http_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with PulsoStreamApi(base_url="https://example.test", transport=httpx.MockTransport(handler)) as api:
        with pytest.raises(httpx.HTTPStatusError):
            api.observations_page(cursor=None, limit=10)


def test_pulso_stream_api_surfaces_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with PulsoStreamApi(base_url="https://example.test", transport=httpx.MockTransport(handler)) as api:
        with pytest.raises(httpx.ReadTimeout):
            api.observations_page(cursor=None, limit=10)


def test_timeout_generates_api_error_and_does_not_advance_cursor() -> None:
    request = httpx.Request("GET", "https://example.test/v1/stream/observations")
    api = FakeApi({None: httpx.ReadTimeout("timeout", request=request)})
    store = MemoryStore()

    with pytest.raises(httpx.ReadTimeout):
        sync_observation_stream(api, store)

    assert store.confirmed == []
    assert store.validation_results[0].events[0].event_type == "api_error"
    assert store.validation_results[0].events[0].severity == "critical"


def test_http_500_generates_api_error_and_does_not_advance_cursor() -> None:
    request = httpx.Request("GET", "https://example.test/v1/stream/observations")
    response = httpx.Response(500, request=request)
    api = FakeApi({None: httpx.HTTPStatusError("server error", request=request, response=response)})
    store = MemoryStore()

    with pytest.raises(httpx.HTTPStatusError):
        sync_observation_stream(api, store)

    assert store.confirmed == []
    assert store.validation_results[0].events[0].event_type == "api_error"
    assert store.validation_results[0].events[0].reason_codes == ("http_5xx",)


def test_api_error_persistence_failure_prevents_cursor_confirmation() -> None:
    request = httpx.Request("GET", "https://example.test/v1/stream/observations")
    api = FakeApi({None: httpx.ReadTimeout("timeout", request=request)})
    store = MemoryStore(fail_validation=True)

    with pytest.raises(StreamSyncError):
        sync_observation_stream(api, store)

    assert store.confirmed == []
    assert store.validation_results == []


def test_invalid_json_generates_api_error_and_does_not_advance_cursor() -> None:
    api = FakeApi({None: StreamSyncError("stream API returned invalid JSON")})
    store = MemoryStore()

    with pytest.raises(StreamSyncError):
        sync_observation_stream(api, store)

    assert store.confirmed == []
    assert store.validation_results[0].events[0].event_type == "api_error"
    assert store.validation_results[0].events[0].reason_codes == ("invalid_json",)


def test_quarantine_replay_successful() -> None:
    event = QuarantinedRuntimeEvent(
        event_id="evt-1",
        raw_payload={"station_id": "03000", "observed_at": "2026-09-09T05:00:00Z", "demand": "91"},
    )
    store = MemoryStore(known_station_ids={"03000"})
    store.quarantine_events.append(event)

    result = replay_quarantined_events(store)

    assert result.resolved == 1
    assert result.rows_written == 1
    assert store.rows[("03000", "2026-09-09T05:00:00Z")] == ROW_1
    assert store.replay_marks["evt-1"]["resolution_status"] == "resolved"


def test_quarantine_replay_still_invalid() -> None:
    event = QuarantinedRuntimeEvent(
        event_id="evt-2",
        raw_payload={"station_id": "03000", "observed_at": "2026-09-09T05:00:00Z", "demand": -1},
    )
    store = MemoryStore(known_station_ids={"03000"})
    store.quarantine_events.append(event)

    result = replay_quarantined_events(store)

    assert result.still_invalid == 1
    assert store.rows == {}
    assert store.replay_marks["evt-2"]["resolution_status"] == "pending"


def test_quarantine_replay_is_idempotent_after_resolution() -> None:
    event = QuarantinedRuntimeEvent(
        event_id="evt-3",
        raw_payload={"station_id": "03000", "observed_at": "2026-09-09T05:00:00Z", "demand": 91},
    )
    store = MemoryStore(known_station_ids={"03000"})
    store.quarantine_events.append(event)

    first = replay_quarantined_events(store)
    second = replay_quarantined_events(store)

    assert first.resolved == 1
    assert second.scanned == 0
    assert store.rows_written == 1


def test_new_station_quarantine_replay_passes_after_catalog_update() -> None:
    event = QuarantinedRuntimeEvent(
        event_id="evt-4",
        raw_payload={"station_id": "99999", "observed_at": "2026-09-09T05:00:00Z", "demand": 17},
        reason_codes=("new_station",),
    )
    store = MemoryStore(known_station_ids={"99999"})
    store.quarantine_events.append(event)

    result = replay_quarantined_events(store)

    assert result.resolved == 1
    assert store.rows[("99999", "2026-09-09T05:00:00Z")].demand == 17
