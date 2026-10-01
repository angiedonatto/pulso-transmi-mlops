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


@pytest.fixture(autouse=True)
def _no_sleep_and_no_default_reconcile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries must not slow the suite; legacy cursor tests run without the reconciliation pass
    (the reconciliation tests enable it explicitly)."""
    import stream_sync

    monkeypatch.setattr(stream_sync, "_sleep", lambda seconds: None)
    monkeypatch.setenv("PULSO_STREAM_RECONCILE_LOOKBACK_HOURS", "0")


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


# ---------------------------------------------------------------------------------------
# Robustness: late arrivals, revisions, retries, repeated pages, order, abrupt changes,
# reconciliation, gaps and freshness
# ---------------------------------------------------------------------------------------

from stream_sync import (  # noqa: E402
    RetryPolicy,
    SupabaseRestObservationStore,
    freshness_for_cutoff,
    reconcile_recent_observations,
    recent_completeness,
)

STATIONS = {"03000", "02300"}


def slot(index: int) -> str:
    return (datetime(2026, 9, 18, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z")


def stream_row(station: str, index: int, demand: int, released: str = "2026-09-30T00:00:00Z") -> dict:
    return {"station_id": station, "observed_at": slot(index), "demand": demand, "released_at": released}


class HistoryStore(MemoryStore):
    """MemoryStore that also exposes what the Supabase store offers for gaps and freshness."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("known_station_ids", set(STATIONS))
        super().__init__(**kwargs)

    def recent_observation_history(self, limit: int = 10000) -> list[ObservationRow]:
        return sorted(self.rows.values(), key=lambda row: row.observed_at, reverse=True)[:limit]


class SequenceApi:
    """Returns (or raises) the queued responses of a cursor in order; the last one repeats."""

    def __init__(self, responses: dict[str | None, list]) -> None:
        self.responses = responses
        self.calls: list[str | None] = []

    def observations_page(self, *, cursor: str | None, limit: int) -> StreamPage:
        self.calls.append(cursor)
        queue = self.responses[cursor]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.test/v1/stream/observations")
    return httpx.HTTPStatusError("error", request=request, response=httpx.Response(status, request=request))


def test_late_arrival_is_written_and_last_observed_never_moves_back() -> None:
    api = FakeApi({"c1": StreamPage([stream_row("03000", 2, 50)], "c2"), "c2": StreamPage([], None)})
    store = HistoryStore(cursor="c1")
    store.last_observed_at = slot(10)

    result = sync_observation_stream(api, store)

    assert store.rows[("03000", slot(2))].demand == 50
    assert result.last_observed_at == slot(10)
    ordering = {c.check_name: c for c in store.validation_results[0].checks}["runtime.timestamp.ordering"]
    assert ordering.details["late_arrivals"] == 1


def test_republished_value_overwrites_stored_value_with_audit_event() -> None:
    api = FakeApi({"c1": StreamPage([stream_row("03000", 4, 777, released="2026-10-01T00:00:00Z")], None)})
    store = HistoryStore(cursor="c1")
    store.rows[("03000", slot(4))] = ObservationRow("03000", slot(4), 120)

    sync_observation_stream(api, store)

    assert store.rows[("03000", slot(4))].demand == 777
    revision = next(e for r in store.validation_results for e in r.events if "value_revised" in e.reason_codes)
    assert revision.details["previous_demand"] == 120


def test_transient_timeout_is_retried_and_does_not_persist_an_api_error() -> None:
    request = httpx.Request("GET", "https://example.test/v1/stream/observations")
    api = SequenceApi({None: [httpx.ReadTimeout("t", request=request), http_error(502), StreamPage([stream_row("03000", 0, 9)], None)]})
    store = HistoryStore()

    result = sync_observation_stream(api, store, retry=RetryPolicy(attempts=3, base_delay_s=0))

    assert result.retries == 2 and result.written == 1
    assert all(e.event_type != "api_error" for r in store.validation_results for e in r.events)


def test_persistent_5xx_exhausts_retries_then_records_error_without_confirming() -> None:
    api = SequenceApi({None: [http_error(503)]})
    store = HistoryStore()

    with pytest.raises(httpx.HTTPStatusError):
        sync_observation_stream(api, store, retry=RetryPolicy(attempts=3, base_delay_s=0))

    assert api.calls == [None, None, None]
    assert store.confirmed == []
    assert store.validation_results[-1].events[0].reason_codes == ("http_5xx",)


def test_client_errors_are_not_retried() -> None:
    api = SequenceApi({None: [http_error(401)]})
    store = HistoryStore()

    with pytest.raises(httpx.HTTPStatusError):
        sync_observation_stream(api, store, retry=RetryPolicy(attempts=3, base_delay_s=0))

    assert api.calls == [None]


def test_page_that_returns_its_own_cursor_does_not_loop() -> None:
    api = FakeApi({"c1": StreamPage([stream_row("03000", 1, 10)], "c1")})
    store = HistoryStore(cursor="c1")

    result = sync_observation_stream(api, store)

    assert result.cursor_stalled is True
    assert api.calls == ["c1"]
    assert store.rows[("03000", slot(1))].demand == 10


def test_cursor_cycle_is_detected() -> None:
    api = FakeApi({None: StreamPage([stream_row("03000", 1, 10)], "a"), "a": StreamPage([stream_row("03000", 2, 11)], "b"), "b": StreamPage([stream_row("03000", 3, 12)], "a")})
    store = HistoryStore()

    result = sync_observation_stream(api, store)

    assert result.cursor_stalled is True
    assert api.calls == [None, "a", "b"]
    assert len(store.rows) == 3


def test_repeated_page_is_idempotent() -> None:
    page = [stream_row("03000", 1, 10), stream_row("02300", 1, 20)]
    api = FakeApi({None: StreamPage(page, "a"), "a": StreamPage(page, None)})  # server re-sends the same rows
    store = HistoryStore()

    result = sync_observation_stream(api, store)

    assert result.received == 4
    assert result.written == 2
    assert store.rows_written == 2


def test_out_of_order_rows_are_all_written() -> None:
    rows = [stream_row("03000", 3, 13), stream_row("03000", 1, 11), stream_row("03000", 2, 12)]
    api = FakeApi({None: StreamPage(rows, None)})
    store = HistoryStore()

    sync_observation_stream(api, store)

    assert [store.rows[("03000", slot(i))].demand for i in (1, 2, 3)] == [11, 12, 13]
    ordering = {c.check_name: c for c in store.validation_results[0].checks}["runtime.timestamp.ordering"]
    assert ordering.details["out_of_order"] >= 1


def test_abrupt_change_is_stored_not_discarded() -> None:
    store = HistoryStore(cursor="c1")
    for day in range(1, 15):
        when = datetime(2026, 9, 18, 4, 0, tzinfo=timezone.utc) - timedelta(days=day)
        store.rows[("03000", when.isoformat().replace("+00:00", "Z"))] = ObservationRow("03000", when.isoformat().replace("+00:00", "Z"), 100 + day % 3)
    api = FakeApi({"c1": StreamPage([stream_row("03000", 16, 2900)], None)})  # 04:00Z, ~29x the usual level

    sync_observation_stream(api, store)

    assert store.rows[("03000", slot(16))].demand == 2900
    assert all(e.severity != "quarantine" for r in store.validation_results for e in r.events)


def test_reconciliation_fills_missing_rows_and_applies_revisions_without_moving_the_cursor() -> None:
    full = [stream_row(s, i, 100 + i) for i in range(8) for s in sorted(STATIONS)]
    full[-1] = {**full[-1], "demand": 999}
    api = FakeApi({None: StreamPage(full[:8], "p2"), "p2": StreamPage(full[8:], None)})
    store = HistoryStore(cursor="confirmed")
    for row in full:
        if row["observed_at"] != slot(5):  # slot 5 never arrived
            store.rows[(row["station_id"], row["observed_at"])] = ObservationRow(row["station_id"], row["observed_at"], 100 + int(row["observed_at"][14:16]) // 15 + 4 * int(row["observed_at"][11:13]))
    stored_last = store.rows[(full[-1]["station_id"], full[-1]["observed_at"])].demand

    result = reconcile_recent_observations(api, store, lookback_hours=6, retry=RetryPolicy(base_delay_s=0))

    assert result.error is None
    assert result.missing_filled == 2 and result.revised == 1
    assert store.rows[(full[-1]["station_id"], full[-1]["observed_at"])].demand == 999 != stored_last
    assert store.cursor == "confirmed" and store.confirmed == []


def test_reconciliation_only_touches_the_lookback_window() -> None:
    rows = [stream_row("03000", i, 10) for i in range(0, 40)]
    api = FakeApi({None: StreamPage(rows, None)})
    store = HistoryStore()

    result = reconcile_recent_observations(api, store, lookback_hours=2, retry=RetryPolicy(base_delay_s=0))

    assert result.in_window == 8  # 2h = 8 slots ending at the newest slot
    assert len(store.rows) == 8


def test_reconciliation_failure_never_breaks_the_sync() -> None:
    api = SequenceApi({"c1": [StreamPage([stream_row("03000", 1, 10)], None)], None: [http_error(503)]})
    store = HistoryStore(cursor="c1")

    result = sync_observation_stream(api, store, reconcile_lookback_hours=6, retry=RetryPolicy(attempts=2, base_delay_s=0))

    assert result.written == 1 and store.cursor == "c1"
    assert result.reconcile is not None and "HTTPStatusError" in result.reconcile.error
    assert all(e.event_type != "api_error" for r in store.validation_results for e in r.events)


def test_completeness_detects_gaps_across_pages() -> None:
    rows = [ObservationRow(s, slot(i), 1) for i in range(8) for s in STATIONS if not (s == "02300" and i == 5)]
    report = recent_completeness(rows, STATIONS, lookback_hours=2)

    assert report.missing_slots == (("02300", slot(5)),)
    assert report.observed_through == slot(4)
    assert report.expected_slots == 16


def test_freshness_for_cutoff_flags_a_missing_last_slot() -> None:
    rows = [ObservationRow(s, slot(i), 1) for i in range(8) for s in STATIONS]
    assert freshness_for_cutoff(rows, STATIONS, slot(7), lookback_hours=2)["fresh"] is True
    stale = freshness_for_cutoff(rows, STATIONS, slot(8), lookback_hours=2)
    assert stale["fresh"] is False
    assert stale["observed_through"] == slot(7)
    assert ["03000", slot(8)] in stale["missing_slots"] and ["02300", slot(8)] in stale["missing_slots"]


def test_required_through_fails_when_data_is_stale() -> None:
    api = FakeApi({None: StreamPage([stream_row(s, i, 5) for i in range(4) for s in STATIONS], None)})
    store = HistoryStore()

    with pytest.raises(StreamSyncError, match="complete only through"):
        sync_observation_stream(api, store, reconcile_lookback_hours=0, required_through=slot(5))
    # the rows were written and the cursor confirmed before the freshness check failed
    assert len(store.rows) == 8
    assert store.confirmed == [None] and store.is_exhausted is True


def test_confirm_cursor_refreshes_confirmed_at() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(201, json=[])

    with SupabaseRestObservationStore("https://example.test", "key", transport=httpx.MockTransport(handler)) as store:
        store.confirm_cursor(stream_name="observations", cursor="c", rows_written=1, last_observed_at=slot(0), is_exhausted=True)

    assert b"confirmed_at" in captured["body"]
