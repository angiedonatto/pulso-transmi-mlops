from __future__ import annotations

import pytest

from stream_sync import CursorState, ObservationRow, StreamPage, StreamSyncError, sync_observation_stream


class FakeApi:
    def __init__(self, pages: dict[str | None, StreamPage]) -> None:
        self.pages = pages
        self.calls: list[str | None] = []

    def observations_page(self, *, cursor: str | None, limit: int) -> StreamPage:
        self.calls.append(cursor)
        return self.pages[cursor]


class MemoryStore:
    def __init__(self, cursor: str | None = None, fail_write: bool = False, is_exhausted: bool = False) -> None:
        self.cursor = cursor
        self.fail_write = fail_write
        self.is_exhausted = is_exhausted
        self.rows: dict[tuple[str, str], ObservationRow] = {}
        self.confirmed: list[str | None] = []
        self.rows_written = 0
        self.last_observed_at: str | None = None

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


def test_exhausted_stream_rechecks_cursor_without_rewriting_confirmed_rows() -> None:
    api = FakeApi({"cursor-1": StreamPage([ROW_TERMINAL], None)})
    store = MemoryStore(cursor="cursor-1", is_exhausted=True)
    store.last_observed_at = "2026-09-10T16:30:00Z"

    result = sync_observation_stream(api, store)

    assert result.received == 1
    assert result.written == 0
    assert result.pages == 1
    assert result.confirmed_cursor == "cursor-1"
    assert result.last_observed_at == "2026-09-10T16:30:00Z"
    assert result.is_exhausted is True
    assert api.calls == ["cursor-1"]
    assert store.rows == {}
