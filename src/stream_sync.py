from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
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
    rows: list[ObservationRow]
    next_cursor: str | None


@dataclass(frozen=True)
class SyncResult:
    received: int
    written: int
    pages: int
    confirmed_cursor: str | None
    last_observed_at: str | None
    is_exhausted: bool = False


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
        payload = response.json()
        rows = [_validate_observation(row) for row in payload.get("data", [])]
        return StreamPage(rows=rows, next_cursor=payload.get("next_cursor"))


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


def sync_observation_stream(
    api: StreamApi,
    store: ObservationStore,
    *,
    stream_name: str = STREAM_NAME,
    limit: int = 5000,
    max_pages: int | None = None,
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
        page = api.observations_page(cursor=cursor, limit=limit)
        pages += 1
        if not page.rows:
            break

        rows_to_write = _rows_after_cutoff(page.rows, last_observed_at) if is_exhausted else page.rows
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


def _validate_observation(row: dict[str, Any]) -> ObservationRow:
    station_id = row.get("station_id")
    observed_at = row.get("observed_at")
    demand = row.get("demand")

    if not isinstance(station_id, str) or not STATION_ID_RE.fullmatch(station_id):
        raise StreamSyncError(f"invalid station_id: {station_id!r}")
    parsed = _parse_aware_utc(observed_at)
    if not isinstance(demand, int) or demand < 0:
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


def _rows_after_cutoff(rows: list[ObservationRow], cutoff: str | None) -> list[ObservationRow]:
    if cutoff is None:
        return rows
    cutoff_time = _parse_aware_utc(cutoff)
    return [row for row in rows if _parse_aware_utc(row.observed_at) > cutoff_time]


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
