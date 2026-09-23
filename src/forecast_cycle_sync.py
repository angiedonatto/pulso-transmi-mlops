from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

import httpx


DEFAULT_BASE_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
STATION_ID_RE = re.compile(r"^[0-9]{5}$")


class ForecastCycleError(RuntimeError):
    """Raised when the official forecast cycle cannot be handled safely."""


@dataclass(frozen=True)
class CycleTarget:
    station_id: str
    target_at: str
    horizon_minutes: int
    target_order: int
    raw_target: dict[str, Any]


@dataclass(frozen=True)
class ForecastCycle:
    cycle_id: str
    data_cutoff: str
    closes_at: str
    expected_predictions: int
    targets: list[CycleTarget]
    raw_cycle: dict[str, Any]


@dataclass(frozen=True)
class CycleSyncResult:
    status: Literal["stored", "no_open_cycle"]
    cycle_id: str | None = None
    expected_predictions: int = 0
    targets_written: int = 0


class ForecastCycleApi(Protocol):
    def validate_identity(self) -> dict[str, Any]:
        ...

    def current_cycle(self) -> ForecastCycle | None:
        ...


class ForecastCycleStore(Protocol):
    def persist_cycle(self, cycle: ForecastCycle) -> None:
        ...

    def targets_for_cycle(self, cycle_id: str) -> list[CycleTarget]:
        ...


class PulsoForecastCycleApi:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_url = base_url or os.getenv("PULSO_API_URL", DEFAULT_BASE_URL)
        resolved_key = api_key or os.getenv("PULSO_API_KEY")
        headers = {"User-Agent": "pulso-transmi-cycle-sync/0.1.0"}
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

    def __enter__(self) -> "PulsoForecastCycleApi":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def validate_identity(self) -> dict[str, Any]:
        response = self._client.get("/v1/me")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ForecastCycleError(f"identity validation failed: HTTP {response.status_code}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise ForecastCycleError("/v1/me returned a non-object payload")
        return payload

    def current_cycle(self) -> ForecastCycle | None:
        response = self._client.get("/v1/forecast-cycles/current")
        if response.status_code == 404:
            payload = response.json()
            if _error_code(payload) == "no_open_cycle":
                return None
            raise ForecastCycleError(f"unexpected 404 from current cycle: {_error_code(payload)!r}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ForecastCycleError(f"current cycle request failed: HTTP {response.status_code}") from exc
        return _parse_cycle(response.json())


class SupabaseRestForecastCycleStore:
    def __init__(
        self,
        supabase_url: str | None = None,
        service_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        resolved_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        resolved_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not resolved_url or not resolved_key:
            raise ForecastCycleError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
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

    def __enter__(self) -> "SupabaseRestForecastCycleStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def persist_cycle(self, cycle: ForecastCycle) -> None:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        cycle_response = self._client.post(
            "/forecast_cycles",
            params={"on_conflict": "cycle_id"},
            headers={"Prefer": "resolution=merge-duplicates"},
            json={
                "cycle_id": cycle.cycle_id,
                "data_cutoff": cycle.data_cutoff,
                "closes_at": cycle.closes_at,
                "expected_predictions": cycle.expected_predictions,
                "raw_cycle": cycle.raw_cycle,
                "last_seen_at": now,
            },
        )
        cycle_response.raise_for_status()

        delete_response = self._client.delete("/forecast_cycle_targets", params={"cycle_id": f"eq.{cycle.cycle_id}"})
        delete_response.raise_for_status()
        if not cycle.targets:
            return
        targets_response = self._client.post(
            "/forecast_cycle_targets",
            json=[
                {
                    "cycle_id": cycle.cycle_id,
                    "station_id": target.station_id,
                    "target_at": target.target_at,
                    "horizon_minutes": target.horizon_minutes,
                    "target_order": target.target_order,
                    "raw_target": target.raw_target,
                }
                for target in cycle.targets
            ],
        )
        targets_response.raise_for_status()

    def targets_for_cycle(self, cycle_id: str) -> list[CycleTarget]:
        response = self._client.get(
            "/forecast_cycle_targets",
            params={
                "cycle_id": f"eq.{cycle_id}",
                "select": "station_id,target_at,horizon_minutes,target_order,raw_target",
                "order": "target_order.asc",
            },
        )
        response.raise_for_status()
        return [_parse_stored_target(row) for row in response.json()]


def sync_current_forecast_cycle(api: ForecastCycleApi, store: ForecastCycleStore) -> CycleSyncResult:
    api.validate_identity()
    cycle = api.current_cycle()
    if cycle is None:
        return CycleSyncResult(status="no_open_cycle")

    store.persist_cycle(cycle)
    persisted_targets = store.targets_for_cycle(cycle.cycle_id)
    _assert_targets_match(cycle.targets, persisted_targets)
    return CycleSyncResult(
        status="stored",
        cycle_id=cycle.cycle_id,
        expected_predictions=cycle.expected_predictions,
        targets_written=len(persisted_targets),
    )


def _parse_cycle(payload: Any) -> ForecastCycle:
    if not isinstance(payload, dict):
        raise ForecastCycleError("current cycle payload must be an object")
    raw_cycle = dict(payload)
    targets_payload = payload.get("targets")
    if not isinstance(targets_payload, list):
        raise ForecastCycleError("current cycle payload requires targets[]")

    data_cutoff = _normalize_timestamp(payload.get("data_cutoff"), "data_cutoff")
    closes_at = _normalize_timestamp(payload.get("closes_at"), "closes_at")
    cycle = ForecastCycle(
        cycle_id=_required_string(payload, "cycle_id"),
        data_cutoff=data_cutoff,
        closes_at=closes_at,
        expected_predictions=_required_positive_int(payload, "expected_predictions"),
        targets=[
            _parse_target(target, target_order=idx, data_cutoff=data_cutoff)
            for idx, target in enumerate(targets_payload)
        ],
        raw_cycle=raw_cycle,
    )
    if cycle.expected_predictions != len(cycle.targets):
        raise ForecastCycleError(
            f"expected_predictions={cycle.expected_predictions} does not match targets={len(cycle.targets)}"
        )
    return cycle


def _parse_target(payload: Any, *, target_order: int, data_cutoff: str) -> CycleTarget:
    if not isinstance(payload, dict):
        raise ForecastCycleError("target payload must be an object")
    station_id = _required_string(payload, "station_id")
    if not STATION_ID_RE.fullmatch(station_id):
        raise ForecastCycleError(f"invalid target station_id: {station_id!r}")
    target_at = _normalize_timestamp(_target_timestamp(payload), "target_at")
    horizon = payload.get("horizon_minutes")
    if horizon is None:
        horizon = _minutes_between(data_cutoff, target_at)
    if not isinstance(horizon, int) or horizon <= 0:
        raise ForecastCycleError(f"invalid horizon_minutes: {horizon!r}")
    return CycleTarget(
        station_id=station_id,
        target_at=target_at,
        horizon_minutes=horizon,
        target_order=target_order,
        raw_target=dict(payload),
    )


def _parse_stored_target(payload: dict[str, Any]) -> CycleTarget:
    target_order = payload.get("target_order")
    if not isinstance(target_order, int):
        raise ForecastCycleError(f"stored target has invalid target_order: {target_order!r}")
    return CycleTarget(
        station_id=_required_string(payload, "station_id"),
        target_at=_normalize_timestamp(payload.get("target_at"), "target_at"),
        horizon_minutes=_required_positive_int(payload, "horizon_minutes"),
        target_order=target_order,
        raw_target=payload.get("raw_target") if isinstance(payload.get("raw_target"), dict) else {},
    )


def _assert_targets_match(expected: list[CycleTarget], actual: list[CycleTarget]) -> None:
    expected_key = [(target.station_id, target.target_at) for target in expected]
    actual_key = [(target.station_id, target.target_at) for target in actual]
    if expected_key != actual_key:
        raise ForecastCycleError("persisted targets do not match server targets exactly")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ForecastCycleError(f"missing or invalid {key}")
    return value


def _required_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ForecastCycleError(f"missing or invalid {key}")
    return value


def _target_timestamp(payload: dict[str, Any]) -> Any:
    for key in ("target_at", "target_observed_at", "observed_at"):
        if key in payload:
            return payload[key]
    raise ForecastCycleError("target payload requires target_at")


def _normalize_timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ForecastCycleError(f"missing or invalid {field_name}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastCycleError(f"invalid {field_name}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ForecastCycleError(f"{field_name} must include timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _minutes_between(start: str, end: str) -> int:
    delta = _to_datetime(end) - _to_datetime(start)
    minutes = int(delta.total_seconds() // 60)
    if minutes <= 0 or delta.total_seconds() % 60:
        raise ForecastCycleError("target_at must be an exact positive minute horizon after data_cutoff")
    return minutes


def _to_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _error_code(payload: Any) -> str | None:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("code"), str):
            return detail["code"]
    return None


def main() -> None:
    with PulsoForecastCycleApi() as api:
        api.validate_identity()
        cycle = api.current_cycle()
        if cycle is None:
            result = CycleSyncResult(status="no_open_cycle")
        else:
            with SupabaseRestForecastCycleStore() as store:
                store.persist_cycle(cycle)
                persisted_targets = store.targets_for_cycle(cycle.cycle_id)
                _assert_targets_match(cycle.targets, persisted_targets)
            result = CycleSyncResult(
                status="stored",
                cycle_id=cycle.cycle_id,
                expected_predictions=cycle.expected_predictions,
                targets_written=len(persisted_targets),
            )
    print(
        "forecast cycle sync complete: "
        f"status={result.status} cycle_id={result.cycle_id} "
        f"expected_predictions={result.expected_predictions} targets_written={result.targets_written}"
    )


if __name__ == "__main__":
    main()
