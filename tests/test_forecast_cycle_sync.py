from __future__ import annotations

import httpx
import pytest

from forecast_cycle_sync import (
    CycleTarget,
    ForecastCycle,
    ForecastCycleError,
    PulsoForecastCycleApi,
    sync_current_forecast_cycle,
)


OPEN_CYCLE_PAYLOAD = {
    "cycle_id": "cycle-2026-09-23T03:45Z",
    "data_cutoff": "2026-09-10T16:30:00Z",
    "closes_at": "2026-09-23T04:00:00Z",
    "expected_predictions": 2,
    "targets": [
        {"station_id": "03000", "target_at": "2026-09-10T16:45:00Z"},
        {"station_id": "02300", "target_at": "2026-09-10T17:00:00Z", "horizon_minutes": 30},
    ],
}


class MemoryCycleStore:
    def __init__(self, mismatch: bool = False) -> None:
        self.cycles: dict[str, ForecastCycle] = {}
        self.targets: dict[str, list[CycleTarget]] = {}
        self.persist_calls = 0
        self.mismatch = mismatch

    def persist_cycle(self, cycle: ForecastCycle) -> None:
        self.persist_calls += 1
        self.cycles[cycle.cycle_id] = cycle
        self.targets[cycle.cycle_id] = list(cycle.targets)

    def targets_for_cycle(self, cycle_id: str) -> list[CycleTarget]:
        targets = list(self.targets[cycle_id])
        if self.mismatch:
            return targets[:-1]
        return targets


def api_with(handler) -> PulsoForecastCycleApi:
    return PulsoForecastCycleApi(base_url="https://example.test", api_key="test-key", transport=httpx.MockTransport(handler))


def test_open_cycle_is_persisted_and_validated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/me":
            return httpx.Response(200, json={"team_id": "team-1"})
        if request.url.path == "/v1/forecast-cycles/current":
            return httpx.Response(200, json=OPEN_CYCLE_PAYLOAD)
        return httpx.Response(404)

    store = MemoryCycleStore()
    with api_with(handler) as api:
        result = sync_current_forecast_cycle(api, store)

    assert result.status == "stored"
    assert result.cycle_id == OPEN_CYCLE_PAYLOAD["cycle_id"]
    assert result.expected_predictions == 2
    assert result.targets_written == 2
    assert store.cycles[OPEN_CYCLE_PAYLOAD["cycle_id"]].raw_cycle == OPEN_CYCLE_PAYLOAD


def test_no_open_cycle_404_finishes_successfully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/me":
            return httpx.Response(200, json={"team_id": "team-1"})
        if request.url.path == "/v1/forecast-cycles/current":
            return httpx.Response(404, json={"detail": {"code": "no_open_cycle", "message": "none"}})
        return httpx.Response(404)

    store = MemoryCycleStore()
    with api_with(handler) as api:
        result = sync_current_forecast_cycle(api, store)

    assert result.status == "no_open_cycle"
    assert store.persist_calls == 0


def test_rerun_is_idempotent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/me":
            return httpx.Response(200, json={"team_id": "team-1"})
        if request.url.path == "/v1/forecast-cycles/current":
            return httpx.Response(200, json=OPEN_CYCLE_PAYLOAD)
        return httpx.Response(404)

    store = MemoryCycleStore()
    with api_with(handler) as api:
        first = sync_current_forecast_cycle(api, store)
        second = sync_current_forecast_cycle(api, store)

    assert first == second
    assert len(store.cycles) == 1
    assert len(store.targets[OPEN_CYCLE_PAYLOAD["cycle_id"]]) == 2


def test_target_mismatch_fails_after_persistence_check() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/me":
            return httpx.Response(200, json={"team_id": "team-1"})
        if request.url.path == "/v1/forecast-cycles/current":
            return httpx.Response(200, json=OPEN_CYCLE_PAYLOAD)
        return httpx.Response(404)

    store = MemoryCycleStore(mismatch=True)
    with api_with(handler) as api:
        with pytest.raises(ForecastCycleError, match="persisted targets"):
            sync_current_forecast_cycle(api, store)
