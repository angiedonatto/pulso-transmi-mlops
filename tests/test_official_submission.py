from __future__ import annotations

import httpx
import pytest

from forecast_cycle_sync import CycleTarget, ForecastCycle
from official_submission import (
    StoredPrediction,
    SubmissionError,
    build_submission_payload,
    stable_idempotency_key,
    stable_payload_hash,
    submit_current_cycle,
)


def cycle() -> ForecastCycle:
    return ForecastCycle(
        cycle_id="cyc_test_1",
        data_cutoff="2026-01-02T00:00:00Z",
        closes_at="2026-01-02T00:10:00Z",
        expected_predictions=2,
        targets=[
            CycleTarget("03000", "2026-01-02T00:15:00Z", 15, 0, {}),
            CycleTarget("02300", "2026-01-02T00:30:00Z", 30, 1, {}),
        ],
        raw_cycle={},
    )


def model() -> dict:
    return {
        "model_id": "11111111-1111-4111-8111-111111111111",
        "run_id": "22222222-2222-4222-8222-222222222222",
        "trained_until": "2026-01-02T00:00:00Z",
        "pipeline_runs": {"git_commit": "abcdef1", "started_at": "2026-01-02T00:01:00Z"},
        "is_active": True,
    }


def predictions() -> list[StoredPrediction]:
    return [
        StoredPrediction("03000", "2026-01-02T00:15:00Z", 10.0),
        StoredPrediction("02300", "2026-01-02T00:30:00Z", 20.5),
    ]


class FakeApi:
    def __init__(self, responses: list[httpx.Response] | None = None, current_cycle: ForecastCycle | None = None) -> None:
        self.responses = responses or []
        self.current = current_cycle
        self.keys: list[str] = []

    def validate_identity(self):
        return {"ok": True}

    def current_cycle(self):
        return self.current

    def post_submission(self, payload, *, idempotency_key):
        self.keys.append(idempotency_key)
        response = self.responses.pop(0)
        response.request = httpx.Request("POST", "https://example.test/v1/submissions")
        return response


class FakeStore:
    def __init__(self) -> None:
        self.receipts: list[dict] = []

    def load_active_champion(self):
        return model()

    def predictions_for_cycle(self, cycle_id, model_id):
        return predictions()

    def persist_receipt(self, receipt):
        self.receipts.append(receipt)


def test_payload_contract_and_stable_key() -> None:
    payload = build_submission_payload(cycle(), model(), predictions())
    payload_hash = stable_payload_hash(payload)

    assert payload == {
        "schema_version": "1.0",
        "cycle_id": "cyc_test_1",
        "client_run_id": "predict-cyc_test_1-7c02567e4b767907",
        "data_cutoff": "2026-01-02T00:00:00Z",
        "model": {
            "version": "11111111-1111-4111-8111-111111111111",
            "trained_at": "2026-01-02T00:01:00Z",
            "training_data_end": "2026-01-02T00:00:00Z",
            "git_commit": "abcdef1",
        },
        "predictions": [
            {"station_id": "03000", "target_at": "2026-01-02T00:15:00Z", "value": 10.0},
            {"station_id": "02300", "target_at": "2026-01-02T00:30:00Z", "value": 20.5},
        ],
    }
    assert stable_idempotency_key("cyc_test_1", model()["model_id"], payload_hash) == stable_idempotency_key(
        "cyc_test_1", model()["model_id"], payload_hash
    )


def test_no_open_cycle_is_success() -> None:
    result = submit_current_cycle(FakeApi(current_cycle=None), FakeStore())

    assert result.status == "no_open_cycle"


def test_training_data_end_must_not_exceed_cutoff() -> None:
    bad_model = {**model(), "trained_until": "2026-01-02T00:15:00Z"}

    with pytest.raises(SubmissionError, match="training_data_end"):
        build_submission_payload(cycle(), bad_model, predictions())


def test_target_set_must_be_exact() -> None:
    with pytest.raises(SubmissionError, match="targets"):
        build_submission_payload(cycle(), model(), list(reversed(predictions())))


def test_success_persists_receipt() -> None:
    store = FakeStore()
    api = FakeApi(
        [httpx.Response(201, json={"submission_id": "sub_1", "accepted_at": "2026-01-02T00:05:00Z", "is_official": True})],
        cycle(),
    )

    result = submit_current_cycle(api, store)

    assert result.status == "accepted"
    assert result.submission_id == "sub_1"
    assert store.receipts[0]["attempt"] == 1
    assert store.receipts[0]["status"] == "accepted"
    assert store.receipts[0]["is_official"] is True


def test_retry_reuses_same_idempotency_key() -> None:
    store = FakeStore()
    api = FakeApi(
        [
            httpx.Response(500, json={"detail": "temporary"}),
            httpx.Response(200, json={"submission_id": "sub_1", "accepted_at": "2026-01-02T00:05:00Z"}),
        ],
        cycle(),
    )

    result = submit_current_cycle(api, store)

    assert result.status == "idempotent_replay"
    assert len(api.keys) == 2
    assert api.keys[0] == api.keys[1]


def test_409_does_not_retry_blindly() -> None:
    store = FakeStore()
    api = FakeApi([httpx.Response(409, json={"detail": {"code": "conflict"}})], cycle())

    with pytest.raises(SubmissionError, match="without blind retry"):
        submit_current_cycle(api, store)

    assert len(api.keys) == 1
    assert store.receipts[0]["status"] == "rejected"
