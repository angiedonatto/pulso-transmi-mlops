from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from cycle_predictions import PredictionStore, generate_predictions_for_cycle
from forecast_cycle_sync import ForecastCycle, PulsoForecastCycleApi


class SubmissionError(RuntimeError):
    """Raised when an official submission cannot be built or sent safely."""


@dataclass(frozen=True)
class StoredPrediction:
    station_id: str
    target_at: str
    prediction: float


@dataclass(frozen=True)
class SubmissionResult:
    status: str
    cycle_id: str | None = None
    submission_id: str | None = None
    payload_hash: str | None = None
    idempotency_key: str | None = None


class SubmissionStore(Protocol):
    def load_active_champion(self) -> dict[str, Any]:
        ...

    def predictions_for_cycle(self, cycle_id: str, model_id: str) -> list[StoredPrediction]:
        ...

    def official_receipt_exists(self, cycle_id: str) -> bool:
        ...

    def persist_receipt(self, receipt: dict[str, Any]) -> None:
        ...


class SubmissionApi(Protocol):
    def validate_identity(self) -> dict[str, Any]:
        ...

    def current_cycle(self) -> ForecastCycle | None:
        ...

    def post_submission(self, payload: dict[str, Any], *, idempotency_key: str) -> httpx.Response:
        ...


class PulsoSubmissionApi(PulsoForecastCycleApi):
    def post_submission(self, payload: dict[str, Any], *, idempotency_key: str) -> httpx.Response:
        return self._client.post("/v1/submissions", json=payload, headers={"Idempotency-Key": idempotency_key})


class SupabaseSubmissionStore:
    def __init__(self, supabase_url: str | None = None, service_key: str | None = None, timeout: float = 30.0) -> None:
        resolved_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        resolved_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not resolved_url or not resolved_key:
            raise SubmissionError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
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

    def __enter__(self) -> "SupabaseSubmissionStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def load_active_champion(self) -> dict[str, Any]:
        response = self._client.get(
            "/model_versions",
            params={
                "is_active": "eq.true",
                "select": "*,pipeline_runs(git_commit,started_at)",
            },
        )
        response.raise_for_status()
        rows = response.json()
        if len(rows) != 1:
            raise SubmissionError(f"expected exactly one active champion, found {len(rows)}")
        return rows[0]

    def predictions_for_cycle(self, cycle_id: str, model_id: str) -> list[StoredPrediction]:
        response = self._client.get(
            "/predictions",
            params={
                "cycle_id": f"eq.{cycle_id}",
                "model_id": f"eq.{model_id}",
                "select": "station_id,target_observed_at,prediction",
                "order": "station_id.asc,target_observed_at.asc",
            },
        )
        response.raise_for_status()
        return [
            StoredPrediction(
                station_id=row["station_id"],
                target_at=_normalize_timestamp(row["target_observed_at"]),
                prediction=float(row["prediction"]),
            )
            for row in response.json()
        ]

    def official_receipt_exists(self, cycle_id: str) -> bool:
        response = self._client.get(
            "/submission_receipts",
            params={
                "cycle_id": f"eq.{cycle_id}",
                "is_official": "eq.true",
                "select": "receipt_id",
                "limit": "1",
            },
        )
        response.raise_for_status()
        return len(response.json()) > 0

    def persist_receipt(self, receipt: dict[str, Any]) -> None:
        response = self._client.post("/submission_receipts", json=receipt)
        response.raise_for_status()


def submit_current_cycle(api: SubmissionApi, store: SubmissionStore, *, max_attempts: int = 3) -> SubmissionResult:
    api.validate_identity()
    cycle = api.current_cycle()
    if cycle is None:
        return SubmissionResult(status="no_open_cycle")
    return submit_for_cycle(cycle, api, store, max_attempts=max_attempts)


def run_official_cycle(
    api: SubmissionApi,
    prediction_store: PredictionStore,
    submission_store: SubmissionStore,
    *,
    max_attempts: int = 3,
) -> SubmissionResult:
    """Sync the current cycle, generate its predictions, and submit them officially.

    Skips generation and submission entirely if an official receipt already
    exists for the open cycle, so re-runs within the same cycle never repost.
    """
    api.validate_identity()
    cycle = api.current_cycle()
    if cycle is None:
        return SubmissionResult(status="no_open_cycle")
    if submission_store.official_receipt_exists(cycle.cycle_id):
        return SubmissionResult(status="already_official", cycle_id=cycle.cycle_id)
    generate_predictions_for_cycle(cycle, prediction_store)
    return submit_for_cycle(cycle, api, submission_store, max_attempts=max_attempts)


def submit_for_cycle(
    cycle: ForecastCycle, api: SubmissionApi, store: SubmissionStore, *, max_attempts: int = 3
) -> SubmissionResult:
    model = store.load_active_champion()
    predictions = store.predictions_for_cycle(cycle.cycle_id, model["model_id"])
    payload = build_submission_payload(cycle, model, predictions)
    payload_hash = stable_payload_hash(payload)
    idempotency_key = stable_idempotency_key(cycle.cycle_id, model["model_id"], payload_hash)

    last_response: httpx.Response | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = api.post_submission(payload, idempotency_key=idempotency_key)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == max_attempts:
                store.persist_receipt(_receipt(cycle, model, payload_hash, idempotency_key, None, "failed", {"error": str(exc)}))
                raise SubmissionError("submission transport failed") from exc
            time.sleep(2 ** (attempt - 1))
            continue

        last_response = response
        if response.status_code in (200, 201):
            body = _json_body(response)
            status = "accepted" if response.status_code == 201 else "idempotent_replay"
            store.persist_receipt(_receipt(cycle, model, payload_hash, idempotency_key, attempt, status, body))
            return SubmissionResult(
                status=status,
                cycle_id=cycle.cycle_id,
                submission_id=_submission_id(body),
                payload_hash=payload_hash,
                idempotency_key=idempotency_key,
            )
        if response.status_code == 404 and _error_code(response) == "no_open_cycle":
            return SubmissionResult(status="no_open_cycle", cycle_id=cycle.cycle_id, payload_hash=payload_hash, idempotency_key=idempotency_key)
        if response.status_code in (409, 422):
            body = _json_body(response)
            store.persist_receipt(_receipt(cycle, model, payload_hash, idempotency_key, None, "rejected", body))
            raise SubmissionError(f"submission rejected without blind retry: HTTP {response.status_code}")
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))
                continue
            body = _json_body(response)
            store.persist_receipt(_receipt(cycle, model, payload_hash, idempotency_key, None, "failed", body))
            raise SubmissionError(f"submission failed after retries: HTTP {response.status_code}")

        body = _json_body(response)
        store.persist_receipt(_receipt(cycle, model, payload_hash, idempotency_key, None, "failed", body))
        raise SubmissionError(f"unexpected submission response: HTTP {response.status_code}")

    raise SubmissionError(f"submission failed after retries: HTTP {last_response.status_code if last_response else 'timeout'}")


def build_submission_payload(
    cycle: ForecastCycle,
    model: dict[str, Any],
    predictions: list[StoredPrediction],
) -> dict[str, Any]:
    _validate_training_cutoff(cycle, model)
    _validate_target_set(cycle, predictions)
    model_trace = _model_trace(model)
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle.cycle_id,
        "client_run_id": stable_client_run_id(cycle.cycle_id, model["model_id"]),
        "data_cutoff": cycle.data_cutoff,
        "model": model_trace,
        "predictions": [
            {"station_id": prediction.station_id, "target_at": prediction.target_at, "value": prediction.prediction}
            for prediction in predictions
        ],
    }
    return payload


def stable_payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_idempotency_key(cycle_id: str, model_id: str, payload_hash: str) -> str:
    return f"pulso:{cycle_id}:{model_id}:{payload_hash[:24]}"[:128]


def stable_client_run_id(cycle_id: str, model_id: str) -> str:
    digest = hashlib.sha256(f"{cycle_id}:{model_id}".encode()).hexdigest()[:16]
    return f"predict-{cycle_id}-{digest}"[:128]


def _validate_training_cutoff(cycle: ForecastCycle, model: dict[str, Any]) -> None:
    trained_until = _normalize_timestamp(model.get("trained_until"))
    if _to_dt(trained_until) > _to_dt(cycle.data_cutoff):
        raise SubmissionError("training_data_end is after cycle data_cutoff")


def _validate_target_set(cycle: ForecastCycle, predictions: list[StoredPrediction]) -> None:
    expected = [(target.station_id, target.target_at) for target in cycle.targets]
    actual = [(prediction.station_id, prediction.target_at) for prediction in predictions]
    if len(predictions) != cycle.expected_predictions:
        raise SubmissionError("prediction count does not match expected_predictions")
    if len(set(actual)) != len(actual):
        raise SubmissionError("duplicate prediction targets")
    if actual != expected:
        raise SubmissionError("prediction targets do not match cycle targets exactly")
    for prediction in predictions:
        if prediction.prediction < 0:
            raise SubmissionError("prediction value is negative")


def _model_trace(model: dict[str, Any]) -> dict[str, Any]:
    pipeline_run = model.get("pipeline_runs") if isinstance(model.get("pipeline_runs"), dict) else {}
    trained_until = _normalize_timestamp(model.get("trained_until"))
    trained_at = model.get("created_at") or pipeline_run.get("started_at") or trained_until
    trace = {
        "version": str(model["model_id"]),
        "trained_at": _normalize_timestamp(trained_at),
        "training_data_end": trained_until,
        "git_commit": model.get("git_commit") or pipeline_run.get("git_commit"),
    }
    if trace["git_commit"] is None:
        trace.pop("git_commit")
    return trace


def _receipt(
    cycle: ForecastCycle,
    model: dict[str, Any],
    payload_hash: str,
    idempotency_key: str,
    attempt: int | None,
    status: str,
    receipt_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "submission_id": _submission_id(receipt_payload),
        "cycle_id": cycle.cycle_id,
        "model_id": model["model_id"],
        "payload_hash": payload_hash,
        "idempotency_key": idempotency_key,
        "attempt": attempt,
        "accepted_at": _accepted_at(receipt_payload, status),
        "client_run_id": model.get("run_id"),
        "status": status,
        "is_official": bool(receipt_payload.get("is_official", False)),
        "receipt_payload": receipt_payload,
    }


def _accepted_at(payload: dict[str, Any], status: str) -> str | None:
    if status not in ("accepted", "idempotent_replay"):
        return None
    return _normalize_timestamp(payload.get("accepted_at") or payload.get("created_at") or _now_iso())


def _submission_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("submission_id") or payload.get("id")
    return value if isinstance(value, str) and value else None


def _json_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {"raw": response.text}
    return body if isinstance(body, dict) else {"data": body}


def _error_code(response: httpx.Response) -> str | None:
    body = _json_body(response)
    detail = body.get("detail")
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return detail["code"]
    return None


def _normalize_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionError("invalid timestamp")
    return _to_dt(value).isoformat().replace("+00:00", "Z")


def _to_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SubmissionError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
