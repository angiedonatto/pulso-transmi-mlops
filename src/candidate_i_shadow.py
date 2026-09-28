from __future__ import annotations

import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Protocol
from uuid import uuid4

import pandas as pd

import candidate_i_snr_gate as ci
from candidate_b_shadow import ShadowCandidateError, SupabaseShadowStore, _ts
from cycle_predictions import CyclePrediction, validate_predictions
from forecast_cycle_sync import CycleTarget, ForecastCycle


# Candidate I shadow mode.
#
# Candidate G stays the only active model and the only official submission (Predict
# workflow). The shadow runs afterwards, in its own process, and:
# - predicts exactly the cycle G predicted: the cycle, targets and data_cutoff G persisted
#   in forecast_cycles / forecast_cycle_targets, only once G's predictions cover them;
# - writes only Candidate I rows (predictions / prediction_traces under its own model_id);
# - never calls the submission API, never writes submission_receipts, never changes
#   model_versions.is_active;
# - is best-effort: `run_candidate_i_shadow` returns a result instead of raising, and a hard
#   time budget stops a slow run before anything is persisted.
DEFAULT_TIME_BUDGET_S = 300.0
FAILURE_STATUSES = {"failed", "timeout"}


class ShadowTimeout(BaseException):
    """BaseException on purpose: the model's own `except Exception` fail-safes must not
    swallow the deadline and let a half-computed cycle be persisted."""


class ShadowGuardError(RuntimeError):
    """A shadow-mode invariant was violated (e.g. the shadow model is active)."""


@dataclass(frozen=True)
class ShadowRunResult:
    status: str
    model_name: str = ci.MODEL_NAME
    cycle_id: str | None = None
    data_cutoff: str | None = None
    model_id: str | None = None
    expected_predictions: int = 0
    predictions_generated: int = 0
    target_validation: str = "not_run"
    fallback_stations: int = 0
    reason: str | None = None
    station_factors: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateIShadowStore(Protocol):
    def candidate_i_model_version(self, version: str = ci.MODEL_VERSION) -> dict[str, Any] | None:
        ...

    def champion_cycle(self, cycle_id: str | None = None) -> tuple[ForecastCycle, list[tuple[str, str, int]]] | None:
        ...

    def shadow_predictions_complete(self, cycle: ForecastCycle, model: dict[str, Any]) -> bool:
        ...

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        ...

    def persist_shadow_predictions(self, cycle: ForecastCycle, model: dict[str, Any], predictions: list[CyclePrediction]) -> int:
        ...

    def persist_candidate_i_traces(self, cycle_id: str, model_id: str, rows: list[dict[str, Any]]) -> None:
        ...


def run_candidate_i_shadow(
    store: CandidateIShadowStore,
    *,
    cycle_id: str | None = None,
    time_budget_s: float | None = DEFAULT_TIME_BUDGET_S,
    log: Callable[[str], None] | None = None,
) -> ShadowRunResult:
    """Best-effort shadow inference for the champion's latest (or given) cycle. Never raises
    an Exception; failures come back as status "failed" / "timeout" with a reason."""
    log = log or _log_stderr
    started = time.perf_counter()
    timings: dict[str, float] = {}
    context: dict[str, Any] = {}
    try:
        with _deadline(time_budget_s):
            outcome = _predict_shadow_cycle(store, cycle_id, timings, context)
        if isinstance(outcome, ShadowRunResult):
            result = outcome
        else:
            cycle, model, predictions, items = outcome
            step = time.perf_counter()
            written = store.persist_shadow_predictions(cycle, model, predictions)
            trace_error = _persist_traces_best_effort(store, cycle, model, predictions, items)
            timings["persist"] = time.perf_counter() - step
            fallbacks = {item.prediction.station_id for item in items if item.signal.fallback_reason}
            result = ShadowRunResult(
                status="shadow_predicted",
                cycle_id=cycle.cycle_id,
                data_cutoff=cycle.data_cutoff,
                model_id=str(model["model_id"]),
                expected_predictions=cycle.expected_predictions,
                predictions_generated=written,
                target_validation="exact",
                fallback_stations=len(fallbacks),
                reason=trace_error,
                station_factors=_station_factors(items),
            )
            if fallbacks:
                log(f"candidate I adaptive layer fell back to factor 1.0 (Candidate F) for {len(fallbacks)} stations")
    except ShadowTimeout as exc:
        result = ShadowRunResult(status="timeout", reason=str(exc), **context)
    except Exception as exc:  # noqa: BLE001 - best-effort by contract
        result = ShadowRunResult(status="failed", reason=f"{type(exc).__name__}: {exc}", **context)
    timings["total"] = time.perf_counter() - started
    result = replace(result, timings_s={key: round(value, 3) for key, value in timings.items()})
    if result.status in FAILURE_STATUSES:
        log(f"candidate I shadow {result.status}: {result.reason} (the official Candidate G path is unaffected)")
    return result


def _predict_shadow_cycle(
    store: CandidateIShadowStore,
    cycle_id: str | None,
    timings: dict[str, float],
    context: dict[str, Any],
) -> ShadowRunResult | tuple[ForecastCycle, dict[str, Any], list[CyclePrediction], list[ci.CandidateIPrediction]]:
    step = time.perf_counter()
    model = store.candidate_i_model_version()
    if model is None:
        return ShadowRunResult(status="skipped", reason="candidate I is not registered in model_versions")
    context["model_id"] = str(model["model_id"])
    if model.get("is_active") is not False:
        raise ShadowGuardError("candidate I must stay is_active=false in shadow mode")
    artifact = ci.load_candidate_i_artifact(model["artifact_uri"])
    registered_version = (model.get("feature_config") or {}).get("version")
    if registered_version != artifact.metadata["version"]:
        raise ShadowGuardError(f"model_versions version {registered_version!r} != artifact version {artifact.metadata['version']!r}")
    found = store.champion_cycle(cycle_id)
    if found is None:
        return ShadowRunResult(status="skipped", reason="no champion cycle with predictions to shadow", model_id=context["model_id"])
    cycle, champion_targets = found
    context.update(cycle_id=cycle.cycle_id, data_cutoff=cycle.data_cutoff, expected_predictions=cycle.expected_predictions)
    expected = sorted((target.station_id, target.target_at, int(target.horizon_minutes)) for target in cycle.targets)
    if len(cycle.targets) != cycle.expected_predictions or sorted(champion_targets) != expected:
        return ShadowRunResult(status="skipped", reason="champion predictions do not cover the cycle targets exactly", **context)
    timings["resolve_cycle"] = time.perf_counter() - step
    if store.shadow_predictions_complete(cycle, model):
        return ShadowRunResult(
            status="already_shadow_predicted",
            predictions_generated=cycle.expected_predictions,
            target_validation="exact",
            **context,
        )
    step = time.perf_counter()
    observations = store.observations_until(cycle.data_cutoff)
    timings["load_observations"] = time.perf_counter() - step
    step = time.perf_counter()
    items = ci.build_candidate_i_predictions(cycle, str(model["model_id"]), artifact, observations)
    timings["predict"] = time.perf_counter() - step
    predictions = [
        CyclePrediction(
            cycle_id=item.prediction.cycle_id,
            model_id=item.prediction.model_id,
            station_id=item.prediction.station_id,
            target_at=item.prediction.target_at,
            horizon_minutes=item.prediction.horizon_minutes,
            prediction=item.prediction.prediction,
            trace=ci.candidate_i_trace(item),
        )
        for item in items
    ]
    validate_predictions(cycle, predictions)
    return cycle, model, predictions, items


def candidate_i_trace_payloads(predictions: list[CyclePrediction], items: list[ci.CandidateIPrediction]) -> list[dict[str, Any]]:
    """prediction_traces rows (existing columns). feedback_samples = resolved history slots
    behind the factor."""
    return [
        {
            "cycle_id": prediction.cycle_id,
            "model_id": prediction.model_id,
            "station_id": prediction.station_id,
            "target_at": prediction.target_at,
            "horizon_minutes": prediction.horizon_minutes,
            "base_prediction": item.base_prediction,
            "calibration_factor": item.signal.factor,
            "final_prediction": prediction.prediction,
            "feedback_samples": item.signal.history_slots,
            "data_cutoff": item.data_cutoff,
        }
        for prediction, item in zip(predictions, items, strict=True)
    ]


def candidate_i_model_version_payload(artifact: ci.CandidateIArtifact) -> dict[str, Any]:
    """model_versions row for the shadow registration. is_active is always False."""
    ci.validate_candidate_i_artifact(artifact.metadata)
    metadata = artifact.metadata
    return {
        "model_name": ci.MODEL_NAME,
        "model_type": ci.MODEL_TYPE,
        "artifact_uri": metadata["artifact_uri"],
        "trained_until": ci.TRAINED_UNTIL,
        "feature_config": {
            "mode": "shadow",
            "version": metadata["version"],
            "artifact_storage": "repo",
            "base_model": metadata["base_model"],
            "history": metadata["history"],
            "data_contract": metadata["data_contract"],
            "artifact_git_commit": metadata.get("git_commit"),
        },
        "params": {
            "adaptive_layer": metadata["adaptive_layer"],
            "base_hyperparameters": metadata["base_model"]["hyperparameters"],
            "trained_until_meaning": "parameters frozen on selection cycles <= trained_until; F base and history refit from observations <= data_cutoff every cycle",
            "selection": {
                "windows": "stable 2026-09-03T00:00Z..2026-09-10T17:00Z, drift_dev 2026-09-10T18:00Z..2026-09-13T11:00Z",
                "sealed_test": "2026-09-13T12:00Z..2026-09-15T03:00Z, +0.420 pp vs G, P(I>G)=0.995",
                "report": "reports/candidate_i/summary.md",
            },
        },
        "is_active": False,
    }


class SupabaseCandidateIShadowStore(SupabaseShadowStore):
    """Supabase access for the Candidate I shadow. Reads the champion's persisted cycle and
    writes only Candidate I rows (predictions / prediction_traces). No submission tables."""

    def candidate_i_model_version(self, version: str = ci.MODEL_VERSION) -> dict[str, Any] | None:
        rows = self._get(
            "/model_versions",
            {"model_type": f"eq.{ci.MODEL_TYPE}", "select": "*,pipeline_runs(started_at,git_commit)", "order": "trained_until.desc"},
        )
        for row in rows:
            feature_config = row.get("feature_config") if isinstance(row.get("feature_config"), dict) else {}
            if feature_config.get("version") == version:
                return row
        return None

    def champion_cycle(self, cycle_id: str | None = None) -> tuple[ForecastCycle, list[tuple[str, str, int]]] | None:
        champions = self._get("/model_versions", {"is_active": "eq.true", "select": "model_id"})
        if len(champions) != 1:
            raise ShadowCandidateError(f"expected exactly one active champion, found {len(champions)}")
        champion_id = champions[0]["model_id"]
        if cycle_id is None:
            latest = self._get(
                "/predictions",
                {"model_id": f"eq.{champion_id}", "select": "cycle_id,created_at", "order": "created_at.desc", "limit": "1"},
            )
            if not latest or not latest[0].get("cycle_id"):
                return None
            cycle_id = str(latest[0]["cycle_id"])
        cycles = self._get("/forecast_cycles", {"cycle_id": f"eq.{cycle_id}", "select": "cycle_id,data_cutoff,closes_at,expected_predictions", "limit": "1"})
        if not cycles:
            return None
        targets = self._get(
            "/forecast_cycle_targets",
            {"cycle_id": f"eq.{cycle_id}", "select": "station_id,target_at,horizon_minutes,target_order", "order": "target_order.asc"},
        )
        champion_rows = self._get(
            "/predictions",
            {"cycle_id": f"eq.{cycle_id}", "model_id": f"eq.{champion_id}", "select": "station_id,target_observed_at,horizon_minutes"},
        )
        row = cycles[0]
        cycle = ForecastCycle(
            cycle_id=str(row["cycle_id"]),
            data_cutoff=_ts(row["data_cutoff"]),
            closes_at=_ts(row["closes_at"]) if row.get("closes_at") else "",
            expected_predictions=int(row["expected_predictions"]),
            targets=[
                CycleTarget(
                    station_id=str(target["station_id"]),
                    target_at=_ts(target["target_at"]),
                    horizon_minutes=int(target["horizon_minutes"]),
                    target_order=int(target["target_order"]),
                    raw_target={},
                )
                for target in targets
            ],
            raw_cycle={},
        )
        champion_targets = [(str(item["station_id"]), _ts(item["target_observed_at"]), int(item["horizon_minutes"])) for item in champion_rows]
        return cycle, champion_targets

    def persist_candidate_i_traces(self, cycle_id: str, model_id: str, rows: list[dict[str, Any]]) -> None:
        delete_response = self._client.delete("/prediction_traces", params={"cycle_id": f"eq.{cycle_id}", "model_id": f"eq.{model_id}"})
        if delete_response.status_code == 404:
            return
        delete_response.raise_for_status()
        if not rows:
            return
        response = self._client.post(
            "/prediction_traces",
            params={"on_conflict": "cycle_id,model_id,station_id,target_at"},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=rows,
        )
        response.raise_for_status()

    def register_candidate_i_model_version(self, payload: dict[str, Any], *, git_commit: str | None) -> dict[str, Any]:
        if payload.get("is_active") is not False:
            raise ShadowGuardError("candidate I shadow registration must use is_active=false")
        existing = self.candidate_i_model_version(payload["feature_config"]["version"])
        if existing is not None:
            return {"status": "already_registered", "model_id": existing["model_id"], "run_id": existing["run_id"], "is_active": existing["is_active"]}
        run_id = str(uuid4())
        model_id = str(uuid4())
        self._post(
            "/pipeline_runs",
            {"run_id": run_id, "status": "running", "git_commit": git_commit, "data_cutoff": payload["trained_until"], "trigger_type": "shadow_registration"},
        )
        self._post("/model_versions", {"model_id": model_id, "run_id": run_id, **payload})
        self._patch("/pipeline_runs", {"status": "success", "finished_at": _now_iso()}, params={"run_id": f"eq.{run_id}"})
        return {"status": "registered", "model_id": model_id, "run_id": run_id, "is_active": False}

    def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()


def _persist_traces_best_effort(
    store: CandidateIShadowStore,
    cycle: ForecastCycle,
    model: dict[str, Any],
    predictions: list[CyclePrediction],
    items: list[ci.CandidateIPrediction],
) -> str | None:
    try:
        store.persist_candidate_i_traces(cycle.cycle_id, str(model["model_id"]), candidate_i_trace_payloads(predictions, items))
    except Exception as exc:  # noqa: BLE001 - traces are observability only
        return f"traces not persisted: {type(exc).__name__}: {exc}"
    return None


def _station_factors(items: list[ci.CandidateIPrediction]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        s = item.signal
        out.setdefault(
            item.prediction.station_id,
            {
                "factor": round(s.factor, 4),
                "gate": round(s.gate, 4),
                "z": round(s.z, 3),
                "history_slots": s.history_slots,
                "history_max_target_at": s.history_max_target_at,
                "fallback_reason": s.fallback_reason,
            },
        )
    return out


@contextmanager
def _deadline(seconds: float | None) -> Iterator[None]:
    usable = seconds is not None and seconds > 0 and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def _expire(_signum: int, _frame: Any) -> None:
        raise ShadowTimeout(f"candidate I shadow exceeded its {seconds:.0f}s budget; nothing was persisted")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _log_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
