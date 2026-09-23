from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx
import pandas as pd

from champion_baseline import ChampionBaselineError, MODEL_TYPE, SupabaseBaselineStore
from forecast_cycle_sync import ForecastCycle, PulsoForecastCycleApi
from stream_sync import _time_slot_payload


class CyclePredictionError(RuntimeError):
    """Raised when cycle predictions cannot be generated safely."""


@dataclass(frozen=True)
class CyclePrediction:
    cycle_id: str
    model_id: str
    station_id: str
    target_at: str
    horizon_minutes: int
    prediction: float


@dataclass(frozen=True)
class CyclePredictionResult:
    status: str
    cycle_id: str | None = None
    data_cutoff: str | None = None
    model_id: str | None = None
    expected_predictions: int = 0
    predictions_generated: int = 0
    min_prediction: float | None = None
    max_prediction: float | None = None
    target_validation: str = "not_run"


class PredictionStore(Protocol):
    def load_active_champion(self) -> dict[str, Any]:
        ...

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        ...

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        ...

    def persist_predictions(self, cycle: ForecastCycle, model: dict[str, Any], predictions: list[CyclePrediction]) -> int:
        ...


def generate_cycle_predictions(api: PulsoForecastCycleApi, store: PredictionStore) -> CyclePredictionResult:
    api.validate_identity()
    cycle = api.current_cycle()
    if cycle is None:
        return CyclePredictionResult(status="no_open_cycle")

    model = store.load_active_champion()
    artifact = _load_artifact(model)
    observations = store.observations_until(cycle.data_cutoff)
    predictions = build_predictions(cycle, model["model_id"], artifact, observations)
    validate_predictions(cycle, predictions)
    store.persist_cycle_targets(cycle)
    written = store.persist_predictions(cycle, model, predictions)
    values = [prediction.prediction for prediction in predictions]
    return CyclePredictionResult(
        status="predicted",
        cycle_id=cycle.cycle_id,
        data_cutoff=cycle.data_cutoff,
        model_id=model["model_id"],
        expected_predictions=cycle.expected_predictions,
        predictions_generated=written,
        min_prediction=min(values) if values else None,
        max_prediction=max(values) if values else None,
        target_validation="exact",
    )


def build_predictions(
    cycle: ForecastCycle,
    model_id: str,
    artifact: dict[str, Any],
    observations: pd.DataFrame,
) -> list[CyclePrediction]:
    if artifact.get("model_type") != MODEL_TYPE:
        raise CyclePredictionError(f"unsupported champion model_type: {artifact.get('model_type')!r}")
    frame = _prepare_observations(observations)
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    if (frame["observed_at"] > cutoff).any():
        raise CyclePredictionError("observations contain rows after data_cutoff")
    lag_lookup = {
        (str(row.station_id), _ts(row.observed_at)): float(row.demand)
        for row in frame.itertuples(index=False)
    }
    slot_means = artifact.get("station_slot_means")
    if not isinstance(slot_means, dict):
        raise CyclePredictionError("champion artifact missing station_slot_means")

    predictions: list[CyclePrediction] = []
    for target in cycle.targets:
        target_at = pd.Timestamp(target.target_at).tz_convert("UTC")
        lag_at = _ts(target_at - pd.to_timedelta(1, unit="D"))
        lag_value = lag_lookup.get((target.station_id, lag_at))
        quarter_hour_block = _quarter_hour_block(target_at)
        fallback = slot_means.get(f"{target.station_id}|{quarter_hour_block}")
        value = lag_value if lag_value is not None else fallback
        if value is None:
            raise CyclePredictionError(f"missing lag and fallback for {target.station_id} {target.target_at}")
        predictions.append(
            CyclePrediction(
                cycle_id=cycle.cycle_id,
                model_id=model_id,
                station_id=target.station_id,
                target_at=target.target_at,
                horizon_minutes=target.horizon_minutes,
                prediction=float(value),
            )
        )
    return predictions


def validate_predictions(cycle: ForecastCycle, predictions: list[CyclePrediction]) -> None:
    expected_pairs = [(target.station_id, target.target_at) for target in cycle.targets]
    actual_pairs = [(prediction.station_id, prediction.target_at) for prediction in predictions]
    if len(predictions) != cycle.expected_predictions:
        raise CyclePredictionError("prediction count does not match expected_predictions")
    if len(set(actual_pairs)) != len(actual_pairs):
        raise CyclePredictionError("duplicate prediction targets")
    if actual_pairs != expected_pairs:
        raise CyclePredictionError("prediction targets do not match cycle targets exactly")
    for prediction in predictions:
        if not math.isfinite(prediction.prediction):
            raise CyclePredictionError("prediction value is not finite")
        if prediction.prediction < 0:
            raise CyclePredictionError("prediction value is negative")


class SupabaseCyclePredictionStore(SupabaseBaselineStore):
    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        frame = self.observations()
        parsed_cutoff = pd.Timestamp(data_cutoff).tz_convert("UTC")
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        return frame[frame["observed_at"] <= parsed_cutoff].copy()

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        now = _now_iso()
        self._post_upsert(
            "/forecast_cycles",
            {
                "cycle_id": cycle.cycle_id,
                "data_cutoff": cycle.data_cutoff,
                "closes_at": cycle.closes_at,
                "expected_predictions": cycle.expected_predictions,
                "raw_cycle": cycle.raw_cycle,
                "last_seen_at": now,
            },
            on_conflict="cycle_id",
        )
        if cycle.targets:
            self._post_upsert(
                "/forecast_cycle_targets",
                [
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
                on_conflict="cycle_id,station_id,target_at",
            )
            self._post_upsert(
                "/time_slots",
                [_time_slot_payload(target.target_at) for target in cycle.targets],
                on_conflict="observed_at",
            )

    def persist_predictions(self, cycle: ForecastCycle, model: dict[str, Any], predictions: list[CyclePrediction]) -> int:
        model_id = model["model_id"]
        run_id = model["run_id"]
        delete_response = self._client.delete(
            "/predictions",
            params={"cycle_id": f"eq.{cycle.cycle_id}", "model_id": f"eq.{model_id}"},
        )
        delete_response.raise_for_status()
        if not predictions:
            return 0
        response = self._client.post(
            "/predictions",
            json=[
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "cycle_id": prediction.cycle_id,
                    "station_id": prediction.station_id,
                    "target_observed_at": prediction.target_at,
                    "horizon_minutes": prediction.horizon_minutes,
                    "prediction": prediction.prediction,
                }
                for prediction in predictions
            ],
        )
        response.raise_for_status()
        return len(predictions)

    def _post_upsert(self, path: str, payload: Any, *, on_conflict: str) -> None:
        response = self._client.post(
            path,
            params={"on_conflict": on_conflict},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=payload,
        )
        response.raise_for_status()


def _load_artifact(model: dict[str, Any]) -> dict[str, Any]:
    active = model.get("is_active")
    if active is not True:
        raise CyclePredictionError("loaded model is not active")
    artifact_uri = model.get("artifact_uri")
    if not isinstance(artifact_uri, str) or not artifact_uri:
        raise CyclePredictionError("active champion missing artifact_uri")
    try:
        return json.loads(Path(artifact_uri).read_text())
    except OSError as exc:
        raise CyclePredictionError(f"cannot read champion artifact: {artifact_uri}") from exc
    except json.JSONDecodeError as exc:
        raise CyclePredictionError(f"invalid champion artifact JSON: {artifact_uri}") from exc


def _prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "observed_at", "demand"}
    missing = required - set(observations.columns)
    if missing:
        raise CyclePredictionError(f"missing observation columns: {sorted(missing)}")
    frame = observations.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = frame["demand"].astype(float)
    return frame


def _quarter_hour_block(timestamp: pd.Timestamp) -> int:
    local = timestamp.tz_convert("America/Bogota")
    return int(local.hour * 4 + local.minute // 15)


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    with PulsoForecastCycleApi() as api, SupabaseCyclePredictionStore() as store:
        result = generate_cycle_predictions(api, store)
    print(
        json.dumps(
            {
                "cycle_status": result.status,
                "cycle_id": result.cycle_id,
                "data_cutoff": result.data_cutoff,
                "model_id": result.model_id,
                "expected_predictions": result.expected_predictions,
                "predictions_generated": result.predictions_generated,
                "min_prediction": result.min_prediction,
                "max_prediction": result.max_prediction,
                "target_validation": result.target_validation,
                "submission_post": "not_executed",
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
