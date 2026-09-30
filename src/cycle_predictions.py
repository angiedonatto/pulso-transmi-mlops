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
    trace: dict[str, Any] | None = None


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
    return generate_predictions_for_cycle(cycle, store)


def generate_predictions_for_cycle(cycle: ForecastCycle, store: PredictionStore) -> CyclePredictionResult:
    model = store.load_active_champion()
    artifact = _load_artifact(model)
    observations = store.observations_until(cycle.data_cutoff)
    feedback_frame = _feedback_history_for_model(store, cycle, model, artifact)
    predictions = build_predictions(cycle, model["model_id"], artifact, observations, feedback_frame=feedback_frame)
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
    artifact: Any,
    observations: pd.DataFrame,
    *,
    feedback_frame: pd.DataFrame | None = None,
) -> list[CyclePrediction]:
    if not isinstance(artifact, dict):
        return _build_non_baseline_predictions(cycle, model_id, artifact, observations, feedback_frame=feedback_frame)
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
                trace=None,
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
                "/time_slots",
                _time_slot_payloads_for_targets(cycle),
                on_conflict="observed_at",
            )
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
        self._persist_prediction_traces(cycle.cycle_id, model_id, predictions)
        return len(predictions)

    def _persist_prediction_traces(self, cycle_id: str, model_id: str, predictions: list[CyclePrediction]) -> None:
        rows = prediction_trace_payloads(predictions)
        delete_response = self._client.delete(
            "/prediction_traces",
            params={"cycle_id": f"eq.{cycle_id}", "model_id": f"eq.{model_id}"},
        )
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

    def feedback_history_until(self, data_cutoff: str, model: dict[str, Any]) -> pd.DataFrame:
        try:
            from candidate_g_adaptive import build_candidate_f_base_frame
            from forecast_cycle_sync import CycleTarget, ForecastCycle
        except ImportError as exc:
            raise CyclePredictionError("candidate G feedback dependencies are unavailable") from exc

        cutoff = pd.Timestamp(data_cutoff).tz_convert("UTC")
        response = self._client.get(
            "/predictions",
            params={
                "model_id": f"eq.{model['model_id']}",
                "target_observed_at": f"lte.{_ts(cutoff)}",
                "select": "cycle_id,station_id,target_observed_at",
                "order": "target_observed_at.desc",
                "limit": "2000",
            },
        )
        response.raise_for_status()
        prediction_rows = response.json()
        if not prediction_rows:
            return pd.DataFrame(columns=["station_id", "target_at", "base_prediction", "actual"])

        observations = self.observations()
        observations["station_id"] = observations["station_id"].astype(str)
        observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
        observations = observations[observations["observed_at"] <= cutoff].copy()
        actual_lookup = {
            (str(row.station_id), _ts(row.observed_at)): float(row.demand)
            for row in observations[["station_id", "observed_at", "demand"]].itertuples(index=False)
        }
        wanted = {
            (str(row["cycle_id"]), str(row["station_id"]), _normalize_timestamp(row["target_observed_at"]))
            for row in prediction_rows
            if row.get("cycle_id")
        }
        by_cycle: dict[str, set[tuple[str, str]]] = {}
        for cycle_id, station_id, target_at in wanted:
            if (station_id, target_at) in actual_lookup:
                by_cycle.setdefault(cycle_id, set()).add((station_id, target_at))
        feedback_rows: list[dict[str, Any]] = []
        for cycle_id, pairs in by_cycle.items():
            cycle_response = self._client.get(
                "/forecast_cycles",
                params={"cycle_id": f"eq.{cycle_id}", "select": "cycle_id,data_cutoff,closes_at,expected_predictions", "limit": "1"},
            )
            cycle_response.raise_for_status()
            cycle_rows = cycle_response.json()
            if not cycle_rows:
                continue
            target_response = self._client.get(
                "/forecast_cycle_targets",
                params={
                    "cycle_id": f"eq.{cycle_id}",
                    "select": "station_id,target_at,horizon_minutes,target_order",
                    "order": "target_order.asc",
                },
            )
            target_response.raise_for_status()
            target_rows = target_response.json()
            cycle_row = cycle_rows[0]
            historical_cycle = ForecastCycle(
                cycle_id=str(cycle_id),
                data_cutoff=_normalize_timestamp(cycle_row["data_cutoff"]),
                closes_at=_normalize_timestamp(cycle_row["closes_at"]) if cycle_row.get("closes_at") else "",
                expected_predictions=int(cycle_row["expected_predictions"]),
                targets=[
                    CycleTarget(
                        station_id=str(row["station_id"]),
                        target_at=_normalize_timestamp(row["target_at"]),
                        horizon_minutes=int(row["horizon_minutes"]),
                        target_order=int(row["target_order"]),
                        raw_target={},
                    )
                    for row in target_rows
                ],
                raw_cycle={},
            )
            hist_cutoff = pd.Timestamp(historical_cycle.data_cutoff).tz_convert("UTC")
            base = build_candidate_f_base_frame(historical_cycle, observations[observations["observed_at"] <= hist_cutoff].copy())
            for row in base.itertuples(index=False):
                target_at = _ts(row.target_at)
                station_id = str(row.station_id)
                if (station_id, target_at) not in pairs:
                    continue
                feedback_rows.append(
                    {
                        "station_id": station_id,
                        "target_at": target_at,
                        "base_prediction": float(row.base_prediction),
                        "actual": actual_lookup[(station_id, target_at)],
                    }
                )
        return pd.DataFrame(feedback_rows).sort_values(["station_id", "target_at"]).reset_index(drop=True)

    def _post_upsert(self, path: str, payload: Any, *, on_conflict: str) -> None:
        response = self._client.post(
            path,
            params={"on_conflict": on_conflict},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=payload,
        )
        response.raise_for_status()


def _load_artifact(model: dict[str, Any]) -> Any:
    active = model.get("is_active")
    if active is not True:
        raise CyclePredictionError("loaded model is not active")
    artifact_uri = model.get("artifact_uri")
    if not isinstance(artifact_uri, str) or not artifact_uri:
        raise CyclePredictionError("active champion missing artifact_uri")
    if artifact_uri.startswith("repo://") or Path(artifact_uri).is_dir():
        try:
            from candidate_b_shadow import MODEL_TYPE as CANDIDATE_B_MODEL_TYPE
            from candidate_b_shadow import load_shadow_artifact
            from candidate_g_adaptive import MODEL_TYPE as CANDIDATE_G_MODEL_TYPE
            from candidate_g_adaptive import load_candidate_g_artifact

            if model.get("model_type") == CANDIDATE_G_MODEL_TYPE:
                return load_candidate_g_artifact(artifact_uri)
            if model.get("model_type") == CANDIDATE_B_MODEL_TYPE:
                return load_shadow_artifact(artifact_uri)
            return load_shadow_artifact(artifact_uri)
        except Exception as exc:
            raise CyclePredictionError(f"cannot load champion artifact: {artifact_uri}") from exc
    try:
        return json.loads(Path(artifact_uri).read_text())
    except OSError as exc:
        raise CyclePredictionError(f"cannot read champion artifact: {artifact_uri}") from exc
    except json.JSONDecodeError as exc:
        raise CyclePredictionError(f"invalid champion artifact JSON: {artifact_uri}") from exc


def _build_non_baseline_predictions(
    cycle: ForecastCycle,
    model_id: str,
    artifact: Any,
    observations: pd.DataFrame,
    *,
    feedback_frame: pd.DataFrame | None = None,
) -> list[CyclePrediction]:
    try:
        from candidate_b_shadow import MODEL_TYPE as CANDIDATE_B_MODEL_TYPE
        from candidate_b_shadow import ShadowArtifact, ShadowCandidateError, build_shadow_predictions
        from candidate_g_adaptive import MODEL_TYPE as CANDIDATE_G_MODEL_TYPE
        from candidate_g_adaptive import CandidateGArtifact, build_candidate_g_predictions
    except ImportError as exc:
        raise CyclePredictionError("candidate model dependencies are unavailable") from exc
    if isinstance(artifact, CandidateGArtifact):
        if artifact.metadata.get("model_type") != CANDIDATE_G_MODEL_TYPE:
            raise CyclePredictionError(f"unsupported candidate G model_type: {artifact.metadata.get('model_type')!r}")
        try:
            traced = build_candidate_g_predictions(cycle, model_id, artifact, observations, feedback_frame=feedback_frame)
            return [
                CyclePrediction(
                    cycle_id=item.prediction.cycle_id,
                    model_id=item.prediction.model_id,
                    station_id=item.prediction.station_id,
                    target_at=item.prediction.target_at,
                    horizon_minutes=item.prediction.horizon_minutes,
                    prediction=item.prediction.prediction,
                    trace={
                        "candidate": "candidate_g_v1",
                        "base_prediction": item.base_prediction,
                        "calibration_factor": item.calibration_factor,
                        "final_prediction": item.prediction.prediction,
                        "feedback_samples": item.feedback_samples,
                        "data_cutoff": item.data_cutoff,
                    },
                )
                for item in traced
            ]
        except ShadowCandidateError as exc:
            raise CyclePredictionError(str(exc)) from exc
    if not isinstance(artifact, ShadowArtifact):
        raise CyclePredictionError(f"unsupported champion artifact type: {type(artifact).__name__}")
    if artifact.metadata.get("model_type") != CANDIDATE_B_MODEL_TYPE:
        raise CyclePredictionError(f"unsupported champion model_type: {artifact.metadata.get('model_type')!r}")
    try:
        return build_shadow_predictions(cycle, model_id, artifact, observations)
    except ShadowCandidateError as exc:
        raise CyclePredictionError(str(exc)) from exc


def _feedback_history_for_model(
    store: PredictionStore,
    cycle: ForecastCycle,
    model: dict[str, Any],
    artifact: Any,
) -> pd.DataFrame | None:
    try:
        from candidate_g_adaptive import CandidateGArtifact
    except ImportError:
        return None
    if not isinstance(artifact, CandidateGArtifact):
        return None
    reader = getattr(store, "feedback_history_until", None)
    if reader is None:
        return None
    return reader(cycle.data_cutoff, model)


def prediction_trace_payloads(predictions: list[CyclePrediction]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        trace = prediction.trace or {}
        if trace.get("candidate") != "candidate_g_v1":
            continue
        rows.append(
            {
                "cycle_id": prediction.cycle_id,
                "model_id": prediction.model_id,
                "station_id": prediction.station_id,
                "target_at": prediction.target_at,
                "horizon_minutes": prediction.horizon_minutes,
                "base_prediction": trace.get("base_prediction"),
                "calibration_factor": trace.get("calibration_factor"),
                "final_prediction": trace.get("final_prediction", prediction.prediction),
                "feedback_samples": trace.get("feedback_samples", 0),
                "data_cutoff": trace.get("data_cutoff"),
            }
        )
    return rows


def _time_slot_payloads_for_targets(cycle: ForecastCycle) -> list[dict[str, Any]]:
    by_observed_at = {
        _time_slot_payload(target.target_at)["observed_at"]: _time_slot_payload(target.target_at)
        for target in cycle.targets
    }
    return [by_observed_at[observed_at] for observed_at in sorted(by_observed_at)]


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


def _normalize_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CyclePredictionError("invalid timestamp")
    return _ts(value)


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
