from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd

from cycle_predictions import CyclePrediction, validate_predictions
from forecast_cycle_sync import ForecastCycle, PulsoForecastCycleApi
from stream_sync import _time_slot_payload


MODEL_VERSION = "candidate_b_v1"
MODEL_NAME = "candidate_b_hist_gradient_boosting_shadow"
MODEL_TYPE = "shadow_hist_gradient_boosting_candidate_b"
RANDOM_STATE = 17
BOGOTA_TZ = "America/Bogota"
HORIZONS = (15, 30, 45, 60)
SHORT_LAGS = (15, 30, 45, 60)
DAILY_LAG_MINUTES = 24 * 60
WEEKLY_LAG_MINUTES = 7 * 24 * 60
MIN_HISTORY_DAYS = 14
DEFAULT_ARTIFACT_URI = f"repo://artifacts/shadow/{MODEL_VERSION}"

FEATURE_SCHEMA = [
    "station_code",
    "horizon_minutes",
    "target_hour",
    "target_minute",
    "target_day_of_week",
    "target_is_weekend",
    "target_quarter_hour_block",
    "lag_15",
    "lag_30",
    "lag_45",
    "lag_60",
    "lag_96",
    "lag_weekly",
    "recent_mean",
    "recent_median",
    "profile_mean",
    "profile_median",
]

HYPERPARAMETERS = {
    "loss": "absolute_error",
    "learning_rate": 0.06,
    "max_iter": 160,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.05,
    "random_state": RANDOM_STATE,
}


class ShadowCandidateError(RuntimeError):
    """Raised when Candidate B shadow training or inference is unsafe."""


@dataclass(frozen=True)
class ShadowArtifact:
    metadata: dict[str, Any]
    model: Any


@dataclass(frozen=True)
class ShadowTrainingResult:
    status: str
    model_id: str | None
    version: str
    artifact_uri: str
    training_data_end: str


@dataclass(frozen=True)
class ShadowPredictionResult:
    status: str
    cycle_id: str | None = None
    data_cutoff: str | None = None
    model_id: str | None = None
    expected_predictions: int = 0
    predictions_generated: int = 0
    artifact_uri: str | None = None
    target_validation: str = "not_run"


@dataclass(frozen=True)
class ShadowCycleEvaluation:
    cycle_id: str
    champion_accuracy: float
    candidate_b_accuracy: float
    delta_pp: float
    stations_candidate_b_wins: list[str]
    stations_candidate_b_loses: list[str]
    coverage: float
    station_metrics: pd.DataFrame


class ShadowStore(Protocol):
    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        ...

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        ...

    def load_shadow_model_version(self, version: str = MODEL_VERSION) -> dict[str, Any]:
        ...

    def shadow_predictions_complete(self, cycle: ForecastCycle, model: dict[str, Any]) -> bool:
        ...

    def persist_shadow_predictions(self, cycle: ForecastCycle, model: dict[str, Any], predictions: list[CyclePrediction]) -> int:
        ...


def train_and_register_candidate_b(
    store: "SupabaseShadowStore",
    *,
    training_data_end: str,
    version: str = MODEL_VERSION,
    artifact_uri: str = DEFAULT_ARTIFACT_URI,
) -> ShadowTrainingResult:
    existing = store.find_shadow_model_version(version)
    if existing is not None:
        return ShadowTrainingResult(
            status="already_registered",
            model_id=existing["model_id"],
            version=version,
            artifact_uri=existing["artifact_uri"],
            training_data_end=_normalize_timestamp(str(existing["trained_until"])),
        )
    observations = store.observations_until(training_data_end)
    artifact = train_candidate_b_artifact(observations, training_data_end=training_data_end, version=version, artifact_uri=artifact_uri)
    save_shadow_artifact(artifact, artifact_uri=artifact_uri)
    registered = store.register_shadow_model_version(artifact.metadata, artifact_uri)
    return ShadowTrainingResult(
        status="registered",
        model_id=registered["model_id"],
        version=version,
        artifact_uri=artifact_uri,
        training_data_end=artifact.metadata["training_data_end"],
    )


def run_shadow_cycle(api: PulsoForecastCycleApi, store: ShadowStore, *, version: str = MODEL_VERSION) -> ShadowPredictionResult:
    api.validate_identity()
    cycle = api.current_cycle()
    if cycle is None:
        return ShadowPredictionResult(status="no_open_cycle")

    model = store.load_shadow_model_version(version)
    if store.shadow_predictions_complete(cycle, model):
        return ShadowPredictionResult(
            status="already_shadow_predicted",
            cycle_id=cycle.cycle_id,
            data_cutoff=cycle.data_cutoff,
            model_id=model["model_id"],
            expected_predictions=cycle.expected_predictions,
            predictions_generated=cycle.expected_predictions,
            artifact_uri=model.get("artifact_uri"),
            target_validation="exact",
        )
    artifact = load_shadow_artifact(model["artifact_uri"])
    observations = store.observations_until(cycle.data_cutoff)
    predictions = build_shadow_predictions(cycle, model["model_id"], artifact, observations)
    validate_predictions(cycle, predictions)
    store.persist_cycle_targets(cycle)
    written = store.persist_shadow_predictions(cycle, model, predictions)
    return ShadowPredictionResult(
        status="shadow_predicted",
        cycle_id=cycle.cycle_id,
        data_cutoff=cycle.data_cutoff,
        model_id=model["model_id"],
        expected_predictions=cycle.expected_predictions,
        predictions_generated=written,
        artifact_uri=model.get("artifact_uri"),
        target_validation="exact",
    )


def train_candidate_b_artifact(
    observations: pd.DataFrame,
    *,
    training_data_end: str,
    version: str = MODEL_VERSION,
    artifact_uri: str = DEFAULT_ARTIFACT_URI,
) -> ShadowArtifact:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        import sklearn
    except ImportError as exc:
        raise ShadowCandidateError("scikit-learn is required for Candidate B shadow mode") from exc

    frame = _prepare_observations(observations)
    cutoff = pd.Timestamp(training_data_end).tz_convert("UTC")
    eligible = frame[frame["observed_at"] <= cutoff].copy()
    if eligible.empty:
        raise ShadowCandidateError("no observations at or before training_data_end")
    station_mapping = {station_id: idx for idx, station_id in enumerate(sorted(eligible["station_id"].unique()))}
    samples = _training_samples(eligible, cutoff)
    if samples.empty:
        raise ShadowCandidateError("no supervised training samples available for Candidate B")
    model = HistGradientBoostingRegressor(**HYPERPARAMETERS)
    model.fit(_feature_matrix(samples, station_mapping), samples["actual"].to_numpy())
    metadata = {
        "version": version,
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "trained_at": _now_iso(),
        "training_data_end": _ts(cutoff),
        "artifact_uri": artifact_uri,
        "feature_schema": FEATURE_SCHEMA,
        "hyperparameters": HYPERPARAMETERS,
        "station_mapping": station_mapping,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(samples)),
        "training_cutoff_start": _ts(samples["data_cutoff"].min()),
        "training_cutoff_end": _ts(samples["data_cutoff"].max()),
        "sklearn_version": sklearn.__version__,
        "git_commit": _git_commit(),
    }
    return ShadowArtifact(metadata=metadata, model=model)


def save_shadow_artifact(artifact: ShadowArtifact, *, artifact_uri: str = DEFAULT_ARTIFACT_URI) -> str:
    artifact_dir = _resolve_artifact_uri(artifact_uri)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(artifact.metadata, indent=2, sort_keys=True) + "\n")
    with (artifact_dir / "model.pkl").open("wb") as handle:
        pickle.dump(artifact.model, handle)
    return artifact_uri


def load_shadow_artifact(artifact_uri: str | Path) -> ShadowArtifact:
    artifact_dir = _resolve_artifact_uri(str(artifact_uri))
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    with (artifact_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    _validate_artifact_metadata(metadata)
    return ShadowArtifact(metadata=metadata, model=model)


def build_shadow_predictions(
    cycle: ForecastCycle,
    model_id: str,
    artifact: ShadowArtifact,
    observations: pd.DataFrame,
) -> list[CyclePrediction]:
    _validate_artifact_metadata(artifact.metadata)
    frame = _prepare_observations(observations)
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    if (frame["observed_at"] > cutoff).any():
        raise ShadowCandidateError("observations contain rows after data_cutoff")
    training_data_end = pd.Timestamp(artifact.metadata["training_data_end"]).tz_convert("UTC")
    if training_data_end > cutoff:
        raise ShadowCandidateError("shadow artifact training_data_end is after cycle data_cutoff")

    feature_rows = _inference_samples(cycle, frame, artifact.metadata["station_mapping"])
    values = np.maximum(0.0, artifact.model.predict(feature_rows[FEATURE_SCHEMA].astype(float)))
    predictions: list[CyclePrediction] = []
    for row, value in zip(feature_rows.itertuples(index=False), values, strict=True):
        if not math.isfinite(float(value)):
            raise ShadowCandidateError("prediction value is not finite")
        predictions.append(
            CyclePrediction(
                cycle_id=cycle.cycle_id,
                model_id=model_id,
                station_id=str(row.station_id),
                target_at=row.target_at,
                horizon_minutes=int(row.horizon_minutes),
                prediction=float(value),
            )
        )
    return predictions


class SupabaseShadowStore:
    def __init__(self, supabase_url: str | None = None, service_key: str | None = None, timeout: float = 30.0) -> None:
        resolved_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        resolved_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not resolved_url or not resolved_key:
            raise ShadowCandidateError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{resolved_url}/rest/v1",
            headers={"apikey": resolved_key, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupabaseShadowStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        cutoff = pd.Timestamp(data_cutoff).tz_convert("UTC")
        rows: list[dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while True:
            response = self._client.get(
                "/observations",
                params={
                    "observed_at": f"lte.{_ts(cutoff)}",
                    "select": "station_id,observed_at,demand",
                    "order": "observed_at.asc,station_id.asc",
                    "limit": str(page_size),
                    "offset": str(offset),
                },
            )
            response.raise_for_status()
            page = response.json()
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return pd.DataFrame(rows)

    def find_shadow_model_version(self, version: str = MODEL_VERSION) -> dict[str, Any] | None:
        response = self._client.get(
            "/model_versions",
            params={
                "model_type": f"eq.{MODEL_TYPE}",
                "is_active": "eq.false",
                "select": "*,pipeline_runs(started_at,git_commit)",
                "order": "trained_until.desc",
            },
        )
        response.raise_for_status()
        for row in response.json():
            feature_config = row.get("feature_config") if isinstance(row.get("feature_config"), dict) else {}
            if feature_config.get("version") == version:
                return row
        return None

    def load_shadow_model_version(self, version: str = MODEL_VERSION) -> dict[str, Any]:
        model = self.find_shadow_model_version(version)
        if model is None:
            raise ShadowCandidateError(f"shadow model version is not registered: {version}")
        if model.get("is_active") is not False:
            raise ShadowCandidateError("shadow model must not be active")
        return model

    def register_shadow_model_version(self, artifact: dict[str, Any], artifact_uri: str) -> dict[str, Any]:
        existing = self.find_shadow_model_version(str(artifact["version"]))
        if existing is not None:
            return existing
        run_id = str(uuid4())
        model_id = str(uuid4())
        self._post(
            "/pipeline_runs",
            {
                "run_id": run_id,
                "status": "running",
                "git_commit": artifact.get("git_commit"),
                "data_cutoff": artifact["training_data_end"],
                "trigger_type": "shadow_training",
            },
        )
        self._post(
            "/model_versions",
            {
                "model_id": model_id,
                "run_id": run_id,
                "model_name": MODEL_NAME,
                "model_type": MODEL_TYPE,
                "artifact_uri": artifact_uri,
                "trained_until": artifact["training_data_end"],
                "feature_config": {
                    "mode": "shadow",
                    "version": artifact["version"],
                    "trained_at": artifact["trained_at"],
                    "training_data_end": artifact["training_data_end"],
                    "feature_schema": artifact["feature_schema"],
                    "station_mapping": artifact["station_mapping"],
                    "artifact_storage": "repo",
                },
                "params": {
                    "hyperparameters": artifact["hyperparameters"],
                    "random_state": artifact["random_state"],
                    "training_rows": artifact["training_rows"],
                    "training_cutoff_start": artifact["training_cutoff_start"],
                    "training_cutoff_end": artifact["training_cutoff_end"],
                    "sklearn_version": artifact.get("sklearn_version"),
                },
                "is_active": False,
            },
        )
        self._patch("/pipeline_runs", {"status": "success", "finished_at": _now_iso()}, params={"run_id": f"eq.{run_id}"})
        return {"run_id": run_id, "model_id": model_id, "is_active": False, "artifact_uri": artifact_uri}

    def shadow_predictions_complete(self, cycle: ForecastCycle, model: dict[str, Any]) -> bool:
        response = self._client.get(
            "/predictions",
            params={
                "cycle_id": f"eq.{cycle.cycle_id}",
                "model_id": f"eq.{model['model_id']}",
                "select": "station_id,target_observed_at",
                "order": "station_id.asc,target_observed_at.asc",
            },
        )
        response.raise_for_status()
        rows = response.json()
        actual = {(row["station_id"], _normalize_timestamp(row["target_observed_at"])) for row in rows}
        expected = {(target.station_id, target.target_at) for target in cycle.targets}
        return actual == expected and len(rows) == cycle.expected_predictions

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
            self._post_upsert("/time_slots", _time_slot_payloads_for_targets(cycle), on_conflict="observed_at")
            self._post_upsert(
                "/forecast_cycle_targets",
                [
                    {
                        "cycle_id": target_cycle.cycle_id,
                        "station_id": target.station_id,
                        "target_at": target.target_at,
                        "horizon_minutes": target.horizon_minutes,
                        "target_order": target.target_order,
                        "raw_target": target.raw_target,
                    }
                    for target_cycle in [cycle]
                    for target in target_cycle.targets
                ],
                on_conflict="cycle_id,station_id,target_at",
            )

    def persist_shadow_predictions(self, cycle: ForecastCycle, model: dict[str, Any], predictions: list[CyclePrediction]) -> int:
        model_id = model["model_id"]
        run_id = model["run_id"]
        delete_response = self._client.delete("/predictions", params={"cycle_id": f"eq.{cycle.cycle_id}", "model_id": f"eq.{model_id}"})
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

    def evaluate_shadow_cycle(self, cycle_id: str, *, version: str = MODEL_VERSION) -> ShadowCycleEvaluation | None:
        model = self.load_shadow_model_version(version)
        official_model_id = self._official_model_id_for_cycle(cycle_id)
        if official_model_id is None:
            return None
        response = self._client.get(
            "/predictions",
            params={
                "cycle_id": f"eq.{cycle_id}",
                "select": "model_id,station_id,target_observed_at,prediction",
                "order": "station_id.asc,target_observed_at.asc",
            },
        )
        response.raise_for_status()
        rows = response.json()
        champion_rows = [row for row in rows if row.get("model_id") == official_model_id]
        shadow_rows = [row for row in rows if row.get("model_id") == model["model_id"]]
        if not champion_rows or not shadow_rows:
            return None
        actuals = self._actuals_for_predictions(rows)
        champion_scored = _score_rows(champion_rows, actuals)
        shadow_scored = _score_rows(shadow_rows, actuals)
        if champion_scored.empty or shadow_scored.empty:
            return None
        champion_metrics = _station_metrics(champion_scored).rename(columns={"wape": "champion_wape", "accuracy": "champion_accuracy"})
        shadow_metrics = _station_metrics(shadow_scored).rename(columns={"wape": "candidate_b_wape", "accuracy": "candidate_b_accuracy"})
        station_metrics = champion_metrics.merge(shadow_metrics, on="station_id", how="inner")
        station_metrics["delta_pp"] = station_metrics["candidate_b_accuracy"] - station_metrics["champion_accuracy"]
        champion_accuracy = float(station_metrics["champion_accuracy"].mean())
        candidate_accuracy = float(station_metrics["candidate_b_accuracy"].mean())
        expected_pairs = {(row["station_id"], _normalize_timestamp(row["target_observed_at"])) for row in shadow_rows}
        actual_pairs = set(actuals)
        coverage = len(expected_pairs & actual_pairs) / len(expected_pairs) if expected_pairs else 0.0
        return ShadowCycleEvaluation(
            cycle_id=cycle_id,
            champion_accuracy=champion_accuracy,
            candidate_b_accuracy=candidate_accuracy,
            delta_pp=candidate_accuracy - champion_accuracy,
            stations_candidate_b_wins=station_metrics.loc[station_metrics["delta_pp"] > 0, "station_id"].tolist(),
            stations_candidate_b_loses=station_metrics.loc[station_metrics["delta_pp"] < 0, "station_id"].tolist(),
            coverage=coverage,
            station_metrics=station_metrics.sort_values("delta_pp"),
        )

    def shadow_cycle_evaluations(self, *, limit: int = 20, version: str = MODEL_VERSION) -> list[ShadowCycleEvaluation]:
        cycle_ids = self._cycle_ids_with_shadow_predictions(limit=limit, version=version)
        return [evaluation for cycle_id in cycle_ids if (evaluation := self.evaluate_shadow_cycle(cycle_id, version=version)) is not None]

    def _cycle_ids_with_shadow_predictions(self, *, limit: int, version: str) -> list[str]:
        model = self.load_shadow_model_version(version)
        response = self._client.get(
            "/predictions",
            params={"model_id": f"eq.{model['model_id']}", "select": "cycle_id,created_at", "order": "created_at.desc", "limit": str(limit * 100)},
        )
        response.raise_for_status()
        seen: list[str] = []
        for row in response.json():
            cycle_id = row.get("cycle_id")
            if isinstance(cycle_id, str) and cycle_id not in seen:
                seen.append(cycle_id)
            if len(seen) >= limit:
                break
        return seen

    def _official_model_id_for_cycle(self, cycle_id: str) -> str | None:
        response = self._client.get(
            "/submission_receipts",
            params={"cycle_id": f"eq.{cycle_id}", "is_official": "eq.true", "select": "model_id,created_at", "order": "created_at.desc", "limit": "1"},
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None
        model_id = rows[0].get("model_id")
        return model_id if isinstance(model_id, str) and model_id else None

    def _actuals_for_predictions(self, prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
        pairs = sorted({(row["station_id"], _normalize_timestamp(row["target_observed_at"])) for row in prediction_rows})
        actuals: dict[tuple[str, str], float] = {}
        for station_id, target_at in pairs:
            response = self._client.get(
                "/observations",
                params={"station_id": f"eq.{station_id}", "observed_at": f"eq.{target_at}", "select": "station_id,observed_at,demand", "limit": "1"},
            )
            response.raise_for_status()
            rows = response.json()
            if rows:
                actuals[(station_id, _normalize_timestamp(rows[0]["observed_at"]))] = float(rows[0]["demand"])
        return actuals

    def _post(self, path: str, payload: Any) -> None:
        response = self._client.post(path, json=payload)
        response.raise_for_status()

    def _patch(self, path: str, payload: dict[str, Any], *, params: dict[str, str]) -> None:
        response = self._client.patch(path, params=params, json=payload)
        response.raise_for_status()

    def _post_upsert(self, path: str, payload: Any, *, on_conflict: str) -> None:
        response = self._client.post(path, params={"on_conflict": on_conflict}, headers={"Prefer": "resolution=merge-duplicates"}, json=payload)
        response.raise_for_status()


def _training_samples(observations: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    train_start = observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    samples = _supervised_samples(observations)
    samples = samples[(samples["target_at"] <= cutoff) & (samples["data_cutoff"] >= train_start)].copy()
    return samples.dropna(subset=["profile_mean"])


def _supervised_samples(observations: pd.DataFrame) -> pd.DataFrame:
    base = observations.sort_values(["station_id", "observed_at"]).copy()
    base = base.rename(columns={"observed_at": "target_at", "demand": "actual"})
    by_station = base.groupby("station_id", sort=False)
    for lag in SHORT_LAGS:
        base[f"lag_{lag}"] = by_station["actual"].shift(lag // 15)
    base["lag_96"] = by_station["actual"].shift(DAILY_LAG_MINUTES // 15)
    base["lag_weekly"] = by_station["actual"].shift(WEEKLY_LAG_MINUTES // 15)
    by_slot = base.groupby(["station_id", "quarter_hour_block"], sort=False)["actual"]
    base["profile_mean"] = by_slot.transform(lambda values: values.shift(1).expanding().mean())
    base["profile_median"] = by_slot.transform(lambda values: values.shift(1).expanding().median())
    base = base.rename(columns=_TIME_FEATURE_RENAMES)
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        part = base.copy()
        part["horizon_minutes"] = horizon
        part["data_cutoff"] = part["target_at"] - pd.Timedelta(minutes=horizon)
        for lag in SHORT_LAGS:
            if lag < horizon:
                part[f"lag_{lag}"] = np.nan
        _add_recent_features(part)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    return result[_SAMPLE_COLUMNS].sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)


def _inference_samples(cycle: ForecastCycle, observations: pd.DataFrame, station_mapping: dict[str, int]) -> pd.DataFrame:
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    lookup = {(str(row.station_id), row.observed_at): float(row.demand) for row in observations[["station_id", "observed_at", "demand"]].itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for target in cycle.targets:
        if target.station_id not in station_mapping:
            raise ShadowCandidateError(f"station_id missing from station_mapping: {target.station_id}")
        target_at = pd.Timestamp(target.target_at).tz_convert("UTC")
        row: dict[str, Any] = {
            "station_id": target.station_id,
            "target_at": _ts(target_at),
            "target_order": target.target_order,
            "horizon_minutes": target.horizon_minutes,
            "station_code": station_mapping[target.station_id],
            **_target_time_features(target_at),
        }
        for lag in SHORT_LAGS:
            source_at = target_at - pd.Timedelta(minutes=lag)
            row[f"lag_{lag}"] = lookup.get((target.station_id, source_at), np.nan) if source_at <= cutoff else np.nan
        row["lag_96"] = lookup.get((target.station_id, target_at - pd.Timedelta(minutes=DAILY_LAG_MINUTES)), np.nan)
        row["lag_weekly"] = lookup.get((target.station_id, target_at - pd.Timedelta(minutes=WEEKLY_LAG_MINUTES)), np.nan)
        lag_values = [row[f"lag_{lag}"] for lag in SHORT_LAGS if not pd.isna(row[f"lag_{lag}"])]
        row["recent_mean"] = float(np.mean(lag_values)) if lag_values else np.nan
        row["recent_median"] = float(np.median(lag_values)) if lag_values else np.nan
        profile = observations[
            (observations["station_id"] == target.station_id)
            & (observations["quarter_hour_block"] == row["target_quarter_hour_block"])
            & (observations["observed_at"] <= cutoff)
        ]["demand"]
        row["profile_mean"] = float(profile.mean()) if not profile.empty else np.nan
        row["profile_median"] = float(profile.median()) if not profile.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("target_order").reset_index(drop=True)


def _feature_matrix(samples: pd.DataFrame, station_mapping: dict[str, int]) -> pd.DataFrame:
    frame = samples.copy()
    frame["station_code"] = frame["station_id"].map(station_mapping).astype(float)
    if frame["station_code"].isna().any():
        missing = sorted(frame.loc[frame["station_code"].isna(), "station_id"].unique())
        raise ShadowCandidateError(f"station_id missing from station_mapping: {missing}")
    return frame[FEATURE_SCHEMA].astype(float)


def _score_rows(rows: list[dict[str, Any]], actuals: dict[tuple[str, str], float]) -> pd.DataFrame:
    scored = []
    for row in rows:
        key = (row["station_id"], _normalize_timestamp(row["target_observed_at"]))
        if key in actuals:
            scored.append({"station_id": row["station_id"], "actual": actuals[key], "prediction": float(row["prediction"])})
    return pd.DataFrame(scored)


def _station_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.assign(abs_error=(scored["actual"] - scored["prediction"]).abs())
    grouped = frame.groupby("station_id", as_index=False).agg(abs_error=("abs_error", "sum"), actual_sum=("actual", "sum"))
    grouped["wape"] = np.where(grouped["actual_sum"] == 0, 0.0, grouped["abs_error"] / grouped["actual_sum"])
    grouped["accuracy"] = 100.0 * (1.0 - grouped["wape"]).clip(lower=0.0)
    return grouped[["station_id", "wape", "accuracy"]]


def _prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "observed_at", "demand"}
    missing = required - set(observations.columns)
    if missing:
        raise ShadowCandidateError(f"missing observation columns: {sorted(missing)}")
    frame = observations.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = frame["demand"].astype(float)
    local = frame["observed_at"].dt.tz_convert(BOGOTA_TZ)
    frame["hour"] = local.dt.hour
    frame["minute"] = local.dt.minute
    frame["day_of_week"] = local.dt.dayofweek
    frame["is_weekend"] = local.dt.dayofweek.isin([5, 6]).astype(int)
    frame["quarter_hour_block"] = frame["hour"] * 4 + frame["minute"] // 15
    return frame.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def _target_time_features(target_at: pd.Timestamp) -> dict[str, int]:
    local = target_at.tz_convert(BOGOTA_TZ)
    return {
        "target_hour": int(local.hour),
        "target_minute": int(local.minute),
        "target_day_of_week": int(local.dayofweek),
        "target_is_weekend": int(local.dayofweek in (5, 6)),
        "target_quarter_hour_block": int(local.hour * 4 + local.minute // 15),
    }


def _add_recent_features(frame: pd.DataFrame) -> None:
    lag_columns = [f"lag_{lag}" for lag in SHORT_LAGS]
    frame["recent_mean"] = frame[lag_columns].mean(axis=1, skipna=True)
    frame["recent_median"] = frame[lag_columns].median(axis=1, skipna=True)


def _validate_artifact_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("model_type") != MODEL_TYPE:
        raise ShadowCandidateError(f"unsupported shadow model_type: {metadata.get('model_type')!r}")
    if metadata.get("version") != MODEL_VERSION:
        raise ShadowCandidateError("shadow artifact version mismatch")
    if metadata.get("feature_schema") != FEATURE_SCHEMA:
        raise ShadowCandidateError("shadow artifact feature schema mismatch")
    if metadata.get("random_state") != RANDOM_STATE:
        raise ShadowCandidateError("shadow artifact random_state mismatch")
    if not isinstance(metadata.get("station_mapping"), dict) or not metadata["station_mapping"]:
        raise ShadowCandidateError("shadow artifact missing station_mapping")


def _time_slot_payloads_for_targets(cycle: ForecastCycle) -> list[dict[str, Any]]:
    by_observed_at = {_time_slot_payload(target.target_at)["observed_at"]: _time_slot_payload(target.target_at) for target in cycle.targets}
    return [by_observed_at[observed_at] for observed_at in sorted(by_observed_at)]


def _resolve_artifact_uri(artifact_uri: str) -> Path:
    if artifact_uri.startswith("repo://"):
        return Path(__file__).resolve().parents[1] / artifact_uri.removeprefix("repo://")
    return Path(artifact_uri)


def _normalize_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ShadowCandidateError("invalid timestamp")
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


_TIME_FEATURE_RENAMES = {
    "hour": "target_hour",
    "minute": "target_minute",
    "day_of_week": "target_day_of_week",
    "is_weekend": "target_is_weekend",
    "quarter_hour_block": "target_quarter_hour_block",
}

_SAMPLE_COLUMNS = [
    "station_id",
    "data_cutoff",
    "target_at",
    "horizon_minutes",
    "actual",
    "target_hour",
    "target_minute",
    "target_day_of_week",
    "target_is_weekend",
    "target_quarter_hour_block",
    "lag_15",
    "lag_30",
    "lag_45",
    "lag_60",
    "lag_96",
    "lag_weekly",
    "recent_mean",
    "recent_median",
    "profile_mean",
    "profile_median",
]
